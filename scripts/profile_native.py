#!/usr/bin/env python3
"""Build a separately instrumented engine and report disjoint stage wall times and counters.

Instrumentation never changes the production source. The profiled stages are
mutually exclusive (disjoint) within step(), summing to whole_step:
- input_engram_non_proj: token embedding gather, lane replication, engram hash/conv (non-proj), rope setup
- engram_projections: engram linear projections (ti+1, ti+2)
- qkvg_projections: fused or individual attention projections (q, k, v, gate)
- attention_core: QK norm, RoPE application, KV store, GQA attention, softmax, V accumulation, sigmoid gate
- attn_out_and_mlp: attention output projection, post-attention RMS, Hadamard MLP
- mhc_routing_and_mixing: mHC normalization, linear projections, Sinkhorn routing, residual mixing
- lm_head: final RMS and LM head projection
- overhead_remaining: residual difference (whole_step - sum(stages))
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--tokens', type=int, default=64)
    p.add_argument('--matmul', choices=['fp32', 'sdot'], default='fp32')
    p.add_argument('--kv-cache', choices=['fp32', 'int8'], default='fp32')
    p.add_argument('--output', default='reports/native_profile.json')
    a = p.parse_args()

    dest = ROOT / 'artifacts/profiler'
    dest.mkdir(parents=True, exist_ok=True)

    cq = (ROOT / 'needle2/csrc/cq.cpp').read_text()
    eng = (ROOT / 'needle2/csrc/engine.cpp').read_text()

    profiler_header = '''#include <chrono>
#include <cstdint>

enum ProfileStage {
    PROF_INPUT_ENGRAM_RAW = 0,
    PROF_ENGRAM_PROJ = 1,
    PROF_QKVG_PROJ = 2,
    PROF_ATTENTION_CORE = 3,
    PROF_ATTN_OUT_AND_MLP = 4,
    PROF_MHC = 5,
    PROF_LM_HEAD = 6,
    PROF_WHOLE_STEP = 7,
    PROF_COUNT = 8
};

static double profile_ms[PROF_COUNT] = {};
static int64_t profile_counters[8] = {};

struct ProfileScope {
    int id;
    std::chrono::steady_clock::time_point t;
    ProfileScope(int i) : id(i), t(std::chrono::steady_clock::now()) {}
    ~ProfileScope() {
        profile_ms[id] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
    }
};

extern "C" void needle2_profile_get(double *times, int64_t *counters) {
    for (int i = 0; i < PROF_COUNT; ++i) times[i] = profile_ms[i];
    for (int i = 0; i < 8; ++i) counters[i] = profile_counters[i];
}

extern "C" void needle2_profile_reset() {
    for (auto &x : profile_ms) x = 0;
    for (auto &c : profile_counters) c = 0;
}
'''

    # Instrument engrams() to isolate engram linear projections
    engrams_marker = '            activation_quant(e.data(),D,abits);\n            linear(ti+1,e.data(),ek.data()+s*D);\n            linear(ti+2,e.data(),rawv.data());'
    engrams_repl = '''            activation_quant(e.data(),D,abits);
            {
                ProfileScope s_ep(PROF_ENGRAM_PROJ);
                linear(ti+1,e.data(),ek.data()+s*D);
                linear(ti+2,e.data(),rawv.data());
            }'''
    if eng.count(engrams_marker) != 1:
        raise ValueError('Could not find exact engrams marker in engine.cpp')
    eng = eng.replace(engrams_marker, engrams_repl)

    # Instrument step() with disjoint scopes
    step_marker = '''    void step(int token,float*out,float*hidden_out) {
        int D=c.dim,N=c.lanes,H=c.heads,KV=c.kvheads,HD=c.head_dim,A=H*HD,K=KV*HD;
        if(token<0||token>=c.vocab)throw std::runtime_error("token outside vocabulary");
        if(!c.window&&position>=capacity)throw std::runtime_error("maximum context reached");
        history_ring[position%history_cap]=token;
        row(0,token,z.data());
        for(int n=0;n<N;++n)for(int d=0;d<D;++d)x[n*D+d]=z[d]*std::sqrt(float(D));
        engrams();
        for(int j=0;j<HD/2;++j){float a=position/rope_divisor[j];rope_cos[j]=std::cos(a);rope_sin[j]=std::sin(a);}
        int mh=1+14*c.layers;
        for(int l=0;l<c.layers;++l) {
            int ti=1+14*l;
            rms(x.data(),nx.data(),N*D);
            linear(mh+6,nx.data(),hpre.data(),l*N,N);
            linear(mh+7,nx.data(),hpost.data(),l*N,N);
            linear(mh+8,nx.data(),hres.data(),l*N*N,N*N);
            for(int n=0;n<N;++n) {
                hpre[n]=sigmoid(dense(mh)[l]*hpre[n]+dense(mh+3)[l*N+n]+(n==l%N?4:-4));
                hpost[n]=2*sigmoid(dense(mh+1)[l]*hpost[n]+dense(mh+4)[l*N+n]+(n==l%N?0:-4));
            }
            for(int d=0;d<D;++d){float sum=0;for(int n=0;n<N;++n)sum+=hpre[n]*x[n*D+d];u[d]=bx[d]=sum;}
            for(int s=0;s<c.num_sites;++s)if(c.sites[s]==l) {
                rms(u.data(),z.data(),D);rms(ek.data()+s*D,proj.data(),D);
                float alpha=sigmoid(dot_f32(z.data(),proj.data(),D)/std::sqrt(float(D)));
                for(int d=0;d<D;++d)bx[d]+=alpha*ev[s*D+d];
            }
            rms(bx.data(),z.data(),D,dense(ti));activation_quant(z.data(),D,abits);
            attention_projections(ti,z.data());
            for(int h=0;h<H;++h)rms(q.data()+h*HD,q.data()+h*HD,HD,dense(ti+4));
            for(int h=0;h<KV;++h)rms(k.data()+h*HD,k.data()+h*HD,HD,dense(ti+5));
            for(int j=0;j<HD/2;++j) {
                float co=rope_cos[j],si=rope_sin[j];
                for(int h=0;h<H;++h) {float a=q[h*HD+j],b=q[h*HD+j+HD/2];q[h*HD+j]=a*co-b*si;q[h*HD+j+HD/2]=b*co+a*si;}
                for(int h=0;h<KV;++h){float a=k[h*HD+j],b=k[h*HD+j+HD/2];k[h*HD+j]=a*co-b*si;k[h*HD+j+HD/2]=b*co+a*si;}
            }
            int slot=cache_slot(position);
            if(!int8_kv_enabled) {
                auto *kc=keys.data()+size_t(l)*capacity*K,*vc=values.data()+size_t(l)*capacity*K;
                std::copy(k.begin(),k.end(),kc+slot*K);std::copy(v.begin(),v.end(),vc+slot*K);
            } else {
                auto *kc_i8=keys_i8.data()+size_t(l)*capacity*K,*vc_i8=values_i8.data()+size_t(l)*capacity*K;
                auto *ks=k_scales.data()+size_t(l)*capacity*KV,*vs=v_scales.data()+size_t(l)*capacity*KV;
                for(int kh=0;kh<KV;++kh) {
                    float max_k=0,max_v=0;
                    const float*kp=k.data()+kh*HD,*vp=v.data()+kh*HD;
                    for(int d=0;d<HD;++d){max_k=std::max(max_k,std::abs(kp[d]));max_v=std::max(max_v,std::abs(vp[d]));}
                    float scale_k=max_k>0?max_k/127.0f:1.0f,scale_v=max_v>0?max_v/127.0f:1.0f;
                    float inv_k=max_k>0?127.0f/max_k:0.0f,inv_v=max_v>0?127.0f/max_v:0.0f;
                    ks[slot*KV+kh]=scale_k;vs[slot*KV+kh]=scale_v;
                    int8_t*k_dst=kc_i8+slot*K+kh*HD,*v_dst=vc_i8+slot*K+kh*HD;
                    for(int d=0;d<HD;++d){
                        k_dst[d]=int8_t(std::max(-127.0f,std::min(127.0f,std::nearbyint(kp[d]*inv_k))));
                        v_dst[d]=int8_t(std::max(-127.0f,std::min(127.0f,std::nearbyint(vp[d]*inv_v))));
                    }
                }
            }
            compute_attention(l, position, q.data(), att.data());
            sigmoid_gate(att.data(),gate.data(),A);
            activation_quant(att.data(),A,abits);linear(ti+7,att.data(),proj.data());
            rms(proj.data(),proj.data(),D,dense(ti+8));float ag=sigmoid(dense(ti+9)[0]);
            for(int d=0;d<D;++d)bx[d]+=ag*proj[d];
            rms(bx.data(),z.data(),D,dense(ti+10));
            for(int d=0;d<c.hada;++d)mlp[d]=d<D?z[d]*dense(ti+11)[d]:0;
            hadamard(mlp.data(),c.hada);
            silu_diagonal(mlp.data(),dense(ti+12),c.hada);
            hadamard(mlp.data(),c.hada);
            for(int d=0;d<D;++d)bx[d]=bx[d]+dense(ti+13)[d]*mlp[d]-u[d];
            for(int ij=0;ij<N*N;++ij)hres[ij]=hres[ij]*dense(mh+2)[l]+dense(mh+5)[l*N*N+ij];
            sinkhorn(hres.data(),N);
            for(int n=0;n<N;++n)for(int d=0;d<D;++d){float val=hpost[n]*bx[d];for(int j=0;j<N;++j)val+=hres[n*N+j]*x[j*D+d];newx[n*D+d]=val;}
            x.swap(newx);
            for(int d=0;d<D;++d){float sum=0;for(int n=0;n<N;++n)sum+=x[n*D+d];hidden[l*D+d]=sum/N;}
        }
        std::copy(hidden.end()-D,hidden.end(),z.begin());
        rms(z.data(),z.data(),D,dense(1+14*c.layers+9+4*c.num_sites));
        if(out){activation_quant(z.data(),D,abits);linear(0,z.data(),out);}'''

    step_repl = '''    void step(int token,float*out,float*hidden_out) {
        ProfileScope s_step(PROF_WHOLE_STEP);
        profile_counters[1]++;
        profile_counters[2] = c.window ? std::min(prefix,position+1)+std::min(c.window,std::max(0,position+1-prefix)) : position+1;
        int D=c.dim,N=c.lanes,H=c.heads,KV=c.kvheads,HD=c.head_dim,A=H*HD,K=KV*HD;
        if(token<0||token>=c.vocab)throw std::runtime_error("token outside vocabulary");
        if(!c.window&&position>=capacity)throw std::runtime_error("maximum context reached");
        {
            ProfileScope s_in(PROF_INPUT_ENGRAM_RAW);
            history_ring[position%history_cap]=token;
            row(0,token,z.data());
            for(int n=0;n<N;++n)for(int d=0;d<D;++d)x[n*D+d]=z[d]*std::sqrt(float(D));
            engrams();
            for(int j=0;j<HD/2;++j){float a=position/rope_divisor[j];rope_cos[j]=std::cos(a);rope_sin[j]=std::sin(a);}
        }
        int mh=1+14*c.layers;
        for(int l=0;l<c.layers;++l) {
            int ti=1+14*l;
            {
                ProfileScope s_mhc(PROF_MHC);
                rms(x.data(),nx.data(),N*D);
                linear(mh+6,nx.data(),hpre.data(),l*N,N);
                linear(mh+7,nx.data(),hpost.data(),l*N,N);
                linear(mh+8,nx.data(),hres.data(),l*N*N,N*N);
                for(int n=0;n<N;++n) {
                    hpre[n]=sigmoid(dense(mh)[l]*hpre[n]+dense(mh+3)[l*N+n]+(n==l%N?4:-4));
                    hpost[n]=2*sigmoid(dense(mh+1)[l]*hpost[n]+dense(mh+4)[l*N+n]+(n==l%N?0:-4));
                }
                for(int d=0;d<D;++d){float sum=0;for(int n=0;n<N;++n)sum+=hpre[n]*x[n*D+d];u[d]=bx[d]=sum;}
                for(int s=0;s<c.num_sites;++s)if(c.sites[s]==l) {
                    rms(u.data(),z.data(),D);rms(ek.data()+s*D,proj.data(),D);
                    float alpha=sigmoid(dot_f32(z.data(),proj.data(),D)/std::sqrt(float(D)));
                    for(int d=0;d<D;++d)bx[d]+=alpha*ev[s*D+d];
                }
                rms(bx.data(),z.data(),D,dense(ti));activation_quant(z.data(),D,abits);
            }
            {
                ProfileScope s_qkvg(PROF_QKVG_PROJ);
                attention_projections(ti,z.data());
            }
            {
                ProfileScope s_att(PROF_ATTENTION_CORE);
                for(int h=0;h<H;++h)rms(q.data()+h*HD,q.data()+h*HD,HD,dense(ti+4));
                for(int h=0;h<KV;++h)rms(k.data()+h*HD,k.data()+h*HD,HD,dense(ti+5));
                for(int j=0;j<HD/2;++j) {
                    float co=rope_cos[j],si=rope_sin[j];
                    for(int h=0;h<H;++h) {float a=q[h*HD+j],b=q[h*HD+j+HD/2];q[h*HD+j]=a*co-b*si;q[h*HD+j+HD/2]=b*co+a*si;}
                    for(int h=0;h<KV;++h){float a=k[h*HD+j],b=k[h*HD+j+HD/2];k[h*HD+j]=a*co-b*si;k[h*HD+j+HD/2]=b*co+a*si;}
                }
                int slot=cache_slot(position);
                if(!int8_kv_enabled) {
                    auto *kc=keys.data()+size_t(l)*capacity*K,*vc=values.data()+size_t(l)*capacity*K;
                    std::copy(k.begin(),k.end(),kc+slot*K);std::copy(v.begin(),v.end(),vc+slot*K);
                } else {
                    auto *kc_i8=keys_i8.data()+size_t(l)*capacity*K,*vc_i8=values_i8.data()+size_t(l)*capacity*K;
                    auto *ks=k_scales.data()+size_t(l)*capacity*KV,*vs=v_scales.data()+size_t(l)*capacity*KV;
                    for(int kh=0;kh<KV;++kh) {
                        float max_k=0,max_v=0;
                        const float*kp=k.data()+kh*HD,*vp=v.data()+kh*HD;
                        for(int d=0;d<HD;++d){max_k=std::max(max_k,std::abs(kp[d]));max_v=std::max(max_v,std::abs(vp[d]));}
                        float scale_k=max_k>0?max_k/127.0f:1.0f,scale_v=max_v>0?max_v/127.0f:1.0f;
                        float inv_k=max_k>0?127.0f/max_k:0.0f,inv_v=max_v>0?127.0f/max_v:0.0f;
                        ks[slot*KV+kh]=scale_k;vs[slot*KV+kh]=scale_v;
                        int8_t*k_dst=kc_i8+slot*K+kh*HD,*v_dst=vc_i8+slot*K+kh*HD;
                        for(int d=0;d<HD;++d){
                            k_dst[d]=int8_t(std::max(-127.0f,std::min(127.0f,std::nearbyint(kp[d]*inv_k))));
                            v_dst[d]=int8_t(std::max(-127.0f,std::min(127.0f,std::nearbyint(vp[d]*inv_v))));
                        }
                    }
                }
                compute_attention(l, position, q.data(), att.data());
                sigmoid_gate(att.data(),gate.data(),A);
            }
            {
                ProfileScope s_mlp(PROF_ATTN_OUT_AND_MLP);
                activation_quant(att.data(),A,abits);linear(ti+7,att.data(),proj.data());
                rms(proj.data(),proj.data(),D,dense(ti+8));float ag=sigmoid(dense(ti+9)[0]);
                for(int d=0;d<D;++d)bx[d]+=ag*proj[d];
                rms(bx.data(),z.data(),D,dense(ti+10));
                for(int d=0;d<c.hada;++d)mlp[d]=d<D?z[d]*dense(ti+11)[d]:0;
                hadamard(mlp.data(),c.hada);
                silu_diagonal(mlp.data(),dense(ti+12),c.hada);
                hadamard(mlp.data(),c.hada);
                for(int d=0;d<D;++d)bx[d]=bx[d]+dense(ti+13)[d]*mlp[d]-u[d];
            }
            {
                ProfileScope s_mhc2(PROF_MHC);
                for(int ij=0;ij<N*N;++ij)hres[ij]=hres[ij]*dense(mh+2)[l]+dense(mh+5)[l*N*N+ij];
                sinkhorn(hres.data(),N);
                for(int n=0;n<N;++n)for(int d=0;d<D;++d){float val=hpost[n]*bx[d];for(int j=0;j<N;++j)val+=hres[n*N+j]*x[j*D+d];newx[n*D+d]=val;}
                x.swap(newx);
                for(int d=0;d<D;++d){float sum=0;for(int n=0;n<N;++n)sum+=x[n*D+d];hidden[l*D+d]=sum/N;}
            }
        }
        std::copy(hidden.end()-D,hidden.end(),z.begin());
        rms(z.data(),z.data(),D,dense(1+14*c.layers+9+4*c.num_sites));
        if(out){
            ProfileScope s_lm(PROF_LM_HEAD);
            profile_counters[0] += c.vocab;
            activation_quant(z.data(),D,abits);
            linear(0,z.data(),out);
        }'''

    if eng.count(step_marker) != 1:
        raise ValueError('Could not find exact step marker in engine.cpp')
    eng = eng.replace(step_marker, step_repl)

    (dest / 'engine.cpp').write_text(eng)
    (dest / 'cq.cpp').write_text(profiler_header + cq)

    libpath = dest / 'profile.so'
    import platform
    flags = ['-O3', '-DNDEBUG', '-std=c++17', '-fPIC', '-shared', '-pthread']
    if platform.system() == 'Darwin':
        from needle2.native import _darwin_openmp_flags
        flags += _darwin_openmp_flags()
        # Let the selected compiler resolve its matching C++ headers and SDK.
        # Injecting CLT libc++ headers can mix them with an active Xcode SDK.
    else:
        flags.append('-fopenmp')
        if platform.machine() in ('aarch64', 'arm64'):
            flags.append('-march=armv8.2-a+dotprod')

    compile_cmd = [
        os.environ.get('CXX', 'c++'), *flags,
        f'-I{dest}', f'-I{ROOT / "needle2/csrc"}',
        str(dest / 'cq.cpp'), '-o', str(libpath)
    ]
    subprocess.run(compile_cmd, check=True)

    import numpy as np
    import needle2.native as native
    from needle2.tokenizer import RefTokenizer
    from needle2.prompt import render_prompt

    native.build_native = lambda: libpath
    engine = native.NativeEngine(
        ROOT / 'artifacts/official/needle2.cact',
        threads=a.threads,
        matmul=a.matmul,
        kv_cache=a.kv_cache
    )
    tok = RefTokenizer.from_cact(ROOT / 'artifacts/official/needle2.cact')
    tools = json.loads((ROOT / 'examples/tools.json').read_text())
    ids = [2] + tok.encode(render_prompt('Turn on the kitchen light.', tools))
    engine.prefill(ids, last_only=True)

    lib = engine._lib
    lib.needle2_profile_get.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.needle2_profile_reset()

    # Step generation loop
    for i in range(a.tokens):
        engine.step((100 + i) % 8192)

    raw_times = np.empty(8, dtype=np.float64)
    counters = np.empty(8, dtype=np.int64)
    lib.needle2_profile_get(raw_times.ctypes.data, counters.ctypes.data)

    # Convert to disjoint components:
    # input_engram_raw includes engram_proj; subtract it to obtain pure non-proj time.
    input_engram_raw = raw_times[0]
    engram_proj = raw_times[1]
    input_engram_non_proj = max(0.0, input_engram_raw - engram_proj)
    qkvg_proj = raw_times[2]
    attention_core = raw_times[3]
    attn_out_and_mlp = raw_times[4]
    mhc = raw_times[5]
    lm_head = raw_times[6]
    whole_step = raw_times[7]

    subtotal = (input_engram_non_proj + engram_proj + qkvg_proj +
                attention_core + attn_out_and_mlp + mhc + lm_head)
    overhead_remaining = max(0.0, whole_step - subtotal)

    disjoint_times = {
        'input_engram_non_proj': input_engram_non_proj / a.tokens,
        'engram_projections': engram_proj / a.tokens,
        'qkvg_projections': qkvg_proj / a.tokens,
        'attention_core': attention_core / a.tokens,
        'attn_out_and_mlp': attn_out_and_mlp / a.tokens,
        'mhc_routing_and_mixing': mhc / a.tokens,
        'lm_head': lm_head / a.tokens,
        'overhead_remaining': overhead_remaining / a.tokens,
        'whole_step': whole_step / a.tokens,
    }

    result = {
        'matmul': a.matmul,
        'kv_cache': a.kv_cache,
        'threads': a.threads,
        'tokens': a.tokens,
        'tps': a.tokens / (whole_step / 1000.0) if whole_step > 0 else 0.0,
        'disjoint_ms_per_token': disjoint_times,
        'counters': {
            'lm_head_projected_rows': int(counters[0]),
            'step_calls': int(counters[1]),
            'final_kv_length': int(counters[2]),
        },
        'note': 'Profiled stages are mutually exclusive (disjoint) within step() and sum to whole_step.'
    }

    out_path = Path(a.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
