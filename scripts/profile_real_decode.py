"""Profile actual grammar-constrained requests with matching SDOT/INT8 KV settings."""
from pathlib import Path
import ctypes as ct
import hashlib
import json
import os
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
BASE='32005c7818aa44bdd9a3f6750e9f92214c1b5931'
STAGES=['input_engram','mhc_pre_and_input_norm','qkvg','attention_norm_rope_kv_gate','out_projection_mlp','sinkhorn_residual_hidden','final_norm_open_lm_head','candidate_projection']

def build(variant, compile_library=True):
    dest=ROOT/'artifacts/real_decode_profile'/variant;dest.mkdir(parents=True,exist_ok=True)
    for p in (ROOT/'needle2/csrc').glob('*.cpp'):
        data=subprocess.check_output(['git','show',f'{BASE}:needle2/csrc/{p.name}'],cwd=ROOT) if variant=='baseline' else p.read_bytes()
        (dest/p.name).write_bytes(data)
    cq=dest/'cq.cpp';cq.write_text('''#include <chrono>
static bool profile_enabled=false;
static double profile_times[8]={};
static long long profile_prepares=0;
using ProfileClock=std::chrono::steady_clock;
static void profile_mark(ProfileClock::time_point& t,int stage){
    auto now=ProfileClock::now();if(profile_enabled)profile_times[stage]+=std::chrono::duration<double,std::milli>(now-t).count();t=now;
}
extern "C" void real_profile_reset(){for(auto&x:profile_times)x=0;profile_prepares=0;profile_enabled=true;}
extern "C" long long real_profile_get(double*out){profile_enabled=false;for(int i=0;i<8;++i)out[i]=profile_times[i];return profile_prepares;}
'''+cq.read_text())
    sd=dest/'sdot.cpp';sd.write_text(sd.read_text().replace('    void prepare(const float *input, Prepared &output) const {','    void prepare(const float *input, Prepared &output) const {\n        if(profile_enabled)++profile_prepares;'))
    p=dest/'engine.cpp';s=p.read_text();start=s.index('    void step(int token');end=s.index('    void prefill(',start);step=s[start:end]
    step=step.replace('    void step(int token,float*out,float*hidden_out) {','    void step(int token,float*out,float*hidden_out) {\n        auto profile_time=ProfileClock::now();')
    markers=[('        for(int l=0;l<c.layers;++l) {',0),('            attention_projections(ti,z.data());',1),('            for(int h=0;h<H;++h)rms(q.data()',2),('            activation_quant(att.data(),A,abits);',3),('            for(int ij=0;ij<N*N;++ij)hres[ij]=',4)]
    for marker,stage in markers:
        assert step.count(marker)==1,marker
        step=step.replace(marker,f'        profile_mark(profile_time,{stage});\n'+marker)
    marker='''        }
        std::copy(hidden.end()-D'''
    assert marker in step
    step=step.replace(marker,'''            profile_mark(profile_time,5);
        }
        std::copy(hidden.end()-D''')
    step=step.replace('        ++position;','        ++position;\n        profile_mark(profile_time,6);')
    s=s[:start]+step+s[end:]
    s=s.replace('    void project_candidates(const int* candidates, int num_candidates, float* candidate_logits) {','''    void project_candidates(const int* candidates, int num_candidates, float* candidate_logits) {
        struct Scope {ProfileClock::time_point t=ProfileClock::now();~Scope(){profile_mark(t,7);}} scope;''')
    p.write_text(s)
    lib=dest/'lib.so'
    if not compile_library:return lib
    subprocess.run(['c++','-O3','-DNDEBUG','-std=c++17','-fPIC','-shared','-pthread','-fopenmp',str(cq),'-o',str(lib)],check=True)
    return lib

def run(variant, extra_reader=None):
    import numpy as np
    os.sched_setaffinity(0,{0,1,2,3})
    os.environ['NEEDLE2_NATIVE_LIBRARY']=str(ROOT/'artifacts/real_decode_profile'/variant/'lib.so')
    from needle2.native import NativeEngine
    from needle2.tokenizer import RefTokenizer
    from needle2.grammar import ToolGrammar,compile_tool_dfa
    from needle2.prompt import render_prompt,parse_response
    from needle2.official import strict_json_equal
    from benchmark_official import pin_all_threads
    tok=RefTokenizer.from_cact(ROOT/'artifacts/official/needle2.cact')
    dest=ROOT/'artifacts/real_decode_profile'/variant
    report={'variant':variant,'baseline_revision':BASE,'threads':4,'matmul':'sdot','kv_cache':'int8',
            'library_sha256':hashlib.sha256((dest/'lib.so').read_bytes()).hexdigest(),
            'instrumented_sources_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.glob('*.cpp')},
            'measurement':'one warmup and three measured actual requests per case; stages additive; instrumented, not a speed benchmark',
            'results':{}}
    reference=json.loads((ROOT/'reports/decode_projection_experiment.json').read_text())
    for suite,tf,cf in [('basic','examples/tools.json','benchmarks/cases.jsonl'),('expanded','benchmarks/expanded_tools.json','benchmarks/expanded_cases.jsonl')]:
        tools=json.loads((ROOT/tf).read_text());cases=[json.loads(s) for s in (ROOT/cf).read_text().splitlines() if s.strip()]
        engine=NativeEngine(ROOT/'artifacts/official/needle2.cact',threads=4,matmul='sdot',kv_cache='int8');pin_all_threads({0,1,2,3})
        dfa=compile_tool_dfa(tools,tok)
        ids=[2]+tok.encode(render_prompt(cases[0]['query'],tools));split=ids.index(tok.p2id['</tools>'])+1
        engine.reset(prefix_len=split);engine.prefill(ids[:split],last_only=True);engine.cache_prefix()
        lib=engine._lib;lib.real_profile_get.argtypes=[ct.c_void_p];lib.real_profile_get.restype=ct.c_longlong
        expected={r['id']:r['results']['baseline']['token_ids'] for r in reference[suite]['rows']}
        rows=[]
        for c in cases:
            suffix=([2]+tok.encode(render_prompt(c['query'],tools)))[split:]
            for rep in range(4):
                engine.reset_to_prefix();logits=engine.prefill(suffix,last_only=True)
                first=ToolGrammar(tools,tok).select(logits)
                lib.real_profile_reset();started=time.perf_counter();tokens=engine.decode(first,max_new_tokens=192,grammar_dfa=dfa);wall=(time.perf_counter()-started)*1000
                values=np.empty(8,np.float64);prepares=lib.real_profile_get(values.ctypes.data)
                assert tokens==expected[c['id']],c['id']
                assert strict_json_equal(parse_response(tok.decode(tokens))['function_calls'],c['expected_calls'])
                if rep:
                    row=dict(id=c['id'],rep=rep,decode_ms=wall,steps=len(tokens)-1,prepare_calls=prepares,stages_ms=dict(zip(STAGES,values.tolist())),unattributed_ms=wall-float(values.sum()))
                    if extra_reader:row['attention_detail']=extra_reader(lib)
                    rows.append(row)
        # Means over all tokens preserve additivity; individual medians would not.
        total_steps=sum(r['steps'] for r in rows)
        report['results'][suite]=dict(prefix_tokens=split,rows=rows,ms_per_token={s:sum(r['stages_ms'][s] for r in rows)/total_steps for s in STAGES},prepare_calls_per_token=sum(r['prepare_calls'] for r in rows)/total_steps,decode_ms_per_token=sum(r['decode_ms'] for r in rows)/total_steps)
        print(variant,suite,report['results'][suite]['ms_per_token'],flush=True)
        del engine
    (ROOT/f'reports/real_decode_profile_{variant}.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':
    if sys.argv[1]=='build':print(build(sys.argv[2]))
    else:run(sys.argv[2])
