"""Isolated decode variants; production sources are never modified by this script."""
from pathlib import Path
import json, os, subprocess, sys, statistics, time
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/decode_experiments'
BASE_REVISION='32005c7818aa44bdd9a3f6750e9f92214c1b5931'
sys.path.insert(0,str(ROOT))

def build(name):
    dest=OUT/name;dest.mkdir(parents=True,exist_ok=True)
    for p in (ROOT/'needle2/csrc').glob('*.cpp'):
        (dest/p.name).write_bytes(p.read_bytes() if name=='mhc_final' else subprocess.check_output(['git','show',f'{BASE_REVISION}:needle2/csrc/{p.name}'],cwd=ROOT))
    eng=(dest/'engine.cpp').read_text();sdot=(dest/'sdot.cpp').read_text()
    if name in ('mhc_pair', 'mhc_parallel'):
        helper = '''    void mhc_projections(int mh, int layer, const float* input) {
        const TensorDesc* matrices[3] = {&t[mh+6], &t[mh+7], &t[mh+8]};
        int rows[3] = {c.lanes, c.lanes, c.lanes*c.lanes};
        float* outputs[3] = {hpre.data(), hpost.data(), hres.data()};
        if (matrices[0]->cq || matrices[1]->cq || matrices[2]->cq) {
            for (int m = 0; m < 3; ++m)
                linear(mh+6+m, input, outputs[m], layer*rows[m], rows[m]);
            return;
        }
        int blocks[3] = {(rows[0]+1)/2, (rows[1]+1)/2, (rows[2]+1)/2};
        auto work = [&](int tid, int begin, int end) {
            int offset = 0;
            for (int m = 0; m < 3; ++m) {
                int first = std::max(0, begin-offset), last = std::min(blocks[m], end-offset);
                offset += blocks[m];
                for (int b = first; b < last; ++b) {
                    int r = 2*b, cols = matrices[m]->cols;
                    const float* w = matrices[m]->data + size_t(layer*rows[m]+r)*cols;
                    if (r+1 < rows[m]) dot_pair(w, w+cols, input, cols, outputs[m][r], outputs[m][r+1]);
                    else outputs[m][r] = dot_f32(w, input, cols);
                }
            }
        };
        work(0, 0, blocks[0]+blocks[1]+blocks[2]);
    }
'''
        eng=eng.replace('    void attention_projections',helper+'    void attention_projections',1)
        eng=eng.replace('            linear(mh+6,nx.data(),hpre.data(),l*N,N);\n            linear(mh+7,nx.data(),hpost.data(),l*N,N);\n            linear(mh+8,nx.data(),hres.data(),l*N*N,N*N);','            mhc_projections(mh, l, nx.data());',1)
    if name == 'mhc_parallel':
        eng=eng.replace('        work(0, 0, blocks[0]+blocks[1]+blocks[2]);','        parallel_for(blocks[0]+blocks[1]+blocks[2], work);')
    if name == 'dense7680':
        eng=eng.replace('    Prepared sdot_input;', '    Prepared sdot_input;\n    std::vector<float> candidate_full_logits;')
    if name.startswith('dense'):
        threshold=int(name[5:].split('_')[0])
        marker='            auto* q = sdot[0].get();'
        eng=eng.replace(marker, f'''            if(num_candidates >= {threshold}) {{
                thread_local std::vector<float> full_logits;full_logits.resize(c.vocab);
                linear(0,z.data(),full_logits.data());
                for(int i=0;i<num_candidates;++i)candidate_logits[i]=full_logits[candidates[i]];
                return;
            }}
'''+marker)
    if name == 'dense7680':
        eng=eng.replace('num_candidates >= 7680)', 'num_candidates >= 7680 && num_candidates >= c.vocab - c.vocab / 16)')
        eng=eng.replace('thread_local std::vector<float> full_logits;full_logits.resize(c.vocab);', 'candidate_full_logits.resize(c.vocab);')
        eng=eng.replace('linear(0,z.data(),full_logits.data());','linear(0,z.data(),candidate_full_logits.data());').replace('candidate_logits[i]=full_logits[candidates[i]];', 'candidate_logits[i]=candidate_full_logits[candidates[i]];')
    if name.startswith('batch4'):
        start=sdot.index('    void row4(int r,');end=sdot.index('\n    void ',start+10)
        fn=sdot[start:end].replace('void row4(int r,','void row4_ids(const int* ids,')
        for i in range(4): fn=fn.replace(f'r + {i}',f'ids[{i}]')
        # Preserve the per-function target attribute for ARM SDOT intrinsics.
        sdot=sdot[:end]+'\n#ifdef __aarch64__\n    __attribute__((target("arch=armv8.2-a+dotprod")))\n#endif\n'+fn+sdot[end:]
        start=eng.index('            if (num_candidates >= 128)',eng.index('    void project_candidates'))
        end=eng.index('        } else if (t[0].cq)',start)
        eng=eng[:start]+'''            int blocks=num_candidates/4;
            auto run=[&](int tid,int start,int end) {
                for(int b=start;b<end;++b)q->row4_ids(candidates+4*b,sdot_input,candidate_logits+4*b);
            };
            if(num_candidates>=128)parallel_for(blocks,run);else run(0,0,blocks);
            for(int i=blocks*4;i<num_candidates;++i)candidate_logits[i]=q->row(candidates[i],sdot_input);
'''+eng[end:]
    if 'serial' in name or 'short' in name:
        eng=eng.replace('const float* q_in, float* att_out) {','const float* q_in, float* att_out, bool decode=false) {').replace('compute_attention(l, position, q.data(), att.data());','compute_attention(l, position, q.data(), att.data(), true);')
        eng=eng.replace('        parallel_for(int(head_work.size()), [&](int tid, int start_g, int end_g) {','        auto attention_work = [&](int tid, int start_g, int end_g) {')
        marker='''        });
    }
    void step(int token'''
        limit='decode' if 'serial' in name else 'decode && length <= 64'
        eng=eng.replace(marker,'''        };
        if('''+limit+''')attention_work(0,0,int(head_work.size()));
        else parallel_for(int(head_work.size()),attention_work);
    }
    void step(int token''')
    eng+='''
extern "C" void experiment_project(void*p,const int*ids,int n,float*out){static_cast<Engine*>(p)->project_candidates(ids,n,out);}
'''
    (dest/'engine.cpp').write_text(eng);(dest/'sdot.cpp').write_text(sdot)
    lib=dest/'lib.so'
    subprocess.run(['c++','-O3','-DNDEBUG','-std=c++17','-fPIC','-shared','-pthread','-fopenmp',str(dest/'cq.cpp'),'-o',str(lib)],check=True)
    return lib

def micro(name, threads=4, order="random"):
    label=f"{name}_{threads}_{order}"
    import ctypes as ct, numpy as np
    os.sched_setaffinity(0,{0,1,2,3})
    os.environ['NEEDLE2_NATIVE_LIBRARY']=str(OUT/name/'lib.so')
    from needle2.native import NativeEngine,_library
    engine=NativeEngine(ROOT/'artifacts/official/needle2.cact',threads=threads,matmul='sdot',kv_cache='int8')
    from benchmark_official import pin_all_threads
    pin_all_threads({0,1,2,3})
    engine.prefill([2,20,30,40],last_only=True)
    lib=_library();lib.experiment_project.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_void_p]
    rng=np.random.default_rng(42);rows={};outputs={}
    for n in [128,512,1024,2048,4096,6144,8037,8192]:
        ids=rng.choice(8192,n,replace=False).astype(np.int32)
        if order=="sorted":ids.sort()
        out=np.empty(n,np.float32)
        def run():lib.experiment_project(engine._handle,ids.ctypes.data,n,out.ctypes.data)
        for _ in range(5):run()
        times=[]
        for _ in range(31):
            t=time.perf_counter();run();times.append((time.perf_counter()-t)*1000)
        rows[n]=dict(median_ms=statistics.median(times),samples_ms=times)
        outputs[str(n)]=out.copy()
    np.savez(OUT/(label+'_logits.npz'),**outputs)
    (OUT/(label+'_micro.json')).write_text(json.dumps(rows,indent=2))
    print(label,{k:round(v['median_ms'],3) for k,v in rows.items()},flush=True)

def worker_main():
    os.environ['NEEDLE2_NATIVE_LIBRARY']=str(OUT/sys.argv[2]/'lib.so')
    from benchmark_backends import worker
    config=json.loads(sys.stdin.readline())
    class Pipe:
        def recv(self):return json.loads(sys.stdin.readline())
        def send(self,msg):print(json.dumps(msg),flush=True)
        def close(self):pass
    worker(Pipe(),dict(name=sys.argv[2],kind='sdot',threads=4),config)

def bench(names,report,repeat):
    from needle2.tokenizer import RefTokenizer
    from needle2.prompt import render_prompt
    tok=RefTokenizer.from_cact(ROOT/'artifacts/official/needle2.cact')
    results={}
    for suite,tf,cf in [('basic','examples/tools.json','benchmarks/cases.jsonl'),('expanded','benchmarks/expanded_tools.json','benchmarks/expanded_cases.jsonl')]:
        tools=json.loads((ROOT/tf).read_text());cases=[json.loads(s) for s in (ROOT/cf).read_text().splitlines() if s.strip()]
        for c in cases:
            ids=[2]+tok.encode(render_prompt(c['query'],tools));split=ids.index(tok.p2id['</tools>'])+1
            prefix=ids[:split];c['suffix_ids']=ids[split:]
        config=dict(model=str(ROOT/'artifacts/official/needle2.cact'),tools=tools,prefix_ids=prefix,cpus=[0,1,2,3],max_new_tokens=192,kv_cache='int8')
        procs={};rows=[]
        def request(n,c):
            time.sleep(.02);p=procs[n];p.stdin.write(json.dumps(c)+'\n');p.stdin.flush()
            msg=json.loads(p.stdout.readline())
            if 'error' in msg:raise RuntimeError(msg['error'])
            return msg['result']
        try:
            for n in names:
                p=subprocess.Popen([sys.executable,__file__,'worker',n],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
                procs[n]=p;p.stdin.write(json.dumps(config)+'\n');p.stdin.flush()
                msg=json.loads(p.stdout.readline())
                if 'error' in msg:raise RuntimeError(msg['error'])
            for ci,c in enumerate(cases):
                for n in names:request(n,c)
                for rep in range(repeat):
                    offset=(ci+rep)%len(names);order=names[offset:]+names[:offset]
                    r={n:request(n,c) for n in order}
                    assert len({tuple(v['token_ids']) for v in r.values()})==1,c['id']
                    assert all(v['exact_call_match'] for v in r.values()),c['id']
                    rows.append(dict(id=c['id'],rep=rep,results=r))
                print(suite,c['id'],flush=True)
        finally:
            for p in procs.values():
                if p.poll() is None:p.stdin.write('null\n');p.stdin.flush();p.wait(timeout=20)
        summary={n:{k:statistics.median(r['results'][n][k] for r in rows) for k in ['wall_ms','prefill_ms','decode_ms','decode_tps']} for n in names}
        results[suite]=dict(rows=rows,summary=summary)
        (ROOT/report).write_text(json.dumps(results,indent=2)+'\n')
        print(suite,summary,flush=True)

def summarize_micro():
    import numpy as np
    import hashlib
    import platform
    report = {}
    for threads in [1, 2, 4]:
        for order in ['sorted', 'random']:
            key = f'{threads}_{order}'
            expected = np.load(OUT / f'baseline_{key}_logits.npz')
            measurements = {}
            for name in ['baseline', 'dense6144', 'batch4']:
                actual = np.load(OUT / f'{name}_{key}_logits.npz')
                for count in expected.files:
                    np.testing.assert_array_equal(actual[count], expected[count])
                measurements[name] = json.loads((OUT / f'{name}_{key}_micro.json').read_text())
            report[key] = dict(bitwise_equal=True, measurements=measurements)
    (ROOT / 'reports/decode_projection_micro.json').write_text(json.dumps(report, indent=2) + '\n')
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    names = ['baseline', 'dense512', 'batch4', 'dense6144', 'dense6144_serial', 'dense6144_short', 'dense7680']
    manifest = dict(baseline=BASE_REVISION, platform=platform.platform(), cpus=[0,1,2,3],
                    model_sha256=sha(ROOT/'artifacts/official/needle2.cact'),
                    compiler=subprocess.check_output(['c++','--version'],text=True).splitlines()[0],
                    request_threads=4, request_repeats=5, request_warmups=1,
                    request_backend='sdot', request_kv_cache='int8',
                    request_order='serial interleaved persistent processes; rotating variant order',
                    micro_repeats=31, micro_warmups=5,
                    micro_order='separate serial processes, baseline then dense6144 then batch4',
                    environment={k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','OMP_WAIT_POLICY']},
                    libraries={n:dict(library_sha256=sha(OUT/n/'lib.so'),
                                      sources={p.name:sha(p) for p in sorted((OUT/n).glob('*.cpp'))}) for n in names})
    (ROOT/'reports/decode_experiment_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':
    sys.path.insert(0,str(ROOT/'scripts'))
    if sys.argv[1]=='build':
        for n in sys.argv[2:]:print(build(n),flush=True)
    elif sys.argv[1]=='micro':micro(sys.argv[2],int(sys.argv[3]) if len(sys.argv)>3 else 4,sys.argv[4] if len(sys.argv)>4 else 'random')
    elif sys.argv[1]=='worker':worker_main()
    elif sys.argv[1]=='summarize':summarize_micro()
    elif sys.argv[1]=='bench':
        names=sys.argv[3:];repeat=5
        if '--repeat' in names:
            index=names.index('--repeat');repeat=int(names[index+1]);del names[index:index+2]
        bench(names,sys.argv[2],repeat)
    else:raise SystemExit('Expected build, micro, worker, summarize, or bench')
