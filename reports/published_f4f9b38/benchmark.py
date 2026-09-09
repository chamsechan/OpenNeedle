"""Paired persistent-process benchmark: baseline and optimized.
All use the same model, tools, native worker, CPUs, SDOT and INT8 KV.
"""
from pathlib import Path
import json, os, statistics, subprocess, sys, time
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
BASE=Path('/tmp/needle2-e809ffc')

def worker_main():
    source=Path(sys.argv[2]); override=sys.argv[3]
    sys.path[:0]=[str(source),str(source/'scripts')]
    import needle2.native as native
    if override!='-': native.build_native=lambda: Path(override)
    from benchmark_backends import worker
    config=json.loads(sys.stdin.readline())
    class Pipe:
        def recv(self): return json.loads(sys.stdin.readline())
        def send(self,msg): print(json.dumps(msg),flush=True)
        def close(self): pass
    worker(Pipe(),dict(name='native',kind='sdot',threads=4),config)

def main():
    baseline_lib=subprocess.check_output([sys.executable,'-c','from needle2.native import build_native; print(build_native())'],cwd=BASE,text=True).strip()
    sys.path.insert(0,str(ROOT))
    from needle2.tokenizer import RefTokenizer
    from needle2.prompt import render_prompt
    tok=RefTokenizer.from_cact(ROOT/'artifacts/official/needle2.cact')
    report=dict(baseline='e809ffc',results={},source_diff=subprocess.check_output(['git','diff'],cwd=ROOT,text=True))
    for suite,toolfile,casefile in [('basic','examples/tools.json','benchmarks/cases.jsonl'),('expanded','benchmarks/expanded_tools.json','benchmarks/expanded_cases.jsonl')]:
        tools=json.loads((ROOT/toolfile).read_text())
        cases=[json.loads(s) for s in (ROOT/casefile).read_text().splitlines() if s.strip()]
        for c in cases:
            ids=[2]+tok.encode(render_prompt(c['query'],tools));split=ids.index(tok.p2id['</tools>'])+1
            prefix=ids[:split];c['suffix_ids']=ids[split:]
        config=dict(model=str(ROOT/'artifacts/official/needle2.cact'),tools=tools,prefix_ids=prefix,cpus=[0,1,2,3],max_new_tokens=192,kv_cache='int8')
        procs={}; rows=[]; startup={}
        def request(name,c):
            time.sleep(.02)
            p=procs[name];p.stdin.write(json.dumps(c)+'\n');p.stdin.flush()
            msg=json.loads(p.stdout.readline())
            if 'error' in msg: raise RuntimeError(msg['error'])
            return msg['result']
        try:
            for name,src,lib in [('baseline',BASE,'-'),('optimized',ROOT,'-')]:
                p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker',str(src),str(lib)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,cwd=ROOT)
                procs[name]=p;p.stdin.write(json.dumps(config)+'\n');p.stdin.flush()
                msg=json.loads(p.stdout.readline())
                if 'error' in msg: raise RuntimeError(msg['error'])
                startup[name]=msg['ready']
            for ci,c in enumerate(cases):
                for name in procs: request(name,c)
                for rep in range(9):
                    order=list(procs);off=(ci+rep)%len(order);order=order[off:]+order[:off]
                    row=dict(id=c['id'],rep=rep,results={n:request(n,c) for n in order})
                    assert len({tuple(r['token_ids']) for r in row['results'].values()})==1, c['id']
                    assert all(r['exact_call_match'] for r in row['results'].values()), c['id']
                    rows.append(row)
                print(suite,c['id'],flush=True)
        finally:
            for p in procs.values():
                if p.poll() is None:
                    p.stdin.write('null\n');p.stdin.flush();p.wait(timeout=10)
        summary={n:{k:statistics.median(r['results'][n][k] for r in rows) for k in ['wall_ms','prefill_ms','decode_ms','decode_tps']} for n in procs}
        report['results'][suite]=dict(rows=rows,startup=startup,summary=summary)
        (OUT/'benchmark_confirm.json').write_text(json.dumps(report,indent=2)+'\n')
        print(suite,json.dumps(summary),flush=True)
    assert report['source_diff']==subprocess.check_output(['git','diff'],cwd=ROOT,text=True)
if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--worker':worker_main()
    else:main()
