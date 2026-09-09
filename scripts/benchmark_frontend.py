"""Cold frontend setup and warm request comparison against commit 0988b81."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
BASE='0988b81'


def main():
    os.sched_setaffinity(0,{0,1,2,3})
    from needle2.inference import InferenceSession
    from needle2.official import strict_json_equal
    from needle2.grammar import _cached_tool_dfa
    dest=ROOT/'artifacts/frontend/inference_before.py'
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_bytes(subprocess.check_output(['git','show',f'{BASE}:needle2/inference.py'],cwd=ROOT))
    spec=importlib.util.spec_from_file_location('needle2._frontend_before',dest)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    config=dict(threads=4,matmul='sdot',kv_cache='int8')
    path=ROOT/'artifacts/official/needle2.cact'
    report=dict(baseline=BASE,cpus=[0,1,2,3],config=config,repeats=5,
        note='Cold = first tool/schema request in an already initialized session, with imports/library/page cache warm. Warm = serial interleaved requests. Same current neural engine in both variants.',
        sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in [ROOT/'needle2/frontend.py',ROOT/'needle2/inference.py',ROOT/'needle2/native.py',ROOT/'needle2/csrc/frontend.cpp',ROOT/'needle2/csrc/engine.cpp']},suites={})
    for suite,tf,cf in [('basic','examples/tools.json','benchmarks/cases.jsonl'),('expanded','benchmarks/expanded_tools.json','benchmarks/expanded_cases.jsonl')]:
        tools=json.loads((ROOT/tf).read_text());cases=[json.loads(s) for s in (ROOT/cf).read_text().splitlines() if s]
        _cached_tool_dfa.cache_clear()
        sessions={'python':old.InferenceSession(path,**config),'cpp':InferenceSession(path,**config)}
        cold={}
        for name,s in sessions.items():
            start=time.perf_counter();r=s.generate(cases[0]['query'],tools=tools,max_new_tokens=192)
            cold[name]=dict(wall_ms=(time.perf_counter()-start)*1000,result=r,session_setup_ms=s.setup_seconds*1000)
        assert cold['python']['result']['token_ids']==cold['cpp']['result']['token_ids']
        rows=[]
        for ci,case in enumerate(cases):
            for s in sessions.values():s.generate(case['query'],tools=tools,max_new_tokens=192)
            for rep in range(5):
                outputs={}
                for name in (['python','cpp'] if (ci+rep)%2==0 else ['cpp','python']):
                    time.sleep(.02)
                    start=time.perf_counter();r=sessions[name].generate(case['query'],tools=tools,max_new_tokens=192)
                    outputs[name]=dict(wall_ms=(time.perf_counter()-start)*1000,result=r)
                    assert strict_json_equal(r['function_calls'],case['expected_calls']),case['id']
                assert outputs['python']['result']['token_ids']==outputs['cpp']['result']['token_ids'],case['id']
                rows.append(dict(id=case['id'],rep=rep,outputs=outputs))
            print(suite,case['id'],flush=True)
        summary={name:dict(wall_ms=statistics.median(r['outputs'][name]['wall_ms'] for r in rows),
                          prepare_ms=statistics.median(r['outputs'][name]['result']['prepare_seconds']*1000 for r in rows)) for name in sessions}
        report['suites'][suite]=dict(cold=cold,rows=rows,summary=summary)
        (ROOT/'reports/frontend_benchmark.json').write_text(json.dumps(report,indent=2)+'\n')
        print(suite,summary,'cold prepare', {n:cold[n]['result']['prepare_seconds']*1000 for n in cold},flush=True)

if __name__=='__main__':main()
