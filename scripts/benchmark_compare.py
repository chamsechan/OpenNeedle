#!/usr/bin/env python3
"""Interleave official and independent tool decoding on the same allowed CPUs.

Both prepare the tools prefix outside timed requests. Official internal prompt,
integer arithmetic, grammar, confidence and token counts still differ. This is
an application comparison, not an identical-operator microbenchmark.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default='artifacts/official/needle2.cact')
    p.add_argument('--library',default='artifacts/official/python/libneedle.so')
    p.add_argument('--tools',default='examples/tools.json');p.add_argument('--cases',default='benchmarks/cases.jsonl')
    p.add_argument('--threads',type=int,default=2);p.add_argument('--repeat',type=int,default=5)
    p.add_argument('--matmul',choices=['fp32','sdot'],default='fp32')
    p.add_argument('--affinity',default='0,1,2,3');p.add_argument('--output',default='reports/comparison.json')
    a=p.parse_args()
    if a.repeat<1:p.error('repeat must be positive')
    cpus={int(s) for s in a.affinity.split(',')};os.sched_setaffinity(0,cpus)
    import numpy as np
    from needle2.archive import Archive
    from needle2.native import NativeEngine
    from needle2.tokenizer import RefTokenizer
    from needle2.grammar import ToolGrammar
    from needle2.prompt import render_prompt,parse_response
    from needle2.official import OfficialEngine,sha256,strict_json_equal
    from benchmark_official import hardware,workload
    tools=json.loads(Path(a.tools).read_text());cases=[json.loads(l) for l in Path(a.cases).read_text().splitlines() if l.strip()]
    arc=Archive.load(a.model);tok=RefTokenizer.from_cact(a.model)
    engine=NativeEngine(arc,threads=a.threads,matmul=a.matmul)
    source_hashes={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted((ROOT/'needle2/csrc').glob('*.cpp'))}
    report=dict(hardware=hardware(),workload_before=workload(),model_sha256=arc.sha256,official_library_sha256=sha256(a.library),native_source_sha256=source_hashes,
                native_threads=a.threads,native_matmul=a.matmul,official_threads='automatic',repeat=a.repeat,cases=[],
                notes=['Requests are interleaved; tool prefix setup is performed once and excluded from request timing in both paths.',
                       'Native uses cache_prefix/reset_to_prefix. Request wall time includes prefix restoration/reset in both engines.',
                       ('Official performs its own confidence/retrieval/grammar; independent path uses bounded grammar without confidence and ' +
                        ('public FP32 math.' if a.matmul == 'fp32' else 'approximate SDOT centroid rounding / rotated A8 with FP32 KV.')),
                       'Internal prompts and generated token counts differ: TPS ratio is not equal-work kernel speedup.',
                       'Native decode timing includes grammar selection and token forward. Official TPS is its own reported field.'])
    cached_prefix_ids=None
    with OfficialEngine(a.library,a.model,tools) as official:
        report['official_load_ms']=official.load_ms
        report['official_prefix_setup_ms']=official.init_ms
        for task in Path('/proc/self/task').iterdir():
            try:os.sched_setaffinity(int(task.name),cpus)
            except ProcessLookupError:pass
        for c in cases:
            ids=[2]+tok.encode(render_prompt(c['query'],tools));prefix=ids.index(tok.p2id['</tools>'])+1
            prefix_setup_ms=0.0
            if cached_prefix_ids!=ids[:prefix]:
                start=time.perf_counter()
                engine.reset(prefix_len=prefix)
                for token in ids[:prefix]:engine.step(token,compute_logits=False)
                engine.cache_prefix()
                prefix_setup_ms=(time.perf_counter()-start)*1000
                cached_prefix_ids=ids[:prefix]
            def ours():
                grammar=ToolGrammar(tools,tok);out=[]
                cpu0=time.process_time();t0=time.perf_counter()
                engine.reset_to_prefix()
                restore_ms=(time.perf_counter()-t0)*1000
                tp=time.perf_counter()
                logits=engine.prefill(ids[prefix:],last_only=True)
                prefill_s=time.perf_counter()-tp
                td=time.perf_counter();steps=0
                for i in range(128):
                    t=grammar.select(logits);grammar.accept(t);out.append(t)
                    if t in (1,5) or grammar.finished or i==127:break
                    logits=engine.step(t);steps+=1
                decode_s=time.perf_counter()-td
                wall=time.perf_counter()-t0
                result=parse_response(tok.decode(out))
                return dict(response=result,wall_ms=wall*1000,process_cpu_ms=(time.process_time()-cpu0)*1000,
                            prefix_restore_ms=restore_ms,
                            decode_tps=steps/decode_s,prefill_tps=(len(ids)-prefix)/prefill_s,
                            generated_tokens=len(out),prompt_tokens=len(ids),prefix_tokens=prefix,
                            exact_call_match=strict_json_equal(result['function_calls'],c.get('expected_calls')))
            def theirs():
                cpu=time.process_time();start=time.perf_counter()
                official.reset();r=official.complete(c['query'],128)
                r['wall_ms']=(time.perf_counter()-start)*1000
                r['process_cpu_ms']=(time.process_time()-cpu)*1000
                r['decode_tps']=r['response'].get('decode_tps')
                r['exact_call_match']=strict_json_equal(r['response'].get('function_calls'),c.get('expected_calls'))
                return r
            ours();theirs()
            entry=dict(**c,native_prefix_setup_ms=prefix_setup_ms,native=[],official=[])
            for repeat in range(a.repeat):
                for name,fn in ([('native',ours),('official',theirs)] if repeat%2==0 else [('official',theirs),('native',ours)]):
                    entry[name].append(fn())
            report['cases'].append(entry)
            print(c['id'],flush=True)
    report['workload_after']=workload();report['summary']={}
    for name in ('native','official'):
        rows=[r for c in report['cases'] for r in c[name]]
        report['summary'][name]=dict(decode_tps_median=statistics.median(r['decode_tps'] for r in rows),
                                    wall_ms_median=statistics.median(r['wall_ms'] for r in rows),
                                    exact_call_matches=sum(r['exact_call_match'] for r in rows),runs=len(rows))
    dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['summary'],indent=2))

if __name__=='__main__':main()
