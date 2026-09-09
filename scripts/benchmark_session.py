"""Compare one-shot initialization with a persistent session, including Python work."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = '2664107'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--output', default='reports/session_benchmark.json')
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error('--repeat must be positive')
    os.sched_setaffinity(0, {0, 1, 2, 3})
    from needle2.inference import InferenceSession, generate
    from needle2.official import strict_json_equal
    source = subprocess.check_output(['git', 'show', f'{BASE}:needle2/inference.py'], cwd=ROOT)
    path = ROOT/'artifacts/session_benchmark/inference_before.py'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(source)
    spec = importlib.util.spec_from_file_location('needle2._session_baseline', path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    model = ROOT/'artifacts/official/needle2.cact'
    config = dict(threads=4, matmul='sdot', kv_cache='int8')
    session = InferenceSession(model, **config)
    results = dict(baseline=BASE, repeats=args.repeat, session_setup_seconds=session.setup_seconds,
                   note='Same process, warm imports/library/page cache. One-shot includes archive and engine initialization; not a fresh process cold-start measurement.',
                   source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                                  for p in ['needle2/inference.py', 'needle2/grammar.py', 'needle2/tokenizer.py',
                                            'pyproject.toml', 'needle2/csrc/engine.cpp']}, suites={})
    for suite, tool_path, case_path in [('basic','examples/tools.json','benchmarks/cases.jsonl'),
                                       ('expanded','benchmarks/expanded_tools.json','benchmarks/expanded_cases.jsonl')]:
        tools = json.loads((ROOT/tool_path).read_text())
        cases = [json.loads(line) for line in (ROOT/case_path).read_text().splitlines() if line]
        rows = []
        for case in cases:
            opts = dict(tools=tools, max_new_tokens=192)
            initial = session.generate(case['query'], **opts)
            # Check current one-shot compatibility, outside measured repetitions.
            one_shot = generate(model, case['query'], **config, **opts)
            assert initial['token_ids'] == one_shot['token_ids'], case['id']
            for rep in range(args.repeat):
                outputs = {}
                for name in (['before', 'session'] if rep % 2 == 0 else ['session', 'before']):
                    started = time.perf_counter()
                    result = (old.generate(model, case['query'], **config, **opts)
                              if name == 'before' else session.generate(case['query'], **opts))
                    elapsed = time.perf_counter()-started
                    assert strict_json_equal(result['function_calls'], case['expected_calls']), case['id']
                    outputs[name] = dict(wall_seconds=elapsed, result=result)
                assert outputs['before']['result']['token_ids'] == outputs['session']['result']['token_ids'], case['id']
                rows.append(dict(id=case['id'], rep=rep, outputs=outputs))
            print(suite, case['id'], flush=True)
        summary = {name: statistics.median(row['outputs'][name]['wall_seconds'] for row in rows)*1000
                   for name in ['before', 'session']}
        phases = ['prepare_seconds', 'prefix_restore_seconds', 'prefill_seconds', 'decode_seconds',
                  'native_decode_call_seconds', 'parse_seconds']
        summary['session_phase_ms'] = {p:statistics.median(row['outputs']['session']['result'][p] for row in rows)*1000 for p in phases}
        results['suites'][suite] = dict(rows=rows, summary_ms=summary)
        (ROOT/args.output).write_text(json.dumps(results, indent=2)+'\n')
        print(suite, summary, flush=True)


if __name__ == '__main__':
    main()
