#!/usr/bin/env python3
"""Compare official, packed native FP32/SDOT and eager PyTorch on CPU.

Persistent, separate processes isolate runtime thread pools. Requests are serial
and interleaved, sharing the same allowed CPUs. Tool prefixes are initialized
once. The official internal prompt/grammar/TPS definitions remain its own.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def worker(connection, spec, config):
    official = None
    try:
        if hasattr(os, 'sched_setaffinity'):
            os.sched_setaffinity(0, set(config['cpus']))
        import numpy as np
        from needle2.official import OfficialEngine, strict_json_equal
        from needle2.tokenizer import RefTokenizer
        from needle2.prompt import parse_response
        from needle2.grammar import ToolGrammar
        from benchmark_official import pin_all_threads

        name, kind, threads = spec['name'], spec['kind'], spec['threads']
        tokenizer = RefTokenizer.from_cact(config['model'])
        prefix_ids = config['prefix_ids']
        startup = {'pid': os.getpid(), 'threads': threads, 'backend': kind}
        if kind == 'official':
            official = OfficialEngine(config['library'], config['model'], config['tools'])
            startup.update(load_ms=official.load_ms, prefix_setup_ms=official.init_ms,
                           prefix_tokens='internal official prompt')
        elif kind == 'torch':
            import torch
            from needle2.convert import load_torch_model
            torch.set_num_threads(threads)
            torch.set_num_interop_threads(1)
            start = time.perf_counter()
            model = load_torch_model(config['model']).eval()
            startup.update(load_ms=(time.perf_counter() - start)*1000,
                           torch_version=torch.__version__, interop_threads=1,
                           weight_dtype='float32', device='cpu', compiled=False)

            @torch.inference_mode()
            def consume(tokens, cache=None, *, prefix=False, output_logits=True):
                tensor = torch.tensor([tokens], dtype=torch.long)
                sinks = torch.ones_like(tensor, dtype=torch.bool) if prefix else None
                # Reuse the model's exact eager operations, avoiding discarded
                # vocabulary projections at earlier prefill positions.
                hidden, new_cache, _ = model._run(tensor, cache, None, sinks, False)
                if not output_logits:
                    return None, new_cache
                logits = model._linear('embedding', model._aq(hidden[:, -1:]).float(),
                                       model._w('embedding').float())
                return logits[0, -1].numpy(), new_cache

            start = time.perf_counter()
            _, prefix_cache = consume(prefix_ids, prefix=True, output_logits=False)
            startup.update(prefix_setup_ms=(time.perf_counter()-start)*1000,
                           prefix_tokens=prefix_cache.position)
        else:
            from needle2.native import NativeEngine, build_native
            from needle2.grammar import compile_tool_dfa, GrammarTooLarge
            start = time.perf_counter()
            compiled = build_native()
            startup['compile_cache_ms'] = (time.perf_counter()-start)*1000
            start = time.perf_counter()
            engine = NativeEngine(config['model'], threads=threads, matmul=kind, kv_cache=config.get('kv_cache', 'fp32'))
            startup.update(load_ms=(time.perf_counter()-start)*1000,
                           native_library=str(compiled))
            start = time.perf_counter()
            engine.reset(prefix_len=len(prefix_ids))
            engine.prefill(prefix_ids, last_only=True)
            engine.cache_prefix()
            try:
                native_dfa = compile_tool_dfa(config['tools'], tokenizer)
            except GrammarTooLarge:
                native_dfa = None
            startup.update(prefix_setup_ms=(time.perf_counter()-start)*1000,
                           prefix_tokens=len(prefix_ids))
        startup['kv_cache'] = config.get('kv_cache', 'fp32') if kind in ('fp32', 'sdot') else 'fp32' if kind == 'torch' else 'official_internal'
        startup['grammar_backend'] = ('native_dfa' if native_dfa is not None else 'python_regex') if kind in ('fp32', 'sdot') else 'python_regex' if kind == 'torch' else 'official_internal'
        pin_all_threads(set(config['cpus']))
        connection.send({'ready': startup})
        while True:
            case = connection.recv()
            if case is None:
                break
            if kind == 'official':
                cpu_start = time.process_time()
                start = time.perf_counter()
                official.reset()
                result = official.complete(case['query'], config['max_new_tokens'])
                result['wall_ms'] = (time.perf_counter()-start)*1000
                result['process_cpu_ms'] = (time.process_time()-cpu_start)*1000
                result['decode_tps'] = result['response'].get('decode_tps')
                result['prefill_tps'] = result['response'].get('prefill_tps')
                result['exact_call_match'] = strict_json_equal(
                    result['response'].get('function_calls'), case['expected_calls'])
            else:
                grammar = ToolGrammar(config['tools'], tokenizer)
                output = []
                cpu_start = time.process_time()
                start = time.perf_counter()
                if kind == 'torch':
                    # NeedleModel creates new cache tensors on each forward;
                    # its input prefix cache is read-only and can be shared.
                    cache = prefix_cache
                else:
                    engine.reset_to_prefix()
                restore_ms = (time.perf_counter()-start)*1000
                prefill_start = time.perf_counter()
                if kind == 'torch':
                    logits, cache = consume(case['suffix_ids'], cache)
                else:
                    logits = engine.prefill(case['suffix_ids'], last_only=True)
                prefill_ms = (time.perf_counter()-prefill_start)*1000
                decode_start = time.perf_counter()
                forward_seconds = 0.0
                steps = 0
                if kind != 'torch' and native_dfa is not None:
                    first_token = grammar.select(logits)
                    output = engine.decode(first_token, max_new_tokens=config['max_new_tokens'], grammar_dfa=native_dfa)
                    decode_wall = time.perf_counter() - decode_start
                    steps = max(0, len(output) - 1)
                    forward_seconds = decode_wall
                else:
                    candidates = None
                    for index in range(config['max_new_tokens']):
                        if candidates is not None:
                            token = grammar.select_candidate(candidates, logits)
                        else:
                            token = grammar.select(logits)
                        grammar.accept(token)
                        output.append(token)
                        if token in (1, 5) or grammar.finished or index == config['max_new_tokens']-1:
                            break
                        forward_start = time.perf_counter()
                        if kind == 'torch':
                            logits, cache = consume([token], cache)
                        else:
                            candidates = grammar.candidate_tokens()
                            logits = engine.step(token) if candidates is None else engine.step_candidates(token, candidates)
                        forward_seconds += time.perf_counter()-forward_start
                        steps += 1
                decode_ms = (time.perf_counter()-decode_start)*1000
                wall_ms = (time.perf_counter()-start)*1000
                process_cpu_ms = (time.process_time()-cpu_start)*1000
                response = parse_response(tokenizer.decode(output))
                result = dict(response=response, wall_ms=wall_ms, process_cpu_ms=process_cpu_ms,
                              prefix_restore_ms=restore_ms, prefill_ms=prefill_ms,
                              prefill_tps=len(case['suffix_ids'])*1000/prefill_ms,
                              decode_ms=decode_ms, decode_forward_steps=steps,
                              decode_tps=steps*1000/decode_ms,
                              decode_forward_tps=steps/forward_seconds if steps else None,
                              generated_tokens=len(output), token_ids=output,
                              prompt_tokens=len(prefix_ids)+len(case['suffix_ids']),
                              exact_call_match=strict_json_equal(response['function_calls'],case['expected_calls']))
                if kind == 'torch' and prefix_cache.position != len(prefix_ids):
                    raise AssertionError('PyTorch input prefix cache was mutated')
            result.update(kv_cache=startup['kv_cache'], grammar_backend=startup['grammar_backend'])
            connection.send({'result': result})
    except BaseException:
        connection.send({'error': traceback.format_exc()})
    finally:
        if official is not None:
            official.close()
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='artifacts/official/needle2.cact')
    default_lib = 'artifacts/official/python/libneedle.so'
    if not Path(default_lib).exists() and Path('artifacts/official/libneedle.dylib').exists():
        default_lib = 'artifacts/official/libneedle.dylib'
    parser.add_argument('--library', default=default_lib)
    parser.add_argument('--tools', default='examples/tools.json')
    parser.add_argument('--cases', default='benchmarks/cases.jsonl')
    parser.add_argument('--native-threads', type=int, default=4)
    parser.add_argument('--torch-threads', default='1,2,4')
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--kv-cache', choices=['fp32', 'int8'], default='fp32')
    parser.add_argument('--affinity', default='0,1,2,3')
    parser.add_argument('--output', default='reports/backend_comparison.json')
    args = parser.parse_args()
    torch_threads = [int(value) for value in args.torch_threads.split(',')]
    if args.repeat < 1 or args.native_threads < 1 or not torch_threads or min(torch_threads) < 1 or args.max_new_tokens < 2:
        parser.error('positive repeat and threads, max_new_tokens >= 2 required')
    cpus = [int(value) for value in args.affinity.split(',')]
    if hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, set(cpus))
    from needle2.official import sha256
    from needle2.tokenizer import RefTokenizer
    from needle2.prompt import render_prompt
    from benchmark_official import hardware, workload
    tools = json.loads(Path(args.tools).read_text())
    cases = [json.loads(line) for line in Path(args.cases).read_text().splitlines() if line.strip()]
    tokenizer = RefTokenizer.from_cact(args.model)
    prefix_ids = None
    for case in cases:
        ids = [2]+tokenizer.encode(render_prompt(case['query'], tools))
        prefix = ids.index(tokenizer.p2id['</tools>'])+1
        if prefix_ids is not None and prefix_ids != ids[:prefix]:
            raise ValueError('cases must share a tools prefix')
        prefix_ids = ids[:prefix]
        case['suffix_ids'] = ids[prefix:]
    specs = [dict(name='official',kind='official',threads='automatic'),
             dict(name='native_fp32',kind='fp32',threads=args.native_threads),
             dict(name='native_sdot',kind='sdot',threads=args.native_threads)]
    specs += [dict(name=f'pytorch_t{threads}',kind='torch',threads=threads) for threads in torch_threads]
    config = dict(model=str(Path(args.model).resolve()),library=str(Path(args.library).resolve()),
                  tools=tools,prefix_ids=prefix_ids,cpus=cpus,max_new_tokens=args.max_new_tokens,
                  kv_cache=args.kv_cache)
    sources = [Path(__file__), *sorted((ROOT/'needle2').rglob('*.py')),*sorted((ROOT/'needle2/csrc').glob('*.cpp'))]
    source_hashes = {str(p.relative_to(ROOT)):sha256(p) for p in sources}
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),hardware=hardware(),
                  workload_before=workload(),model_sha256=sha256(args.model),
                  official_library_sha256=sha256(args.library),source_sha256=source_hashes,
                  tools=tools,cases_sha256=sha256(args.cases),prefix_tokens=len(prefix_ids),
                  native_kv_cache=args.kv_cache,repeat=args.repeat,backends=specs,startup={},cases=[],
                  environment={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OMP_WAIT_POLICY','GOMP_SPINCOUNT','OPENBLAS_NUM_THREADS']},
                  notes=[
                      'Each backend has its own persistent process; identical CPU affinity, serial interleaved requests, no simultaneous inference.',
                      '20 ms idle between requests allows inactive runtime thread pools to settle; idle and IPC are excluded from worker wall time.',
                      'Each engine prepares and reuses its tools prefix. Timed requests include cache restoration, query prefill, grammar selection and decode.',
                      'Native token DFA and PyTorch regex gate implement the same bounded schema and UTF-8 language; native falls back to regex on compilation size limits. Actual grammar/KV modes are recorded per result. Official uses its built-in prompt/grammar/confidence.',
                      'PyTorch uses the deployed CACT dequantized to dense FP32, eval/inference_mode on CPU, no torch.compile.',
                      'PyTorch reuses unchanged model operations but skips unused LM head positions: no prefix logits; final query position and each decode position only.',
                      'Independent decode TPS is actual forward steps divided by grammar-inclusive decode time; official TPS is self-reported, not identical operator work.',
                      'Initialization is one cold observation per worker, excluding Python imports; warm request metrics use repeated medians.',
                  ])
    context = mp.get_context('spawn')
    workers = {}
    dest = Path(args.output)
    dest.parent.mkdir(parents=True,exist_ok=True)
    try:
        for spec in specs:
            parent, child = context.Pipe()
            process = context.Process(target=worker,args=(child,spec,config))
            process.start()
            child.close()
            workers[spec['name']] = (parent,process)
            message = parent.recv()
            if 'error' in message:
                raise RuntimeError(message['error'])
            report['startup'][spec['name']] = message['ready']
            print('ready', spec['name'],flush=True)

        def request(name,case):
            time.sleep(.02)
            connection,_ = workers[name]
            connection.send(case)
            message = connection.recv()
            if 'error' in message:
                raise RuntimeError(message['error'])
            return message['result']

        names = list(workers)
        for case_index,case in enumerate(cases):
            for name in names:
                warm = request(name,case)
                if not warm['exact_call_match']:
                    print('warmup call mismatch:', name,case['id'],flush=True)
            entry = dict(**case,results={name:[] for name in names},execution_order=[])
            report['cases'].append(entry)
            for repeat in range(args.repeat):
                offset=(case_index+repeat)%len(names)
                order=names[offset:]+names[:offset]
                entry['execution_order'].append(order)
                for name in order:
                    entry['results'][name].append(request(name,case))
                print(case['id'],repeat+1,'/',args.repeat,flush=True)
        report['summary'] = {}
        for name in names:
            rows = [row for case in report['cases'] for row in case['results'][name]]
            report['summary'][name] = dict(
                decode_tps_median=statistics.median(row['decode_tps'] for row in rows),
                wall_ms_median=statistics.median(row['wall_ms'] for row in rows),
                wall_ms_min=min(row['wall_ms'] for row in rows),
                wall_ms_max=max(row['wall_ms'] for row in rows),
                process_cpu_ms_median=statistics.median(row['process_cpu_ms'] for row in rows),
                prefill_tps_median=statistics.median(row['prefill_tps'] for row in rows),
                exact_call_matches=sum(row['exact_call_match'] for row in rows),runs=len(rows))
        report['independent_token_agreement'] = all(
            row['token_ids']==case['results']['native_fp32'][0]['token_ids']
            for case in report['cases'] for name in names if name!='official' for row in case['results'][name])
        print(json.dumps(report['summary'],indent=2),flush=True)
    except BaseException:
        report['error'] = traceback.format_exc()
        raise
    finally:
        for connection, process in workers.values():
            if process.is_alive():
                try:
                    connection.send(None)
                except (BrokenPipeError,EOFError):
                    pass
            process.join(timeout=3)
            if process.is_alive():
                process.terminate()
                process.join(timeout=3)
            connection.close()
        report['workload_after']=workload()
        report['source_changed_during_run']=source_hashes!={str(p.relative_to(ROOT)):sha256(p) for p in sources}
        dest.write_text(json.dumps(report,indent=2)+'\n')
    if any(row['exact_call_matches']!=row['runs'] for row in report['summary'].values()):
        raise SystemExit('one or more output correctness checks failed; see report')


if __name__ == '__main__':
    main()
