#!/usr/bin/env python3
"""Fixed-context, fixed-token end-to-end FP32 projection lookup comparison."""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from needle2.archive import Archive
from needle2.native import NativeEngine


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="artifacts/official/needle2.cact")
    p.add_argument("--output", default="reports/lookup_engine_benchmark.json")
    p.add_argument("--context", type=int, default=190)
    p.add_argument("--decode", type=int, default=32)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--affinity", default="0,1,2,3")
    args = p.parse_args()
    os.sched_setaffinity(0, {int(v) for v in args.affinity.split(",")})
    archive = Archive.load(args.model)
    rng = np.random.default_rng(1309)
    ids = rng.integers(1, archive.metadata["vocab_size"], args.context + args.decode)
    cases = [(threads, mode) for threads in (1, 2) for mode in (0, 3, 4)]
    expected = None
    results = []
    for repeat in range(args.repeats):
        for threads, mode in cases if repeat % 2 == 0 else reversed(cases):
            engine = NativeEngine(archive, threads=threads, projection_lookup=mode)
            engine.reset(prefix_len=16)
            engine.prefill(ids[:args.context], last_only=True)
            elapsed, cpu, outputs = [], [], []
            for token in ids[args.context:]:
                c = time.process_time_ns()
                t = time.perf_counter_ns()
                outputs.append(engine.step(int(token)))
                elapsed.append((time.perf_counter_ns() - t) / 1e6)
                cpu.append((time.process_time_ns() - c) / 1e6)
            actual = np.stack(outputs)
            if expected is None:
                expected = actual
            item = dict(repeat=repeat, threads=threads, mode=mode,
                        tokens_per_second=1000/statistics.mean(elapsed), mean_ms=statistics.mean(elapsed),
                        median_ms=statistics.median(elapsed), p10_ms=float(np.quantile(elapsed,.1)),
                        p90_ms=float(np.quantile(elapsed,.9)), cpu_mean_ms=statistics.mean(cpu),
                        max_abs_error=float(np.abs(actual-expected).max()),
                        rms_error=float(np.sqrt(np.mean((actual-expected)**2))),
                        argmax_agreement=float(np.mean(actual.argmax(-1)==expected.argmax(-1))))
            results.append(item)
            print(json.dumps(item), flush=True)
            del engine
    result = dict(model_sha256=archive.sha256, affinity=sorted(os.sched_getaffinity(0)),
                  context=args.context, decode=args.decode, prefix=16, token_ids=ids.tolist(), results=results)
    output = Path(args.output)
    output.parent.mkdir(exist_ok=True, parents=True)
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
