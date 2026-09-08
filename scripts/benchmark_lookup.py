#!/usr/bin/env python3
"""Compare exact FP32 CQ activation tables against the baseline weight LUT.

The default remains fused NEON. Alternate lookup modes do not quantize inputs.
"""
import argparse
import ctypes as ct
import json
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from needle2.archive import Archive
from needle2.native import NativeCQ, _library


def measure(fn, count):
    for _ in range(5):
        fn()
    values, cpus = [], []
    for _ in range(count):
        cpu = time.process_time_ns()
        start = time.perf_counter_ns()
        fn()
        values.append((time.perf_counter_ns() - start) / 1e3)
        cpus.append((time.process_time_ns() - cpu) / 1e3)
    return dict(mean_us=statistics.mean(values), median_us=statistics.median(values),
                cpu_mean_us=statistics.mean(cpus), p90_us=float(np.quantile(values, .9)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="artifacts/official/needle2.cact")
    p.add_argument("--output", default="reports/cq_lookup_benchmark.json")
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--affinity", default="0,1,2,3")
    args = p.parse_args()
    os.sched_setaffinity(0, {int(v) for v in args.affinity.split(",")})
    archive = Archive.load(args.model)
    rng = np.random.default_rng(2026)
    report = dict(model_sha256=archive.sha256, affinity=sorted(os.sched_getaffinity(0)),
                  modes={"0": "fused NEON weight-byte LUT", "1": "activation byte LUT",
                         "2": "activation nibble LUT", "3": "activation group byte LUT",
                         "4": "per-thread activation group LUT"}, matrices=[], fused=[])
    for name in ("layer00.q_proj", "layer00.k_proj", "embedding"):
        q = NativeCQ.from_record(archive.tensors[name])
        x = rng.normal(size=q.shape[1]).astype(np.float32)
        expected = q.linear(x)
        for threads in (1, 2):
            for mode in (0, 1, 2, 3):
                actual = q.linear(x, threads, lookup=mode)
                item = measure(lambda: q.linear(x, threads, lookup=mode), args.iterations)
                item.update(name=name, threads=threads, mode=mode,
                            max_abs_error=float(np.max(np.abs(actual-expected))))
                report["matrices"].append(item)
    matrices = [NativeCQ.from_record(archive.tensors[f"layer00.{k}_proj"]) for k in ("q", "k", "v", "gate")]
    inputs = rng.normal(size=matrices[0].shape[1]).astype(np.float32)
    outputs = [np.empty(m.shape[0], np.float32) for m in matrices]
    expected = [m.linear(inputs) for m in matrices]
    handles = (ct.c_void_p * 4)(*[m._handle for m in matrices])
    pointers = (ct.c_void_p * 4)(*[y.ctypes.data for y in outputs])
    lib = _library()
    lib.needle2_cq_linear_many.argtypes = [ct.c_void_p, ct.c_int, ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_int]
    lib.needle2_cq_linear_many.restype = None
    for threads in (1, 2, 4):
        for mode in (-1, 0, 1, 2, 3, 4):
            def op():
                if mode == -1:
                    for i, matrix in enumerate(matrices):
                        outputs[i][:] = matrix.linear(inputs, threads)
                else:
                    lib.needle2_cq_linear_many(handles, 4, inputs.ctypes.data, pointers, threads, mode)
            item = measure(op, args.iterations)
            item.update(threads=threads, mode=mode,
                        max_abs_error=max(float(np.max(np.abs(a-b))) for a, b in zip(outputs, expected)))
            report["fused"].append(item)
    output = Path(args.output)
    output.parent.mkdir(exist_ok=True, parents=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
