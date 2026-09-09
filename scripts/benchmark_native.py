#!/usr/bin/env python3
"""Reproducible CQ microkernels, fixed-token decode, and initial prefill benchmark.

Run without other CPU-intensive jobs. This measures this open implementation;
comparison to the official tool runtime must use its API and equivalent work.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from needle2.archive import Archive
from needle2.native import NativeCQ, NativeEngine, features


def measure(fn, iterations, warmup=5):
    for _ in range(warmup):
        fn()
    samples, cpu_samples = [], []
    for _ in range(iterations):
        cpu_t = time.process_time_ns()
        t = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - t) / 1e6)
        cpu_samples.append((time.process_time_ns() - cpu_t) / 1e6)
    return {"iterations": iterations, "mean_ms": statistics.mean(samples),
            "median_ms": statistics.median(samples), "p10_ms": float(np.quantile(samples, .1)),
            "p90_ms": float(np.quantile(samples, .9)), "max_ms": max(samples),
            "process_cpu_mean_ms": statistics.mean(cpu_samples),
            "cpu_wall_ratio": sum(cpu_samples) / sum(samples)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="artifacts/official/needle2.cact")
    p.add_argument("--output", default="artifacts/reports/native_kernel_benchmark.json")
    p.add_argument("--threads", default="1,2,4")
    p.add_argument("--affinity", help="comma-separated CPU IDs, e.g. 0,1,2,3")
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--decode-tokens", type=int, default=64)
    p.add_argument("--prefill-tokens", type=int, default=190)
    args = p.parse_args()
    if args.affinity:
        os.sched_setaffinity(0, {int(cpu) for cpu in args.affinity.split(",")})
    threads = [int(s) for s in args.threads.split(",")]
    archive = Archive.load(args.model)
    torch.set_num_threads(1)
    rng = np.random.default_rng(417)
    result = {"model_sha256": archive.sha256, "platform": platform.platform(),
              "architecture": platform.machine(), "cpu_count": os.cpu_count(),
              "affinity_cpus": sorted(os.sched_getaffinity(0)), "requested_affinity": args.affinity, "native_features": features(),
              "torch_version": torch.__version__, "numpy_version": np.__version__,
              "measurement": "wall clock; float32 inputs/accumulation; no activation or KV int8 rounding",
              "matrices": [], "decode": [], "prefill": []}
    for name in ("embedding", "layer00.q_proj", "layer00.k_proj"):
        rec = archive.tensors[name]
        q = NativeCQ.from_record(rec)
        x = rng.normal(size=rec.shape[-1]).astype(np.float32)
        dense = torch.from_numpy(rec.dequantize())
        xt = torch.from_numpy(x)
        expected = (dense @ xt).numpy()
        for th in threads:
            actual = q.linear(x, th)
            stats = measure(lambda: q.linear(x, th), args.iterations)
            stats.update(name=name, shape=list(rec.shape), bits=rec.bits, threads=th,
                         max_abs_error=float(np.max(np.abs(actual - expected))),
                         payload_bytes=len(rec.blob))
            result["matrices"].append(stats)
        for th in threads:
            torch.set_num_threads(th)
            stats = measure(lambda: dense @ xt, args.iterations)
            stats.update(name=name, shape=list(rec.shape), backend="torch_dense_fp32", threads=th)
            result["matrices"].append(stats)
        del q, dense
    # Teacher-forced IDs make every implementation perform identical decode work.
    ids = rng.integers(1, archive.metadata["vocab_size"], args.decode_tokens + 16, dtype=np.int64)
    prompt = rng.integers(1, archive.metadata["vocab_size"], args.prefill_tokens, dtype=np.int64)
    for th in threads:
        torch.set_num_threads(th)
        e = NativeEngine(archive, threads=th)
        for token in ids[:16]:
            e.step(int(token), compute_logits=False)
        it = iter(ids[16:])
        stats = measure(lambda: e.step(int(next(it))), args.decode_tokens, warmup=0)
        stats.update(threads=th, tokens_per_second=1000 / stats["mean_ms"], prefix_tokens=16,
                     token_ids=ids.tolist())
        result["decode"].append(stats)
        for backend in ("native", "torch"):
            samples, cpu_samples = [], []
            for repeat in range(3):
                e.reset(prefix_len=16)
                cpu_t = time.process_time()
                t = time.perf_counter()
                output = e.prefill(prompt, last_only=True, backend=backend)
                samples.append((time.perf_counter() - t) * 1000)
                cpu_samples.append((time.process_time() - cpu_t) * 1000)
            result["prefill"].append({"threads": th, "backend": backend, "tokens": len(prompt),
                                      "first_ms": samples[0], "reuse_mean_ms": statistics.mean(samples[1:]),
                                      "samples_ms": samples, "process_cpu_samples_ms": cpu_samples, "argmax": int(output.argmax()),
                                      "logits_last_only": True,
                                      "dense_weights_retained": backend == "torch"})
        del e
        gc.collect()
        print(json.dumps({"threads": th, "decode": stats["tokens_per_second"]}), flush=True)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
