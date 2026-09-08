#!/usr/bin/env python3
"""Measure an opt-in ARM SDOT approximation; never changes production weights.

Both shared codebook centroids and group-rotated activations are quantized to
signed INT8. Weights remain packed CQ2/CQ4. Results are local matrix tests, not
evidence of unchanged end-to-end tool-calling accuracy.
"""
from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from needle2.archive import Archive
from needle2.native import NativeCQ
from needle2.quantize import hadamard, unpack_indices


def build():
    if platform.machine().lower() not in ("aarch64", "arm64"):
        raise RuntimeError("This experiment requires an ARM64 CPU with DotProd")
    features = Path("/proc/cpuinfo").read_text().lower()
    if "asimddp" not in features and "dotprod" not in features:
        raise RuntimeError("CPU does not advertise ARM DotProd; refusing unsupported instructions")
    source = ROOT / "experimental/sdot.cpp"
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    flags = ["-O3", "-DNDEBUG", "-std=c++17", "-shared", "-fPIC", "-march=armv8.2-a+dotprod"]
    digest = hashlib.sha256(source.read_bytes() + repr((compiler, flags)).encode()).hexdigest()[:16]
    target = ROOT / "artifacts/experimental" / f"sdot-{digest}.so"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        subprocess.run(compiler + flags + [str(source), "-o", str(target)], check=True, capture_output=True)
    lib = ct.CDLL(str(target))
    lib.needle2_sdot_supported.restype = ct.c_int
    if not lib.needle2_sdot_supported():
        raise RuntimeError("runtime HWCAP_ASIMDDP check failed")
    lib.needle2_sdot_create.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_void_p]
    lib.needle2_sdot_create.restype = ct.c_void_p
    lib.needle2_sdot_destroy.argtypes = [ct.c_void_p]
    lib.needle2_sdot_linear.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p]
    lib.needle2_sdot_prepare.argtypes = [ct.c_void_p, ct.c_void_p]
    lib.needle2_sdot_prepare.restype = ct.c_void_p
    lib.needle2_sdot_prepared_destroy.argtypes = [ct.c_void_p]
    lib.needle2_sdot_multiply.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p]
    return lib, target


class SdotMatrix:
    def __init__(self, lib, record):
        self.lib, self.record = lib, record
        self.packed = np.ascontiguousarray(record.packed, dtype=np.uint8)
        self.norms = np.ascontiguousarray(record.norms, dtype=np.float16)
        self.codebook = np.ascontiguousarray(record.codebook, dtype=np.float32)
        self.handle = lib.needle2_sdot_create(self.packed.ctypes.data, self.norms.ctypes.data,
                                             *record.shape, record.bits, record.group_size,
                                             self.codebook.ctypes.data)
        if not self.handle:
            raise RuntimeError("could not construct SDOT matrix")

    def __del__(self):
        if getattr(self, "handle", None):
            self.lib.needle2_sdot_destroy(self.handle)
            self.handle = None

    def linear(self, vector):
        x = np.ascontiguousarray(vector, dtype=np.float32)
        if x.shape != (self.record.shape[1],):
            raise ValueError(f"expected vector of length {self.record.shape[1]}")
        y = np.empty(self.record.shape[0], dtype=np.float32)
        self.lib.needle2_sdot_linear(self.handle, x.ctypes.data, y.ctypes.data)
        return y


def integer_reference(record, vector):
    """Independent NumPy calculation using the original continuous bit order."""
    group = record.group_size
    x = np.pad(vector, (0, record.in_pad - len(vector))).reshape(-1, group)
    rotated = hadamard(x)
    scale = np.max(np.abs(rotated), axis=-1) / np.float32(127)
    scale = np.where(scale > 0, scale, np.float32(1))
    qx = np.rint(rotated / scale[:, None]).clip(-127, 127).astype(np.int32)
    cb_scale = np.max(np.abs(record.codebook)) / np.float32(127)
    cb = np.rint(record.codebook / cb_scale).clip(-127, 127).astype(np.int32)
    indices = unpack_indices(record.packed, record.bits, record.in_pad)
    qweight = cb[indices].reshape(record.shape[0], -1, group)
    dots = np.sum(qweight * qx[None], axis=-1, dtype=np.int32)
    return np.sum(dots.astype(np.float32) * record.norms.astype(np.float32) * cb_scale * scale,
                  axis=-1, dtype=np.float32)


def error_metrics(actual, expected):
    a, e = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    difference = a - e
    norms = np.linalg.norm(a) * np.linalg.norm(e)
    return {"max_abs": float(np.max(np.abs(difference))), "mse": float(np.mean(difference ** 2)),
            "relative_l2": float(np.linalg.norm(difference) / max(np.linalg.norm(e), 1e-30)),
            "cosine_similarity": float(np.sum(a * e) / norms) if norms else 1.0}


def real_activations(archive):
    import torch
    from needle2.model import NeedleModel
    torch.set_num_threads(1)
    model = NeedleModel.from_archive(archive).eval()
    captured = {}

    def collect(name, x, weight):
        if name in ("layer00.q_proj", "embedding"):
            captured[name] = x.detach().cpu().numpy().reshape(-1, x.shape[-1]).copy()
        return torch.nn.functional.linear(x, weight)

    model.set_linear_backend(collect)
    with torch.inference_mode():
        model(torch.tensor([[2, 176, 45, 3, 128, 52, 765, 1298]]))
    return captured


def timed_pair(baseline, approximate, iterations, repeats):
    """Interleave order; CPU time distinguishes arithmetic from VM scheduling."""
    measurements = {"fp32": [], "sdot": []}
    for _ in range(8):
        baseline()
        approximate()
    for repeat in range(repeats):
        calls = [("fp32", baseline), ("sdot", approximate)]
        if repeat % 2:
            calls.reverse()
        for name, call in calls:
            cpu, wall = time.process_time_ns(), time.perf_counter_ns()
            for _ in range(iterations):
                call()
            measurements[name].append({"wall_us": (time.perf_counter_ns() - wall) / iterations / 1000,
                                       "cpu_us": (time.process_time_ns() - cpu) / iterations / 1000})
    report = {}
    for name, rows in measurements.items():
        report[name] = {"median_wall_us": float(np.median([row["wall_us"] for row in rows])),
                        "median_cpu_us": float(np.median([row["cpu_us"] for row in rows])),
                        "samples": rows}
    report["wall_speedup"] = report["fp32"]["median_wall_us"] / report["sdot"]["median_wall_us"]
    report["cpu_speedup"] = report["fp32"]["median_cpu_us"] / report["sdot"]["median_cpu_us"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts/official/needle2.cact")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/sdot_experiment.json")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--real-activations", action="store_true")
    parser.add_argument("--accuracy-only", action="store_true")
    parser.add_argument("--cpu", type=int, default=None)
    args = parser.parse_args()
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    lib, binary = build()
    archive = Archive.load(args.model)
    captured = real_activations(archive) if args.real_activations else {}
    rng = np.random.default_rng(218)
    report = {"experimental": True, "production_backend_changed": False,
              "platform": platform.platform(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
              "model_sha256": archive.sha256, "compiler_flag": "-march=armv8.2-a+dotprod",
              "shared_library": str(binary), "matrices": []}
    try:
        disassembly = subprocess.run(["objdump", "-d", str(binary)], check=True, capture_output=True, text=True).stdout
        report["sdot_in_disassembly"] = "sdot\t" in disassembly or "sdot " in disassembly
    except (OSError, subprocess.CalledProcessError):
        report["sdot_in_disassembly"] = None
    for name in ("layer00.q_proj", "embedding"):
        record = archive.tensors[name]
        baseline, approximate = NativeCQ.from_record(record), SdotMatrix(lib, record)
        random_vectors = rng.normal(size=(8, record.shape[1])).astype(np.float32)
        source = {"gaussian": random_vectors}
        if name in captured:
            source["real_model_activations"] = captured[name]
        results = {"name": name, "shape": list(record.shape), "bits": record.bits, "group_size": record.group_size,
                   "packed_payload_bytes": len(record.blob),
                   "extra_norm_scale_bytes": record.norms.size * 4,
                   "centroids_int8": np.rint(record.codebook / (np.abs(record.codebook).max() / 127)).astype(int).tolist(),
                   "accuracy": {}}
        for label, vectors in source.items():
            expected = np.stack([baseline.linear(x) for x in vectors])
            actual = np.stack([approximate.linear(x) for x in vectors])
            results["accuracy"][label] = error_metrics(actual, expected)
            results["accuracy"][label]["top1_agreement"] = float(np.mean(actual.argmax(-1) == expected.argmax(-1)))
        # Check independent integer math and both zero/one-hot activations.
        for vector in (random_vectors[0], np.zeros(record.shape[1], np.float32),
                       np.eye(1, record.shape[1], record.shape[1] - 1, dtype=np.float32)[0]):
            integer = integer_reference(record, vector)
            actual = approximate.linear(vector)
            np.testing.assert_allclose(actual, integer, atol=4e-4, rtol=2e-5)
        results["integer_reference_verified"] = True
        if not args.accuracy_only:
            x = np.ascontiguousarray(source.get("real_model_activations", random_vectors)[-1])
            y0, y1 = np.empty(record.shape[0], np.float32), np.empty(record.shape[0], np.float32)
            exact_call = lambda: baseline._lib.needle2_cq_linear(baseline._handle, x.ctypes.data, y0.ctypes.data, 1, 1)
            sdot_call = lambda: lib.needle2_sdot_linear(approximate.handle, x.ctypes.data, y1.ctypes.data)
            results["including_activation_preparation"] = timed_pair(exact_call, sdot_call, args.iterations, args.repeats)
            prepared = lib.needle2_sdot_prepare(approximate.handle, x.ctypes.data)
            try:
                only_kernel = lambda: lib.needle2_sdot_multiply(approximate.handle, prepared, y1.ctypes.data)
                results["fp32_full_vs_sdot_prepared_lower_bound"] = timed_pair(exact_call, only_kernel, args.iterations, args.repeats)
            finally:
                lib.needle2_sdot_prepared_destroy(prepared)
        report["matrices"].append(results)
        print(json.dumps(results, indent=2), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
