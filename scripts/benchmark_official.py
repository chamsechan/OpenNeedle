#!/usr/bin/env python3
"""Benchmark the official C ABI without importing needle or executing tools."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from needle2.official import (ENGINE_VERSION, HF_REVISION, OfficialEngine,
                             fetch_official_library, platform_tag, sha256, strict_json_equal)


def hardware():
    info = {"platform": platform.platform(), "machine": platform.machine(),
            "python": platform.python_version(), "logical_cpus": os.cpu_count(),
            "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None}
    try:
        result = subprocess.run(["lscpu", "-J"], capture_output=True, text=True, check=True)
        info["lscpu"] = json.loads(result.stdout)["lscpu"]
    except (OSError, subprocess.SubprocessError, ValueError, KeyError):
        info["processor"] = platform.processor()
    return info


def workload():
    result = {"load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None}
    for name in ("cpu.max", "cpu.stat"):
        path = Path("/sys/fs/cgroup") / name
        if path.exists():
            result[name] = path.read_text()
    return result


def pin_all_threads(cpus):
    tasks = Path("/proc/self/task")
    if tasks.exists():
        for task in tasks.iterdir():
            try:
                os.sched_setaffinity(int(task.name), cpus)
            except ProcessLookupError:
                pass


def thread_affinities():
    tasks = Path("/proc/self/task")
    if not tasks.exists():
        return None
    result = {}
    for task in tasks.iterdir():
        try:
            result[task.name] = sorted(os.sched_getaffinity(int(task.name)))
        except ProcessLookupError:
            pass
    return result


def summary(values):
    values = sorted(float(x) for x in values)
    if not values:
        return None
    return {"min": values[0], "median": statistics.median(values),
            "mean": statistics.mean(values), "max": values[-1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "artifacts/official/needle2.cact")
    parser.add_argument("--library", type=Path)
    parser.add_argument("--fetch-library", action="store_true", help="Explicitly download the pinned official wheel")
    parser.add_argument("--tools", type=Path, default=ROOT / "examples/tools.json")
    parser.add_argument("--cases", type=Path, default=ROOT / "benchmarks/cases.jsonl")
    parser.add_argument("--query", help="Run one custom query instead of the case file")
    parser.add_argument("--system", default="")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--cpu-affinity", help="Pin process and native workers to comma-separated CPU IDs, e.g. 0")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/reports/official_benchmark.json")
    args = parser.parse_args()
    if args.repeat < 1 or args.warmup < 0:
        parser.error("repeat must be >= 1 and warmup >= 0")
    if args.cpu_affinity:
        if not hasattr(os, "sched_setaffinity"):
            parser.error("CPU affinity is not supported on this platform")
        try:
            cpus = {int(value) for value in args.cpu_affinity.split(",")}
            os.sched_setaffinity(0, cpus)
        except (ValueError, OSError) as exc:
            parser.error(f"invalid CPU affinity: {exc}")
    os.environ["NEEDLE_TELEMETRY"] = "0"
    os.environ["DO_NOT_TRACK"] = "1"
    destination = ROOT / "artifacts/official/python"
    library = args.library or destination / platform_tag()[1]
    if args.fetch_library:
        if args.library:
            parser.error("--library and --fetch-library are mutually exclusive")
        library = fetch_official_library(destination)
    if not library.is_file():
        parser.error(f"Library missing: {library}; use --fetch-library once")
    tools = json.loads(args.tools.read_text())
    cases = ([{"id": "custom", "query": args.query}] if args.query is not None else
             [json.loads(line) for line in args.cases.read_text().splitlines() if line.strip()])
    if not cases:
        parser.error("no benchmark cases")
    report = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "implementation": "official_c_api", "engine_version_expected": ENGINE_VERSION,
              "hf_revision_expected": HF_REVISION, "hardware": hardware(),
              "workload_before": workload(),
              "weights": {"path": str(args.weights.resolve()), "sha256": sha256(args.weights)},
              "library": {"path": str(library.resolve()), "sha256": sha256(library)},
              "settings": {"repeat": args.repeat, "warmup": args.warmup,
                           "max_new_tokens": args.max_new_tokens, "system": args.system,
                           "reset_before_every_run": True, "tools": tools,
                           "telemetry": False, "execute_function_calls": False,
                           "grammar": "official default", "retrieval": "official default",
                           "thread_count": "official automatic selection"},
              "notes": ["Small deterministic smoke cases; not an accuracy benchmark.",
                        "Throughput fields are reported by the official engine; wall_ms uses perf_counter.",
                        "No logits or token-step C ABI is provided by the official header.",
                        "Expected revision/version are provenance targets; verify library hash/metadata.",
                        "Process RSS includes Python and the loaded library, not only model state."],
              "cases": []}
    metadata_path = library.parent / "official_library.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("library_sha256") == report["library"]["sha256"]:
            report["library"]["download_metadata"] = metadata
    with OfficialEngine(library, args.weights, tools, system=args.system) as engine:
        if args.cpu_affinity:
            pin_all_threads(cpus)
        report["initialization"] = {"load_ms": engine.load_ms, "init_ms": engine.init_ms,
                                    "prefix_tokens": engine.prefix_tokens}
        report["thread_affinities"] = thread_affinities()
        for case in cases:
            print(f"official: {case['id']}", file=sys.stderr, flush=True)
            for _ in range(args.warmup):
                engine.reset()
                engine.complete(case["query"], args.max_new_tokens)
            runs = []
            for _ in range(args.repeat):
                engine.reset()
                run = engine.complete(case["query"], args.max_new_tokens)
                if "expected_calls" in case:
                    run["exact_call_match"] = strict_json_equal(run["response"].get("function_calls"), case["expected_calls"])
                runs.append(run)
            entry = {**case, "runs": runs, "wall_ms": summary(r["wall_ms"] for r in runs)}
            for field in ("prefill_tps", "decode_tps", "peak_ram_mb"):
                entry[field] = summary(r["response"][field] for r in runs
                                       if isinstance(r["response"].get(field), (int, float)))
            entry["deterministic_calls"] = all(r["response"].get("function_calls") ==
                                                runs[0]["response"].get("function_calls") for r in runs)
            report["cases"].append(entry)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["process_peak_rss_bytes"] = rss if platform.system() == "Darwin" else rss * 1024
    report["workload_after"] = workload()
    all_runs = [r for case in report["cases"] for r in case["runs"]]
    scored = [r["exact_call_match"] for r in all_runs if "exact_call_match" in r]
    report["summary"] = {"measured_runs": len(all_runs),
                         "exact_call_matches": sum(scored) if scored else None,
                         "scored_runs": len(scored),
                         "wall_ms": summary(r["wall_ms"] for r in all_runs),
                         "decode_tps": summary(r["response"]["decode_tps"] for r in all_runs
                                                if "decode_tps" in r["response"])}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), **report["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
