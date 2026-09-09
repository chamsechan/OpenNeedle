#!/usr/bin/env python3
"""Compare converted .cact files using the same official C ABI and fixed cases."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from needle2.official import OfficialEngine, HF_REVISION, ENGINE_VERSION, sha256, strict_json_equal
from benchmark_official import hardware, workload, pin_all_threads, thread_affinities

TIMING_FIELDS = {"prefill_tps", "decode_tps"}
RUNTIME_FIELDS = TIMING_FIELDS | {"peak_ram_mb"}


def without(response, fields):
    return {key: value for key, value in response.items() if key not in fields}


def differences(left, right):
    return [key for key in sorted(set(left) | set(right)) if left.get(key) != right.get(key)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=ROOT / "artifacts/official/python/libneedle.so")
    parser.add_argument("--models", type=Path, nargs="+", default=[ROOT / "artifacts/official/needle2.cact",
                                                                  ROOT / "artifacts/roundtrip.cact",
                                                                  ROOT / "artifacts/from_master.cact"])
    parser.add_argument("--tools", type=Path, default=ROOT / "examples/tools.json")
    parser.add_argument("--cases", type=Path, default=ROOT / "benchmarks/cases.jsonl")
    parser.add_argument("--cpu-affinity", help="Comma-separated CPU IDs")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/reports/conversion_parity.json")
    args = parser.parse_args()
    if len(args.models) < 2:
        parser.error("at least two models are required; the first is the reference")
    cpus = {int(x) for x in args.cpu_affinity.split(",")} if args.cpu_affinity else None
    if cpus:
        os.sched_setaffinity(0, cpus)
    cases = [json.loads(line) for line in args.cases.read_text().splitlines() if line.strip()]
    tools = json.loads(args.tools.read_text())
    report = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "hardware": hardware(), "workload_before": workload(),
              "official_engine_version_expected": ENGINE_VERSION, "hf_revision_expected": HF_REVISION,
              "library_sha256": sha256(args.library),
              "settings": {"tools": tools, "cases": cases, "max_new_tokens": args.max_new_tokens,
                           "reset_before_each_case": True, "execute_function_calls": False,
                           "telemetry": False},
              "notes": ["These are three smoke cases, not full benchmark quality equivalence.",
                        "response_except_timing_match retains peak_ram_mb, which can change across loads.",
                        "semantic_response_match excludes only prefill_tps, decode_tps, peak_ram_mb.",
                        "Timing is diagnostic under concurrent host workload, not a speed comparison."],
              "models": []}
    baseline = None
    for path in args.models:
        print(f"official conversion parity: {path.name}", file=sys.stderr, flush=True)
        entry = {"path": str(path.resolve()), "sha256": sha256(path), "cases": []}
        with OfficialEngine(args.library, path, tools) as engine:
            if cpus:
                pin_all_threads(cpus)
            entry["thread_affinities"] = thread_affinities()
            for case in cases:
                engine.reset()
                run = engine.complete(case["query"], args.max_new_tokens)
                expected = case.get("expected_calls")
                entry["cases"].append({"id": case["id"], **run,
                                       "expected_call_match": strict_json_equal(run["response"].get("function_calls"), expected)
                                       if expected is not None else None})
        if baseline is None:
            baseline = entry
        else:
            for actual, reference in zip(entry["cases"], baseline["cases"]):
                a, b = actual["response"], reference["response"]
                actual["reference_call_match"] = strict_json_equal(a.get("function_calls"), b.get("function_calls"))
                actual["response_except_timing_match"] = without(a, TIMING_FIELDS) == without(b, TIMING_FIELDS)
                actual["semantic_response_match"] = without(a, RUNTIME_FIELDS) == without(b, RUNTIME_FIELDS)
                actual["different_response_fields"] = differences(a, b)
            entry["all_calls_match_reference"] = all(x["reference_call_match"] for x in entry["cases"])
            entry["all_semantic_responses_match_reference"] = all(x["semantic_response_match"] for x in entry["cases"])
        report["models"].append(entry)
    report["workload_after"] = workload()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "models": [
        {"path": m["path"], "all_calls_match_reference": m.get("all_calls_match_reference"),
         "all_semantic_responses_match_reference": m.get("all_semantic_responses_match_reference")}
        for m in report["models"]]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
