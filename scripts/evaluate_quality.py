#!/usr/bin/env python3
"""Small manually specified tool cases: independent runtime vs official C ABI."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from needle2.inference import generate
from needle2.official import OfficialEngine, HF_REVISION, ENGINE_VERSION, sha256, strict_json_equal
from benchmark_official import hardware, workload, pin_all_threads


def source_hashes():
    files = [ROOT / "needle2" / name for name in
             ("native.py", "inference.py", "grammar.py", "prompt.py", "archive.py", "tokenizer.py")]
    files.extend(sorted((ROOT / "needle2/csrc").glob("*.cpp")))
    return {str(path.relative_to(ROOT)): sha256(path) for path in files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "artifacts/official/needle2.cact")
    parser.add_argument("--library", type=Path, default=ROOT / "artifacts/official/python/libneedle.so")
    parser.add_argument("--tools", type=Path, default=ROOT / "benchmarks/quality_tools.json")
    parser.add_argument("--cases", type=Path, default=ROOT / "benchmarks/quality_cases.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/quality.json")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--prefill-backend", choices=("native", "torch"), default="native")
    parser.add_argument("--matmul", choices=("fp32", "sdot"), default="fp32")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cpu-affinity")
    args = parser.parse_args()
    cpus = {int(x) for x in args.cpu_affinity.split(",")} if args.cpu_affinity else None
    if cpus:
        os.sched_setaffinity(0, cpus)
    if args.prefill_backend == "torch":
        import torch
        torch.set_num_threads(args.threads)
    tools = json.loads(args.tools.read_text())
    cases = [json.loads(line) for line in args.cases.read_text().splitlines() if line.strip()]
    if args.limit is not None:
        cases = cases[:args.limit]
    from needle2.native import build_native
    native_library = build_native()
    report = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "hardware": hardware(), "workload_before": workload(),
              "model_sha256": sha256(args.weights), "official_library_sha256": sha256(args.library),
              "independent_source_sha256": source_hashes(),
              "independent_library_sha256": sha256(native_library),
              "official_version_expected": ENGINE_VERSION, "official_hf_revision_expected": HF_REVISION,
              "tools": tools, "cases_sha256": sha256(args.cases),
              "settings": {"threads_independent": args.threads, "prefill_backend": args.prefill_backend,"matmul":args.matmul,
                           "max_new_tokens": args.max_new_tokens, "independent_activation_bits": 0,
                           "rotated_matmul_activation_bits": 8 if args.matmul == "sdot" else 32,
                           "independent_kv_arithmetic": "fp32",
                           "independent_grammar": "restricted byte-level schema grammar; canonical property order",
                           "official_grammar": "official default", "execute_function_calls": False,
                           "telemetry": False, "reset_before_each_official_case": True},
              "notes": ["Manually authored diagnostic cases, not a representative public accuracy benchmark.",
                        "Expected calls were fixed in the case file before evaluating either model.",
                        "Each expected result is an ordered exact match of function names and every argument.",
                        "Both engines use the same three tool schemas and query; their prompt rendering differs.",
                        "Scores compare raw function_calls; no confidence threshold or validation-based suppression is applied.",
                        "Official envelopes include negation/ungrounded flags and confidence that a product may use to reject candidates.",
                        "Independent inference includes every tool and does not reproduce official tool retrieval.",
                        ("Independent SDOT approximates rotated weights/activations with int8; KV remains FP32."
                         if args.matmul == "sdot" else
                         "Independent FP32 activation/KV arithmetic differs from the official int8 path."),
                        "Wall times include setup and may reflect host contention; not a performance comparison.",
                        "One sample per case, greedy/default decoding; agreement does not establish general parity."],
              "cases": []}
    # Run official cases together. The C ABI has no destructor, so its worker
    # pool may remain allocated; none of these times are performance evidence.
    references = []
    with OfficialEngine(args.library, args.weights, tools) as engine:
        if cpus:
            pin_all_threads(cpus)
        for case in cases:
            print(f"quality official: {case['id']}", file=sys.stderr, flush=True)
            engine.reset()
            references.append(engine.complete(case["query"], args.max_new_tokens))
    for case, reference in zip(cases, references):
        print(f"quality independent: {case['id']}", file=sys.stderr, flush=True)
        started = time.perf_counter()
        try:
            independent = generate(args.weights, case["query"], tools=tools, backend="native",
                                   threads=args.threads, prefill_backend=args.prefill_backend,
                                   max_new_tokens=args.max_new_tokens, constrain=True,matmul=args.matmul)
            error = None
        except Exception as exc:
            independent = {}
            error = f"{type(exc).__name__}: {exc}"
        official_calls = reference["response"].get("function_calls")
        independent_calls = independent.get("function_calls")
        entry = {**case, "official": reference, "independent": independent,
                 "independent_wall_ms_including_setup": 1000 * (time.perf_counter() - started),
                 "independent_error": error,
                 "independent_parse_success": error is None and independent.get("parse_error") is None
                                              and isinstance(independent_calls, list),
                 "official_expected_match": strict_json_equal(official_calls, case["expected_calls"]),
                 "independent_expected_match": error is None and strict_json_equal(independent_calls, case["expected_calls"]),
                 "call_agreement": error is None and strict_json_equal(independent_calls, official_calls)}
        report["cases"].append(entry)
        # Persist completed cases during longer evaluation runs.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    report["workload_after"] = workload()
    report["source_changed_during_run"] = report["independent_source_sha256"] != source_hashes()
    report["summary"] = {"cases": len(cases),
                         "official_expected_matches": sum(c["official_expected_match"] for c in report["cases"]),
                         "independent_expected_matches": sum(c["independent_expected_match"] for c in report["cases"]),
                         "call_agreements": sum(c["call_agreement"] for c in report["cases"]),
                         "independent_parse_successes": sum(c["independent_parse_success"] for c in report["cases"]),
                         "independent_errors": sum(c["independent_error"] is not None for c in report["cases"])}
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), **report["summary"]}))
    if report['summary']['independent_errors']:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
