import json
from pathlib import Path
import pytest
import numpy as np

from needle2.archive import Archive
from needle2.tokenizer import RefTokenizer, parse_tokenizer_blob
from needle2.grammar import compile_tool_dfa, NativeGrammarDFA
from needle2.native import NativeEngine, available as native_available
from needle2.official import OfficialEngine, strict_json_equal
from needle2.prompt import render_prompt, parse_response

MODEL_PATH = Path("artifacts/official/needle2.cact")
OFFICIAL_LIB_PATH = Path("artifacts/official/libneedle.dylib")
TOOLS_PATH = Path("benchmarks/expanded_tools.json")
CASES_PATH = Path("benchmarks/expanded_cases.jsonl")

requires_native = pytest.mark.skipif(not native_available(), reason="Native library not available")
requires_model = pytest.mark.skipif(not MODEL_PATH.is_file(), reason="Artifact needle2.cact not found")
requires_official = pytest.mark.skipif(not OFFICIAL_LIB_PATH.is_file(), reason="Official library not found")


@pytest.fixture(scope="module")
def tokenizer():
    if not MODEL_PATH.is_file():
        pytest.skip("Model artifact not found")
    archive = Archive.load(MODEL_PATH)
    return RefTokenizer(parse_tokenizer_blob(archive.tensors["tokenizer"].blob))


@pytest.fixture(scope="module")
def expanded_tools():
    return json.loads(TOOLS_PATH.read_text())


@pytest.fixture(scope="module")
def expanded_cases():
    return [json.loads(line) for line in CASES_PATH.read_text().splitlines() if line.strip()]


def test_expanded_dataset_integrity(expanded_tools, expanded_cases):
    assert len(expanded_tools) == 8
    assert len(expanded_cases) == 16
    tool_names = {t["name"] for t in expanded_tools}
    for case in expanded_cases:
        assert case["expected_calls"], f"Case {case['id']} must expect function calls"
        for call in case["expected_calls"]:
            assert call["name"] in tool_names, f"Tool {call['name']} not in expanded_tools"


@requires_native
@requires_model
def test_native_engine_expanded_suite_precision(tokenizer, expanded_tools, expanded_cases):
    dfa = compile_tool_dfa(expanded_tools, tokenizer)
    engine = NativeEngine(MODEL_PATH, threads=4, matmul="sdot")

    matches = 0
    for case in expanded_cases:
        prompt = render_prompt(case["query"], expanded_tools)
        ids = [2] + tokenizer.encode(prompt)
        prefix_len = ids.index(tokenizer.p2id["</tools>"]) + 1
        engine.reset(prefix_len=prefix_len)
        logits = engine.prefill(ids, last_only=True)
        first_tok = int(logits.argmax())

        tokens = engine.decode(first_tok, max_new_tokens=128, grammar_dfa=dfa)
        parsed = parse_response(tokenizer.decode(tokens))
        calls = parsed.get("function_calls")
        assert strict_json_equal(calls, case["expected_calls"]), f"Case {case['id']} mismatch: {calls} != {case['expected_calls']}"
        matches += 1

    assert matches == len(expanded_cases)
