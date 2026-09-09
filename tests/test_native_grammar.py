import json
from pathlib import Path
import pytest
import numpy as np

from needle2.archive import Archive
from needle2.tokenizer import RefTokenizer, parse_tokenizer_blob
from needle2.grammar import compile_tool_dfa, NativeGrammarDFA
from needle2.native import NativeEngine, available as native_available, sdot_available
from needle2.prompt import render_prompt, parse_response

MODEL_PATH = Path("artifacts/official/needle2.cact")
TOOLS_PATH = Path("examples/tools.json")
CASES_PATH = Path("benchmarks/cases.jsonl")

requires_native = pytest.mark.skipif(not native_available(), reason="Native library not available")
requires_model = pytest.mark.skipif(not MODEL_PATH.is_file(), reason="Artifact needle2.cact not found")


@pytest.fixture(scope="module")
def tokenizer():
    if not MODEL_PATH.is_file():
        pytest.skip("Model artifact not found")
    archive = Archive.load(MODEL_PATH)
    return RefTokenizer(parse_tokenizer_blob(archive.tensors["tokenizer"].blob))


@pytest.fixture(scope="module")
def sample_tools():
    return json.loads(TOOLS_PATH.read_text())


def test_compile_tool_dfa_structure(tokenizer, sample_tools):
    dfa = compile_tool_dfa(sample_tools, tokenizer)
    assert isinstance(dfa, NativeGrammarDFA)
    assert dfa.num_states > 0
    assert dfa.initial_state == 0
    assert dfa.eos_id == 1
    assert dfa.tool_start_id == 10
    assert dfa.tool_end_id == 11
    assert len(dfa.state_types) == dfa.num_states
    assert len(dfa.fallback_next_states) == dfa.num_states
    assert len(dfa.candidate_offsets) == dfa.num_states + 1
    assert len(dfa.candidate_tokens) == len(dfa.next_states)
    assert dfa.candidate_offsets[0] == 0
    assert dfa.candidate_offsets[-1] == len(dfa.candidate_tokens)


@requires_native
@requires_model
def test_native_engine_decode_loop_unconstrained(tokenizer):
    engine = NativeEngine(MODEL_PATH, threads=2, matmul="fp32")
    engine.reset()
    tokens = [2] + tokenizer.encode("Hello world")
    logits = engine.prefill(tokens, last_only=True)
    first_tok = int(np.argmax(logits))
    out = engine.decode(first_tok, max_new_tokens=8, grammar_dfa=None)
    assert len(out) >= 1
    assert out[0] == first_tok


@requires_native
@requires_model
@pytest.mark.skipif(not sdot_available(), reason="SDOT unavailable")
def test_native_engine_decode_with_grammar(tokenizer, sample_tools):
    cases = [json.loads(line) for line in CASES_PATH.read_text().splitlines() if line.strip()]
    dfa = compile_tool_dfa(sample_tools, tokenizer)
    engine = NativeEngine(MODEL_PATH, threads=4, matmul="sdot")

    for case in cases:
        prompt = render_prompt(case["query"], sample_tools)
        ids = [2] + tokenizer.encode(prompt)
        prefix_len = ids.index(tokenizer.p2id["</tools>"]) + 1
        engine.reset(prefix_len=prefix_len)
        logits = engine.prefill(ids, last_only=True)
        first_tok = int(np.argmax(logits))

        tokens = engine.decode(first_tok, max_new_tokens=32, grammar_dfa=dfa)
        assert len(tokens) > 5
        text = tokenizer.decode(tokens)
        parsed = parse_response(text)
        assert parsed["function_calls"] == case["expected_calls"]
