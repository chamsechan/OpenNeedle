import json
from pathlib import Path

import numpy as np
import pytest
import regex

from needle2.grammar import ToolGrammar, compile_tools


def tool(parameters):
    return [{"name": "do", "parameters": parameters}]


def matches(tools, arguments):
    expression = regex.compile(compile_tools(tools), regex.VERSION1)
    data = json.dumps([{"name": "do", "arguments": arguments}], ensure_ascii=False).encode()
    return expression.fullmatch(data) is not None


def test_required_optional_properties_and_canonical_order():
    tools = tool({"type": "object", "properties": {"a": {"type": "integer"},
                 "b": {"type": "boolean"}, "c": {"type": "string"}}, "required": ["b"]})
    assert matches(tools, {"b": True})
    assert matches(tools, {"a": 1, "b": False})
    assert matches(tools, {"b": True, "c": "yes"})
    assert matches(tools, {"a": -1, "b": False, "c": "yes"})
    assert not matches(tools, {})
    assert not matches(tools, {"b": "true"})
    assert not matches(tools, {"c": "yes", "b": True})
    assert not matches(tools, {"b": True, "unknown": 1})


def test_empty_optional_objects_no_leading_or_trailing_comma():
    tools = tool({"type": "object", "properties": {"x": {"type": "integer"}}})
    expression = regex.compile(compile_tools(tools), regex.VERSION1)
    assert matches(tools, {})
    assert matches(tools, {"x": 1})
    for invalid in (b'{,"x":1}', b'{"x":1,}', b'{"x":1,"x":2}'):
        assert expression.fullmatch(b'[{"name":"do","arguments":' + invalid + b'}]') is None


def test_nested_arrays_and_optional_state_does_not_leak():
    tools = tool({"type": "object", "properties": {"rows": {"type": "array", "items": {
        "type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "string"}}
    }}}, "required": ["rows"]})
    assert matches(tools, {"rows": [{"x": 2}, {}, {"y": "hello"}, {"x": 1.5, "y": "好"}]})
    assert matches(tools, {"rows": []})
    assert not matches(tools, {"rows": [{"x": True}]})


def test_enums_array_bounds_and_multiple_calls():
    tools = tool({"type": "object", "properties": {
        "mode": {"type": "string", "enum": ["on", "off"]},
        "ids": {"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 2}
    }, "required": ["mode", "ids"]})
    assert matches(tools, {"mode": "on", "ids": [1, 2]})
    assert not matches(tools, {"mode": "auto", "ids": [1]})
    assert not matches(tools, {"mode": "on", "ids": []})
    assert not matches(tools, {"mode": "on", "ids": [1, 2, 3]})
    expression = regex.compile(compile_tools(tools), regex.VERSION1)
    call = {"name": "do", "arguments": {"mode": "off", "ids": [3]}}
    assert expression.fullmatch(json.dumps([call, call]).encode())
    assert expression.fullmatch(b"[]")


@pytest.mark.parametrize("schema", [
    {"type": "string", "pattern": "x"}, {"type": "integer", "minimum": 0},
    {"type": ["string", "null"]}, {"$ref": "#/x"},
    {"type": "array", "items": {}}, {"type": "string", "enum": [True]},
])
def test_unsupported_schema_fails_explicitly(schema):
    with pytest.raises((ValueError, TypeError)):
        compile_tools(tool({"type": "object", "properties": {"value": schema}}))


def test_optional_expression_size_is_linear():
    small = compile_tools(tool({"type": "object", "properties": {
        f"p{i:03d}": {"type": "integer"} for i in range(10)}}))
    large = compile_tools(tool({"type": "object", "properties": {
        f"p{i:03d}": {"type": "integer"} for i in range(100)}}))
    assert len(large) < 11 * len(small)


class ByteTokenizer:
    def __init__(self):
        self.pieces = ["<pad>", "</s>", "<s>", "<unk>"] + [f"<control{i}>" for i in range(4, 10)]
        self.pieces += ["<tool_call>", "</tool_call>"]
        self.types = [2] * len(self.pieces)
        self.pieces += [f"<0x{i:02X}>" for i in range(256)]
        self.types += [4] * 256
        self.p2id = {piece: i for i, piece in enumerate(self.pieces)}

    def encode_bytes(self, data):
        return [12 + value for value in data]


def test_byte_prefix_unicode_escaping_and_close_token():
    tokenizer = ByteTokenizer()
    tools = tool({"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]})
    grammar = ToolGrammar(tools, tokenizer)
    assert not grammar.active
    grammar.accept(10)
    assert not grammar.can_accept(11)
    output = json.dumps([{"name": "do", "arguments": {"text": '你 "yes"\n'}}], ensure_ascii=False).encode()
    for token in tokenizer.encode_bytes(output):
        assert grammar.can_accept(token)
        grammar.accept(token)
    assert grammar.json_complete
    grammar.accept(11)
    assert grammar.finished
    assert grammar.can_accept(1)


def test_invalid_utf8_and_invalid_top_logit_are_rejected():
    tokenizer = ByteTokenizer()
    grammar = ToolGrammar(tool({"type": "object", "properties": {}}), tokenizer)
    grammar.accept(10)
    scores = np.full(len(tokenizer.pieces), -100.0)
    scores[12 + ord("x")] = 20
    scores[12 + ord("[")] = 10
    assert grammar.select(scores) == 12 + ord("[")
    assert not grammar.can_accept(12 + 0xFF)
    assert not grammar.can_accept(1)


def test_unknown_tool_and_wrong_argument_key_are_rejected():
    expression = regex.compile(compile_tools(tool({"type": "object", "properties": {
        "minutes": {"type": "integer"}}, "required": ["minutes"]})), regex.VERSION1)
    assert expression.fullmatch(b'[{"name":"other"', partial=True) is None
    assert expression.fullmatch(b'[{"name":"do","arguments":{"time_human"', partial=True) is None


def test_official_tokenizer_prefixes_when_available():
    path = Path(__file__).resolve().parents[1] / "artifacts/official/needle2.cact"
    if not path.exists():
        pytest.skip("official model not downloaded")
    from needle2.tokenizer import RefTokenizer
    tokenizer = RefTokenizer.from_cact(path)
    tools = json.loads((path.parents[2] / "examples/tools.json").read_text())
    grammar = ToolGrammar(tools, tokenizer)
    ids = tokenizer.encode('<tool_call>[{"name":"set_timer","arguments":{"minutes":5}}]</tool_call>')
    for token in ids:
        grammar.accept(token)
    assert grammar.finished


def test_candidate_tokens_and_select_candidate():
    path = Path(__file__).resolve().parents[1] / "artifacts/official/needle2.cact"
    if not path.exists():
        pytest.skip("official model not downloaded")
    from needle2.tokenizer import RefTokenizer
    tokenizer = RefTokenizer.from_cact(path)
    tools = json.loads((path.parents[2] / "examples/tools.json").read_text())
    grammar = ToolGrammar(tools, tokenizer)
    # Before <tool_call>, candidates is None
    assert grammar.candidate_tokens() is None
    grammar.accept(grammar.start_id)
    assert grammar.active
    # In active state, candidates is a non-empty subset
    cands = grammar.candidate_tokens()
    assert cands is not None and len(cands) > 0
    # Every token in candidates must be acceptable
    for c in cands:
        assert grammar.can_accept(c)
    # select_candidate with fake logits picks argmax
    fake_logits = np.arange(len(cands), dtype=np.float32)
    best = grammar.select_candidate(cands, fake_logits)
    assert best == cands[-1]
    # When finished, candidate is only EOS
    grammar.finished = True
    assert grammar.candidate_tokens() == [grammar.eos_id]

