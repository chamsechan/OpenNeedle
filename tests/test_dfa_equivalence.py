"""Independent regex oracle checks for the byte/token DFA and FFI validation."""
import json
import numpy as np
import pytest
from needle2.grammar import ToolGrammar, compile_tool_dfa, NativeGrammarDFA
from test_grammar import ByteTokenizer


def edges(dfa, state):
    a, b = dfa.candidate_offsets[state:state + 2]
    return dict(zip(dfa.candidate_tokens[a:b], dfa.next_states[a:b]))


def check_prefixes(tools, data, tok=None):
    tok = tok or ByteTokenizer()
    grammar = ToolGrammar(tools, tok)
    dfa = compile_tool_dfa(tools, tok)
    state = edges(dfa, 0)[tok.p2id['<tool_call>']]
    grammar.accept(tok.p2id['<tool_call>'])
    for token in tok.encode_bytes(data) + [tok.p2id['</tool_call>']]:
        actual = edges(dfa, state)
        expected = {t for t in range(len(tok.pieces)) if grammar.can_accept(t)}
        assert set(actual) == expected, grammar.prefix
        if token not in expected:
            return False
        grammar.accept(token)
        state = actual[token]
    assert dfa.state_types[state] == 4
    return True


@pytest.mark.parametrize('args,valid', [
    ({'b': True}, True), ({'a': 1, 'b': False, 'c': '你好😀'}, True),
    ({'b': False, 'c': 'quote"\\\n'}, True), ({}, False),
    ({'b': 'true'}, False), ({'c': 'x', 'b': True}, False), ({'a': 1}, False),
])
def test_optional_required_order_and_utf8(args, valid):
    tools = [{'name': 'do', 'parameters': {'type': 'object', 'properties': {
        'a': {'type': 'integer'}, 'b': {'type': 'boolean'}, 'c': {'type': 'string'}}, 'required': ['b']}}]
    data = json.dumps([{'name': 'do', 'arguments': args}], ensure_ascii=False).encode()
    assert check_prefixes(tools, data) == valid


@pytest.mark.parametrize('args,valid', [
    ({'xs': []}, True), ({'xs': [{}, {'n': -1.2e10, 's': 'on'}, {'z': None}]}, True),
    ({'xs': [{'s': 'other'}]}, False), ({'xs': [1]}, False),
    ({'xs': [{}] * 4}, False),
])
def test_nested_arrays_objects_enums(args, valid):
    tools = [{'name': 'do', 'parameters': {'type': 'object', 'properties': {
        'xs': {'type': 'array', 'maxItems': 3, 'items': {'type': 'object', 'properties': {
            'n': {'type': 'number'}, 's': {'enum': ['on', 'off']}, 'z': {'type': 'null'}}}}}, 'required': ['xs']}}]
    data = json.dumps([{'name': 'do', 'arguments': args}]).encode()
    assert check_prefixes(tools, data) == valid


@pytest.mark.parametrize('minimum,maximum', [(0, 0), (0, 2), (1, 1), (2, 3), (0, None), (3, None)])
def test_array_counts_and_multiple_calls(minimum, maximum):
    array = {'type': 'array', 'items': {'type': 'integer'}, 'minItems': minimum}
    if maximum is not None: array['maxItems'] = maximum
    tools = [{'name': 'do', 'parameters': {'type': 'object', 'properties': {'xs': array}, 'required': ['xs']}}]
    assert check_prefixes(tools, b'[]')
    for count in range(5):
        call = {'name': 'do', 'arguments': {'xs': list(range(count))}}
        assert check_prefixes(tools, json.dumps([call, call]).encode()) == (count >= minimum and (maximum is None or count <= maximum))


def test_empty_tools_and_empty_arguments():
    assert check_prefixes([], b' [ ] ')
    assert not check_prefixes([], b'[{')
    assert check_prefixes([{'name': 'do'}], b'[{"name":"do","arguments":{}}]')


@pytest.mark.parametrize('data', [b'\xc0\xaf', b'\xed\xa0\x80', b'\xf4\x90\x80\x80', b'\xff', b'\xe4"', b'\n', b'\\x'])
def test_invalid_utf8_and_string_escapes(data):
    tools = [{'name': 'do', 'parameters': {'type': 'object', 'properties': {'s': {'type': 'string'}}, 'required': ['s']}}]
    assert not check_prefixes(tools, b'[{"name":"do","arguments":{"s":"' + data + b'"}}]')


def test_merged_tokens_and_remapped_ids():
    tok = ByteTokenizer()
    tok.pieces += ['[{"name":"', 'do","arguments":{}', '}}]', '▁[', '你好']
    tok.types += [0] * 5
    # Oracle compares merged-token candidates at every byte prefix as well.
    assert check_prefixes([{'name': 'do'}], b'[{"name":"do","arguments":{}}]', tok)
    tok.pieces[10], tok.pieces[8] = tok.pieces[8], tok.pieces[10]
    tok.pieces[11], tok.pieces[9] = tok.pieces[9], tok.pieces[11]
    tok.p2id = {p: i for i, p in enumerate(tok.pieces)}
    assert check_prefixes([{'name': 'do'}], b'[{"name":"do","arguments":{}}]', tok)


def desc(**changes):
    data = dict(num_states=2, initial_state=0, eos_id=1, stop_id=5, tool_start_id=10, tool_end_id=11,
                state_types=[1, 4], fallback_next_states=[-1, -1], candidate_offsets=[0, 1, 1],
                candidate_tokens=[3], next_states=[1])
    data.update(changes)
    return NativeGrammarDFA(**data)


@pytest.mark.parametrize('changes', [
    {'num_states': 0}, {'initial_state': 2}, {'state_types': [0]}, {'state_types': [2, 4]},
    {'candidate_offsets': [0, 2, 2]}, {'candidate_offsets': [1, 1, 1]},
    {'candidate_offsets': [0, -1, 1]}, {'fallback_next_states': [-1, 2]},
    {'next_states': [2]}, {'candidate_tokens': [-1]}, {'candidate_tokens': [2.5]},
    {'candidate_tokens': [2**32]}, {'candidate_tokens': [[3]]}, {'eos_id': -1},
])
def test_reject_invalid_descriptor(changes):
    with pytest.raises(ValueError): desc(**changes)


def test_decode_validates_vocab_and_mutated_descriptor_before_forward():
    from test_runtime_regressions import archive
    from needle2.native import NativeEngine
    e = NativeEngine(archive())
    dfa = desc(candidate_tokens=[31])
    with pytest.raises(ValueError, match='vocabulary'): e.decode(2, 3, dfa)
    dfa = desc()
    with pytest.raises(AttributeError, match='immutable'):
        dfa.candidate_offsets = np.array([0, 1000, 1000], dtype=np.int32)
    assert e.position == 0
    np.testing.assert_array_equal(e.step(2), NativeEngine(archive()).step(2))


def test_large_exact_grammar_requests_constrained_fallback():
    from needle2.grammar import GrammarTooLarge
    array = {'type': 'array', 'maxItems': 1024, 'items': {'type': 'array', 'maxItems': 1024, 'items': {'type': 'integer'}}}
    tools = [{'name': 'do', 'parameters': {'type': 'object', 'properties': {'xs': array}}}]
    with pytest.raises(GrammarTooLarge): compile_tool_dfa(tools, ByteTokenizer())


@pytest.mark.parametrize('value', [b'00', b'+1', b'.5', b'1.', b'1e', b'true', b'hello', b'--1'])
def test_invalid_number_prefixes(value):
    tools = [{'name': 'do', 'parameters': {'type': 'object', 'properties': {'n': {'type': 'number'}}, 'required': ['n']}}]
    assert not check_prefixes(tools, b'[{"name":"do","arguments":{"n":' + value + b'}}]')


def test_dfa_storage_and_metadata_are_immutable(monkeypatch):
    source = np.array([3], dtype=np.int32)
    dfa = desc(candidate_tokens=source)
    source[0] = 999
    assert dfa.candidate_tokens.tolist() == [3]
    for name in dfa._arrays:
        view = getattr(dfa, name)
        with pytest.raises(ValueError): view.setflags(write=True)
        with pytest.raises(ValueError): view[0] = 999
        view.shape = (1, -1)
        view.dtype = np.uint8
        assert getattr(dfa, name).ndim == 1
        assert getattr(dfa, name).dtype == np.int32
        assert isinstance(view.base, bytes)
    with pytest.raises(AttributeError): dfa.initial_state = 100
    with pytest.raises(AttributeError): del dfa.candidate_tokens
    with pytest.raises(AttributeError): dfa._frozen = False
    descriptor = dfa._desc
    descriptor.num_states = 1000
    assert dfa._desc.num_states == 2
    # Graph scans are unnecessary after construction, including across vocabularies.
    def fail(*args, **kwargs): raise AssertionError("repeated graph validation")
    monkeypatch.setattr(np, 'unique', fail)
    monkeypatch.setattr(np, 'isin', fail)
    for _ in range(3): dfa.validate(32)
    with pytest.raises(ValueError, match='vocabulary'): dfa.validate(11)
