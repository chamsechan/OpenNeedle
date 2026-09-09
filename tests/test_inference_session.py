"""Request isolation, prefix invalidation and the Python/native boundary."""
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from needle2.inference import InferenceSession, generate

MODEL = Path('artifacts/official/needle2.cact')
requires_model = pytest.mark.skipif(not MODEL.is_file(), reason='model absent')
TOOLS = [{'name': 'add', 'parameters': {'type': 'object', 'properties': {
    'x': {'type': 'integer'}, 'y': {'type': 'integer'}}, 'required': ['x', 'y']}}]


@requires_model
@pytest.mark.parametrize('kv', ['fp32', 'int8'])
def test_session_reuses_resources_and_invalidates_prefix(kv):
    session = InferenceSession(MODEL, threads=2, kv_cache=kv)
    first = session.generate('Add 2 and 3.', tools=TOOLS, max_new_tokens=64)
    assert not first['prefix_cache_hit']
    assert first['prefill_tokens'] == first['prompt_tokens']
    # Warm requests must not reload the model, rebuild token tables or compile DFA.
    with patch('needle2.inference.Archive.load', side_effect=AssertionError('reload')), \
         patch('needle2.grammar.ToolGrammar.__init__', side_effect=AssertionError('grammar rebuilt')), \
         patch('needle2.grammar.compile_tool_dfa', side_effect=AssertionError('DFA recompiled')):
        session.generate('Add 7 and 4.', tools=TOOLS, max_new_tokens=64)
        again = session.generate('Add 2 and 3.', tools=TOOLS, max_new_tokens=64)
    assert again['token_ids'] == first['token_ids']
    assert again['prefix_cache_hit']
    assert again['prefill_tokens'] == again['prompt_tokens'] - again['prefix_tokens']
    assert again['native_decode_call_seconds'] is not None
    changed = session.generate('Add 2 and 3.', tools=TOOLS, system='Be concise.', max_new_tokens=4)
    assert not changed['prefix_cache_hit']
    changed_tools = [{'name': 'ping', 'parameters': {'type': 'object', 'properties': {}}}]
    assert not session.generate('Ping.', tools=changed_tools, max_new_tokens=4)['prefix_cache_hit']
    session.generate('Hello.', max_new_tokens=3)
    assert not session.generate('Add 2 and 3.', tools=TOOLS, max_new_tokens=0)['prefix_cache_hit']
    assert session.generate('Add 2 and 3.', tools=TOOLS, max_new_tokens=64)['token_ids'] == first['token_ids']


@requires_model
def test_fallback_grammar_resets_between_requests(monkeypatch):
    from needle2.grammar import GrammarTooLarge
    def too_large(*args, **kwargs):
        raise GrammarTooLarge('forced fallback')
    monkeypatch.setattr('needle2.grammar.compile_tool_dfa', too_large)
    session = InferenceSession(MODEL, threads=2)
    first = session.generate('Add 2 and 3.', tools=TOOLS, max_new_tokens=64)
    session.generate('Add 7 and 4.', tools=TOOLS, max_new_tokens=64)
    again = session.generate('Add 2 and 3.', tools=TOOLS, max_new_tokens=64)
    assert again['token_ids'] == first['token_ids']
    assert again['grammar_backend'] == 'python_regex'
    assert again['native_decode_call_seconds'] is None


@pytest.mark.parametrize('value', [-1, 1.2, True])
def test_invalid_limit_rejected_before_model_load(value):
    with pytest.raises(ValueError, match='nonnegative integer'):
        generate('missing.cact', 'test', max_new_tokens=value)


@requires_model
def test_native_session_without_torch_or_safetensors():
    source = '''
import importlib.abc
import sys
class BlockTraining(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'safetensors'):
            raise AssertionError('native inference imported ' + fullname)
sys.meta_path.insert(0, BlockTraining())
from needle2.inference import InferenceSession
s = InferenceSession('artifacts/official/needle2.cact')
a = s.generate('Hello', max_new_tokens=2)
assert len(a['token_ids']) > 0
'''
    subprocess.run([sys.executable, '-c', source], check=True)


@requires_model
def test_cached_prefix_tokenization_matches_full_text():
    from needle2.prompt import render_prompt
    session = InferenceSession(MODEL)
    tools = TOOLS + [{'name': 'echo', 'description': '中文说明 😀',
                     'parameters': {'type': 'object', 'properties': {}}}]
    for system in [None, '请简洁回答。']:
        for query in ['', 'Hello', ' 中文 😀\n', '<tool_call> literal </tools>']:
            text = render_prompt(query, tools, system)
            expected = [2] + session.tokenizer.encode(text)
            seen = []
            original = session.engine.prefill
            def capture(tokens, **kwargs):
                seen.extend(tokens)
                return original(tokens, **kwargs)
            with patch.object(session.engine, 'prefill', side_effect=capture):
                result = session.generate(query, tools=tools, system=system, max_new_tokens=0)
            prefix = expected[:result['prefix_tokens']] if result['prefix_cache_hit'] else []
            assert prefix + seen == expected


@requires_model
def test_failed_decode_does_not_contaminate_next_request():
    session = InferenceSession(MODEL, threads=2)
    expected = session.generate('Add 2 and 3.', tools=TOOLS, max_new_tokens=32)
    def fail_after_step(token, **kwargs):
        session.engine.step(token)
        raise RuntimeError('injected decode failure')
    with patch.object(session.engine, 'decode', side_effect=fail_after_step):
        with pytest.raises(RuntimeError, match='injected'):
            session.generate('Add 9 and 1.', tools=TOOLS, max_new_tokens=32)
    assert session.generate('Add 2 and 3.', tools=TOOLS, max_new_tokens=32)['token_ids'] == expected['token_ids']


@requires_model
@pytest.mark.parametrize('backend,prefill', [('torch', 'native'), ('native', 'torch')])
def test_torch_session_resets_request_cache(backend, prefill):
    session = InferenceSession(MODEL, backend=backend, prefill_backend=prefill, threads=2)
    first = session.generate('Add 2 and 3.', tools=TOOLS, max_new_tokens=3)
    session.generate('Add 7 and 4.', tools=TOOLS, max_new_tokens=3)
    again = session.generate('Add 2 and 3.', tools=TOOLS, max_new_tokens=3)
    assert again['token_ids'] == first['token_ids']
    assert not again['prefix_cache_hit']
