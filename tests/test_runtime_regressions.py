"""Regression coverage for the eee07942 optimization review."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
from needle2.archive import TensorRecord, FP32
from needle2.native import NativeEngine
from needle2.grammar import compile_tool_dfa, ToolGrammar
from needle2.tokenizer import RefTokenizer
from test_model import tiny_model


def archive():
    m = tiny_model(window=5)
    records = {n: TensorRecord(n, FP32, tuple(t.shape), t.numpy().tobytes()) for n,t in m.canonical_state_dict().items()}
    meta=m.config.to_dict(); meta['hada_n']=16
    return SimpleNamespace(metadata=meta,tensors=records)


def test_single_candidate_hidden_contract():
    e=NativeEngine(archive())
    logits, hidden=e.step_candidates(2,[3],return_hidden=True)
    assert isinstance(hidden,np.ndarray) and hidden.shape==(3,12)


def test_single_candidate_bounds():
    e=NativeEngine(archive())
    with pytest.raises((ValueError,IndexError)):
        e.step_candidates(2,[31])


def test_kv_toggle_position():
    e=NativeEngine(archive()); e.prefill([2,3,4]); e.set_int8_kv(True)
    # The C++ setter resets the sequence; wrapper must expose that reset.
    assert e.position==0


@pytest.mark.parametrize('threads',[1,2,4])
@pytest.mark.parametrize('kv',['fp32','int8'])
def test_long_prefill_prefix_restore_and_decode(threads,kv):
    a=archive(); x=NativeEngine(a,threads=threads,kv_cache=kv); y=NativeEngine(a,threads=threads,kv_cache=kv)
    ids=np.random.default_rng(23).integers(2,31,310)
    for e in (x,y): e.reset(prefix_len=9)
    expected=np.stack([x.step(int(t)) for t in ids[:9]])
    np.testing.assert_allclose(y.prefill(ids[:9]),expected,atol=2e-6,rtol=5e-5)
    for e in (x,y): e.cache_prefix()
    expected=np.stack([x.step(int(t)) for t in ids[9:]])
    np.testing.assert_allclose(y.prefill(ids[9:]),expected,atol=2e-6,rtol=5e-5)
    for e in (x,y): e.reset_to_prefix()
    expected=np.stack([x.step(int(t)) for t in ids[9:30]])
    np.testing.assert_allclose(y.prefill(ids[9:30]),expected,atol=2e-6,rtol=5e-5)
    first=int(expected[-1].argmax()); out=[first]
    while len(out)<8 and out[-1] not in (1,5): out.append(int(x.step(out[-1]).argmax()))
    assert y.decode(first,8)==out
    assert y.position==x.position


@pytest.fixture(scope='module')
def tok():
    if not Path('artifacts/official/needle2.cact').is_file():
        pytest.skip('official tokenizer absent')
    return RefTokenizer.from_cact('artifacts/official/needle2.cact')


def walk(dfa, ids):
    state=dfa.initial_state
    for token in ids:
        start,end=dfa.candidate_offsets[state:state+2]
        trans=dict(zip(dfa.candidate_tokens[start:end],dfa.next_states[start:end]))
        if token in trans: state=int(trans[token])
        elif dfa.state_types[state]==0 and dfa.fallback_next_states[state]>=0: state=int(dfa.fallback_next_states[state])
        else: return False
    return True


@pytest.mark.parametrize('props,args',[
    ({},{}),
    ({'on':{'type':'boolean'}},{'on':True}),
    ({'city':{'type':'string'}},{'city':'Paris'}),
    ({'n':{'type':'integer'}},{'n':2}),
    ({'xs':{'type':'array','items':{'type':'integer'}}},{'xs':[1]}),
])
def test_dfa_accepts_valid_schema_output(tok,props,args):
    tools=[{'name':'test','parameters':{'type':'object','properties':props,'required':list(props)}}]
    payload=json.dumps([{'name':'test','arguments':args}],separators=(',',':'))
    tokens=tok.encode('<tool_call>'+payload+'</tool_call>')
    grammar=ToolGrammar(tools,tok)
    for token in tokens:
        assert grammar.can_accept(token)
        grammar.accept(token)
    assert walk(compile_tool_dfa(tools,tok),tokens),payload


def test_dfa_integer_rejects_letters(tok):
    tools=[{'name':'test','parameters':{'type':'object','properties':{'n':{'type':'integer'}},'required':['n']}}]
    ids=tok.encode('<tool_call>[{"name":"test","arguments":{"n":hello')
    assert not walk(compile_tool_dfa(tools,tok),ids)


def test_supported_long_engram_order():
    import torch
    from needle2.model import NeedleModel
    a=archive(); a.metadata.update(engram_orders=(2,257),kv_window=300,max_seq_len=400)
    reference=NeedleModel.from_archive(a).eval(); e=NativeEngine(a)
    ids=np.random.default_rng(72).integers(2,31,270)
    with torch.no_grad(): expected=reference(torch.tensor(ids)[None]).numpy()[0]
    np.testing.assert_allclose(e.prefill(ids),expected,atol=2e-6,rtol=5e-5)


def test_pool_wakeup_after_idle():
    import time
    a=archive(); x=NativeEngine(a,threads=1); y=NativeEngine(a,threads=4)
    for i in range(40):
        time.sleep(.01)
        np.testing.assert_array_equal(y.prefill([2,3,4]),x.prefill([2,3,4]))


@pytest.mark.skipif(not Path("artifacts/official/needle2.cact").is_file(), reason="model absent")
def test_cached_embedding_can_be_used_by_generate():
    from unittest.mock import patch
    from needle2.inference import generate
    tools=[{'name':'ping','parameters':{'type':'object','properties':{}},'_embedding':np.ones(128,dtype=np.float32)}]
    with patch('needle2.inference.retrieve_tools',return_value=(tools,[(1.,tools[0])])):
        result=generate('artifacts/official/needle2.cact','Ping',tools=tools,retrieval=True,max_new_tokens=0)
    assert result['retrieval_enabled']


@pytest.mark.parametrize('bad', [[-1], [31], [2, -1], [2, 31], [2.5, 3], [2**32 + 2], [True]])
def test_candidate_validation_precedes_forward(bad):
    e = NativeEngine(archive())
    with pytest.raises(ValueError):
        e.step_candidates(2, bad)
    assert e.position == 0
    np.testing.assert_array_equal(e.step(3), NativeEngine(archive()).step(3))


def test_native_candidate_validation_precedes_forward():
    import ctypes as ct
    e = NativeEngine(archive())
    f = e._lib.needle2_engine_step_candidates
    f.argtypes = [ct.c_void_p, ct.c_int, ct.c_void_p, ct.c_int, ct.c_void_p, ct.c_void_p]
    f.restype = ct.c_int
    candidates = np.array([2, 31], dtype=np.int32)
    logits = np.empty(2, dtype=np.float32)
    assert f(e._handle, 2, candidates.ctypes.data, 2, logits.ctypes.data, None) == -1
    np.testing.assert_array_equal(e.step(3), NativeEngine(archive()).step(3))


def test_int8_kv_releases_fp32_including_snapshot():
    e = NativeEngine(archive())
    e.reset(prefix_len=3)
    e.prefill([2, 3, 4]); e.cache_prefix()
    baseline = e.kv_storage_bytes
    e.set_int8_kv(True)
    assert e.position == 0
    e.prefill([2, 3, 4]); e.cache_prefix()
    # One byte per element + one FP32 scale per head, versus FP32 elements.
    expected_ratio = (1 + 4 / e.metadata['head_dim']) / 4
    assert e.kv_storage_bytes <= baseline * expected_ratio
    fp32_exact = 2 * e.metadata['num_layers'] * (e.metadata['kv_window'] + 6) * e.metadata['num_kv_heads'] * e.metadata['head_dim'] * 4
    assert e.kv_storage_bytes == pytest.approx(fp32_exact * expected_ratio)
    e.set_int8_kv(False)
    e.prefill([2, 3, 4]); e.cache_prefix()
    assert e.kv_storage_bytes == fp32_exact


def test_long_history_torch_import_and_snapshot():
    import torch
    torch.set_num_threads(1)
    a = archive()
    a.metadata.update(engram_orders=(2, 257), kv_window=300, max_seq_len=400)
    x, y = NativeEngine(a), NativeEngine(a)
    ids = np.random.default_rng(42).integers(2, 31, 300)
    for e in (x, y): e.reset(prefix_len=280)
    x.prefill(ids[:280]); y.prefill(ids[:280], backend='torch')
    for e in (x, y): e.cache_prefix()
    for _ in range(2):
        np.testing.assert_allclose(y.prefill(ids[280:]), x.prefill(ids[280:]), atol=2e-6, rtol=5e-5)
        for e in (x, y): e.reset_to_prefix()


def test_attention_worklist_above_64_groups():
    import torch
    from needle2.model import NeedleModel, NeedleConfig
    config = NeedleConfig(vocab_size=31, d_model=12, num_heads=130, num_kv_heads=130,
                          num_layers=1, head_dim=2, mhc_lanes=2, engram_layers=(), max_seq_len=64)
    torch.manual_seed(1)
    m = NeedleModel(config)
    records = {n: TensorRecord(n, FP32, tuple(t.shape), t.detach().numpy().tobytes())
               for n, t in m.canonical_state_dict().items()}
    meta = config.to_dict(); meta['hada_n'] = 16
    e = NativeEngine(SimpleNamespace(metadata=meta, tensors=records), threads=4)
    ids = torch.tensor([[2, 3, 4]])
    with torch.no_grad(): expected = m(ids).numpy()[0]
    np.testing.assert_allclose(e.prefill(ids.numpy()[0]), expected, atol=2e-6, rtol=5e-5)


@pytest.mark.parametrize('k', [0, -1, True, 1.5])
def test_retrieval_rejects_invalid_k_before_loading(k):
    from needle2.inference import generate, retrieve_tools
    with pytest.raises(ValueError, match='positive integer'):
        generate('missing.cact', 'test', top_k_tools=k)
    with pytest.raises(ValueError, match='positive integer'):
        retrieve_tools(None, None, 'test', [], top_k=k)


@pytest.mark.skipif(not Path('artifacts/official/needle2.cact').is_file(), reason='model absent')
def test_native_generation_ignores_python_dfa_compiler(monkeypatch):
    from needle2.inference import generate
    from needle2.grammar import GrammarTooLarge
    tools = [{'name': 'sum_numbers', 'description': 'Sum numbers', 'parameters': {'type': 'object',
              'properties': {'numbers': {'type': 'array', 'items': {'type': 'integer'}}}, 'required': ['numbers']}}]
    kwargs = dict(tools=tools, threads=2, max_new_tokens=64)
    actual = generate('artifacts/official/needle2.cact', 'Sum the numbers 1 and 2.', **kwargs)
    def too_large(*args, **kwargs): raise GrammarTooLarge('test size limit')
    monkeypatch.setattr('needle2.grammar.compile_tool_dfa', too_large)
    fallback = generate('artifacts/official/needle2.cact', 'Sum the numbers 1 and 2.', **kwargs)
    assert actual['grammar_backend'] == 'native_dfa'
    assert fallback['grammar_backend'] == 'native_dfa'
    assert fallback['grammar_compiler'] == 'cpp'
    assert actual['function_calls'] == [{'name': 'sum_numbers', 'arguments': {'numbers': [1, 2]}}]
    assert actual['token_ids'] == fallback['token_ids']
