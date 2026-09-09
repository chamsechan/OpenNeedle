"""Native frontend vs independent Python/SentencePiece semantics, including ABI errors."""
import ctypes as ct
import json
import random
import struct
from pathlib import Path
import numpy as np
import pytest
from needle2.frontend import NativeTokenizer, CompiledGrammar, _bindings
from needle2.tokenizer import RefTokenizer, parse_tokenizer_blob
from needle2.grammar import compile_tool_dfa

MODEL = Path('artifacts/official/needle2.cact')


def small_tokenizer():
    pieces = [('<unk>', 1), ('</s>', 2), ('<s>', 2), ('▁', 0), ('x', 0), ('<stop>', 3),
              ('a', 0), ('b', 0), ('ab', 0), ('ba', 0), ('<tool_call>', 3), ('</tool_call>', 3),
              ('</tools>', 3)] + [(f'<0x{b:02X}>', 4) for b in range(256)]
    blob = struct.pack('<IIIIIBBH', len(pieces), len(pieces), 1, 2, 0, 1, 1, 0)
    for p, kind in pieces:
        b = p.encode();blob += struct.pack('<fBH', 1.0, kind, len(b))+b
    return blob


def test_tokenizer_unicode_ties_specials_and_invalid_bytes():
    blob=small_tokenizer();a=NativeTokenizer(blob);b=RefTokenizer(parse_tokenizer_blob(blob))
    rng=random.Random(472)
    texts=['', 'abababa', '  ab', '中文😀é', 'a</tools> ba', '\x00\n\t']
    texts += [''.join(rng.choice(['a','b',' ','中','😀','</tools>','\n']) for _ in range(30)) for _ in range(60)]
    for text in texts:
        for dummy in [None,False,True]:
            assert a.encode(text,add_dummy_prefix=dummy)==b.encode(text,add_dummy_prefix=dummy)
    for _ in range(200):
        ids=[a.byte_id[rng.randrange(256)] for _ in range(12)]
        assert a.decode(ids)==b.decode(ids)
    assert a.decode(a.encode(' 中文😀')) == b.decode(b.encode(' 中文😀'))


def compare_automata(a,b):
    desc=a._lib.needle2_grammar_dfa(a._handle).contents
    queue=[(0,0)];seen=set()
    while queue:
        x,y=queue.pop()
        if (x,y) in seen:continue
        seen.add((x,y))
        assert desc.state_types[x]==b.state_types[y]
        lo,hi=desc.candidate_offsets[x],desc.candidate_offsets[x+1]
        ae={desc.candidate_tokens[i]:desc.next_states[i] for i in range(lo,hi)}
        lo,hi=b.candidate_offsets[y:y+2]
        be=dict(zip(b.candidate_tokens[lo:hi],b.next_states[lo:hi]))
        assert ae.keys()==be.keys(),(x,y,ae.keys()^be.keys())
        queue.extend((ae[t],be[t]) for t in ae)


@pytest.mark.parametrize('schema', [
    {'type':'string'}, {'type':'integer'}, {'type':'number'}, {'type':'boolean'}, {'type':'null'},
    {'enum':['中文',None,True,1,1.5]},
    {'type':'array','items':{'type':'integer'},'minItems':2,'maxItems':3},
    {'type':'array','items':{'type':'string'},'maxItems':0},
    {'type':'array','items':{'type':'string'}},
    {'type':'object','properties':{'a':{'type':'string'},'b':{'type':'integer'},'c':{'type':'boolean'}},'required':['b']},
])
def test_compiled_grammar_candidate_language_matches_reference(schema):
    a=NativeTokenizer(small_tokenizer());b=RefTokenizer(parse_tokenizer_blob(small_tokenizer()))
    tools=[{'name':'fn','parameters':{'type':'object','properties':{'value':schema},'required':['value']}}]
    compare_automata(CompiledGrammar(a,json.dumps(tools,ensure_ascii=False,separators=(',',':'))),compile_tool_dfa(tools,b))


@pytest.mark.parametrize('tools', [[],[{'type':'function','function':{'name':'fn'}}]])
def test_empty_tools_and_wrappers(tools):
    a=NativeTokenizer(small_tokenizer());b=RefTokenizer(parse_tokenizer_blob(small_tokenizer()))
    compare_automata(CompiledGrammar(a,json.dumps(tools)),compile_tool_dfa(tools,b))


@pytest.mark.parametrize('schema', [
    {'type':'string','pattern':'x'}, {'type':[]},
    {'type':'array','items':{'type':'string'},'minItems':3,'maxItems':2},
    {'type':'object','additionalProperties':True}, {'enum':[]},
    {'type':'object','required':['missing']},
])
def test_unsupported_schema_fails_without_silent_relaxation(schema):
    tok=NativeTokenizer(small_tokenizer())
    with pytest.raises(ValueError):
        CompiledGrammar(tok,json.dumps([{'name':'fn','parameters':{'type':'object','properties':{'x':schema}}}]))


def test_raw_abi_lengths_buffers_and_errors():
    lib=_bindings()
    assert not lib.needle2_tokenizer_create(b'bad',3)
    tok=NativeTokenizer(small_tokenizer());buf=(ct.c_int*1)(-123)
    n=lib.needle2_tokenizer_encode(tok._handle,b'xxxxxxxx',8,-1,buf,1)
    assert n>1 and buf[0]==-123
    assert lib.needle2_tokenizer_encode(tok._handle,b'\xff',1,-1,buf,1)==-1
    assert not lib.needle2_grammar_compile(tok._handle,b'[{',2)
    assert not lib.needle2_grammar_compile(tok._handle,b'[] junk',7)
    for ids in [[-1],[2**32],[len(tok.pieces)]]:
        with pytest.raises(ValueError):tok.decode(ids)


@pytest.mark.skipif(not MODEL.is_file(),reason='model absent')
def test_released_tokenizer_matches_reference_and_sentencepiece():
    a=NativeTokenizer.from_cact(MODEL);b=RefTokenizer.from_cact(MODEL)
    rng=random.Random(23)
    texts=[''.join(rng.choice(['hello',' ','中文','😀','é','\n','<tool_call>']) for _ in range(30)) for _ in range(80)]
    texts += ['abababab','x\x00y','<|im_start|>user\n<tools>[]</tools>\n中文<|im_end|>']
    for text in texts:
        assert a.encode(text)==b.encode(text)
        assert a.decode(a.encode(text))==b.decode(b.encode(text))
    model=Path('artifacts/official/tokenizer/tokenizer.model')
    if model.exists():
        spm=pytest.importorskip('sentencepiece').SentencePieceProcessor(model_file=str(model))
        for text in texts:assert a.encode(text)==spm.encode(text)


def test_native_union_and_anyof_match_same_scalar_enum_language():
    tok=NativeTokenizer(small_tokenizer())
    def compiled(value):
        return CompiledGrammar(tok,json.dumps([{'name':'fn','parameters':{'type':'object','properties':{'x':value},'required':['x']}}]))
    a=compiled({'type':['boolean','null']})
    tools=[{'name':'fn','parameters':{'type':'object','properties':{'x':{'enum':[True,False,None]}},'required':['x']}}]
    reference=compile_tool_dfa(tools,RefTokenizer(parse_tokenizer_blob(small_tokenizer())))
    compare_automata(a,reference)
    compare_automata(compiled({'anyOf':[{'type':'boolean'},{'type':'null'}]}),reference)
