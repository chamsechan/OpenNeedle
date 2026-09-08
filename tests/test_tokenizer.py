from pathlib import Path
import pytest
from needle2.tokenizer import RefTokenizer
from needle2.prompt import render_prompt,parse_response

@pytest.mark.integration
def test_embedded_tokenizer_matches_sentencepiece():
    spm=pytest.importorskip('sentencepiece')
    model=Path('artifacts/official/tokenizer/tokenizer.model')
    archive=Path('artifacts/official/needle2.cact')
    if not model.exists() or not archive.exists(): pytest.skip('download official tokenizer')
    sp=spm.SentencePieceProcessor(model_file=str(model))
    tok=RefTokenizer.from_cact(archive)
    for text in ('hello world','打开客厅的灯。','café déjà vu','emoji 🙂🚀','one\ntwo',
                 '<|im_start|>user\n<tools>[]</tools>\nset timer<|im_end|>\n',
                 ' leading and  repeated spaces '):
        assert tok.encode(text)==sp.encode(text)
        assert tok.decode(tok.encode(text))==sp.decode(sp.encode(text))

def test_public_prompt_and_parse():
    s=render_prompt('hi',[{'type':'function','function':{'name':'f','parameters':{}}}])
    assert s=='<|im_start|>user\n<tools>[{"name":"f","parameters":{}}]</tools>\nhi<|im_end|>\n<|im_start|>assistant\n'
    r=parse_response('<think>reason</think><tool_call>[{"name":"f","arguments":{}}]</tool_call>')
    assert r['function_calls']==[{'name':'f','arguments':{}}]
    assert r['reasoning']=='reason'
    assert parse_response('<tool_call>{bad}</tool_call>')['parse_error']
