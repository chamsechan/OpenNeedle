import struct
from pathlib import Path
import numpy as np
import pytest
from needle2.archive import Archive, CQ, HEADER, RECORD
from needle2.convert import import_model, export_model, read_official_checkpoint
from needle2.quantize import quantize_matrix, dequantize_matrix, pack_indices, unpack_indices, hadamard, hadamard_matrix, fake_quantize

OFFICIAL=Path('artifacts/official/needle2.cact')

@pytest.mark.parametrize('bits',[2,3,4])
def test_bitstream_known_indices(bits):
    idx=np.array([[i % (1<<bits) for i in range(16)]],dtype=np.uint8)
    p=pack_indices(idx,bits)
    expected=sum(int(i)<<(j*bits) for j,i in enumerate(idx[0]))
    assert p.tobytes()==expected.to_bytes(16*bits//8,'little')
    np.testing.assert_array_equal(unpack_indices(p,bits,16),idx)

@pytest.mark.parametrize('bits',[2,3,4])
def test_cq_geometry_and_zero(bits):
    w=np.random.default_rng(42).normal(size=(7,139)).astype(np.float32)
    w[0]=0
    p,s=quantize_matrix(w,bits)
    d=dequantize_matrix(p,s,w.shape,bits)
    assert p.nbytes+s.nbytes==7*(256*bits//8+4)
    np.testing.assert_array_equal(d[0],0)
    assert np.linalg.norm(w-d)/np.linalg.norm(w)<.4
    np.testing.assert_allclose(hadamard(w[:,:128]),w[:,:128]@hadamard_matrix(128),atol=2e-6)

def test_qat_has_gradient():
    import torch
    w=torch.randn(3,128,requires_grad=True)
    q=fake_quantize(w)
    q.sum().backward()
    assert torch.equal(w.grad,torch.ones_like(w))
    assert not torch.equal(q,w)

@pytest.mark.integration
@pytest.mark.skipif(not OFFICIAL.exists(),reason='download official artifacts')
def test_real_archive_roundtrip(tmp_path):
    a=Archive.load(OFFICIAL)
    assert len(a.tensors)==405
    assert a.stats()['parameters']==43634423
    import_model(OFFICIAL,tmp_path/'torch')
    result=export_model(tmp_path/'torch',tmp_path/'out.cact')
    assert result['byte_identical']
    assert (tmp_path/'out.cact').read_bytes()==OFFICIAL.read_bytes()

@pytest.mark.integration
@pytest.mark.skipif(not OFFICIAL.exists(),reason='download official artifacts')
def test_modified_pytorch_weight_is_requantized(tmp_path):
    from safetensors.numpy import load_file,save_file
    import_model(OFFICIAL,tmp_path/'torch')
    weights=load_file(str(tmp_path/'torch/weights.safetensors'))
    weights['layer00.q_proj'] *= 1.2
    save_file(weights,str(tmp_path/'torch/weights.safetensors'))
    result=export_model(tmp_path/'torch',tmp_path/'out.cact')
    assert result['requantized_tensors']==1
    assert result['reused_tensors']==403
    old,new=Archive.load(OFFICIAL),Archive.load(tmp_path/'out.cact')
    assert new.tensors['layer00.q_proj'].blob!=old.tensors['layer00.q_proj'].blob
    assert new.tensors['embedding'].blob==old.tensors['embedding'].blob

@pytest.mark.integration
@pytest.mark.skipif(not OFFICIAL.exists(),reason='download official artifacts')
def test_archive_corruption_rejected():
    raw=bytearray(OFFICIAL.read_bytes())
    for data in (b'',raw[:100],raw[:-100]):
        with pytest.raises(ValueError): Archive(data)
    struct.pack_into('<Q',raw,120+112+20,1)
    with pytest.raises(ValueError,match='unaligned'): Archive(raw)

def test_pickle_disallows_code_execution(tmp_path):
    import pickle
    p=tmp_path/'bad.pkl'; p.write_bytes(pickle.dumps(eval))
    with pytest.raises(pickle.UnpicklingError):read_official_checkpoint(p)
