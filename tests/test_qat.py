import numpy as np
import torch
from needle2.model import NeedleModel,NeedleConfig
from needle2.archive import TensorRecord,CQ,Archive
from needle2.quantize import quantize_matrix,codebook
from needle2.qat import enable_qat,disable_qat

def test_qat_preserves_master_and_backpropagates():
    torch.set_num_threads(1)
    config=NeedleConfig(vocab_size=24,d_model=16,num_heads=2,num_kv_heads=1,head_dim=8,
                        num_layers=1,mhc_lanes=2,engram_layers=(),engram_orders=(),
                        num_engram_tables=0,max_seq_len=16)
    model=NeedleModel(config)
    master=model.embedding.detach().clone()
    p,s=quantize_matrix(master.numpy(),2,128)
    record=TensorRecord('embedding',CQ,tuple(master.shape),p.tobytes()+s.tobytes(),128,2,codebook(2))
    arc=object.__new__(Archive);arc.tensors={'embedding':record}
    enable_qat(model,arc)
    assert not torch.equal(model.embedding,master)
    model(torch.tensor([[2,3,4]])).square().mean().backward()
    original=model.parametrizations.embedding.original
    assert original.grad is not None and torch.isfinite(original.grad).all()
    assert original.grad.abs().sum()>0
    disable_qat(model)
    assert torch.equal(model.embedding,master)
    assert 'embedding' in model.canonical_state_dict()
