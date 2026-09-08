"""Optional CQ quantization-aware fine tuning using floating master weights.

This supplies the upstream straight-through CQ operation; it cannot guarantee
arbitrary fine tunes retain the released model's task accuracy.
"""
import torch
from torch import nn
from torch.nn.utils import parametrize
from .archive import Archive,CQ
from .quantize import fake_quantize

class _CQParametrization(nn.Module):
    def __init__(self, record):
        super().__init__()
        if record.bits not in (2,3,4):raise ValueError('QAT requires CQ2/3/4')
        self.bits,self.group_size=record.bits,record.group_size
        self.register_buffer('centroids',torch.from_numpy(record.codebook.copy()))

    def forward(self, weight):
        return fake_quantize(weight,self.bits,self.group_size,centroids=self.centroids)

def enable_qat(model, archive):
    """Attach STE parametrizations using the archive's actual precision/codebook.

    Call after moving the model to its training device and before making an
    optimizer. Remove with ``disable_qat`` before serializing master weights.
    """
    archive=archive if isinstance(archive,Archive) else Archive.load(archive)
    if getattr(model,'_cq_qat_enabled',False):raise ValueError('CQ QAT already enabled')
    specs=[]
    for name,record in archive.tensors.items():
        if record.dtype!=CQ:continue
        parent,_,key=name.rpartition('.')
        module=model.get_submodule(parent)
        w=getattr(module,key)
        if tuple(w.shape)!=record.shape or record.bits not in (2,3,4):raise ValueError(f'incompatible QAT tensor {name}')
        specs.append((module,key,record))
    for module,key,record in specs:
        parametrize.register_parametrization(module,key,_CQParametrization(record).to(getattr(module,key).device),unsafe=True)
    model._cq_qat_enabled=True
    return model

def disable_qat(model, *, keep_quantized=False):
    """Restore master weights by default, ready for the official CQ exporter."""
    for module in list(model.modules()):
        if hasattr(module,'parametrizations'):
            for name in list(module.parametrizations.keys()):
                if any(isinstance(p,_CQParametrization) for p in module.parametrizations[name]):
                    parametrize.remove_parametrizations(module,name,leave_parametrized=keep_quantized)
    model._cq_qat_enabled=False
    return model
