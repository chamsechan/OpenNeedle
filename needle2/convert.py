"""Lossless archive bridges and explicit CQ requantization of trained weights."""
import hashlib
import io
import json
from pathlib import Path
import pickle
import numpy as np
from safetensors.numpy import load_file, save_file
from .archive import Archive, RAW, CQ, LAYER_NAMES, MHC_NAMES

class _NumpyUnpickler(pickle.Unpickler):
    """Read upstream's dictionary/NumPy checkpoint without general pickle globals."""
    def find_class(self, module, name):
        allowed = {('numpy','ndarray'):np.ndarray, ('numpy','dtype'):np.dtype}
        if module in ('numpy.core.multiarray','numpy._core.multiarray') and name == '_reconstruct':
            try:
                from numpy._core.multiarray import _reconstruct
            except ImportError:  # NumPy 1.x predates the _core package rename.
                from numpy.core.multiarray import _reconstruct
            return _reconstruct
        if (module,name) in allowed:
            return allowed[module,name]
        raise pickle.UnpicklingError(f"unsupported checkpoint global {module}.{name}")

def read_official_checkpoint(path):
    with open(path,'rb') as f:
        ckpt = _NumpyUnpickler(f).load()
    if not isinstance(ckpt,dict) or ckpt.get('format_version') != 2 or not isinstance(ckpt.get('config'),dict):
        raise ValueError('expected official format-v2 dictionary/NumPy checkpoint')
    return ckpt

def canonical_weights(params, metadata):
    """JAX [in,out] / scanned layer arrays -> deployment [out,in] names."""
    w = {'embedding': params['embedding']['embedding']}
    stack = params['stack']
    b = stack['layers']['block']
    sa, ha = b['self_attn'], b['hadamard_mlp']
    for i in range(metadata['num_layers']):
        values = [b['ZCRMSNorm_0']['scale'][i], sa['q_proj']['kernel'][i].T,
                  sa['k_proj']['kernel'][i].T, sa['v_proj']['kernel'][i].T,
                  sa['q_norm']['scale'][i], sa['k_norm']['scale'][i],
                  sa['gate_proj']['kernel'][i].T, sa['out_proj']['kernel'][i].T,
                  b['post_attn_norm']['scale'][i], np.asarray(b['attn_gate'][i]).reshape(1),
                  b['pre_hada_norm']['scale'][i], ha['d1'][i],ha['d2'][i],ha['d3'][i]]
        w.update({f'layer{i:02d}.{name}':v for name,v in zip(LAYER_NAMES,values)})
    for n in MHC_NAMES:
        a = stack[n]
        w[n] = a.transpose(0,2,1).reshape(-1,a.shape[1]) if n.startswith('mhc_phi') else a
    for i in range(len(metadata['engram_layers'])):
        eg = params[f'engrams_{i}']
        w[f'engram{i}.tables'] = eg['embedding'].reshape(-1,eg['embedding'].shape[-1])
        w[f'engram{i}.key_proj'] = eg['key_proj']['kernel'].T
        w[f'engram{i}.value_proj'] = eg['value_proj']['kernel'].T
        w[f'engram{i}.taps'] = eg['taps']
    w['final_norm'] = stack['final_norm']['scale']
    heads = [(n,c) for n,c in (('contrastive_head',1),('confidence_head',2)) if n in params]
    if heads:
        w['heads.manifest'] = np.asarray([c for n,c in heads],dtype=np.float32)
        for n,c in heads:
            p = params[n]
            w[n+'.probes'] = p['probes']
            w[n+'.proj'] = p['proj']['kernel'].T
            w[n+'.bias'] = p['proj'].get('bias',np.zeros(w[n+'.proj'].shape[0],np.float32))
    return {k: np.ascontiguousarray(v,dtype=np.float32) for k,v in w.items()}

def _fingerprint(a):
    return hashlib.sha256(np.ascontiguousarray(a,dtype=np.float32).tobytes()).hexdigest()

def import_model(source, output, template=None):
    source, output = Path(source), Path(output)
    if source.suffix == '.cact':
        arc = Archive.load(source)
        weights = {n:t.dequantize() for n,t in arc.tensors.items() if t.dtype != RAW}
        origin = 'dequantized_deployment'
    else:
        if template is None:
            raise ValueError('official .pkl import requires --template model.cact for exact geometry, codebooks, tokenizer and precision map')
        arc = Archive.load(template)
        ckpt = read_official_checkpoint(source)
        weights = canonical_weights(ckpt['params'],arc.metadata)
        for k in ('d_model','vocab_size','num_layers','num_heads','num_kv_heads'):
            if ckpt['config'].get(k) != arc.metadata[k]:
                raise ValueError(f'checkpoint/template geometry mismatch: {k}')
        origin = 'official_fp16_master'
    expected = {n for n,t in arc.tensors.items() if t.dtype != RAW}
    if set(weights) != expected:
        raise ValueError(f'weight names differ from archive: {set(weights)^expected}')
    for n,w in weights.items():
        if tuple(w.shape) != arc.tensors[n].shape or not np.isfinite(w).all():
            raise ValueError(f'invalid weight {n}')
    output.mkdir(parents=True,exist_ok=True)
    save_file(weights,str(output/'weights.safetensors'),metadata={'format':'needle2-canonical-v1','origin':origin})
    (output/'source.cact').write_bytes(arc.raw)
    config = dict(format='needle2-canonical-v1',origin=origin,geometry=arc.metadata,
                  source_sha256=arc.sha256, source_file_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  precision={n:t.bits for n,t in arc.tensors.items() if t.dtype == CQ},
                  weight_fingerprints={n:_fingerprint(w) for n,w in weights.items()})
    (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    return dict(output=str(output),origin=origin,parameters=sum(w.size for w in weights.values()),**{'archive':arc.stats()})

def load_checkpoint(path):
    path = Path(path)
    config = json.loads((path/'config.json').read_text())
    if config.get('format') != 'needle2-canonical-v1':
        raise ValueError('unsupported PyTorch checkpoint format')
    weights = load_file(str(path/'weights.safetensors'))
    return weights,config

def export_model(checkpoint, output, *, force_requantize=False):
    weights,config = load_checkpoint(checkpoint)
    arc = Archive.load(Path(checkpoint)/'source.cact')
    if arc.sha256 != config['source_sha256']:
        raise ValueError('source.cact hash differs from checkpoint provenance')
    if config['geometry'] != arc.metadata:
        raise ValueError('checkpoint geometry differs from source archive')
    expected = {n for n,t in arc.tensors.items() if t.dtype != RAW}
    if set(weights) != expected:
        raise ValueError(f'checkpoint keys differ: {set(weights)^expected}')
    replacements = {}
    reused, requantized = [],[]
    for n,w in weights.items():
        t = arc.tensors[n]
        if tuple(w.shape) != t.shape or not np.isfinite(w).all():
            raise ValueError(f'{n}: shape mismatch or nonfinite values')
        unchanged = not force_requantize and config['origin'] == 'dequantized_deployment' and _fingerprint(w) == config['weight_fingerprints'].get(n)
        # Fingerprints alone are not sufficient when user edits the manifest.
        if unchanged:
            unchanged = np.array_equal(w,t.dequantize())
        if unchanged:
            reused.append(n)
        else:
            replacements[n] = t.quantized_like(w)
            requantized.append(n)
    raw = arc.rebuild(replacements)
    out = Path(output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_bytes(raw)
    return dict(output=str(out),reused_tensors=len(reused),requantized_tensors=len(requantized),byte_identical=raw == arc.raw,**Archive(raw).stats())

def load_torch_model(path, **kwargs):
    import torch
    from .model import NeedleModel
    path = Path(path)
    if path.is_file():
        arc = Archive.load(path)
        tensors = {n:torch.from_numpy(t.dequantize()) for n,t in arc.tensors.items() if t.dtype != RAW}
        return NeedleModel(arc.metadata,tensors,**kwargs)
    weights, config = load_checkpoint(path)
    return NeedleModel(config['geometry'],{n:torch.from_numpy(w) for n,w in weights.items()},**kwargs)

def save_torch_weights(model, checkpoint):
    """Save updated model Parameters; keep original codebook/format provenance."""
    if getattr(model,'_cq_qat_enabled',False):
        raise ValueError('call disable_qat(model) to restore master weights before saving')
    weights = {n:w.detach().cpu().float().contiguous().numpy() for n,w in model.canonical_state_dict().items()}
    # Model may treat the head manifest as metadata, so preserve its stored copy.
    previous,_ = load_checkpoint(checkpoint)
    for n in previous:
        if n not in weights and n == 'heads.manifest':
            weights[n] = previous[n]
    if weights.keys() != previous.keys():
        raise ValueError('model canonical state differs from checkpoint')
    save_file(weights,str(Path(checkpoint)/'weights.safetensors'))
