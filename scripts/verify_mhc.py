"""Compare full logits and hidden states against the isolated pre-change library."""
from pathlib import Path
import json,os,subprocess,sys,hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
OUT=ROOT/'artifacts/decode_experiments'

def dump(name):
    os.environ['NEEDLE2_NATIVE_LIBRARY']=str(OUT/name/'lib.so')
    os.sched_setaffinity(0,{0,1,2,3})
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from needle2.native import NativeEngine
    from test_model import tiny_model
    from needle2.archive import TensorRecord,CQ,FP32
    from needle2.quantize import quantize_matrix,codebook
    from types import SimpleNamespace
    results={}
    for matmul in ['fp32','sdot']:
        for kv in ['fp32','int8']:
            for threads in [1,4]:
                e=NativeEngine(ROOT/'artifacts/official/needle2.cact',threads=threads,matmul=matmul,kv_cache=kv)
                e.reset(prefix_len=8);e.prefill([2,31,72,16,3,9,81,42],last_only=True);e.cache_prefix()
                for i in range(280):
                    probe=i in [0,1,15,63,247,248,249,255,256,279]
                    value=e.step(16+(i*37)%8000,compute_logits=probe,return_hidden=probe)
                    if probe:
                        logits,hidden=value;key=f'{matmul}_{kv}_{threads}_{i}'
                        results[key+'_logits']=logits;results[key+'_hidden']=hidden
                e.reset_to_prefix();results[f'{matmul}_{kv}_{threads}_restore']=e.step(91)
                del e
    for layout in ['shared','mixed_bits','mixed_groups','dense']:
        model=tiny_model(window=5);records={}
        for key,tensor in model.canonical_state_dict().items():
            a=tensor.numpy()
            if key.startswith('mhc_phi') and layout!='dense':
                bits=2 if layout=='mixed_bits' and key=='mhc_phi_post' else 4
                group=64 if layout=='mixed_groups' and key=='mhc_phi_res' else 128
                packed,norms=quantize_matrix(a,bits=bits,group_size=group)
                records[key]=TensorRecord(key,CQ,a.shape,packed.tobytes()+norms.tobytes(),group,bits,codebook(bits,group))
            else:records[key]=TensorRecord(key,FP32,a.shape,a.tobytes())
        meta=model.config.to_dict();meta['hada_n']=16
        e=NativeEngine(SimpleNamespace(metadata=meta,tensors=records),threads=2,matmul='sdot')
        for i in range(12):results[f'tiny_{layout}_{i}']=e.step(i+1)
        del e
    np.savez(OUT/f'{name}_mhc_validation.npz',**results)
    print(name,len(results),'arrays',flush=True)

def main(reuse=False):
    import numpy as np
    names=['baseline','mhc_pair','mhc_parallel','mhc_final']
    if not reuse:
        for name in names:
            subprocess.run([sys.executable,__file__,'dump',name],check=True)
    a=np.load(OUT/'baseline_mhc_validation.npz')
    for name in names[1:]:
        b=np.load(OUT/f'{name}_mhc_validation.npz')
        assert a.files==b.files
        for key in a.files:np.testing.assert_array_equal(a[key],b[key],err_msg=name+':'+key)
    report=dict(bitwise_equal=True,arrays_per_variant=len(a.files),elements_per_variant=sum(a[k].size for k in a.files),
                coverage='FP32/SDOT x FP32/INT8 KV x 1/4 threads; 280 steps with pinned prefix and window wrap; prefix restore; tiny two-lane archives with different mHC storage layouts (all dequantized by the loader)',
                library_sha256={n:hashlib.sha256((OUT/n/'lib.so').read_bytes()).hexdigest() for n in names})
    (ROOT/'reports/mhc_numerical_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report)
if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='dump':dump(sys.argv[2])
    else:main(reuse=len(sys.argv)>1 and sys.argv[1]=='compare')
