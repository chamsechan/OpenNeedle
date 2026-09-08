#!/usr/bin/env python3
"""Validate real weights, round trips and native/PyTorch model numerics."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default=str(ROOT/'artifacts/official/needle2.cact'))
    p.add_argument('--output',default=str(ROOT/'reports/validation.json'))
    p.add_argument('--tokens',type=int,default=192)
    p.add_argument('--threads',type=int,default=1)
    args=p.parse_args()
    if args.tokens<2 or args.threads<1:p.error('tokens>=2 and threads>=1 required')
    import numpy as np
    import torch
    from needle2.archive import Archive,CQ
    from needle2.convert import load_torch_model
    from needle2.native import NativeEngine,features,build_native
    from needle2.tokenizer import RefTokenizer
    from needle2.prompt import render_prompt
    torch.set_num_threads(args.threads)
    a=Archive.load(args.model)
    tokenizer=RefTokenizer.from_cact(args.model)
    tools=json.loads((ROOT/'examples/tools.json').read_text())
    ids=([2]+tokenizer.encode(render_prompt('Turn on the kitchen light.',tools)))[:args.tokens]
    model=load_torch_model(args.model).eval()
    native=NativeEngine(a,threads=args.threads)
    started=time.perf_counter()
    with torch.inference_mode():reference=model(torch.tensor([ids]))[0].numpy()
    torch_seconds=time.perf_counter()-started
    started=time.perf_counter(); actual=native.prefill(ids); native_seconds=time.perf_counter()-started
    diff=actual.astype(np.float64)-reference
    cosine=float(np.dot(actual.ravel().astype(np.float64),reference.ravel())/(np.linalg.norm(actual.astype(np.float64))*np.linalg.norm(reference.astype(np.float64))))
    agreement=float(np.mean(actual.argmax(-1)==reference.argmax(-1)))
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),model=a.stats(),
                source_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in sorted((ROOT/'needle2').rglob('*.py'))+sorted((ROOT/'needle2/csrc').glob('*.cpp'))},
                native_library_sha256=hashlib.sha256(build_native().read_bytes()).hexdigest(),
                native_features=features(),threads=args.threads,affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
                numerics=dict(tokens=ids,logits_shape=list(actual.shape),max_absolute_error=float(abs(diff).max()),
                              root_mean_square_error=float(np.mean(diff**2)**.5),cosine_similarity=cosine,
                              top1_agreement=agreement,torch_seconds=torch_seconds,native_seconds=native_seconds),
                notes=['FP32 public-reference arithmetic, not the closed production A8/KV8 arithmetic.',
                       'Timing here is validation overhead, not a controlled performance benchmark.'])
    roundtrip=ROOT/'artifacts/roundtrip.cact'
    if roundtrip.exists():result['archive_roundtrip_byte_identical']=roundtrip.read_bytes()==a.raw
    master=ROOT/'artifacts/from_master.cact'
    if master.exists():
        b=Archive.load(master); different=[]; packed_bytes=scale_entries=0
        for n,t in a.tensors.items():
            u=b.tensors[n]
            if t.blob!=u.blob:
                item=dict(name=n,bytes_different=int(np.count_nonzero(np.frombuffer(t.blob,'u1')!=np.frombuffer(u.blob,'u1'))))
                if t.dtype==CQ:
                    item.update(packed_bytes_different=int(np.count_nonzero(t.packed!=u.packed)),scale_entries_different=int(np.count_nonzero(t.norms!=u.norms)))
                    packed_bytes+=item['packed_bytes_different'];scale_entries+=item['scale_entries_different']
                different.append(item)
        result['master_requantization']=dict(sha256=b.sha256,differences=different,packed_bytes_different=packed_bytes,scale_entries_different=scale_entries)
    result['passed']=cosine>.999999 and agreement==1.0 and np.isfinite(actual).all().item()
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit(1)

if __name__=='__main__':main()
