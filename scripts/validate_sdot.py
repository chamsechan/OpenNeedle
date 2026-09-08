#!/usr/bin/env python3
"""Measure whole-network SDOT approximation error; does not certify parity."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tokens',type=int,default=64)
    p.add_argument('--output',default='reports/sdot_model_error.json');a=p.parse_args()
    import numpy as np
    from needle2.archive import Archive
    from needle2.native import NativeEngine,sdot_available
    from needle2.prompt import render_prompt
    from needle2.tokenizer import RefTokenizer
    if not sdot_available():raise SystemExit('ARM DotProd instructions unavailable')
    model=ROOT/'artifacts/official/needle2.cact';arc=Archive.load(model);tok=RefTokenizer.from_cact(model)
    tools=json.loads((ROOT/'examples/tools.json').read_text())
    ids=([2]+tok.encode(render_prompt('Turn on the kitchen light.',tools)))[:a.tokens]
    full=NativeEngine(arc,threads=1,matmul='fp32').prefill(ids)
    fast=NativeEngine(arc,threads=1,matmul='sdot').prefill(ids)
    error=fast.astype(np.float64)-full
    x,y=full.ravel().astype(np.float64),fast.ravel().astype(np.float64)
    differing=np.flatnonzero(full.argmax(-1)!=fast.argmax(-1)).tolist()
    out=dict(model_sha256=arc.sha256,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'needle2/csrc').glob('*.cpp')},
             reference='independent FP32',candidate='CQ codebook INT8 + rotated groupwise A8 SDOT; KV FP32',
             input_tokens=ids,logits_shape=list(fast.shape),max_absolute_error=float(abs(error).max()),
             root_mean_square_error=float(np.mean(error**2)**.5),relative_l2_error=float(np.linalg.norm(error)/np.linalg.norm(x)),
             cosine_similarity=float(np.dot(x,y)/np.linalg.norm(x)/np.linalg.norm(y)),
             top1_agreement=float(np.mean(full.argmax(-1)==fast.argmax(-1))),differing_positions=differing,
             note='Approximation diagnostics on a fixed prefix, not task accuracy or production-int8 parity.')
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
    if not np.isfinite(fast).all():raise SystemExit(1)

if __name__=='__main__':main()
