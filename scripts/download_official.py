#!/usr/bin/env python3
"""Download reproducible public reference artifacts, never execute them."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

REVISION = '32e9e3a93b205f786929697446ae669cf0a84579'
FILES = ['needle2.cact','checkpoints/needle2.pkl','config.json','LICENSE',
         'tokenizer/tokenizer.model','tokenizer/tokenizer.vocab']

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='artifacts/official')
    p.add_argument('--platform',choices=['linux-arm64','linux-x86_64','macos-arm64'])
    args=p.parse_args()
    files=FILES + ([args.platform+'/'+n for n in ('needle','needle.h','libneedle.a')] if args.platform else [])
    root=Path(args.output); root.mkdir(parents=True,exist_ok=True)
    def get(name):
        dest=root/name; dest.parent.mkdir(parents=True,exist_ok=True)
        temp=dest.with_suffix(dest.suffix+'.part')
        url=f'https://huggingface.co/Cactus-Compute/needle2/resolve/{REVISION}/{name}'
        urllib.request.urlretrieve(url,temp)
        sha=hashlib.sha256(temp.read_bytes()).hexdigest()
        if name=='needle2.cact' and sha!='b43aabfcaf1a6db6acf488076eab71d823c08697c7af4521fc1d174b60ede5ba':
            raise ValueError('official archive hash mismatch')
        temp.replace(dest)
        print(f'{name}: {dest.stat().st_size} bytes',flush=True)
        return name,{'sha256':sha,'bytes':dest.stat().st_size}
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=dict(pool.map(get,files))
    (root/'provenance.json').write_text(json.dumps(dict(repo='Cactus-Compute/needle2',revision=REVISION,files=results),indent=2)+'\n')

if __name__=='__main__': main()
