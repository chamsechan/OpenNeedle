#!/usr/bin/env python3
"""Build a separately instrumented engine and report nested stage wall times.

Instrumentation never changes the production source. Timings overlap (Engram
contains projections), and should guide optimization rather than replace TPS.
"""
import argparse
import ctypes
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--tokens',type=int,default=64);p.add_argument('--output',default='reports/native_profile.json')
    a=p.parse_args()
    dest=ROOT/'artifacts/profiler';dest.mkdir(parents=True,exist_ok=True)
    cq=(ROOT/'needle2/csrc/cq.cpp').read_text();eng=(ROOT/'needle2/csrc/engine.cpp').read_text()
    profiler='''#include <chrono>
static double profile_ms[7]={};
struct ProfileScope{int i; std::chrono::steady_clock::time_point t=std::chrono::steady_clock::now(); ProfileScope(int id):i(id){} ~ProfileScope(){profile_ms[i]+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count();}};
extern "C" void needle2_profile_get(double*p){for(int i=0;i<7;++i)p[i]=profile_ms[i];}
extern "C" void needle2_profile_reset(){for(auto &x:profile_ms)x=0;}
'''
    replacements={
        'static void sinkhorn(float *a,int n) {':'static void sinkhorn(float *a,int n) { ProfileScope prof(4);',
        'void linear(int ti,const float*in,float*out,int first=0,int rows=-1) {':'void linear(int ti,const float*in,float*out,int first=0,int rows=-1) { ProfileScope prof(0);',
        'void row(int ti,int ri,float*out){':'void row(int ti,int ri,float*out){ ProfileScope prof(1);',
        'void engrams() {':'void engrams() { ProfileScope prof(2);',
        'void step(int token,float*out,float*hidden_out) {':'void step(int token,float*out,float*hidden_out) { ProfileScope prof(3);',
    }
    for before,after in replacements.items():
        if eng.count(before)!=1:raise ValueError('instrumentation marker changed: '+before)
        eng=eng.replace(before,after)
    # The fused projection signature can change independently; record it when present.
    marker='void attention_projections('
    if marker in eng:
        pos=eng.index('{',eng.index(marker))+1;eng=eng[:pos]+' ProfileScope prof(5);'+eng[pos:]
    (dest/'engine.cpp').write_text(eng)
    (dest/'cq.cpp').write_text(profiler+cq)
    libpath=dest/'profile.so'
    subprocess.run(['c++','-O3','-DNDEBUG','-std=c++17','-fPIC','-shared','-fopenmp',str(dest/'cq.cpp'),'-o',str(libpath)],check=True)
    import numpy as np
    import needle2.native as native
    from needle2.tokenizer import RefTokenizer
    from needle2.prompt import render_prompt
    native.build_native=lambda:libpath
    engine=native.NativeEngine(ROOT/'artifacts/official/needle2.cact',threads=a.threads)
    tok=RefTokenizer.from_cact(ROOT/'artifacts/official/needle2.cact')
    tools=json.loads((ROOT/'examples/tools.json').read_text())
    ids=[2]+tok.encode(render_prompt('Turn on the kitchen light.',tools))
    engine.prefill(ids,last_only=True)
    lib=engine._lib;lib.needle2_profile_get.argtypes=[ctypes.c_void_p];lib.needle2_profile_reset()
    for i in range(a.tokens):engine.step((100+i)%8192)
    values=np.empty(7,dtype=np.float64);lib.needle2_profile_get(values.ctypes.data)
    names=['other_projections','embedding_rows','engram_including_projections','whole_step','sinkhorn','fused_qkvg','unused']
    result=dict(threads=a.threads,tokens=a.tokens,per_token_ms=dict(zip(names,(values/a.tokens).tolist())),note='Nested wall-clock durations overlap; profiling-only build.')
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
