"""Separate attention wall stages from summed worker work; do not add both together."""
from pathlib import Path
import ctypes as ct
import json
import subprocess
import sys
import profile_real_decode as base
VARIANT='attention_detail'
MAIN=['qk_norm','rope','kv_store','attention_dispatch_and_compute','sigmoid_gate']
WORKER=['qk_scores','softmax','weighted_values']

def build():
    lib=base.build(VARIANT,compile_library=False);dest=lib.parent
    cq=dest/'cq.cpp';s=cq.read_text();marker='static bool profile_enabled=false;'
    s=s.replace(marker,marker+'''
using AttentionClock=std::chrono::steady_clock;
static double attention_main[5]={};
struct alignas(64) AttentionWorker {double t[3]={};};
static AttentionWorker attention_workers[256];
static void attention_mark(AttentionClock::time_point& t,double* output,int stage){
    auto now=AttentionClock::now();if(profile_enabled)output[stage]+=std::chrono::duration<double,std::milli>(now-t).count();t=now;
}
extern "C" void attention_profile_reset(){for(auto&v:attention_main)v=0;for(auto&w:attention_workers)for(auto&v:w.t)v=0;}
extern "C" void attention_profile_get(double*main,double*workers){
    for(int i=0;i<5;++i)main[i]=attention_main[i];
    for(int i=0;i<3;++i){workers[i]=0;for(auto&w:attention_workers)workers[i]+=w.t[i];}
}
''')
    s=s.replace('extern "C" void real_profile_reset(){','extern "C" void real_profile_reset(){attention_profile_reset();')
    cq.write_text(s)
    p=dest/'engine.cpp';s=p.read_text();a=s.index('    void step(');b=s.index('    void prefill(',a);step=s[a:b]
    marker='            for(int h=0;h<H;++h)rms(q.data()'
    step=step.replace(marker,'            auto attention_time=AttentionClock::now();\n'+marker)
    for marker,stage in [('            for(int j=0;j<HD/2;++j) {',0),('            int slot=cache_slot(position);',1),('            compute_attention(l, position, q.data(), att.data());',2),('            sigmoid_gate(att.data(),gate.data(),A);',3)]:
        assert step.count(marker)==1,marker
        step=step.replace(marker,f'            attention_mark(attention_time,attention_main,{stage});\n'+marker)
    marker='            sigmoid_gate(att.data(),gate.data(),A);'
    step=step.replace(marker,marker+'\n            attention_mark(attention_time,attention_main,4);')
    s=s[:a]+step+s[b:]
    a=s.index('    void compute_attention(');b=s.index('    void step(',a);func=s[a:b]
    func=func.replace('                float max0=', '                auto worker_time=AttentionClock::now();\n                float max0=')
    marker='                    softmax_inplace(s0,length,max0);if(count==2)softmax_inplace(s1,length,max1);'
    assert func.count(marker)==2
    func=func.replace(marker,'                    attention_mark(worker_time,attention_workers[tid].t,0);\n'+marker+'\n                    attention_mark(worker_time,attention_workers[tid].t,1);')
    marker='''                }
            }
        });'''
    assert func.count(marker)==1
    func=func.replace(marker,'''                }
                attention_mark(worker_time,attention_workers[tid].t,2);
            }
        });''')
    p.write_text(s[:a]+func+s[b:])
    subprocess.run(['c++','-O3','-DNDEBUG','-std=c++17','-fPIC','-shared','-pthread','-fopenmp',str(cq),'-o',str(lib)],check=True)
    print(lib)

def read(lib):
    import numpy as np
    main=np.empty(5,np.float64);workers=np.empty(3,np.float64)
    lib.attention_profile_get.argtypes=[ct.c_void_p,ct.c_void_p]
    lib.attention_profile_get(main.ctypes.data,workers.ctypes.data)
    return {'wall_stages_ms':dict(zip(MAIN,main.tolist())),'summed_worker_ms':dict(zip(WORKER,workers.tolist()))}

def run():
    base.run(VARIANT,extra_reader=read)
    p=base.ROOT/f'reports/real_decode_profile_{VARIANT}.json';r=json.loads(p.read_text())
    for suite,d in r['results'].items():
        steps=sum(row['steps'] for row in d['rows'])
        d['attention_summary']={group:{key:sum(row['attention_detail'][group][key] for row in d['rows'])/steps for key in keys} for group,keys in [('wall_stages_ms',MAIN),('summed_worker_ms',WORKER)]}
        print(suite,d['attention_summary'])
    r['worker_timing_note']='Worker times sum overlapping per-thread durations. They are not critical-path wall times and must not be added to main wall stages; clocks and scheduling perturb measurements.'
    p.write_text(json.dumps(r,indent=2)+'\n')
if __name__=='__main__':
    if sys.argv[1]=='build':build()
    else:run()
