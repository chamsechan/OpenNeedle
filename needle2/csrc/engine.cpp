// Included by cq.cpp: single-stream Needle 2 forward, using the public architecture.
#include <memory>
#include <stdexcept>

struct TensorDesc {void *cq; const float *data; int rows; int cols;};
struct EngineConfig {
    int vocab,dim,heads,kvheads,layers,head_dim,max_seq,hada,lanes,window;
    int slots,subdim,tables,taps,dilation,num_orders,orders[4],num_sites,sites[4];
    float rope_theta;
};
static float dot_f32(const float *a,const float *b,int n) {
    int i=0;
#ifdef __aarch64__
    auto x=vdupq_n_f32(0),y=vdupq_n_f32(0),z=vdupq_n_f32(0),w=vdupq_n_f32(0);
    for(;i+16<=n;i+=16) {
        x=vfmaq_f32(x,vld1q_f32(a+i),vld1q_f32(b+i));
        y=vfmaq_f32(y,vld1q_f32(a+i+4),vld1q_f32(b+i+4));
        z=vfmaq_f32(z,vld1q_f32(a+i+8),vld1q_f32(b+i+8));
        w=vfmaq_f32(w,vld1q_f32(a+i+12),vld1q_f32(b+i+12));
    }
    float s=vaddvq_f32(vaddq_f32(vaddq_f32(x,y),vaddq_f32(z,w)));
#else
    float s=0;
#endif
    for(;i<n;++i)s+=a[i]*b[i];return s;
}
static float sigmoid(float x) {return 1/(1+std::exp(-x));}
static void dot_pair(const float*q0,const float*q1,const float*k,int n,float&out0,float&out1) {
    int d=0;
#ifdef __aarch64__
    auto a=vdupq_n_f32(0),b=vdupq_n_f32(0),c=vdupq_n_f32(0),e=vdupq_n_f32(0);
    auto a1=a,b1=b,c1=c,e1=e;
    for(;d+16<=n;d+=16) {
        auto k0=vld1q_f32(k+d),k1=vld1q_f32(k+d+4),k2=vld1q_f32(k+d+8),k3=vld1q_f32(k+d+12);
        a=vfmaq_f32(a,k0,vld1q_f32(q0+d));b=vfmaq_f32(b,k1,vld1q_f32(q0+d+4));
        c=vfmaq_f32(c,k2,vld1q_f32(q0+d+8));e=vfmaq_f32(e,k3,vld1q_f32(q0+d+12));
        a1=vfmaq_f32(a1,k0,vld1q_f32(q1+d));b1=vfmaq_f32(b1,k1,vld1q_f32(q1+d+4));
        c1=vfmaq_f32(c1,k2,vld1q_f32(q1+d+8));e1=vfmaq_f32(e1,k3,vld1q_f32(q1+d+12));
    }
    out0=vaddvq_f32(vaddq_f32(vaddq_f32(a,b),vaddq_f32(c,e)));
    out1=vaddvq_f32(vaddq_f32(vaddq_f32(a1,b1),vaddq_f32(c1,e1)));
#else
    out0=out1=0;
#endif
    for(;d<n;++d){out0+=q0[d]*k[d];out1+=q1[d]*k[d];}
}
#ifdef __aarch64__
// Range reduction leaves r in [-log(2)/2, log(2)/2]. A degree-six
// Taylor polynomial has < 2e-7 relative truncation error on this interval.
// Clamping only removes tails far below FP32 softmax significance.
static inline float32x4_t exp4(float32x4_t x) {
    x=vmaxq_f32(vdupq_n_f32(-80),vminq_f32(vdupq_n_f32(80),x));
    auto nf=vrndnq_f32(vmulq_n_f32(x,1.4426950408889634f));
    auto r=vfmsq_n_f32(x,nf,0.693145751953125f);
    r=vfmsq_n_f32(r,nf,1.428606765330187e-6f);
    auto p=vdupq_n_f32(1.0f/720);
    p=vfmaq_f32(vdupq_n_f32(1.0f/120),p,r);
    p=vfmaq_f32(vdupq_n_f32(1.0f/24),p,r);
    p=vfmaq_f32(vdupq_n_f32(1.0f/6),p,r);
    p=vfmaq_f32(vdupq_n_f32(.5f),p,r);
    p=vfmaq_f32(vdupq_n_f32(1),p,r);
    p=vfmaq_f32(vdupq_n_f32(1),p,r);
    auto exponent=vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(nf),vdupq_n_s32(127)),23);
    return vmulq_f32(p,vreinterpretq_f32_s32(exponent));
}
static inline float32x4_t sigmoid4(float32x4_t x) {
    return vdivq_f32(vdupq_n_f32(1),vaddq_f32(vdupq_n_f32(1),exp4(vnegq_f32(x))));
}
#endif
static void sigmoid_gate(float*x,const float*g,int n) {
    int i=0;
#ifdef __aarch64__
    for(;i+4<=n;i+=4)vst1q_f32(x+i,vmulq_f32(vld1q_f32(x+i),sigmoid4(vld1q_f32(g+i))));
#endif
    for(;i<n;++i)x[i]*=sigmoid(g[i]);
}
static void silu_diagonal(float*x,const float*d,int n) {
    int i=0;
#ifdef __aarch64__
    for(;i+4<=n;i+=4){auto a=vmulq_f32(vld1q_f32(x+i),vld1q_f32(d+i));vst1q_f32(x+i,vmulq_f32(a,sigmoid4(a)));}
#endif
    for(;i<n;++i){float a=x[i]*d[i];x[i]=a*sigmoid(a);}
}
static void softmax_inplace(float*x,int n,float maximum) {
    int i=0;float den=0;
#ifdef __aarch64__
    auto sums=vdupq_n_f32(0);
    for(;i+4<=n;i+=4){auto e=exp4(vsubq_f32(vld1q_f32(x+i),vdupq_n_f32(maximum)));vst1q_f32(x+i,e);sums=vaddq_f32(sums,e);}
    den=vaddvq_f32(sums);
#endif
    for(;i<n;++i){x[i]=std::exp(x[i]-maximum);den+=x[i];}
    float inv=1/den;
    for(i=0;i<n;++i)x[i]*=inv;
}
// Algebraically the same 20 alternating normalizations as log-space
// Sinkhorn. Stabilize the first exponential once, then normalize positive
// entries. Fall back to log space for extreme learned routing ranges.
static void sinkhorn(float *a,int n) {
    float maximum=*std::max_element(a,a+n*n);
    float minimum=*std::min_element(a,a+n*n);
    if(!std::isfinite(maximum)||maximum-minimum>60) {
        for(int iteration=0;iteration<20;++iteration) {
            for(int i=0;i<n;++i){float m=-INFINITY;for(int j=0;j<n;++j)m=std::max(m,a[i*n+j]);float sum=0;for(int j=0;j<n;++j)sum+=std::exp(a[i*n+j]-m);float lse=m+std::log(sum);for(int j=0;j<n;++j)a[i*n+j]-=lse;}
            for(int j=0;j<n;++j){float m=-INFINITY;for(int i=0;i<n;++i)m=std::max(m,a[i*n+j]);float sum=0;for(int i=0;i<n;++i)sum+=std::exp(a[i*n+j]-m);float lse=m+std::log(sum);for(int i=0;i<n;++i)a[i*n+j]-=lse;}
        }
        for(int i=0;i<n*n;++i)a[i]=std::exp(a[i]);
        return;
    }
    for(int i=0;i<n*n;++i)a[i]=std::exp(a[i]-maximum);
#ifdef __aarch64__
    if(n==4) {
        auto r0=vld1q_f32(a),r1=vld1q_f32(a+4),r2=vld1q_f32(a+8),r3=vld1q_f32(a+12);
        for(int iteration=0;iteration<20;++iteration) {
            r0=vmulq_n_f32(r0,1/vaddvq_f32(r0));r1=vmulq_n_f32(r1,1/vaddvq_f32(r1));
            r2=vmulq_n_f32(r2,1/vaddvq_f32(r2));r3=vmulq_n_f32(r3,1/vaddvq_f32(r3));
            auto inverse=vdivq_f32(vdupq_n_f32(1),vaddq_f32(vaddq_f32(r0,r1),vaddq_f32(r2,r3)));
            r0=vmulq_f32(r0,inverse);r1=vmulq_f32(r1,inverse);r2=vmulq_f32(r2,inverse);r3=vmulq_f32(r3,inverse);
        }
        vst1q_f32(a,r0);vst1q_f32(a+4,r1);vst1q_f32(a+8,r2);vst1q_f32(a+12,r3);
        return;
    }
#endif
    for(int iteration=0;iteration<20;++iteration) {
        for(int i=0;i<n;++i){float sum=0;for(int j=0;j<n;++j)sum+=a[i*n+j];float inv=1/sum;for(int j=0;j<n;++j)a[i*n+j]*=inv;}
        for(int j=0;j<n;++j){float sum=0;for(int i=0;i<n;++i)sum+=a[i*n+j];float inv=1/sum;for(int i=0;i<n;++i)a[i*n+j]*=inv;}
    }
}
static void rms(const float*x,float*y,int n,const float*scale=nullptr) {
    float r=1/std::sqrt(dot_f32(x,x,n)/n+1e-6f);
    for(int i=0;i<n;++i)y[i]=x[i]*r*(scale?1+scale[i]:1);
}
static void activation_quant(float*x,int n,int bits) {
    if(!bits)return;
    float a=0;for(int i=0;i<n;++i)a=std::max(a,std::abs(x[i]));
    if(!a)return;
    int max=(1<<(bits-1))-1;float scale=a/max;
    for(int i=0;i<n;++i)x[i]=std::max(float(-max-1),std::min(float(max),std::nearbyint(x[i]/scale)))*scale;
}
struct PrefixSnapshot {
    int position;
    std::vector<int> history;
    std::vector<float> keys,values,engvalues;
};
struct Engine {
    EngineConfig c;
    std::vector<float> activation_lut;
    bool sdot_enabled=false;
    std::unique_ptr<PrefixSnapshot> prefix_snapshot;
    std::vector<std::unique_ptr<SdotCQ>> sdot;
    Prepared sdot_input;
    std::vector<TensorDesc> t;
    int threads,abits,position=0,capacity,engring,prefix=0,projection_lookup=-1;
    std::vector<int> history;
    std::vector<float> keys,values,engvalues;
    std::vector<float> x,nx,u,bx,z,q,k,v,gate,att,proj,mlp,newx,hpre,hpost,hres;
    std::vector<float> ek,ev,e,rawv,logits,hidden,scores,rot,rope_cos,rope_sin,rope_divisor;
    Engine(const EngineConfig& cfg,const TensorDesc *td,int count,int th,int ab):c(cfg),t(td,td+count),threads(th),abits(ab) {
        int D=c.dim,N=c.lanes,A=c.heads*c.head_dim,K=c.kvheads*c.head_dim;
        sdot.resize(t.size());
        capacity=c.window?c.window:c.max_seq;engring=(c.taps-1)*c.dilation+1;
        keys.resize(size_t(c.layers)*capacity*K);values.resize(keys.size());
        engvalues.resize(size_t(c.num_sites)*engring*D);
        x.resize(N*D);nx.resize(N*D);u.resize(D);bx.resize(D);z.resize(D);
        q.resize(A);k.resize(K);v.resize(K);gate.resize(A);att.resize(A);proj.resize(D);mlp.resize(c.hada);
        newx.resize(N*D);hpre.resize(N);hpost.resize(N);hres.resize(N*N);
        ek.resize(c.num_sites*D);ev.resize(c.num_sites*D);e.resize(D);rawv.resize(D);
        logits.resize(c.vocab);hidden.resize(c.layers*D);scores.resize(2*capacity);int rotation_size=std::max({c.hada,N*D,A});
        for(const auto &td:t)if(td.cq)rotation_size=std::max(rotation_size,static_cast<CQ*>(td.cq)->padded);
        rot.resize(rotation_size);
        rope_cos.resize(c.head_dim/2);rope_sin.resize(c.head_dim/2);rope_divisor.resize(c.head_dim/2);
        for(int j=0;j<c.head_dim/2;++j)rope_divisor[j]=std::pow(c.rope_theta,float(2*j)/c.head_dim);
    }
    bool configure_sdot() {
        if(!runtime_dotprod())return false;
        for(size_t i=0;i<t.size();++i)if(t[i].cq) {
            auto*q=static_cast<CQ*>(t[i].cq);
            if((q->bits==2||q->bits==4)&&q->group>=64&&q->group<=131072)
                sdot[i]=std::make_unique<SdotCQ>(q->packed,q->norms,q->out,q->in,q->bits,q->group,q->cb.data());
        }
        sdot_enabled=true;return true;
    }
    void reset(int prefix_len){
        prefix_snapshot.reset();
        position=0;history.clear();prefix=prefix_len;
        capacity=c.window?c.window+prefix:c.max_seq;
        keys.resize(size_t(c.layers)*capacity*c.kvheads*c.head_dim);values.resize(keys.size());scores.resize(2*capacity);
    }
    void cache_prefix() {
        if(position<=0||position!=prefix)throw std::runtime_error("prefix snapshot requires position == prefix_len > 0");
        auto snapshot=std::make_unique<PrefixSnapshot>();snapshot->position=position;
        snapshot->history=history;snapshot->engvalues=engvalues;
        size_t K=size_t(c.kvheads)*c.head_dim;
        snapshot->keys.resize(size_t(c.layers)*prefix*K);snapshot->values.resize(snapshot->keys.size());
        for(int l=0;l<c.layers;++l) {
            const auto *ks=keys.data()+size_t(l)*capacity*K,*vs=values.data()+size_t(l)*capacity*K;
            std::copy(ks,ks+size_t(prefix)*K,snapshot->keys.data()+size_t(l)*prefix*K);
            std::copy(vs,vs+size_t(prefix)*K,snapshot->values.data()+size_t(l)*prefix*K);
        }
        prefix_snapshot=std::move(snapshot);
    }
    int reset_to_prefix() {
        if(!prefix_snapshot)throw std::runtime_error("no cached prefix; call cache_prefix() after consuming the complete prefix");
        int cached=prefix_snapshot->position;
        if(prefix!=cached||capacity!=(c.window?c.window+cached:c.max_seq))throw std::runtime_error("cached prefix geometry no longer matches the current state");
        history=prefix_snapshot->history;engvalues=prefix_snapshot->engvalues;
        size_t K=size_t(c.kvheads)*c.head_dim;
        for(int l=0;l<c.layers;++l) {
            const auto *ks=prefix_snapshot->keys.data()+size_t(l)*cached*K,*vs=prefix_snapshot->values.data()+size_t(l)*cached*K;
            std::copy(ks,ks+size_t(cached)*K,keys.data()+size_t(l)*capacity*K);
            std::copy(vs,vs+size_t(cached)*K,values.data()+size_t(l)*capacity*K);
        }
        position=cached;
        return position;
    }
    int cache_slot(int pos)const{return !c.window?pos:pos<prefix?pos:prefix+(pos-prefix)%c.window;}
    const float* dense(int i)const{return t[i].data;}
    void linear(int ti,const float*in,float*out,int first=0,int rows=-1) {
        const auto &w=t[ti];if(rows<0)rows=w.rows;
        if(sdot_enabled&&sdot[ti]) {
            auto*q=sdot[ti].get();q->prepare(in,sdot_input);
            int n4=rows/4;
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) if(threads>1&&rows>=128) schedule(static)
#endif
            for(int b=0;b<n4;++b)q->row4(first+b*4,sdot_input,out+b*4);
            for(int r=n4*4;r<rows;++r)out[r]=q->row(first+r,sdot_input);
        } else if(w.cq) {
            auto *cq=static_cast<CQ*>(w.cq);
            cq->transform(in,rot.data());
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) if(threads > 1 && rows>=128) schedule(static)
#endif
            for(int r=0;r<rows;++r)out[r]=cq->dot_row(first+r,rot.data());
        } else {
            for(int r=0;r<rows;++r)out[r]=dot_f32(w.data+size_t(first+r)*w.cols,in,w.cols);
        }
    }
    void attention_projections(int ti,const float*input) {
        if(sdot_enabled) {
            SdotCQ*matrices[4]={sdot[ti+1].get(),sdot[ti+2].get(),sdot[ti+3].get(),sdot[ti+6].get()};
            float*outputs[4]={q.data(),k.data(),v.data(),gate.data()};
            bool compatible=matrices[0]!=nullptr;
            for(int i=1;i<4&&compatible;++i)compatible=matrices[i]&&matrices[i]->bits==matrices[0]->bits&&matrices[i]->group==matrices[0]->group&&matrices[i]->columns==matrices[0]->columns;
            if(!compatible){linear(ti+1,input,q.data());linear(ti+2,input,k.data());linear(ti+3,input,v.data());linear(ti+6,input,gate.data());return;}
            matrices[0]->prepare(input,sdot_input);int total=0;for(auto*m:matrices)total+=m->rows;
            int total4=total/4;
#ifdef _OPENMP
#pragma omp parallel num_threads(threads) if(threads>1)
#endif
            {
#ifdef _OPENMP
                int tid=omp_get_thread_num(),nt=omp_get_num_threads();
#else
                int tid=0,nt=1;
#endif
                int begin=total4*tid/nt*4,end=total4*(tid+1)/nt*4,offset=0;
                for(int m=0;m<4;++m){
                    int b=std::max(0,begin-offset),e=std::min(matrices[m]->rows,end-offset);
                    offset+=matrices[m]->rows;
                    for(int r=b;r<e-3;r+=4)matrices[m]->row4(r,sdot_input,outputs[m]+r);
                    for(int r=std::max(b,(e/4)*4);r<e;++r)outputs[m][r]=matrices[m]->row(r,sdot_input);
                }
            }
            return;
        }
        CQ*matrices[4]={static_cast<CQ*>(t[ti+1].cq),static_cast<CQ*>(t[ti+2].cq),static_cast<CQ*>(t[ti+3].cq),static_cast<CQ*>(t[ti+6].cq)};
        float*outputs[4]={q.data(),k.data(),v.data(),gate.data()};
        if(!matrices[0]||!matrices[1]||!matrices[2]||!matrices[3]||!compatible_rotation(matrices,4)) {
            linear(ti+1,input,q.data());linear(ti+2,input,k.data());linear(ti+3,input,v.data());linear(ti+6,input,gate.data());return;
        }
        int lookup=projection_lookup;
        if(lookup<0) {
            int rows=0;for(auto*m:matrices)rows+=m->out;
            lookup=threads<=2&&matrices[0]->bits==2&&matrices[0]->group==128&&rows>=1024?4:0;
        }
        if(lookup&&lookup!=4)activation_lut.resize(size_t(matrices[0]->activation_table_count(lookup))*matrices[0]->activation_table_stride(lookup));
        cq_linear_many(matrices,4,input,outputs,threads,lookup,rot.data(),activation_lut.data());
    }
    void row(int ti,int ri,float*out){
        if(t[ti].cq)static_cast<CQ*>(t[ti].cq)->row(ri,out);
        else std::copy(t[ti].data+size_t(ri)*t[ti].cols,t[ti].data+size_t(ri+1)*t[ti].cols,out);
    }
    void engrams() {
        if(!c.num_sites)return;
        int D=c.dim,base=1+14*c.layers+9,heads=c.tables/c.num_orders;
        for(int s=0;s<c.num_sites;++s) {
            int ti=base+4*s;
            for(int table=0;table<c.tables;++table) {
                int order=c.orders[table/heads];
                float *dst=e.data()+table*c.subdim;
                if(position+1<order || (c.window && order-1>=c.window && position-order+1>=prefix)){std::fill(dst,dst+c.subdim,0);continue;}
                uint32_t acc=uint32_t(0x9e3779b9)*uint32_t(table+1);
                for(int j=0;j<order;++j)acc=(acc^uint32_t(history[position-j]))*uint32_t(0x01000193);
                acc^=acc>>15;
                row(ti,table*c.slots+acc%c.slots,dst);
            }
            activation_quant(e.data(),D,abits);
            linear(ti+1,e.data(),ek.data()+s*D);
            linear(ti+2,e.data(),rawv.data());
            auto *cache=engvalues.data()+size_t(s)*engring*D;
            std::copy(rawv.begin(),rawv.end(),cache+(position%engring)*D);
            for(int d=0;d<D;++d){
                float sum=0;
                for(int tap=0;tap<c.taps;++tap) {
                    int past=position-tap*c.dilation;
                    if(past>=0 && (!c.window || tap*c.dilation<c.window || past<prefix))sum+=dense(ti+3)[tap*D+d]*cache[(past%engring)*D+d];
                }
                ev[s*D+d]=sum;
            }
        }
    }
    void import_state(int pos,int pinned,const int*ids,const int*positions,int count,const float*k,const float*v) {
        reset(pinned);position=pos;history.assign(ids,ids+pos);
        int K=c.kvheads*c.head_dim;
        for(int l=0;l<c.layers;++l)for(int t=0;t<count;++t) {
            int slot=cache_slot(positions[t]);
            const auto *ks=k+(size_t(l)*count+t)*K,*vs=v+(size_t(l)*count+t)*K;
            std::copy(ks,ks+K,keys.data()+(size_t(l)*capacity+slot)*K);
            std::copy(vs,vs+K,values.data()+(size_t(l)*capacity+slot)*K);
        }
        // Only raw values are persistent; recompute the short Engram history.
        for(position=std::max(0,pos-engring);position<pos;++position)engrams();
        position=pos;
    }
    void step(int token,float*out,float*hidden_out) {
        int D=c.dim,N=c.lanes,H=c.heads,KV=c.kvheads,HD=c.head_dim,A=H*HD,K=KV*HD;
        if(token<0||token>=c.vocab)throw std::runtime_error("token outside vocabulary");
        if(!c.window&&position>=capacity)throw std::runtime_error("maximum context reached");
        history.push_back(token);
        row(0,token,z.data());
        for(int n=0;n<N;++n)for(int d=0;d<D;++d)x[n*D+d]=z[d]*std::sqrt(float(D));
        engrams();
        for(int j=0;j<HD/2;++j){float a=position/rope_divisor[j];rope_cos[j]=std::cos(a);rope_sin[j]=std::sin(a);}
        int mh=1+14*c.layers;
        for(int l=0;l<c.layers;++l) {
            int ti=1+14*l;
            rms(x.data(),nx.data(),N*D);
            linear(mh+6,nx.data(),hpre.data(),l*N,N);
            linear(mh+7,nx.data(),hpost.data(),l*N,N);
            linear(mh+8,nx.data(),hres.data(),l*N*N,N*N);
            for(int n=0;n<N;++n) {
                hpre[n]=sigmoid(dense(mh)[l]*hpre[n]+dense(mh+3)[l*N+n]+(n==l%N?4:-4));
                hpost[n]=2*sigmoid(dense(mh+1)[l]*hpost[n]+dense(mh+4)[l*N+n]+(n==l%N?0:-4));
            }
            for(int d=0;d<D;++d){float sum=0;for(int n=0;n<N;++n)sum+=hpre[n]*x[n*D+d];u[d]=bx[d]=sum;}
            for(int s=0;s<c.num_sites;++s)if(c.sites[s]==l) {
                rms(u.data(),z.data(),D);rms(ek.data()+s*D,proj.data(),D);
                float alpha=sigmoid(dot_f32(z.data(),proj.data(),D)/std::sqrt(float(D)));
                for(int d=0;d<D;++d)bx[d]+=alpha*ev[s*D+d];
            }
            rms(bx.data(),z.data(),D,dense(ti));activation_quant(z.data(),D,abits);
            attention_projections(ti,z.data());
            for(int h=0;h<H;++h)rms(q.data()+h*HD,q.data()+h*HD,HD,dense(ti+4));
            for(int h=0;h<KV;++h)rms(k.data()+h*HD,k.data()+h*HD,HD,dense(ti+5));
            for(int j=0;j<HD/2;++j) {
                float co=rope_cos[j],si=rope_sin[j];
                for(int h=0;h<H;++h) {float a=q[h*HD+j],b=q[h*HD+j+HD/2];q[h*HD+j]=a*co-b*si;q[h*HD+j+HD/2]=b*co+a*si;}
                for(int h=0;h<KV;++h){float a=k[h*HD+j],b=k[h*HD+j+HD/2];k[h*HD+j]=a*co-b*si;k[h*HD+j+HD/2]=b*co+a*si;}
            }
            auto *kc=keys.data()+size_t(l)*capacity*K,*vc=values.data()+size_t(l)*capacity*K;
            std::copy(k.begin(),k.end(),kc+cache_slot(position)*K);std::copy(v.begin(),v.end(),vc+cache_slot(position)*K);
            int prefix_count=c.window?std::min(prefix,position+1):0;
            int start=c.window?std::max(prefix,position-c.window+1):0;
            int length=prefix_count+std::max(0,position-start+1);
            for(int h=0;h<H;) {
                int kh=h/(H/KV),count=(h+1<H && (h+1)/(H/KV)==kh)?2:1;
                float max0=-INFINITY,max1=-INFINITY,scale=1/std::sqrt(float(HD));
                auto*s0=scores.data();auto*s1=scores.data()+capacity;
                for(int s=0;s<length;++s) {
                    int pos=cache_slot(s<prefix_count?s:start+s-prefix_count);
                    if(count==2)dot_pair(q.data()+h*HD,q.data()+(h+1)*HD,kc+pos*K+kh*HD,HD,s0[s],s1[s]);
                    else s0[s]=dot_f32(q.data()+h*HD,kc+pos*K+kh*HD,HD);
                    s0[s]*=scale;max0=std::max(max0,s0[s]);
                    if(count==2){s1[s]*=scale;max1=std::max(max1,s1[s]);}
                }
                softmax_inplace(s0,length,max0);if(count==2)softmax_inplace(s1,length,max1);
                auto *dst=att.data()+h*HD,*dst1=dst+HD;std::fill(dst,dst+count*HD,0);
                for(int s=0;s<length;++s) {
                    const auto *src=vc+cache_slot(s<prefix_count?s:start+s-prefix_count)*K+kh*HD;float prob=s0[s],prob1=count==2?s1[s]:0;
                    int d=0;
#ifdef __aarch64__
                    for(;d+4<=HD;d+=4){auto value=vld1q_f32(src+d);vst1q_f32(dst+d,vfmaq_n_f32(vld1q_f32(dst+d),value,prob));if(count==2)vst1q_f32(dst1+d,vfmaq_n_f32(vld1q_f32(dst1+d),value,prob1));}
#endif
                    for(;d<HD;++d){dst[d]+=prob*src[d];if(count==2)dst1[d]+=prob1*src[d];}
                }
                h+=count;
            }
            sigmoid_gate(att.data(),gate.data(),A);
            activation_quant(att.data(),A,abits);linear(ti+7,att.data(),proj.data());
            rms(proj.data(),proj.data(),D,dense(ti+8));float ag=sigmoid(dense(ti+9)[0]);
            for(int d=0;d<D;++d)bx[d]+=ag*proj[d];
            rms(bx.data(),z.data(),D,dense(ti+10));
            for(int d=0;d<c.hada;++d)mlp[d]=d<D?z[d]*dense(ti+11)[d]:0;
            hadamard(mlp.data(),c.hada);
            silu_diagonal(mlp.data(),dense(ti+12),c.hada);
            hadamard(mlp.data(),c.hada);
            for(int d=0;d<D;++d)bx[d]=bx[d]+dense(ti+13)[d]*mlp[d]-u[d];
            for(int ij=0;ij<N*N;++ij)hres[ij]=hres[ij]*dense(mh+2)[l]+dense(mh+5)[l*N*N+ij];
            sinkhorn(hres.data(),N);
            for(int n=0;n<N;++n)for(int d=0;d<D;++d){float val=hpost[n]*bx[d];for(int j=0;j<N;++j)val+=hres[n*N+j]*x[j*D+d];newx[n*D+d]=val;}
            x.swap(newx);
            for(int d=0;d<D;++d){float sum=0;for(int n=0;n<N;++n)sum+=x[n*D+d];hidden[l*D+d]=sum/N;}
        }
        std::copy(hidden.end()-D,hidden.end(),z.begin());
        rms(z.data(),z.data(),D,dense(1+14*c.layers+9+4*c.num_sites));
        if(out){activation_quant(z.data(),D,abits);linear(0,z.data(),out);}
        if(hidden_out)std::copy(hidden.begin(),hidden.end(),hidden_out);
        ++position;
    }
};
static thread_local std::string engine_error;
extern "C" {
void *needle2_engine_create(const EngineConfig*c,const TensorDesc*t,int count,int threads,int abits) {
    try{return new Engine(*c,t,count,threads,abits);}catch(const std::exception&e){engine_error=e.what();return nullptr;}
}
void needle2_engine_destroy(void*p){delete static_cast<Engine*>(p);}
void needle2_engine_set_lookup(void*p,int mode){static_cast<Engine*>(p)->projection_lookup=mode;}
int needle2_engine_set_sdot(void*p) {
    try {if(static_cast<Engine*>(p)->configure_sdot())return 0;engine_error="SDOT requires an ARM64 CPU with DotProd support";return -1;}
    catch(const std::exception&e){engine_error=e.what();return -1;}
}
void needle2_engine_reset(void*p,int prefix){static_cast<Engine*>(p)->reset(prefix);}
int needle2_engine_cache_prefix(void*p) {
    try{static_cast<Engine*>(p)->cache_prefix();return 0;}
    catch(const std::exception&e){engine_error=e.what();return -1;}
}
int needle2_engine_reset_to_prefix(void*p) {
    try{return static_cast<Engine*>(p)->reset_to_prefix();}
    catch(const std::exception&e){engine_error=e.what();return -1;}
}
int needle2_engine_step(void*p,int token,float*logits,float*hidden){
    try{static_cast<Engine*>(p)->step(token,logits,hidden);return 0;}catch(const std::exception&e){engine_error=e.what();return -1;}
}
int needle2_engine_import(void*p,int pos,int prefix,const int*ids,const int*positions,int count,const float*k,const float*v) {
    try{static_cast<Engine*>(p)->import_state(pos,prefix,ids,positions,count,k,v);return 0;}
    catch(const std::exception&e){engine_error=e.what();return -1;}
}
const char*needle2_engine_error(){return engine_error.c_str();}
}
