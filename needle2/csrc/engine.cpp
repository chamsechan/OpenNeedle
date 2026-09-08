// Included by cq.cpp: single-stream Needle 2 forward, using the public architecture.
#include <memory>
#include <stdexcept>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <chrono>

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
static inline float dot_i8_f32(const float* q, const int8_t* k, int n) {
    int d = 0;
#ifdef __aarch64__
    float32x4_t s0 = vdupq_n_f32(0), s1 = vdupq_n_f32(0);
    for (; d + 8 <= n; d += 8) {
        int8x8_t k8 = vld1_s8(k + d);
        int16x8_t k16 = vmovl_s8(k8);
        float32x4_t kf0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(k16)));
        float32x4_t kf1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(k16)));
        s0 = vfmaq_f32(s0, vld1q_f32(q + d), kf0);
        s1 = vfmaq_f32(s1, vld1q_f32(q + d + 4), kf1);
    }
    float sum = vaddvq_f32(vaddq_f32(s0, s1));
#else
    float sum = 0;
#endif
    for (; d < n; ++d) sum += q[d] * float(k[d]);
    return sum;
}
static inline void dot_pair_i8_f32(const float* q0, const float* q1, const int8_t* k, int n, float& out0, float& out1) {
    int d = 0;
#ifdef __aarch64__
    float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0), b0 = vdupq_n_f32(0), b1 = vdupq_n_f32(0);
    for (; d + 8 <= n; d += 8) {
        int8x8_t k8 = vld1_s8(k + d);
        int16x8_t k16 = vmovl_s8(k8);
        float32x4_t kf0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(k16)));
        float32x4_t kf1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(k16)));
        a0 = vfmaq_f32(a0, vld1q_f32(q0 + d), kf0);
        a1 = vfmaq_f32(a1, vld1q_f32(q0 + d + 4), kf1);
        b0 = vfmaq_f32(b0, vld1q_f32(q1 + d), kf0);
        b1 = vfmaq_f32(b1, vld1q_f32(q1 + d + 4), kf1);
    }
    out0 = vaddvq_f32(vaddq_f32(a0, a1));
    out1 = vaddvq_f32(vaddq_f32(b0, b1));
#else
    out0 = out1 = 0;
#endif
    for (; d < n; ++d) {
        float kv = float(k[d]);
        out0 += q0[d] * kv;
        out1 += q1[d] * kv;
    }
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
    std::vector<int> history_ring;
    std::vector<float> keys,values,engvalues;
    bool int8_kv=false;
    std::vector<int8_t> keys_i8, values_i8;
    std::vector<float> k_scales, v_scales;
};

class PersistentThreadPool {
    struct alignas(64) WorkerState {
        int start{0};
        int end{0};
        std::atomic<uint64_t> done{0};
    };
    int num_threads{1};
    std::vector<std::thread> workers;
    std::vector<WorkerState> states;
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> current_task{0};
    std::atomic<int> sleeping_workers{0};
    std::mutex cv_mutex;
    std::condition_variable cv;

    void (*task_fn)(void*, int, int, int){nullptr};
    void* task_ctx{nullptr};

    void worker_loop(int tid) {
        uint64_t my_task = 1;
        int spin_count = 0;
        const int max_spins = 200000;
        while (!stop.load(std::memory_order_relaxed)) {
            if (current_task.load(std::memory_order_acquire) >= my_task) {
                task_fn(task_ctx, tid, states[tid].start, states[tid].end);
                states[tid].done.store(my_task, std::memory_order_release);
                my_task++;
                spin_count = 0;
            } else {
                if (spin_count < max_spins) {
                    spin_count++;
#if defined(__aarch64__)
                    asm volatile("yield" ::: "memory");
#elif defined(__x86_64__) || defined(_M_X64)
                    _mm_pause();
#else
                    std::this_thread::yield();
#endif
                } else {
                    std::unique_lock<std::mutex> lock(cv_mutex);
                    if (current_task.load(std::memory_order_acquire) >= my_task) {
                        spin_count = 0;
                        continue;
                    }
                    sleeping_workers.fetch_add(1, std::memory_order_acq_rel);
                    cv.wait(lock, [&] {
                        return stop.load(std::memory_order_relaxed) ||
                               current_task.load(std::memory_order_acquire) >= my_task;
                    });
                    sleeping_workers.fetch_sub(1, std::memory_order_acq_rel);
                    spin_count = 0;
                }
            }
        }
    }

public:
    explicit PersistentThreadPool(int th) : num_threads(th), states(th) {
        for (int i = 1; i < th; ++i) {
            workers.emplace_back(&PersistentThreadPool::worker_loop, this, i);
        }
    }

    ~PersistentThreadPool() {
        stop.store(true, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lock(cv_mutex);
            cv.notify_all();
        }
        for (auto& w : workers) {
            if (w.joinable()) w.join();
        }
    }

    int size() const { return num_threads; }

    template <typename F>
    void parallel_for(int total, const F& f) {
        if (num_threads <= 1 || total <= 1) {
            f(0, 0, total);
            return;
        }
        for (int i = 0; i < num_threads; ++i) {
            states[i].start = i * total / num_threads;
            states[i].end = (i + 1) * total / num_threads;
        }
        auto wrapper = [](void* ctx, int tid, int start, int end) {
            (*static_cast<const F*>(ctx))(tid, start, end);
        };
        task_fn = wrapper;
        task_ctx = const_cast<void*>(static_cast<const void*>(&f));

        uint64_t target = current_task.load(std::memory_order_relaxed) + 1;
        current_task.store(target, std::memory_order_release);
        if (sleeping_workers.load(std::memory_order_acquire) > 0) {
            std::lock_guard<std::mutex> lock(cv_mutex);
            cv.notify_all();
        }

        // Master executes slice 0
        f(0, states[0].start, states[0].end);

        // Wait for workers
        for (int i = 1; i < num_threads; ++i) {
            int spins = 0;
            while (states[i].done.load(std::memory_order_acquire) < target) {
                if (++spins < 100000) {
#if defined(__aarch64__)
                    asm volatile("yield" ::: "memory");
#elif defined(__x86_64__) || defined(_M_X64)
                    _mm_pause();
#else
                    std::this_thread::yield();
#endif
                } else {
                    {
                        std::lock_guard<std::mutex> lock(cv_mutex);
                        cv.notify_all();
                    }
                    std::this_thread::yield();
                }
            }
        }
    }
};

struct Engine {
    EngineConfig c;
    std::unique_ptr<PersistentThreadPool> pool;
    template <typename F>
    void parallel_for(int total, const F& f) {
        if (pool && total > 1) {
            pool->parallel_for(total, f);
        } else {
            f(0, 0, total);
        }
    }
    std::vector<float> activation_lut;
    bool sdot_enabled=false;
    std::unique_ptr<PrefixSnapshot> prefix_snapshot;
    std::vector<std::unique_ptr<SdotCQ>> sdot;
    Prepared sdot_input;
    std::vector<TensorDesc> t;
    int threads,abits,position=0,capacity,engring,prefix=0,projection_lookup=-1;
    static constexpr int history_cap = 256;
    std::vector<int> history_ring;
    int history_token(int p) const {
        return history_ring[((p % history_cap) + history_cap) % history_cap];
    }
    bool int8_kv_enabled=false;
    std::vector<int8_t> keys_i8, values_i8;
    std::vector<float> k_scales, v_scales;
    std::vector<float> thread_scores;
    std::vector<float> keys,values,engvalues;
    std::vector<float> x,nx,u,bx,z,q,k,v,gate,att,proj,mlp,newx,hpre,hpost,hres;
    std::vector<float> ek,ev,e,rawv,logits,hidden,rot,rope_cos,rope_sin,rope_divisor;
    Engine(const EngineConfig& cfg,const TensorDesc *td,int count,int th,int ab):c(cfg),t(td,td+count),threads(th),abits(ab) {
        if(threads > 1) pool = std::make_unique<PersistentThreadPool>(threads);
        int D=c.dim,N=c.lanes,A=c.heads*c.head_dim,K=c.kvheads*c.head_dim;
        sdot.resize(t.size());
        capacity=c.window?c.window:c.max_seq;engring=(c.taps-1)*c.dilation+1;
        keys.resize(size_t(c.layers)*capacity*K);values.resize(keys.size());
        engvalues.resize(size_t(c.num_sites)*engring*D);
        x.resize(N*D);nx.resize(N*D);u.resize(D);bx.resize(D);z.resize(D);
        q.resize(A);k.resize(K);v.resize(K);gate.resize(A);att.resize(A);proj.resize(D);mlp.resize(c.hada);
        newx.resize(N*D);hpre.resize(N);hpost.resize(N);hres.resize(N*N);
        ek.resize(c.num_sites*D);ev.resize(c.num_sites*D);e.resize(D);rawv.resize(D);
        logits.resize(c.vocab);hidden.resize(c.layers*D);
        thread_scores.resize(size_t(std::max(1, threads))*2*capacity);
        int rotation_size=std::max({c.hada,N*D,A});
        for(const auto &td:t)if(td.cq)rotation_size=std::max(rotation_size,static_cast<CQ*>(td.cq)->padded);
        rot.resize(rotation_size);
        rope_cos.resize(c.head_dim/2);rope_sin.resize(c.head_dim/2);rope_divisor.resize(c.head_dim/2);
        for(int j=0;j<c.head_dim/2;++j)rope_divisor[j]=std::pow(c.rope_theta,float(2*j)/c.head_dim);
        history_ring.resize(history_cap, 0);
    }
    void set_int8_kv(bool en) {
        int8_kv_enabled = en;
        reset(prefix);
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
        position=0;std::fill(history_ring.begin(),history_ring.end(),0);prefix=prefix_len;
        capacity=c.window?c.window+prefix:c.max_seq;
        size_t K=size_t(c.kvheads)*c.head_dim;
        keys.resize(size_t(c.layers)*capacity*K);values.resize(keys.size());
        if(int8_kv_enabled) {
            keys_i8.resize(size_t(c.layers)*capacity*K);values_i8.resize(keys_i8.size());
            k_scales.resize(size_t(c.layers)*capacity*c.kvheads);v_scales.resize(k_scales.size());
        }
        thread_scores.resize(size_t(std::max(1, threads))*2*capacity);
    }
    void cache_prefix() {
        if(position<=0||position!=prefix)throw std::runtime_error("prefix snapshot requires position == prefix_len > 0");
        auto snapshot=std::make_unique<PrefixSnapshot>();snapshot->position=position;
        snapshot->history_ring=history_ring;snapshot->engvalues=engvalues;
        size_t K=size_t(c.kvheads)*c.head_dim;
        snapshot->keys.resize(size_t(c.layers)*prefix*K);snapshot->values.resize(snapshot->keys.size());
        for(int l=0;l<c.layers;++l) {
            const auto *ks=keys.data()+size_t(l)*capacity*K,*vs=values.data()+size_t(l)*capacity*K;
            std::copy(ks,ks+size_t(prefix)*K,snapshot->keys.data()+size_t(l)*prefix*K);
            std::copy(vs,vs+size_t(prefix)*K,snapshot->values.data()+size_t(l)*prefix*K);
        }
        if(int8_kv_enabled) {
            snapshot->int8_kv=true;
            snapshot->keys_i8.resize(size_t(c.layers)*prefix*K);snapshot->values_i8.resize(snapshot->keys_i8.size());
            snapshot->k_scales.resize(size_t(c.layers)*prefix*c.kvheads);snapshot->v_scales.resize(snapshot->k_scales.size());
            for(int l=0;l<c.layers;++l) {
                std::copy(keys_i8.data()+size_t(l)*capacity*K,keys_i8.data()+size_t(l)*capacity*K+size_t(prefix)*K,snapshot->keys_i8.data()+size_t(l)*prefix*K);
                std::copy(values_i8.data()+size_t(l)*capacity*K,values_i8.data()+size_t(l)*capacity*K+size_t(prefix)*K,snapshot->values_i8.data()+size_t(l)*prefix*K);
                std::copy(k_scales.data()+size_t(l)*capacity*c.kvheads,k_scales.data()+size_t(l)*capacity*c.kvheads+size_t(prefix)*c.kvheads,snapshot->k_scales.data()+size_t(l)*prefix*c.kvheads);
                std::copy(v_scales.data()+size_t(l)*capacity*c.kvheads,v_scales.data()+size_t(l)*capacity*c.kvheads+size_t(prefix)*c.kvheads,snapshot->v_scales.data()+size_t(l)*prefix*c.kvheads);
            }
        }
        prefix_snapshot=std::move(snapshot);
    }
    int reset_to_prefix() {
        if(!prefix_snapshot)throw std::runtime_error("no cached prefix; call cache_prefix() after consuming the complete prefix");
        int cached=prefix_snapshot->position;
        if(prefix!=cached||capacity!=(c.window?c.window+cached:c.max_seq))throw std::runtime_error("cached prefix geometry no longer matches the current state");
        history_ring=prefix_snapshot->history_ring;engvalues=prefix_snapshot->engvalues;
        size_t K=size_t(c.kvheads)*c.head_dim;
        for(int l=0;l<c.layers;++l) {
            const auto *ks=prefix_snapshot->keys.data()+size_t(l)*cached*K,*vs=prefix_snapshot->values.data()+size_t(l)*cached*K;
            std::copy(ks,ks+size_t(cached)*K,keys.data()+size_t(l)*capacity*K);
            std::copy(vs,vs+size_t(cached)*K,values.data()+size_t(l)*capacity*K);
        }
        if(prefix_snapshot->int8_kv) {
            for(int l=0;l<c.layers;++l) {
                std::copy(prefix_snapshot->keys_i8.data()+size_t(l)*cached*K,prefix_snapshot->keys_i8.data()+size_t(l)*cached*K+size_t(cached)*K,keys_i8.data()+size_t(l)*capacity*K);
                std::copy(prefix_snapshot->values_i8.data()+size_t(l)*cached*K,prefix_snapshot->values_i8.data()+size_t(l)*cached*K+size_t(cached)*K,values_i8.data()+size_t(l)*capacity*K);
                std::copy(prefix_snapshot->k_scales.data()+size_t(l)*cached*c.kvheads,prefix_snapshot->k_scales.data()+size_t(l)*cached*c.kvheads+size_t(cached)*c.kvheads,k_scales.data()+size_t(l)*capacity*c.kvheads);
                std::copy(prefix_snapshot->v_scales.data()+size_t(l)*cached*c.kvheads,prefix_snapshot->v_scales.data()+size_t(l)*cached*c.kvheads+size_t(cached)*c.kvheads,v_scales.data()+size_t(l)*capacity*c.kvheads);
            }
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
            if (rows >= 128) {
                parallel_for(n4, [&](int tid, int start, int end) {
                    for(int b=start;b<end;++b)q->row4(first+b*4,sdot_input,out+b*4);
                });
            } else {
                for(int b=0;b<n4;++b)q->row4(first+b*4,sdot_input,out+b*4);
            }
            for(int r=n4*4;r<rows;++r)out[r]=q->row(first+r,sdot_input);
        } else if(w.cq) {
            auto *cq=static_cast<CQ*>(w.cq);
            cq->transform(in,rot.data());
            if (rows >= 128) {
                parallel_for(rows, [&](int tid, int start, int end) {
                    for(int r=start;r<end;++r)out[r]=cq->dot_row(first+r,rot.data());
                });
            } else {
                for(int r=0;r<rows;++r)out[r]=cq->dot_row(first+r,rot.data());
            }
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
            parallel_for(total4, [&](int tid, int start4, int end4) {
                int begin=start4*4,end=end4*4,offset=0;
                for(int m=0;m<4;++m){
                    int b=std::max(0,begin-offset),e=std::min(matrices[m]->rows,end-offset);
                    offset+=matrices[m]->rows;
                    for(int r=b;r<e-3;r+=4)matrices[m]->row4(r,sdot_input,outputs[m]+r);
                    for(int r=std::max(b,(e/4)*4);r<e;++r)outputs[m][r]=matrices[m]->row(r,sdot_input);
                }
            });
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
        if (lookup == 0 && pool) {
            matrices[0]->transform(input, rot.data());
            int total = 0;
            for (auto* m : matrices) total += m->out;
            parallel_for(total, [&](int tid, int begin, int end) {
                int offset = 0;
                for (int m = 0; m < 4; ++m) {
                    int b = std::max(0, begin - offset), e = std::min(matrices[m]->out, end - offset);
                    offset += matrices[m]->out;
                    for (int r = b; r < e; ++r) outputs[m][r] = matrices[m]->dot_row(r, rot.data());
                }
            });
        } else {
            cq_linear_many(matrices,4,input,outputs,threads,lookup,rot.data(),activation_lut.data());
        }
    }
    void linear_batch(int ti, const float* in, float* out, int batch, int first = 0, int rows = -1) {
        const auto &w = t[ti];
        if (rows < 0) rows = w.rows;
        if (!w.cq) {
            int in_dim = w.cols;
            parallel_for(rows, [&](int tid, int start, int end) {
                for (int r = start; r < end; ++r) {
                    const float* wt = w.data + size_t(first + r) * in_dim;
                    for (int b = 0; b < batch; ++b) {
                        out[size_t(b) * rows + r] = dot_f32(wt, in + size_t(b) * in_dim, in_dim);
                    }
                }
            });
            return;
        }
        if (sdot_enabled && sdot[ti]) {
            auto* q = sdot[ti].get();
            std::vector<Prepared> prep(batch);
            for (int b = 0; b < batch; ++b) q->prepare(in + size_t(b) * q->columns, prep[b]);
            parallel_for(rows, [&](int tid, int start, int end) {
                for (int r = start; r < end; ++r) {
                    for (int b = 0; b < batch; ++b) {
                        out[size_t(b) * rows + r] = q->row(first + r, prep[b]);
                    }
                }
            });
            return;
        }
        auto* q = static_cast<CQ*>(w.cq);
        std::vector<float> rot_batch(size_t(batch) * q->padded);
        for (int b = 0; b < batch; ++b) q->transform(in + size_t(b) * q->in, rot_batch.data() + size_t(b) * q->padded);
        parallel_for(rows, [&](int tid, int start, int end) {
            for (int r = start; r < end; ++r) {
                int row_idx = first + r;
                const auto* p = q->packed + size_t(row_idx) * q->rowbytes;
                const auto* s = q->norms + size_t(row_idx) * q->groups;
                int b = 0;
                for (; b + 2 <= batch; b += 2) {
                    const float* rot0 = rot_batch.data() + size_t(b) * q->padded;
                    const float* rot1 = rot_batch.data() + size_t(b + 1) * q->padded;
                    float sum0 = 0, sum1 = 0;
                    for (int g = 0; g < q->groups; ++g) {
                        float norm = half_float(s[g]);
                        float g0 = 0, g1 = 0;
                        q->group_dot_pair(p + g * q->group * q->storage_bits / 8, rot0 + g * q->group, rot1 + g * q->group, g0, g1);
                        sum0 += g0 * norm;
                        sum1 += g1 * norm;
                    }
                    out[size_t(b) * rows + r] = sum0;
                    out[size_t(b + 1) * rows + r] = sum1;
                }
                for (; b < batch; ++b) {
                    const float* rot_b = rot_batch.data() + size_t(b) * q->padded;
                    float sum = 0;
                    for (int g = 0; g < q->groups; ++g) {
                        sum += q->group_dot(p + g * q->group * q->storage_bits / 8, rot_b + g * q->group) * half_float(s[g]);
                    }
                    out[size_t(b) * rows + r] = sum;
                }
            }
        });
    }
    void attention_projections_batch(int ti, const float* in, float* q_out, float* k_out, float* v_out, float* gate_out, int batch) {
        if (sdot_enabled) {
            SdotCQ* matrices[4] = {sdot[ti+1].get(), sdot[ti+2].get(), sdot[ti+3].get(), sdot[ti+6].get()};
            bool compatible = matrices[0] != nullptr;
            for (int i = 1; i < 4 && compatible; ++i) {
                compatible = matrices[i] && matrices[i]->bits == matrices[0]->bits && matrices[i]->group == matrices[0]->group && matrices[i]->columns == matrices[0]->columns;
            }
            if (!compatible) {
                linear_batch(ti+1, in, q_out, batch);
                linear_batch(ti+2, in, k_out, batch);
                linear_batch(ti+3, in, v_out, batch);
                linear_batch(ti+6, in, gate_out, batch);
                return;
            }
            std::vector<Prepared> prep(batch);
            for (int b = 0; b < batch; ++b) matrices[0]->prepare(in + size_t(b) * matrices[0]->columns, prep[b]);
            int total = 0;
            for (auto* m : matrices) total += m->rows;
            float* outputs[4] = {q_out, k_out, v_out, gate_out};
            parallel_for(total, [&](int tid, int start, int end) {
                for (int i = start; i < end; ++i) {
                    int m = 0, r = i;
                    while (m < 4 && r >= matrices[m]->rows) { r -= matrices[m]->rows; m++; }
                    for (int b = 0; b < batch; ++b) {
                        outputs[m][size_t(b) * matrices[m]->rows + r] = matrices[m]->row(r, prep[b]);
                    }
                }
            });
            return;
        }
        CQ* matrices[4] = {static_cast<CQ*>(t[ti+1].cq), static_cast<CQ*>(t[ti+2].cq), static_cast<CQ*>(t[ti+3].cq), static_cast<CQ*>(t[ti+6].cq)};
        if (!matrices[0] || !matrices[1] || !matrices[2] || !matrices[3] || !compatible_rotation(matrices, 4)) {
            linear_batch(ti+1, in, q_out, batch);
            linear_batch(ti+2, in, k_out, batch);
            linear_batch(ti+3, in, v_out, batch);
            linear_batch(ti+6, in, gate_out, batch);
            return;
        }
        int total = 0;
        for (int m = 0; m < 4; ++m) total += matrices[m]->out;
        float* outputs[4] = {q_out, k_out, v_out, gate_out};
        std::vector<float> rot_batch(size_t(batch) * matrices[0]->padded);
        for (int b = 0; b < batch; ++b) {
            matrices[0]->transform(in + size_t(b) * matrices[0]->in, rot_batch.data() + size_t(b) * matrices[0]->padded);
        }
        parallel_for(total, [&](int tid, int start, int end) {
            for (int i = start; i < end; ++i) {
                int m = 0, r = i;
                while (m < 4 && r >= matrices[m]->out) { r -= matrices[m]->out; m++; }
                CQ* q = matrices[m];
                const auto* p = q->packed + size_t(r) * q->rowbytes;
                const auto* s = q->norms + size_t(r) * q->groups;
                int b = 0;
                for (; b + 2 <= batch; b += 2) {
                    const float* rot0 = rot_batch.data() + size_t(b) * q->padded;
                    const float* rot1 = rot_batch.data() + size_t(b + 1) * q->padded;
                    float sum0 = 0, sum1 = 0;
                    for (int g = 0; g < q->groups; ++g) {
                        float norm = half_float(s[g]);
                        float g0 = 0, g1 = 0;
                        q->group_dot_pair(p + g * q->group * q->storage_bits / 8, rot0 + g * q->group, rot1 + g * q->group, g0, g1);
                        sum0 += g0 * norm;
                        sum1 += g1 * norm;
                    }
                    outputs[m][size_t(b) * q->out + r] = sum0;
                    outputs[m][size_t(b + 1) * q->out + r] = sum1;
                }
                for (; b < batch; ++b) {
                    const float* rot_b = rot_batch.data() + size_t(b) * q->padded;
                    float sum = 0;
                    for (int g = 0; g < q->groups; ++g) {
                        sum += q->group_dot(p + g * q->group * q->storage_bits / 8, rot_b + g * q->group) * half_float(s[g]);
                    }
                    outputs[m][size_t(b) * q->out + r] = sum;
                }
            }
        });
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
                for(int j=0;j<order;++j)acc=(acc^uint32_t(history_token(position-j)))*uint32_t(0x01000193);
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
        reset(pinned);position=pos;
        for(int i=std::max(0,pos-history_cap);i<pos;++i)history_ring[i%history_cap]=ids[i];
        int K=c.kvheads*c.head_dim,KV=c.kvheads,HD=c.head_dim;
        for(int l=0;l<c.layers;++l)for(int t=0;t<count;++t) {
            int slot=cache_slot(positions[t]);
            const auto *ks=k+(size_t(l)*count+t)*K,*vs=v+(size_t(l)*count+t)*K;
            if(!int8_kv_enabled) {
                std::copy(ks,ks+K,keys.data()+(size_t(l)*capacity+slot)*K);
                std::copy(vs,vs+K,values.data()+(size_t(l)*capacity+slot)*K);
            } else {
                for(int kh=0;kh<KV;++kh) {
                    float max_k=0,max_v=0;
                    const float*kp=ks+kh*HD,*vp=vs+kh*HD;
                    for(int d=0;d<HD;++d){max_k=std::max(max_k,std::abs(kp[d]));max_v=std::max(max_v,std::abs(vp[d]));}
                    float scale_k=max_k>0?max_k/127.0f:1.0f,scale_v=max_v>0?max_v/127.0f:1.0f;
                    float inv_k=max_k>0?127.0f/max_k:0.0f,inv_v=max_v>0?127.0f/max_v:0.0f;
                    k_scales[(size_t(l)*capacity+slot)*KV+kh]=scale_k;
                    v_scales[(size_t(l)*capacity+slot)*KV+kh]=scale_v;
                    int8_t*k_dst=keys_i8.data()+(size_t(l)*capacity+slot)*K+kh*HD;
                    int8_t*v_dst=values_i8.data()+(size_t(l)*capacity+slot)*K+kh*HD;
                    for(int d=0;d<HD;++d){
                        k_dst[d]=int8_t(std::max(-127.0f,std::min(127.0f,std::nearbyint(kp[d]*inv_k))));
                        v_dst[d]=int8_t(std::max(-127.0f,std::min(127.0f,std::nearbyint(vp[d]*inv_v))));
                    }
                }
            }
        }
        // Only raw values are persistent; recompute the short Engram history.
        for(position=std::max(0,pos-engring);position<pos;++position)engrams();
        position=pos;
    }
    void compute_attention(int l, int pos_cur, const float* q_in, float* att_out) {
        int H=c.heads,KV=c.kvheads,HD=c.head_dim,K=KV*HD;
        int prefix_count=c.window?std::min(prefix,pos_cur+1):0;
        int start=c.window?std::max(prefix,pos_cur-c.window+1):0;
        int length=prefix_count+std::max(0,pos_cur-start+1);
        int num_groups=0;
        struct HeadWork {int h,kh,count;};
        HeadWork groups[64];
        for(int h=0;h<H;) {
            int kh=h/(H/KV),count=(h+1<H && (h+1)/(H/KV)==kh)?2:1;
            groups[num_groups++]={h,kh,count};
            h+=count;
        }
        parallel_for(num_groups, [&](int tid, int start_g, int end_g) {
            for(int g=start_g;g<end_g;++g) {
                int h=groups[g].h,kh=groups[g].kh,count=groups[g].count;
                float max0=-INFINITY,max1=-INFINITY,scale=1/std::sqrt(float(HD));
                auto*s0=thread_scores.data()+size_t(tid)*2*capacity;
                auto*s1=s0+capacity;
                if(!int8_kv_enabled) {
                    const auto*kc=keys.data()+size_t(l)*capacity*K;
                    for(int s=0;s<length;++s) {
                        int pos=cache_slot(s<prefix_count?s:start+s-prefix_count);
                        if(count==2)dot_pair(q_in+h*HD,q_in+(h+1)*HD,kc+pos*K+kh*HD,HD,s0[s],s1[s]);
                        else s0[s]=dot_f32(q_in+h*HD,kc+pos*K+kh*HD,HD);
                        s0[s]*=scale;max0=std::max(max0,s0[s]);
                        if(count==2){s1[s]*=scale;max1=std::max(max1,s1[s]);}
                    }
                    softmax_inplace(s0,length,max0);if(count==2)softmax_inplace(s1,length,max1);
                    auto *dst=att_out+h*HD,*dst1=dst+HD;std::fill(dst,dst+count*HD,0);
                    const auto*vc=values.data()+size_t(l)*capacity*K;
                    for(int s=0;s<length;++s) {
                        const auto *src=vc+cache_slot(s<prefix_count?s:start+s-prefix_count)*K+kh*HD;float prob=s0[s],prob1=count==2?s1[s]:0;
                        int d=0;
#ifdef __aarch64__
                        for(;d+4<=HD;d+=4){auto value=vld1q_f32(src+d);vst1q_f32(dst+d,vfmaq_n_f32(vld1q_f32(dst+d),value,prob));if(count==2)vst1q_f32(dst1+d,vfmaq_n_f32(vld1q_f32(dst1+d),value,prob1));}
#endif
                        for(;d<HD;++d){dst[d]+=prob*src[d];if(count==2)dst1[d]+=prob1*src[d];}
                    }
                } else {
                    const auto*kc_i8=keys_i8.data()+size_t(l)*capacity*K;
                    const auto*ks=k_scales.data()+size_t(l)*capacity*KV;
                    for(int s=0;s<length;++s) {
                        int pos=cache_slot(s<prefix_count?s:start+s-prefix_count);
                        float k_scale=ks[pos*KV+kh];
                        if(count==2) {
                            dot_pair_i8_f32(q_in+h*HD,q_in+(h+1)*HD,kc_i8+pos*K+kh*HD,HD,s0[s],s1[s]);
                            s0[s]*=k_scale;s1[s]*=k_scale;
                        } else {
                            s0[s]=dot_i8_f32(q_in+h*HD,kc_i8+pos*K+kh*HD,HD)*k_scale;
                        }
                        s0[s]*=scale;max0=std::max(max0,s0[s]);
                        if(count==2){s1[s]*=scale;max1=std::max(max1,s1[s]);}
                    }
                    softmax_inplace(s0,length,max0);if(count==2)softmax_inplace(s1,length,max1);
                    auto *dst=att_out+h*HD,*dst1=dst+HD;std::fill(dst,dst+count*HD,0);
                    const auto*vc_i8=values_i8.data()+size_t(l)*capacity*K;
                    const auto*vs=v_scales.data()+size_t(l)*capacity*KV;
                    for(int s=0;s<length;++s) {
                        int pos=cache_slot(s<prefix_count?s:start+s-prefix_count);
                        float v_scale=vs[pos*KV+kh];
                        const auto*src_i8=vc_i8+pos*K+kh*HD;
                        float prob=s0[s]*v_scale,prob1=(count==2?s1[s]:0)*v_scale;
                        int d=0;
#ifdef __aarch64__
                        for(;d+8<=HD;d+=8) {
                            int8x8_t v8=vld1_s8(src_i8+d);
                            int16x8_t v16=vmovl_s8(v8);
                            float32x4_t vf0=vcvtq_f32_s32(vmovl_s16(vget_low_s16(v16)));
                            float32x4_t vf1=vcvtq_f32_s32(vmovl_s16(vget_high_s16(v16)));
                            vst1q_f32(dst+d,vfmaq_n_f32(vld1q_f32(dst+d),vf0,prob));
                            vst1q_f32(dst+d+4,vfmaq_n_f32(vld1q_f32(dst+d+4),vf1,prob));
                            if(count==2){
                                vst1q_f32(dst1+d,vfmaq_n_f32(vld1q_f32(dst1+d),vf0,prob1));
                                vst1q_f32(dst1+d+4,vfmaq_n_f32(vld1q_f32(dst1+d+4),vf1,prob1));
                            }
                        }
#endif
                        for(;d<HD;++d){
                            float val=float(src_i8[d]);
                            dst[d]+=prob*val;
                            if(count==2)dst1[d]+=prob1*val;
                        }
                    }
                }
            }
        });
    }
    void step(int token,float*out,float*hidden_out) {
        int D=c.dim,N=c.lanes,H=c.heads,KV=c.kvheads,HD=c.head_dim,A=H*HD,K=KV*HD;
        if(token<0||token>=c.vocab)throw std::runtime_error("token outside vocabulary");
        if(!c.window&&position>=capacity)throw std::runtime_error("maximum context reached");
        history_ring[position%history_cap]=token;
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
            int slot=cache_slot(position);
            if(!int8_kv_enabled) {
                auto *kc=keys.data()+size_t(l)*capacity*K,*vc=values.data()+size_t(l)*capacity*K;
                std::copy(k.begin(),k.end(),kc+slot*K);std::copy(v.begin(),v.end(),vc+slot*K);
            } else {
                auto *kc_i8=keys_i8.data()+size_t(l)*capacity*K,*vc_i8=values_i8.data()+size_t(l)*capacity*K;
                auto *ks=k_scales.data()+size_t(l)*capacity*KV,*vs=v_scales.data()+size_t(l)*capacity*KV;
                for(int kh=0;kh<KV;++kh) {
                    float max_k=0,max_v=0;
                    const float*kp=k.data()+kh*HD,*vp=v.data()+kh*HD;
                    for(int d=0;d<HD;++d){max_k=std::max(max_k,std::abs(kp[d]));max_v=std::max(max_v,std::abs(vp[d]));}
                    float scale_k=max_k>0?max_k/127.0f:1.0f,scale_v=max_v>0?max_v/127.0f:1.0f;
                    float inv_k=max_k>0?127.0f/max_k:0.0f,inv_v=max_v>0?127.0f/max_v:0.0f;
                    ks[slot*KV+kh]=scale_k;vs[slot*KV+kh]=scale_v;
                    int8_t*k_dst=kc_i8+slot*K+kh*HD,*v_dst=vc_i8+slot*K+kh*HD;
                    for(int d=0;d<HD;++d){
                        k_dst[d]=int8_t(std::max(-127.0f,std::min(127.0f,std::nearbyint(kp[d]*inv_k))));
                        v_dst[d]=int8_t(std::max(-127.0f,std::min(127.0f,std::nearbyint(vp[d]*inv_v))));
                    }
                }
            }
            compute_attention(l, position, q.data(), att.data());
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
    void prefill(const int* tokens, int num_tokens, bool last_only, float* logits_out, float* hidden_out = nullptr) {
        if (num_tokens <= 0) return;
        for (int i = 0; i < num_tokens; ++i) {
            if (tokens[i] < 0 || tokens[i] >= c.vocab) throw std::runtime_error("token outside vocabulary");
        }
        if (!c.window && position + num_tokens > capacity) throw std::runtime_error("maximum context reached");

        int D = c.dim, N = c.lanes, H = c.heads, KV = c.kvheads, HD = c.head_dim, A = H * HD, K = KV * HD;
        int mh = 1 + 14 * c.layers;
        int chunk_size = c.window ? std::min(32, c.window) : 32;

        for (int chunk_start = 0; chunk_start < num_tokens; chunk_start += chunk_size) {
            int B = std::min(chunk_size, num_tokens - chunk_start);
            const int* chunk_tokens = tokens + chunk_start;
            int start_pos = position;

            // 1. Token history & Engrams
            std::vector<float> ek_chunk(size_t(B) * c.num_sites * D), ev_chunk(size_t(B) * c.num_sites * D);
            for (int b = 0; b < B; ++b) {
                position = start_pos + b;
                history_ring[position % history_cap] = chunk_tokens[b];
                engrams();
                if (c.num_sites) {
                    std::copy(ek.begin(), ek.end(), ek_chunk.data() + size_t(b) * c.num_sites * D);
                    std::copy(ev.begin(), ev.end(), ev_chunk.data() + size_t(b) * c.num_sites * D);
                }
            }

            // 2. Token embeddings
            std::vector<float> x_chunk(size_t(B) * N * D), newx_chunk(size_t(B) * N * D);
            for (int b = 0; b < B; ++b) {
                row(0, chunk_tokens[b], z.data());
                for (int n = 0; n < N; ++n) {
                    for (int d = 0; d < D; ++d) {
                        x_chunk[(size_t(b) * N + n) * D + d] = z[d] * std::sqrt(float(D));
                    }
                }
            }

            // Buffers for layer processing
            std::vector<float> nx_chunk(size_t(B) * N * D);
            std::vector<float> hpre_chunk(size_t(B) * N), hpost_chunk(size_t(B) * N), hres_chunk(size_t(B) * N * N);
            std::vector<float> z_chunk(size_t(B) * D), u_chunk(size_t(B) * D), bx_chunk(size_t(B) * D);
            std::vector<float> q_chunk(size_t(B) * A), k_chunk(size_t(B) * K), v_chunk(size_t(B) * K), gate_chunk(size_t(B) * A);
            std::vector<float> att_chunk(size_t(B) * A), proj_chunk(size_t(B) * D);
            std::vector<float> mlp_chunk(c.hada);

            // 3. Process layers
            for (int l = 0; l < c.layers; ++l) {
                int ti = 1 + 14 * l;

                // RMSNorm on lane representations
                for (int b = 0; b < B; ++b) {
                    rms(x_chunk.data() + size_t(b) * N * D, nx_chunk.data() + size_t(b) * N * D, N * D);
                }

                // Pre-attention routing GEMMs (weights reused across B tokens)
                linear_batch(mh + 6, nx_chunk.data(), hpre_chunk.data(), B, l * N, N);
                linear_batch(mh + 7, nx_chunk.data(), hpost_chunk.data(), B, l * N, N);
                linear_batch(mh + 8, nx_chunk.data(), hres_chunk.data(), B, l * N * N, N * N);

                for (int b = 0; b < B; ++b) {
                    float* hp = hpre_chunk.data() + b * N;
                    float* hpo = hpost_chunk.data() + b * N;
                    float* xb = x_chunk.data() + size_t(b) * N * D;
                    float* ub = u_chunk.data() + size_t(b) * D;
                    float* bxb = bx_chunk.data() + size_t(b) * D;
                    float* zb = z_chunk.data() + size_t(b) * D;

                    for (int n = 0; n < N; ++n) {
                        hp[n] = sigmoid(dense(mh)[l] * hp[n] + dense(mh + 3)[l * N + n] + (n == l % N ? 4 : -4));
                        hpo[n] = 2 * sigmoid(dense(mh + 1)[l] * hpo[n] + dense(mh + 4)[l * N + n] + (n == l % N ? 0 : -4));
                    }
                    for (int d = 0; d < D; ++d) {
                        float sum = 0;
                        for (int n = 0; n < N; ++n) sum += hp[n] * xb[n * D + d];
                        ub[d] = bxb[d] = sum;
                    }
                    for (int s = 0; s < c.num_sites; ++s) {
                        if (c.sites[s] == l) {
                            const float* eks = ek_chunk.data() + (size_t(b) * c.num_sites + s) * D;
                            const float* evs = ev_chunk.data() + (size_t(b) * c.num_sites + s) * D;
                            rms(ub, zb, D);
                            rms(eks, proj.data(), D);
                            float alpha = sigmoid(dot_f32(zb, proj.data(), D) / std::sqrt(float(D)));
                            for (int d = 0; d < D; ++d) bxb[d] += alpha * evs[d];
                        }
                    }
                    rms(bxb, zb, D, dense(ti));
                    activation_quant(zb, D, abits);
                }

                // Batched Q, K, V, Gate projections (all weights reused across B tokens)
                attention_projections_batch(ti, z_chunk.data(), q_chunk.data(), k_chunk.data(), v_chunk.data(), gate_chunk.data(), B);

                // RMS and RoPE for Q and K for all B tokens
                for (int b = 0; b < B; ++b) {
                    int pos = start_pos + b;
                    float* qb = q_chunk.data() + size_t(b) * A;
                    float* kb = k_chunk.data() + size_t(b) * K;

                    for (int h = 0; h < H; ++h) rms(qb + h * HD, qb + h * HD, HD, dense(ti + 4));
                    for (int h = 0; h < KV; ++h) rms(kb + h * HD, kb + h * HD, HD, dense(ti + 5));

                    for (int j = 0; j < HD / 2; ++j) {
                        float a = pos / rope_divisor[j];
                        float co = std::cos(a), si = std::sin(a);
                        for (int h = 0; h < H; ++h) {
                            float qa = qb[h * HD + j], qb_val = qb[h * HD + j + HD / 2];
                            qb[h * HD + j] = qa * co - qb_val * si;
                            qb[h * HD + j + HD / 2] = qb_val * co + qa * si;
                        }
                        for (int h = 0; h < KV; ++h) {
                            float ka = kb[h * HD + j], kb_val = kb[h * HD + j + HD / 2];
                            kb[h * HD + j] = ka * co - kb_val * si;
                            kb[h * HD + j + HD / 2] = kb_val * co + ka * si;
                        }
                    }
                }

                // Causal KV store & attention per token (token b computes attention before token b+1 can evict an old window slot)
                for (int b = 0; b < B; ++b) {
                    int pos = start_pos + b;
                    float* kb = k_chunk.data() + size_t(b) * K;
                    float* vb = v_chunk.data() + size_t(b) * K;
                    int slot = cache_slot(pos);
                    if (!int8_kv_enabled) {
                        auto* kc = keys.data() + (size_t(l) * capacity + slot) * K;
                        auto* vc = values.data() + (size_t(l) * capacity + slot) * K;
                        std::copy(kb, kb + K, kc);
                        std::copy(vb, vb + K, vc);
                    } else {
                        auto* kc_i8 = keys_i8.data() + (size_t(l) * capacity + slot) * K;
                        auto* vc_i8 = values_i8.data() + (size_t(l) * capacity + slot) * K;
                        auto* ks = k_scales.data() + (size_t(l) * capacity + slot) * KV;
                        auto* vs = v_scales.data() + (size_t(l) * capacity + slot) * KV;
                        for (int kh = 0; kh < KV; ++kh) {
                            float max_k = 0, max_v = 0;
                            const float* kp = kb + kh * HD, * vp = vb + kh * HD;
                            for (int d = 0; d < HD; ++d) {
                                max_k = std::max(max_k, std::abs(kp[d]));
                                max_v = std::max(max_v, std::abs(vp[d]));
                            }
                            float scale_k = max_k > 0 ? max_k / 127.0f : 1.0f;
                            float scale_v = max_v > 0 ? max_v / 127.0f : 1.0f;
                            float inv_k = max_k > 0 ? 127.0f / max_k : 0.0f;
                            float inv_v = max_v > 0 ? 127.0f / max_v : 0.0f;
                            ks[kh] = scale_k;
                            vs[kh] = scale_v;
                            int8_t* k_dst = kc_i8 + kh * HD;
                            int8_t* v_dst = vc_i8 + kh * HD;
                            for (int d = 0; d < HD; ++d) {
                                k_dst[d] = int8_t(std::max(-127.0f, std::min(127.0f, std::nearbyint(kp[d] * inv_k))));
                                v_dst[d] = int8_t(std::max(-127.0f, std::min(127.0f, std::nearbyint(vp[d] * inv_v))));
                            }
                        }
                    }
                    float* qb = q_chunk.data() + size_t(b) * A;
                    float* attb = att_chunk.data() + size_t(b) * A;
                    compute_attention(l, pos, qb, attb);
                }

                // Gating & Activation quant
                for (int b = 0; b < B; ++b) {
                    sigmoid_gate(att_chunk.data() + size_t(b) * A, gate_chunk.data() + size_t(b) * A, A);
                    activation_quant(att_chunk.data() + size_t(b) * A, A, abits);
                }

                // Out projection (weight reused across B tokens)
                linear_batch(ti + 7, att_chunk.data(), proj_chunk.data(), B);

                // Post-attention MLP and Sinkhorn routing
                for (int b = 0; b < B; ++b) {
                    float* projb = proj_chunk.data() + size_t(b) * D;
                    float* bxb = bx_chunk.data() + size_t(b) * D;
                    float* zb = z_chunk.data() + size_t(b) * D;
                    float* ub = u_chunk.data() + size_t(b) * D;
                    float* hr = hres_chunk.data() + size_t(b) * N * N;
                    float* hpo = hpost_chunk.data() + b * N;
                    float* xb = x_chunk.data() + size_t(b) * N * D;
                    float* nxb = newx_chunk.data() + size_t(b) * N * D;

                    rms(projb, projb, D, dense(ti + 8));
                    float ag = sigmoid(dense(ti + 9)[0]);
                    for (int d = 0; d < D; ++d) bxb[d] += ag * projb[d];

                    rms(bxb, zb, D, dense(ti + 10));
                    for (int d = 0; d < c.hada; ++d) mlp_chunk[d] = d < D ? zb[d] * dense(ti + 11)[d] : 0;
                    hadamard(mlp_chunk.data(), c.hada);
                    silu_diagonal(mlp_chunk.data(), dense(ti + 12), c.hada);
                    hadamard(mlp_chunk.data(), c.hada);
                    for (int d = 0; d < D; ++d) bxb[d] = bxb[d] + dense(ti + 13)[d] * mlp_chunk[d] - ub[d];

                    for (int ij = 0; ij < N * N; ++ij) hr[ij] = hr[ij] * dense(mh + 2)[l] + dense(mh + 5)[l * N * N + ij];
                    sinkhorn(hr, N);
                    for (int n = 0; n < N; ++n) {
                        for (int d = 0; d < D; ++d) {
                            float val = hpo[n] * bxb[d];
                            for (int j = 0; j < N; ++j) val += hr[n * N + j] * xb[j * D + d];
                            nxb[n * D + d] = val;
                        }
                    }
                    std::copy(nxb, nxb + N * D, xb);
                    if (b == B - 1) {
                        for (int d = 0; d < D; ++d) {
                            float sum = 0;
                            for (int n = 0; n < N; ++n) sum += xb[n * D + d];
                            hidden[l * D + d] = sum / N;
                        }
                    }
                }
            } // end layers

            // Update persistent engine state with final token of chunk
            std::copy(x_chunk.data() + size_t(B - 1) * N * D, x_chunk.data() + size_t(B) * N * D, this->x.begin());
            position = start_pos + B;

            // Final norm and LM Head
            bool is_final_chunk = (chunk_start + B == num_tokens);
            int final_norm_idx = 1 + 14 * c.layers + 9 + 4 * c.num_sites;

            if (logits_out && !last_only) {
                for (int b = 0; b < B; ++b) {
                    float* zb = z_chunk.data() + size_t(b) * D;
                    float* xb = x_chunk.data() + size_t(b) * N * D;
                    for (int d = 0; d < D; ++d) {
                        float sum = 0;
                        for (int n = 0; n < N; ++n) sum += xb[n * D + d];
                        zb[d] = sum / N;
                    }
                    rms(zb, zb, D, dense(final_norm_idx));
                    activation_quant(zb, D, abits);
                }
                linear_batch(0, z_chunk.data(), logits_out + size_t(chunk_start) * c.vocab, B);
            } else if (logits_out && last_only && is_final_chunk) {
                float* zb = z_chunk.data() + size_t(B - 1) * D;
                float* xb = x_chunk.data() + size_t(B - 1) * N * D;
                for (int d = 0; d < D; ++d) {
                    float sum = 0;
                    for (int n = 0; n < N; ++n) sum += xb[n * D + d];
                    zb[d] = sum / N;
                }
                rms(zb, zb, D, dense(final_norm_idx));
                activation_quant(zb, D, abits);
                linear(0, zb, logits_out);
            }
        } // end chunks

        if (hidden_out) {
            std::copy(hidden.begin(), hidden.end(), hidden_out);
        }
    }
    void project_candidates(const int* candidates, int num_candidates, float* candidate_logits) {
        if (!candidate_logits || num_candidates <= 0) return;
        int D = c.dim;
        activation_quant(z.data(), D, abits);
        if (sdot_enabled && sdot[0]) {
            auto* q = sdot[0].get();
            q->prepare(z.data(), sdot_input);
            if (num_candidates >= 128) {
                parallel_for(num_candidates, [&](int tid, int start, int end) {
                    for (int i = start; i < end; ++i) {
                        candidate_logits[i] = q->row(candidates[i], sdot_input);
                    }
                });
            } else {
                for (int i = 0; i < num_candidates; ++i) candidate_logits[i] = q->row(candidates[i], sdot_input);
            }
        } else if (t[0].cq) {
            auto* cq = static_cast<CQ*>(t[0].cq);
            cq->transform(z.data(), rot.data());
            if (num_candidates >= 128) {
                parallel_for(num_candidates, [&](int tid, int start, int end) {
                    for (int i = start; i < end; ++i) {
                        candidate_logits[i] = cq->dot_row(candidates[i], rot.data());
                    }
                });
            } else {
                for (int i = 0; i < num_candidates; ++i) candidate_logits[i] = cq->dot_row(candidates[i], rot.data());
            }
        } else {
            if (num_candidates >= 128) {
                parallel_for(num_candidates, [&](int tid, int start, int end) {
                    for (int i = start; i < end; ++i) {
                        candidate_logits[i] = dot_f32(t[0].data + size_t(candidates[i]) * t[0].cols, z.data(), t[0].cols);
                    }
                });
            } else {
                for (int i = 0; i < num_candidates; ++i) {
                    candidate_logits[i] = dot_f32(t[0].data + size_t(candidates[i]) * t[0].cols, z.data(), t[0].cols);
                }
            }
        }
    }
    void step_candidates(int token, const int* candidates, int num_candidates, float* candidate_logits, float* hidden_out) {
        step(token, nullptr, hidden_out);
        if (candidates && num_candidates > 0 && candidate_logits) {
            project_candidates(candidates, num_candidates, candidate_logits);
        } else if (candidate_logits && (!candidates || num_candidates < 0)) {
            activation_quant(z.data(), c.dim, abits);
            linear(0, z.data(), candidate_logits);
        }
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
int needle2_engine_prefill(void*p,const int*tokens,int num_tokens,int last_only,float*logits,float*hidden){
    try{static_cast<Engine*>(p)->prefill(tokens,num_tokens,last_only!=0,logits,hidden);return 0;}
    catch(const std::exception&e){engine_error=e.what();return -1;}
}
int needle2_engine_step_candidates(void*p,int token,const int*candidates,int num_candidates,float*candidate_logits,float*hidden){
    try{static_cast<Engine*>(p)->step_candidates(token,candidates,num_candidates,candidate_logits,hidden);return 0;}
    catch(const std::exception&e){engine_error=e.what();return -1;}
}
int needle2_engine_project_candidates(void*p,const int*candidates,int num_candidates,float*candidate_logits){
    try{static_cast<Engine*>(p)->project_candidates(candidates,num_candidates,candidate_logits);return 0;}
    catch(const std::exception&e){engine_error=e.what();return -1;}
}
int needle2_engine_import(void*p,int pos,int prefix,const int*ids,const int*positions,int count,const float*k,const float*v) {
    try{static_cast<Engine*>(p)->import_state(pos,prefix,ids,positions,count,k,v);return 0;}
    catch(const std::exception&e){engine_error=e.what();return -1;}
}
int needle2_engine_set_int8_kv(void*p,int enable) {
    try{static_cast<Engine*>(p)->set_int8_kv(enable!=0);return 0;}
    catch(const std::exception&e){engine_error=e.what();return -1;}
}
const char*needle2_engine_error(){return engine_error.c_str();}
}
