// Included by cq.cpp: single-stream Needle 2 forward, using the public architecture.
#include <memory>
#include <stdexcept>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <limits>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

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

struct DFAStateDesc {
    int num_states;
    int initial_state;
    int eos_id;
    int stop_id;
    int tool_start_id;
    int tool_end_id;
    const int* state_types;
    const int* fallback_next_states;
    const int* candidate_offsets;
    const int* candidate_tokens;
    const int* next_states;
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
    std::vector<Prepared> sdot_batch;
    std::vector<TensorDesc> t;
    int threads,abits,position=0,capacity,engring,prefix=0,projection_lookup=-1;
    int history_cap = 1;
    std::vector<int> history_ring;
    int history_token(int p) const {
        return history_ring[((p % history_cap) + history_cap) % history_cap];
    }
    bool int8_kv_enabled=false;
    std::vector<int8_t> keys_i8, values_i8;
    std::vector<float> k_scales, v_scales;
    std::vector<float> thread_scores;
    struct HeadWork { int h, kh, count; };
    std::vector<HeadWork> head_work;
    std::vector<int> attention_slots;
    int attention_position = -1;
    std::vector<float> keys,values,engvalues;
    std::vector<float> x,nx,u,bx,z,q,k,v,gate,att,proj,mlp,newx,hpre,hpost,hres;
    std::vector<float> ek,ev,e,rawv,logits,hidden,rot,rope_cos,rope_sin,rope_divisor;
    Engine(const EngineConfig& cfg,const TensorDesc *td,int count,int th,int ab):c(cfg),t(td,td+count),threads(th),abits(ab) {
        if(threads > 1) pool = std::make_unique<PersistentThreadPool>(threads);
        int D=c.dim,N=c.lanes,A=c.heads*c.head_dim,K=c.kvheads*c.head_dim;
        sdot.resize(t.size());
        for(int h=0;h<c.heads;) {
            int kh=h/(c.heads/c.kvheads);
            int count=(h+1<c.heads && (h+1)/(c.heads/c.kvheads)==kh)?2:1;
            head_work.push_back({h,kh,count});h+=count;
        }
        capacity=c.window?c.window:c.max_seq;
        int64_t ring_size=int64_t(c.taps-1)*c.dilation+1;
        if(ring_size<1||ring_size>std::numeric_limits<int>::max())throw std::runtime_error("invalid Engram ring size");
        engring=int(ring_size);
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
        int max_order = 1;
        for(int i=0;i<c.num_orders;++i)max_order=std::max(max_order,c.orders[i]);
        // import_state rebuilds engring raw values, including their n-grams.
        int64_t needed=int64_t(engring)+max_order-1;
        if(needed>std::numeric_limits<int>::max())throw std::runtime_error("Engram history too large");
        history_cap=c.num_sites?int(needed):1;
        history_ring.resize(history_cap, 0);
    }
    size_t kv_storage_bytes() const {
        size_t bytes=(keys.capacity()+values.capacity()+k_scales.capacity()+v_scales.capacity())*sizeof(float)
            +keys_i8.capacity()+values_i8.capacity();
        if(prefix_snapshot) {
            const auto&s=*prefix_snapshot;
            bytes+=(s.keys.capacity()+s.values.capacity()+s.k_scales.capacity()+s.v_scales.capacity())*sizeof(float)
                +s.keys_i8.capacity()+s.values_i8.capacity();
        }
        return bytes;
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
        attention_position=-1;
        prefix_snapshot.reset();
        position=0;std::fill(history_ring.begin(),history_ring.end(),0);prefix=prefix_len;
        capacity=c.window?c.window+prefix:c.max_seq;
        size_t K=size_t(c.kvheads)*c.head_dim;
        if(int8_kv_enabled) {
            std::vector<float>().swap(keys);std::vector<float>().swap(values);
            keys_i8.resize(size_t(c.layers)*capacity*K);values_i8.resize(keys_i8.size());
            k_scales.resize(size_t(c.layers)*capacity*c.kvheads);v_scales.resize(k_scales.size());
        } else {
            std::vector<int8_t>().swap(keys_i8);std::vector<int8_t>().swap(values_i8);
            std::vector<float>().swap(k_scales);std::vector<float>().swap(v_scales);
            keys.resize(size_t(c.layers)*capacity*K);values.resize(keys.size());
        }
        thread_scores.resize(size_t(std::max(1, threads))*2*capacity);
    }
    void cache_prefix() {
        if(position<=0||position!=prefix)throw std::runtime_error("prefix snapshot requires position == prefix_len > 0");
        auto snapshot=std::make_unique<PrefixSnapshot>();snapshot->position=position;
        snapshot->history_ring=history_ring;snapshot->engvalues=engvalues;
        size_t K=size_t(c.kvheads)*c.head_dim;
        if(!int8_kv_enabled) {
            snapshot->keys.resize(size_t(c.layers)*prefix*K);snapshot->values.resize(snapshot->keys.size());
            for(int l=0;l<c.layers;++l) {
                const auto *ks=keys.data()+size_t(l)*capacity*K,*vs=values.data()+size_t(l)*capacity*K;
                std::copy(ks,ks+size_t(prefix)*K,snapshot->keys.data()+size_t(l)*prefix*K);
                std::copy(vs,vs+size_t(prefix)*K,snapshot->values.data()+size_t(l)*prefix*K);
            }
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
        if(!prefix_snapshot->int8_kv) {
            for(int l=0;l<c.layers;++l) {
                const auto *ks=prefix_snapshot->keys.data()+size_t(l)*cached*K,*vs=prefix_snapshot->values.data()+size_t(l)*cached*K;
                std::copy(ks,ks+size_t(cached)*K,keys.data()+size_t(l)*capacity*K);
                std::copy(vs,vs+size_t(cached)*K,values.data()+size_t(l)*capacity*K);
            }
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
    void mhc_projections(int mh, int layer, const float* input) {
        const TensorDesc* matrices[3] = {&t[mh+6], &t[mh+7], &t[mh+8]};
        int rows[3] = {c.lanes, c.lanes, c.lanes*c.lanes};
        float* outputs[3] = {hpre.data(), hpost.data(), hres.data()};
        if (matrices[0]->cq || matrices[1]->cq || matrices[2]->cq) {
            for (int m = 0; m < 3; ++m)
                linear(mh+6+m, input, outputs[m], layer*rows[m], rows[m]);
            return;
        }
        int blocks[3] = {(rows[0]+1)/2, (rows[1]+1)/2, (rows[2]+1)/2};
        auto work = [&](int tid, int begin, int end) {
            int offset = 0;
            for (int m = 0; m < 3; ++m) {
                int first = std::max(0, begin-offset), last = std::min(blocks[m], end-offset);
                offset += blocks[m];
                for (int b = first; b < last; ++b) {
                    int r = 2*b, cols = matrices[m]->cols;
                    const float* w = matrices[m]->data + size_t(layer*rows[m]+r)*cols;
                    if (r+1 < rows[m]) dot_pair(w, w+cols, input, cols, outputs[m][r], outputs[m][r+1]);
                    else outputs[m][r] = dot_f32(w, input, cols);
                }
            }
        };
        // Keep small geometries serial. The released four-lane model has
        // 24 output rows of width 2048, enough work to share one dispatch.
        int total_blocks = blocks[0]+blocks[1]+blocks[2];
        int64_t work_size = 0;
        for (int m = 0; m < 3; ++m) work_size += int64_t(rows[m])*matrices[m]->cols;
        if (work_size >= 32768) parallel_for(total_blocks, work);
        else work(0, 0, total_blocks);
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
            sdot_batch.resize(batch);
            auto& prep = sdot_batch;
            for (int b = 0; b < batch; ++b) q->prepare(in + size_t(b) * q->columns, prep[b]);
            parallel_for(rows, [&](int tid, int start, int end) {
                for (int r = start; r < end; ++r) {
                    int b = 0;
                    for (; b + 1 < batch; b += 2)
                        q->row_pair(first+r, prep[b], prep[b+1], out[size_t(b)*rows+r], out[size_t(b+1)*rows+r]);
                    if (b < batch) out[size_t(b)*rows+r] = q->row(first+r, prep[b]);
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
            sdot_batch.resize(batch);
            auto& prep = sdot_batch;
            for (int b = 0; b < batch; ++b) matrices[0]->prepare(in + size_t(b) * matrices[0]->columns, prep[b]);
            int total = 0;
            for (auto* m : matrices) total += m->rows;
            float* outputs[4] = {q_out, k_out, v_out, gate_out};
            parallel_for(total, [&](int tid, int start, int end) {
                for (int i = start; i < end; ++i) {
                    int m = 0, r = i;
                    while (m < 4 && r >= matrices[m]->rows) { r -= matrices[m]->rows; m++; }
                    int b = 0, stride = matrices[m]->rows;
                    for (; b + 1 < batch; b += 2)
                        matrices[m]->row_pair(r, prep[b], prep[b+1], outputs[m][size_t(b)*stride+r], outputs[m][size_t(b+1)*stride+r]);
                    if (b < batch) outputs[m][size_t(b)*stride+r] = matrices[m]->row(r, prep[b]);
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
    template<bool Quantized, bool Paired>
    void accumulate_values(int l, int kh, int length, const float* s0, const float* s1, float* dst) {
        int HD=c.head_dim,K=c.kvheads*HD;
        int d=0;
#ifdef __aarch64__
        // Keep a 32-dimension tile in registers over the entire context. Each
        // dimension sees the same token/FMA order as the untiled kernel.
        for (;d+32<=HD;d+=32) {
            auto a0=vdupq_n_f32(0), b0=a0;
            auto a1=vdupq_n_f32(0), b1=a1;
            auto a2=vdupq_n_f32(0), b2=a2;
            auto a3=vdupq_n_f32(0), b3=a3;
            auto a4=vdupq_n_f32(0), b4=a4;
            auto a5=vdupq_n_f32(0), b5=a5;
            auto a6=vdupq_n_f32(0), b6=a6;
            auto a7=vdupq_n_f32(0), b7=a7;
            for(int s=0;s<length;++s) {
                int slot=attention_slots[s];
                float prob=s0[s], prob1=Paired?s1[s]:0;
                float32x4_t v0,v1,v2,v3,v4,v5,v6,v7;
                if constexpr (Quantized) {
                    float scale=v_scales[(size_t(l)*capacity+slot)*c.kvheads+kh];
                    prob*=scale;prob1*=scale;
                    const auto* src=values_i8.data()+(size_t(l)*capacity+slot)*K+kh*HD+d;
                    auto w0=vmovl_s8(vld1_s8(src+0));
                    v0=vcvtq_f32_s32(vmovl_s16(vget_low_s16(w0)));
                    v1=vcvtq_f32_s32(vmovl_s16(vget_high_s16(w0)));
                    auto w1=vmovl_s8(vld1_s8(src+8));
                    v2=vcvtq_f32_s32(vmovl_s16(vget_low_s16(w1)));
                    v3=vcvtq_f32_s32(vmovl_s16(vget_high_s16(w1)));
                    auto w2=vmovl_s8(vld1_s8(src+16));
                    v4=vcvtq_f32_s32(vmovl_s16(vget_low_s16(w2)));
                    v5=vcvtq_f32_s32(vmovl_s16(vget_high_s16(w2)));
                    auto w3=vmovl_s8(vld1_s8(src+24));
                    v6=vcvtq_f32_s32(vmovl_s16(vget_low_s16(w3)));
                    v7=vcvtq_f32_s32(vmovl_s16(vget_high_s16(w3)));
                } else {
                    const auto* src=values.data()+(size_t(l)*capacity+slot)*K+kh*HD+d;
                    v0=vld1q_f32(src+0);
                    v1=vld1q_f32(src+4);
                    v2=vld1q_f32(src+8);
                    v3=vld1q_f32(src+12);
                    v4=vld1q_f32(src+16);
                    v5=vld1q_f32(src+20);
                    v6=vld1q_f32(src+24);
                    v7=vld1q_f32(src+28);
                }
                a0=vfmaq_n_f32(a0,v0,prob); if constexpr(Paired) b0=vfmaq_n_f32(b0,v0,prob1);
                a1=vfmaq_n_f32(a1,v1,prob); if constexpr(Paired) b1=vfmaq_n_f32(b1,v1,prob1);
                a2=vfmaq_n_f32(a2,v2,prob); if constexpr(Paired) b2=vfmaq_n_f32(b2,v2,prob1);
                a3=vfmaq_n_f32(a3,v3,prob); if constexpr(Paired) b3=vfmaq_n_f32(b3,v3,prob1);
                a4=vfmaq_n_f32(a4,v4,prob); if constexpr(Paired) b4=vfmaq_n_f32(b4,v4,prob1);
                a5=vfmaq_n_f32(a5,v5,prob); if constexpr(Paired) b5=vfmaq_n_f32(b5,v5,prob1);
                a6=vfmaq_n_f32(a6,v6,prob); if constexpr(Paired) b6=vfmaq_n_f32(b6,v6,prob1);
                a7=vfmaq_n_f32(a7,v7,prob); if constexpr(Paired) b7=vfmaq_n_f32(b7,v7,prob1);
            }
            vst1q_f32(dst+d+0,a0); if constexpr(Paired) vst1q_f32(dst+HD+d+0,b0);
            vst1q_f32(dst+d+4,a1); if constexpr(Paired) vst1q_f32(dst+HD+d+4,b1);
            vst1q_f32(dst+d+8,a2); if constexpr(Paired) vst1q_f32(dst+HD+d+8,b2);
            vst1q_f32(dst+d+12,a3); if constexpr(Paired) vst1q_f32(dst+HD+d+12,b3);
            vst1q_f32(dst+d+16,a4); if constexpr(Paired) vst1q_f32(dst+HD+d+16,b4);
            vst1q_f32(dst+d+20,a5); if constexpr(Paired) vst1q_f32(dst+HD+d+20,b5);
            vst1q_f32(dst+d+24,a6); if constexpr(Paired) vst1q_f32(dst+HD+d+24,b6);
            vst1q_f32(dst+d+28,a7); if constexpr(Paired) vst1q_f32(dst+HD+d+28,b7);
        }
#endif
        // Small/odd head dimensions and non-NEON architectures retain a tail.
        for(int j=d;j<HD;++j) {
            float a=0,b=0;
            for(int s=0;s<length;++s) {
                int slot=attention_slots[s];
                float prob=s0[s],prob1=Paired?s1[s]:0,value;
                if constexpr(Quantized) {
                    float scale=v_scales[(size_t(l)*capacity+slot)*c.kvheads+kh];
                    prob*=scale;prob1*=scale;
                    value=float(values_i8[(size_t(l)*capacity+slot)*K+kh*HD+j]);
                } else value=values[(size_t(l)*capacity+slot)*K+kh*HD+j];
                a+=prob*value;if constexpr(Paired)b+=prob1*value;
            }
            dst[j]=a;if constexpr(Paired)dst[HD+j]=b;
        }
    }
    void compute_attention(int l, int pos_cur, const float* q_in, float* att_out) {
        int H=c.heads,KV=c.kvheads,HD=c.head_dim,K=KV*HD;
        int prefix_count=c.window?std::min(prefix,pos_cur+1):0;
        int start=c.window?std::max(prefix,pos_cur-c.window+1):0;
        int length=prefix_count+std::max(0,pos_cur-start+1);
        // The slot order depends only on position/prefix/window, not layer.
        // Decode reuses this map across layers; prompt chunks also avoid doing
        // modular arithmetic for every head and both K and V passes.
        if (attention_position != pos_cur) {
            attention_slots.resize(length);
            for(int s=0;s<length;++s)
                attention_slots[s]=cache_slot(s<prefix_count?s:start+s-prefix_count);
            attention_position=pos_cur;
        }
        parallel_for(int(head_work.size()), [&](int tid, int start_g, int end_g) {
            for(int g=start_g;g<end_g;++g) {
                int h=head_work[g].h,kh=head_work[g].kh,count=head_work[g].count;
                float max0=-INFINITY,max1=-INFINITY,scale=1/std::sqrt(float(HD));
                auto*s0=thread_scores.data()+size_t(tid)*2*capacity;
                auto*s1=s0+capacity;
                if(!int8_kv_enabled) {
                    const auto*kc=keys.data()+size_t(l)*capacity*K;
                    for(int s=0;s<length;++s) {
                        int pos=attention_slots[s];
                        if(count==2)dot_pair(q_in+h*HD,q_in+(h+1)*HD,kc+pos*K+kh*HD,HD,s0[s],s1[s]);
                        else s0[s]=dot_f32(q_in+h*HD,kc+pos*K+kh*HD,HD);
                        s0[s]*=scale;max0=std::max(max0,s0[s]);
                        if(count==2){s1[s]*=scale;max1=std::max(max1,s1[s]);}
                    }
                    softmax_inplace(s0,length,max0);if(count==2)softmax_inplace(s1,length,max1);
                    if(count==2)accumulate_values<false,true>(l,kh,length,s0,s1,att_out+h*HD);
                    else accumulate_values<false,false>(l,kh,length,s0,s1,att_out+h*HD);
                } else {
                    const auto*kc_i8=keys_i8.data()+size_t(l)*capacity*K;
                    const auto*ks=k_scales.data()+size_t(l)*capacity*KV;
                    int s=0;
#ifdef __aarch64__
                    // A constant head dimension lets the compiler unroll the
                    // original dot product without changing its reduction order.
                    if (count==2 && HD==64) {
                        for (;s<length;++s) {
                            int pos=attention_slots[s];
                            dot_pair_i8_f32(q_in+h*HD,q_in+(h+1)*HD,
                                kc_i8+pos*K+kh*HD,64,s0[s],s1[s]);
                            float k_scale=ks[pos*KV+kh];
                            s0[s]*=k_scale;s1[s]*=k_scale;
                            s0[s]*=scale;max0=std::max(max0,s0[s]);
                            s1[s]*=scale;max1=std::max(max1,s1[s]);
                        }
                    }
#endif
                    for(;s<length;++s) {
                        int pos=attention_slots[s];
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
                    if(count==2)accumulate_values<true,true>(l,kh,length,s0,s1,att_out+h*HD);
                    else accumulate_values<true,false>(l,kh,length,s0,s1,att_out+h*HD);
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
            mhc_projections(mh, l, nx.data());
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

            std::vector<float> chunk_cos(size_t(B)*HD/2), chunk_sin(size_t(B)*HD/2);
            for (int b=0;b<B;++b) for (int j=0;j<HD/2;++j) {
                float a=(start_pos+b)/rope_divisor[j];
                chunk_cos[size_t(b)*(HD/2)+j]=std::cos(a);
                chunk_sin[size_t(b)*(HD/2)+j]=std::sin(a);
            }
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
                        float co = chunk_cos[size_t(b)*(HD/2)+j], si = chunk_sin[size_t(b)*(HD/2)+j];
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
    void validate_candidates(const int* candidates, int num_candidates) const {
        if(!candidates || num_candidates<=0)throw std::runtime_error("candidates must be nonempty");
        for(int i=0;i<num_candidates;++i)
            if(candidates[i]<0||candidates[i]>=c.vocab)throw std::runtime_error("candidate outside vocabulary");
    }
    void project_candidates(const int* candidates, int num_candidates, float* candidate_logits) {
        validate_candidates(candidates,num_candidates);
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
        validate_candidates(candidates,num_candidates);
        step(token, nullptr, hidden_out);
        if (candidates && num_candidates > 0 && candidate_logits) {
            project_candidates(candidates, num_candidates, candidate_logits);
        } else if (candidate_logits && (!candidates || num_candidates < 0)) {
            activation_quant(z.data(), c.dim, abits);
            linear(0, z.data(), candidate_logits);
        }
    }
    int decode_loop(int first_token, int max_new_tokens, const DFAStateDesc* dfa, int* output_tokens, int* out_generated_count) {
        if (max_new_tokens <= 0) {
            if (out_generated_count) *out_generated_count = 0;
            return 0;
        }

        if(first_token<0||first_token>=c.vocab)throw std::runtime_error("token outside vocabulary");
        if(dfa) {
            if(dfa->num_states<=0||dfa->initial_state<0||dfa->initial_state>=dfa->num_states ||
               !dfa->state_types||!dfa->fallback_next_states||!dfa->candidate_offsets)
                throw std::runtime_error("invalid DFA states");
            for(int token:{dfa->eos_id,dfa->stop_id,dfa->tool_start_id,dfa->tool_end_id})
                if(token<0||token>=c.vocab)throw std::runtime_error("DFA token outside vocabulary");
            if(dfa->candidate_offsets[0]!=0)throw std::runtime_error("invalid DFA offsets");
            for(int s=0;s<dfa->num_states;++s) {
                int type=dfa->state_types[s],fallback=dfa->fallback_next_states[s];
                int begin=dfa->candidate_offsets[s],end=dfa->candidate_offsets[s+1];
                if((type!=0&&type!=1&&type!=4)||fallback < -1||fallback>=dfa->num_states||begin<0||end<begin)
                    throw std::runtime_error("invalid DFA state");
                if(type==1&&begin==end)throw std::runtime_error("DFA state has no candidates");
                if(end>begin) {
                    if(!dfa->candidate_tokens||!dfa->next_states)throw std::runtime_error("missing DFA transitions");
                    validate_candidates(dfa->candidate_tokens+begin,end-begin);
                }
                for(int i=begin;i<end;++i)
                    if(dfa->next_states[i]<0||dfa->next_states[i]>=dfa->num_states)
                        throw std::runtime_error("invalid DFA transition");
            }
        }
        int eos = dfa ? dfa->eos_id : 1;
        int stop_tok = dfa ? dfa->stop_id : 5;
        int tool_start = dfa ? dfa->tool_start_id : 10;
        int tool_end = dfa ? dfa->tool_end_id : 11;

        output_tokens[0] = first_token;
        int count = 1;
        int current_tok = first_token;

        if (first_token == eos || first_token == stop_tok) {
            if (out_generated_count) *out_generated_count = count;
            return 0;
        }

        int state = 0;
        if (dfa) {
            state = dfa->initial_state;
            int start_idx = dfa->candidate_offsets[state];
            int end_idx = dfa->candidate_offsets[state + 1];
            bool matched = false;
            for (int i = start_idx; i < end_idx; ++i) {
                if (dfa->candidate_tokens[i] == first_token) {
                    state = dfa->next_states[i];
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                if(dfa->state_types[state]!=0)throw std::runtime_error("first token rejected by DFA");
                if(dfa->fallback_next_states[state]>=0)state=dfa->fallback_next_states[state];
            }
        }

        std::vector<float> full_logits(c.vocab);
        std::vector<float> cand_logits(1024);

        while (count < max_new_tokens) {
            if (dfa && dfa->state_types[state] == 4) { // TERMINAL
                break;
            }

            int next_tok = -1;

            if (!dfa || dfa->state_types[state] == 0) { // UNCONSTRAINED / OPEN
                step(current_tok, full_logits.data(), nullptr);
                int best = 0;
                float max_v = full_logits[0];
                for (int i = 1; i < c.vocab; ++i) {
                    if (full_logits[i] > max_v) {
                        max_v = full_logits[i];
                        best = i;
                    }
                }
                next_tok = best;
                if (dfa) {
                    int start_idx = dfa->candidate_offsets[state];
                    int end_idx = dfa->candidate_offsets[state + 1];
                    bool matched = false;
                    for (int i = start_idx; i < end_idx; ++i) {
                        if (dfa->candidate_tokens[i] == next_tok) {
                            state = dfa->next_states[i];
                            matched = true;
                            break;
                        }
                    }
                    if (!matched && dfa->fallback_next_states[state] >= 0) {
                        state = dfa->fallback_next_states[state];
                    }
                }
            } else if (dfa->state_types[state] == 1) { // EXACT_CANDIDATES
                int start_idx = dfa->candidate_offsets[state];
                int end_idx = dfa->candidate_offsets[state + 1];
                int num_cands = end_idx - start_idx;

                if (num_cands == 1) {
                    next_tok = dfa->candidate_tokens[start_idx];
                    step(current_tok, nullptr, nullptr);
                    state = dfa->next_states[start_idx];
                } else if (num_cands > 1) {
                    const int* cands = dfa->candidate_tokens + start_idx;
                    if (cand_logits.size() < size_t(num_cands)) cand_logits.resize(num_cands);
                    step_candidates(current_tok, cands, num_cands, cand_logits.data(), nullptr);

                    int best_idx = 0;
                    float max_v = cand_logits[0];
                    for (int i = 1; i < num_cands; ++i) {
                        if (cand_logits[i] > max_v) {
                            max_v = cand_logits[i];
                            best_idx = i;
                        }
                    }
                    next_tok = cands[best_idx];
                    state = dfa->next_states[start_idx + best_idx];
                } else {
                    throw std::runtime_error("DFA state has no candidates");
                }
            }

            output_tokens[count++] = next_tok;
            current_tok = next_tok;

            if (next_tok == eos || next_tok == stop_tok || (dfa && dfa->state_types[state] == 4)) {
                break;
            }
        }

        if (out_generated_count) *out_generated_count = count;
        return 0;
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
int needle2_engine_decode_loop(void*p,int first_token,int max_new_tokens,const DFAStateDesc*dfa,int*output_tokens,int*out_generated_count){
    try{return static_cast<Engine*>(p)->decode_loop(first_token,max_new_tokens,dfa,output_tokens,out_generated_count);}
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
size_t needle2_engine_kv_storage_bytes(void*p){return static_cast<Engine*>(p)->kv_storage_bytes();}
const char*needle2_engine_error(){return engine_error.c_str();}
}
