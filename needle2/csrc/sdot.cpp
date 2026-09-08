// Explicit approximate DotProd backend. CQ storage remains packed; activations
// after Hadamard and codebook centroids are rounded to INT8. Not the FP32 default.
#if defined(__aarch64__) && defined(__linux__)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#endif
static bool runtime_dotprod() {
#if defined(__aarch64__) && defined(__linux__)
    return (getauxval(AT_HWCAP)&HWCAP_ASIMDDP)!=0;
#else
    return false;
#endif
}
struct Prepared {
    std::vector<int8_t> q,quantized;
    std::vector<float> scales,rotated;
    void resize(int padded,int group){q.resize(padded);scales.resize(padded/group);rotated.resize(padded);quantized.resize(group);}
};
struct SdotCQ {
    const uint8_t *packed;
    int rows, columns, bits, group, padded, groups, rowbytes;
    float centroid_scale;
    alignas(16) int8_t centroids[16] = {};
    std::vector<float> norm_scales;

    SdotCQ(const uint8_t *p, const uint16_t *norms, int out, int in, int b, int g, const float *cb)
        : packed(p), rows(out), columns(in), bits(b), group(g), padded((in + g - 1) / g * g),
          groups(padded / g), rowbytes(padded * b / 8), norm_scales(size_t(out) * groups) {
        float maximum = 0;
        for (int i = 0; i < (1 << bits); ++i) maximum = std::max(maximum, std::abs(cb[i]));
        centroid_scale = maximum > 0 ? maximum / 127 : 1;
        for (int i = 0; i < (1 << bits); ++i)
            centroids[i] = int8_t(std::clamp(std::nearbyint(cb[i] / centroid_scale), -127.0f, 127.0f));
        for (size_t i = 0; i < norm_scales.size(); ++i)
            norm_scales[i] = half_float(norms[i]) * centroid_scale;
    }

    void prepare(const float *input, Prepared &output) const {
        output.resize(padded,group);
        auto &rotated=output.rotated;auto &quantized=output.quantized;
        std::copy(input,input+columns,rotated.begin());std::fill(rotated.begin()+columns,rotated.end(),0);
#ifdef __aarch64__
        const auto lower = vdupq_n_f32(-127), upper = vdupq_n_f32(127);
#endif
        for (int g = 0; g < groups; ++g) {
            float *x = rotated.data() + g * group;
            hadamard(x, group);
#ifdef __aarch64__
            auto maximum = vdupq_n_f32(0);
            for (int i = 0; i < group; i += 4) maximum = vmaxq_f32(maximum, vabsq_f32(vld1q_f32(x + i)));
            const float absmax = vmaxvq_f32(maximum);
            float scale = absmax > 0 ? absmax / 127 : 1;
            if (scale == 0) scale = absmax;  // Subnormal absmax/127 can underflow.
            output.scales[g] = scale;
            // Use division instead of 127/absmax for tiny nonzero values to
            // avoid overflowing a reciprocal before multiplying activations.
            const auto scale_vector = vdupq_n_f32(scale);
            for (int i = 0; i < group; i += 16) {
                int32x4_t q[4];
                for (int lane = 0; lane < 4; ++lane) {
                    auto y = vdivq_f32(vld1q_f32(x + i + 4 * lane), scale_vector);
                    q[lane] = vcvtnq_s32_f32(vmaxq_f32(lower, vminq_f32(upper, y)));
                }
                auto lo = vcombine_s16(vqmovn_s32(q[0]), vqmovn_s32(q[1]));
                auto hi = vcombine_s16(vqmovn_s32(q[2]), vqmovn_s32(q[3]));
                vst1q_s8(quantized.data() + i, vcombine_s8(vqmovn_s16(lo), vqmovn_s16(hi)));
            }
#else
            float absmax=0;for(int i=0;i<group;++i)absmax=std::max(absmax,std::abs(x[i]));
            float scale=absmax>0?absmax/127:1;if(!scale)scale=absmax;output.scales[g]=scale;
            for(int i=0;i<group;++i)quantized[i]=int8_t(std::clamp(std::nearbyint(x[i]/scale),-127.f,127.f));
#endif
            int8_t *dst = output.q.data() + g * group;
            // Permute x once, making packed weight decoding simple bit planes.
            // SDOT sees identical permutations on both sides of each product.
#ifdef __aarch64__
            if (bits == 2) {
                for (int i = 0; i < group; i += 64) {
                    const auto planes = vld4q_s8(quantized.data() + i);
                    for (int plane = 0; plane < 4; ++plane) vst1q_s8(dst + i + plane * 16, planes.val[plane]);
                }
            } else {
                for (int i = 0; i < group; i += 32) {
                    const auto planes = vld2q_s8(quantized.data() + i);
                    vst1q_s8(dst + i, planes.val[0]);
                    vst1q_s8(dst + i + 16, planes.val[1]);
                }
            }
#else
            const int width=bits==2?4:2, chunk=width*16;
            for(int i=0;i<group;++i)dst[(i/chunk)*chunk+(i%width)*16+(i%chunk)/width]=quantized[i];
#endif
        }
    }

#ifdef __aarch64__
    __attribute__((target("arch=armv8.2-a+dotprod")))
#endif
    int32_t dot_group(const uint8_t *p, const int8_t *x) const {
#ifdef __aarch64__
        auto a = vdupq_n_s32(0), b = a, c = a, d = a;
        const auto cb = vld1q_s8(centroids);
        if (bits == 2) {
            const auto mask = vdupq_n_u8(3);
            for (int i = 0; i < group; i += 64) {
                const auto packed_values = vld1q_u8(p + i / 4);
                a = vdotq_s32(a, vqtbl1q_s8(cb, vandq_u8(packed_values, mask)), vld1q_s8(x + i));
                b = vdotq_s32(b, vqtbl1q_s8(cb, vandq_u8(vshrq_n_u8(packed_values, 2), mask)), vld1q_s8(x + i + 16));
                c = vdotq_s32(c, vqtbl1q_s8(cb, vandq_u8(vshrq_n_u8(packed_values, 4), mask)), vld1q_s8(x + i + 32));
                d = vdotq_s32(d, vqtbl1q_s8(cb, vshrq_n_u8(packed_values, 6)), vld1q_s8(x + i + 48));
            }
        } else {
            const auto mask = vdupq_n_u8(15);
            for (int i = 0; i < group; i += 64) {
                const auto lo = vld1q_u8(p + i / 2), hi = vld1q_u8(p + i / 2 + 16);
                a = vdotq_s32(a, vqtbl1q_s8(cb, vandq_u8(lo, mask)), vld1q_s8(x + i));
                b = vdotq_s32(b, vqtbl1q_s8(cb, vshrq_n_u8(lo, 4)), vld1q_s8(x + i + 16));
                c = vdotq_s32(c, vqtbl1q_s8(cb, vandq_u8(hi, mask)), vld1q_s8(x + i + 32));
                d = vdotq_s32(d, vqtbl1q_s8(cb, vshrq_n_u8(hi, 4)), vld1q_s8(x + i + 48));
            }
        }
        return vaddvq_s32(vaddq_s32(vaddq_s32(a, b), vaddq_s32(c, d)));
#else
        int32_t sum=0;int width=bits==2?4:2,chunk=width*16;
        for(int i=0;i<group;++i)sum+=int(centroids[index_at(p,i,bits)])*int(x[(i/chunk)*chunk+(i%width)*16+(i%chunk)/width]);
        return sum;
#endif
    }

    #ifdef __aarch64__
    __attribute__((target("arch=armv8.2-a+dotprod")))
    #endif
    float row(int row,const Prepared &x)const {
        float value=0;
        for(int g=0;g<groups;++g) {
            const auto *p=packed+size_t(row)*rowbytes+g*group*bits/8;
            value+=float(dot_group(p,x.q.data()+g*group))*norm_scales[size_t(row)*groups+g]*x.scales[g];
        }
        return value;
    }
    void multiply(const Prepared&x,float*y,int threads=1)const {
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) if(threads>1) schedule(static)
#endif
        for(int r=0;r<rows;++r)y[r]=row(r,x);
    }
    void linear(const float*x,float*y,int threads=1)const {Prepared p;prepare(x,p);multiply(p,y,threads);}
};

extern "C" {
int needle2_sdot_supported(){return runtime_dotprod();}
void*needle2_sdot_create(const uint8_t*p,const uint16_t*n,int out,int in,int bits,int group,const float*cb) {
    if(!runtime_dotprod()||(bits!=2&&bits!=4)||group<64||group>131072)return nullptr;
    try{return new SdotCQ(p,n,out,in,bits,group,cb);}catch(...){return nullptr;}
}
void needle2_sdot_destroy(void*p){delete static_cast<SdotCQ*>(p);}
void needle2_sdot_linear(void*p,const float*x,float*y,int batch,int threads) {
    auto*q=static_cast<SdotCQ*>(p);Prepared prep;
    for(int b=0;b<batch;++b){q->prepare(x+size_t(b)*q->columns,prep);q->multiply(prep,y+size_t(b)*q->rows,threads);}
}
}
