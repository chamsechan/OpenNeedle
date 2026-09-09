// Explicit approximate DotProd backend. CQ storage remains packed; activations
// after Hadamard and codebook centroids are rounded to INT8. Not the FP32 default.
#if defined(__aarch64__) && defined(__linux__)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#elif defined(__aarch64__) && defined(__APPLE__)
#include <sys/sysctl.h>
#endif
static bool runtime_dotprod() {
#if defined(__aarch64__) && defined(__linux__)
    return (getauxval(AT_HWCAP)&HWCAP_ASIMDDP)!=0;
#elif defined(__aarch64__) && defined(__APPLE__)
    int supported = 0;
    size_t size = sizeof(supported);
    if (sysctlbyname("hw.optional.arm.FEAT_DotProd", &supported, &size, NULL, 0) == 0) {
        return supported != 0;
    }
    return false;
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

    // Two prompt tokens share packed-weight loads and centroid lookup. Integer
    // accumulators and group scaling have the same order as two row() calls.
#ifdef __aarch64__
    __attribute__((target("arch=armv8.2-a+dotprod")))
#endif
    void row_pair(int r, const Prepared& x0, const Prepared& x1, float& out0, float& out1) const {
        out0 = out1 = 0;
        for (int g = 0; g < groups; ++g) {
            const auto* p = packed + size_t(r)*rowbytes + g*group*bits/8;
            const auto* u = x0.q.data() + g*group;
            const auto* v = x1.q.data() + g*group;
#ifdef __aarch64__
            const auto cb = vld1q_s8(centroids);
            auto a0=vdupq_n_s32(0), b0=a0, c0=a0, d0=a0;
            auto a1=a0, b1=a0, c1=a0, d1=a0;
            for (int i=0; i<group; i+=64) {
                int8x16_t w0,w1,w2,w3;
                if (bits == 2) {
                    const auto pv=vld1q_u8(p+i/4), mask=vdupq_n_u8(3);
                    w0=vqtbl1q_s8(cb,vandq_u8(pv,mask));
                    w1=vqtbl1q_s8(cb,vandq_u8(vshrq_n_u8(pv,2),mask));
                    w2=vqtbl1q_s8(cb,vandq_u8(vshrq_n_u8(pv,4),mask));
                    w3=vqtbl1q_s8(cb,vshrq_n_u8(pv,6));
                } else {
                    const auto lo=vld1q_u8(p+i/2), hi=vld1q_u8(p+i/2+16), mask=vdupq_n_u8(15);
                    w0=vqtbl1q_s8(cb,vandq_u8(lo,mask)); w1=vqtbl1q_s8(cb,vshrq_n_u8(lo,4));
                    w2=vqtbl1q_s8(cb,vandq_u8(hi,mask)); w3=vqtbl1q_s8(cb,vshrq_n_u8(hi,4));
                }
                a0=vdotq_s32(a0,w0,vld1q_s8(u+i)); a1=vdotq_s32(a1,w0,vld1q_s8(v+i));
                b0=vdotq_s32(b0,w1,vld1q_s8(u+i+16)); b1=vdotq_s32(b1,w1,vld1q_s8(v+i+16));
                c0=vdotq_s32(c0,w2,vld1q_s8(u+i+32)); c1=vdotq_s32(c1,w2,vld1q_s8(v+i+32));
                d0=vdotq_s32(d0,w3,vld1q_s8(u+i+48)); d1=vdotq_s32(d1,w3,vld1q_s8(v+i+48));
            }
            int32_t v0=vaddvq_s32(vaddq_s32(vaddq_s32(a0,b0),vaddq_s32(c0,d0)));
            int32_t v1=vaddvq_s32(vaddq_s32(vaddq_s32(a1,b1),vaddq_s32(c1,d1)));
#else
            int32_t v0=dot_group(p,u), v1=dot_group(p,v);
#endif
            float norm=norm_scales[size_t(r)*groups+g];
            out0 += float(v0)*norm*x0.scales[g];
            out1 += float(v1)*norm*x1.scales[g];
        }
    }

#ifdef __aarch64__
    __attribute__((target("arch=armv8.2-a+dotprod")))
#endif
    void row4(int r, const Prepared &x, float *out) const {
#ifdef __aarch64__
        if ((bits != 2 && bits != 4) || (group % 64 != 0)) {
            out[0] = row(r + 0, x);
            out[1] = row(r + 1, x);
            out[2] = row(r + 2, x);
            out[3] = row(r + 3, x);
            return;
        }
        float val0 = 0, val1 = 0, val2 = 0, val3 = 0;
        const auto cb = vld1q_s8(centroids);
        const auto *x_data = x.q.data();
        const size_t r0_idx = size_t(r + 0);
        const size_t r1_idx = size_t(r + 1);
        const size_t r2_idx = size_t(r + 2);
        const size_t r3_idx = size_t(r + 3);

        if (bits == 2) {
            const auto mask = vdupq_n_u8(3);
            for (int g = 0; g < groups; ++g) {
                const auto *p0 = packed + r0_idx * rowbytes + g * group * 2 / 8;
                const auto *p1 = packed + r1_idx * rowbytes + g * group * 2 / 8;
                const auto *p2 = packed + r2_idx * rowbytes + g * group * 2 / 8;
                const auto *p3 = packed + r3_idx * rowbytes + g * group * 2 / 8;
                const auto *xg = x_data + g * group;

                auto a0 = vdupq_n_s32(0), b0 = a0, c0 = a0, d0 = a0;
                auto a1 = a0, b1 = a0, c1 = a0, d1 = a0;
                auto a2 = a0, b2 = a0, c2 = a0, d2 = a0;
                auto a3 = a0, b3 = a0, c3 = a0, d3 = a0;

                for (int i = 0; i < group; i += 64) {
                    const auto x0 = vld1q_s8(xg + i);
                    const auto x1 = vld1q_s8(xg + i + 16);
                    const auto x2 = vld1q_s8(xg + i + 32);
                    const auto x3 = vld1q_s8(xg + i + 48);

                    const auto pv0 = vld1q_u8(p0 + i / 4);
                    a0 = vdotq_s32(a0, vqtbl1q_s8(cb, vandq_u8(pv0, mask)), x0);
                    b0 = vdotq_s32(b0, vqtbl1q_s8(cb, vandq_u8(vshrq_n_u8(pv0, 2), mask)), x1);
                    c0 = vdotq_s32(c0, vqtbl1q_s8(cb, vandq_u8(vshrq_n_u8(pv0, 4), mask)), x2);
                    d0 = vdotq_s32(d0, vqtbl1q_s8(cb, vshrq_n_u8(pv0, 6)), x3);

                    const auto pv1 = vld1q_u8(p1 + i / 4);
                    a1 = vdotq_s32(a1, vqtbl1q_s8(cb, vandq_u8(pv1, mask)), x0);
                    b1 = vdotq_s32(b1, vqtbl1q_s8(cb, vandq_u8(vshrq_n_u8(pv1, 2), mask)), x1);
                    c1 = vdotq_s32(c1, vqtbl1q_s8(cb, vandq_u8(vshrq_n_u8(pv1, 4), mask)), x2);
                    d1 = vdotq_s32(d1, vqtbl1q_s8(cb, vshrq_n_u8(pv1, 6)), x3);

                    const auto pv2 = vld1q_u8(p2 + i / 4);
                    a2 = vdotq_s32(a2, vqtbl1q_s8(cb, vandq_u8(pv2, mask)), x0);
                    b2 = vdotq_s32(b2, vqtbl1q_s8(cb, vandq_u8(vshrq_n_u8(pv2, 2), mask)), x1);
                    c2 = vdotq_s32(c2, vqtbl1q_s8(cb, vandq_u8(vshrq_n_u8(pv2, 4), mask)), x2);
                    d2 = vdotq_s32(d2, vqtbl1q_s8(cb, vshrq_n_u8(pv2, 6)), x3);

                    const auto pv3 = vld1q_u8(p3 + i / 4);
                    a3 = vdotq_s32(a3, vqtbl1q_s8(cb, vandq_u8(pv3, mask)), x0);
                    b3 = vdotq_s32(b3, vqtbl1q_s8(cb, vandq_u8(vshrq_n_u8(pv3, 2), mask)), x1);
                    c3 = vdotq_s32(c3, vqtbl1q_s8(cb, vandq_u8(vshrq_n_u8(pv3, 4), mask)), x2);
                    d3 = vdotq_s32(d3, vqtbl1q_s8(cb, vshrq_n_u8(pv3, 6)), x3);
                }
                const float xs = x.scales[g];
                int32_t dot0 = vaddvq_s32(vaddq_s32(vaddq_s32(a0, b0), vaddq_s32(c0, d0)));
                int32_t dot1 = vaddvq_s32(vaddq_s32(vaddq_s32(a1, b1), vaddq_s32(c1, d1)));
                int32_t dot2 = vaddvq_s32(vaddq_s32(vaddq_s32(a2, b2), vaddq_s32(c2, d2)));
                int32_t dot3 = vaddvq_s32(vaddq_s32(vaddq_s32(a3, b3), vaddq_s32(c3, d3)));

                val0 += float(dot0) * norm_scales[r0_idx * groups + g] * xs;
                val1 += float(dot1) * norm_scales[r1_idx * groups + g] * xs;
                val2 += float(dot2) * norm_scales[r2_idx * groups + g] * xs;
                val3 += float(dot3) * norm_scales[r3_idx * groups + g] * xs;
            }
        } else {
            const auto mask = vdupq_n_u8(15);
            for (int g = 0; g < groups; ++g) {
                const auto *p0 = packed + r0_idx * rowbytes + g * group * 4 / 8;
                const auto *p1 = packed + r1_idx * rowbytes + g * group * 4 / 8;
                const auto *p2 = packed + r2_idx * rowbytes + g * group * 4 / 8;
                const auto *p3 = packed + r3_idx * rowbytes + g * group * 4 / 8;
                const auto *xg = x_data + g * group;

                auto a0 = vdupq_n_s32(0), b0 = a0, c0 = a0, d0 = a0;
                auto a1 = a0, b1 = a0, c1 = a0, d1 = a0;
                auto a2 = a0, b2 = a0, c2 = a0, d2 = a0;
                auto a3 = a0, b3 = a0, c3 = a0, d3 = a0;

                for (int i = 0; i < group; i += 64) {
                    const auto x0 = vld1q_s8(xg + i);
                    const auto x1 = vld1q_s8(xg + i + 16);
                    const auto x2 = vld1q_s8(xg + i + 32);
                    const auto x3 = vld1q_s8(xg + i + 48);

                    const auto lo0 = vld1q_u8(p0 + i / 2), hi0 = vld1q_u8(p0 + i / 2 + 16);
                    a0 = vdotq_s32(a0, vqtbl1q_s8(cb, vandq_u8(lo0, mask)), x0);
                    b0 = vdotq_s32(b0, vqtbl1q_s8(cb, vshrq_n_u8(lo0, 4)), x1);
                    c0 = vdotq_s32(c0, vqtbl1q_s8(cb, vandq_u8(hi0, mask)), x2);
                    d0 = vdotq_s32(d0, vqtbl1q_s8(cb, vshrq_n_u8(hi0, 4)), x3);

                    const auto lo1 = vld1q_u8(p1 + i / 2), hi1 = vld1q_u8(p1 + i / 2 + 16);
                    a1 = vdotq_s32(a1, vqtbl1q_s8(cb, vandq_u8(lo1, mask)), x0);
                    b1 = vdotq_s32(b1, vqtbl1q_s8(cb, vshrq_n_u8(lo1, 4)), x1);
                    c1 = vdotq_s32(c1, vqtbl1q_s8(cb, vandq_u8(hi1, mask)), x2);
                    d1 = vdotq_s32(d1, vqtbl1q_s8(cb, vshrq_n_u8(hi1, 4)), x3);

                    const auto lo2 = vld1q_u8(p2 + i / 2), hi2 = vld1q_u8(p2 + i / 2 + 16);
                    a2 = vdotq_s32(a2, vqtbl1q_s8(cb, vandq_u8(lo2, mask)), x0);
                    b2 = vdotq_s32(b2, vqtbl1q_s8(cb, vshrq_n_u8(lo2, 4)), x1);
                    c2 = vdotq_s32(c2, vqtbl1q_s8(cb, vandq_u8(hi2, mask)), x2);
                    d2 = vdotq_s32(d2, vqtbl1q_s8(cb, vshrq_n_u8(hi2, 4)), x3);

                    const auto lo3 = vld1q_u8(p3 + i / 2), hi3 = vld1q_u8(p3 + i / 2 + 16);
                    a3 = vdotq_s32(a3, vqtbl1q_s8(cb, vandq_u8(lo3, mask)), x0);
                    b3 = vdotq_s32(b3, vqtbl1q_s8(cb, vshrq_n_u8(lo3, 4)), x1);
                    c3 = vdotq_s32(c3, vqtbl1q_s8(cb, vandq_u8(hi3, mask)), x2);
                    d3 = vdotq_s32(d3, vqtbl1q_s8(cb, vshrq_n_u8(hi3, 4)), x3);
                }
                const float xs = x.scales[g];
                int32_t dot0 = vaddvq_s32(vaddq_s32(vaddq_s32(a0, b0), vaddq_s32(c0, d0)));
                int32_t dot1 = vaddvq_s32(vaddq_s32(vaddq_s32(a1, b1), vaddq_s32(c1, d1)));
                int32_t dot2 = vaddvq_s32(vaddq_s32(vaddq_s32(a2, b2), vaddq_s32(c2, d2)));
                int32_t dot3 = vaddvq_s32(vaddq_s32(vaddq_s32(a3, b3), vaddq_s32(c3, d3)));

                val0 += float(dot0) * norm_scales[r0_idx * groups + g] * xs;
                val1 += float(dot1) * norm_scales[r1_idx * groups + g] * xs;
                val2 += float(dot2) * norm_scales[r2_idx * groups + g] * xs;
                val3 += float(dot3) * norm_scales[r3_idx * groups + g] * xs;
            }
        }
        out[0] = val0;
        out[1] = val1;
        out[2] = val2;
        out[3] = val3;
#else
        out[0] = row(r + 0, x);
        out[1] = row(r + 1, x);
        out[2] = row(r + 2, x);
        out[3] = row(r + 3, x);
#endif
    }

    void multiply(const Prepared&x,float*y,int threads=1)const {
        int n4 = rows / 4;
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) if(threads>1&&rows>=128) schedule(static)
#endif
        for(int b=0;b<n4;++b)row4(b*4,x,y+b*4);
        for(int r=n4*4;r<rows;++r)y[r]=row(r,x);
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
