// Experimental approximate CQ matvec. Not used by the production FP32 engine.
// Requires ARMv8.2-A DotProd; x and codebook centroids are rounded to INT8.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <new>
#include <vector>
#include <arm_neon.h>
#include <sys/auxv.h>
#include <asm/hwcap.h>

static bool runtime_dotprod() { return (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0; }

static float half_value(uint16_t h) {
    __fp16 value;
    std::memcpy(&value, &h, sizeof(h));
    return float(value);
}

static void walsh(float *v, int n) {
    for (int h = 1; h < n; h *= 2) {
        for (int base = 0; base < n; base += 2 * h) {
            int i = 0;
            for (; i + 4 <= h; i += 4) {
                const auto a = vld1q_f32(v + base + i), b = vld1q_f32(v + base + i + h);
                vst1q_f32(v + base + i, vaddq_f32(a, b));
                vst1q_f32(v + base + i + h, vsubq_f32(a, b));
            }
            for (; i < h; ++i) {
                float a = v[base + i], b = v[base + i + h];
                v[base + i] = a + b;
                v[base + i + h] = a - b;
            }
        }
    }
    const auto scale = vdupq_n_f32(1 / std::sqrt(float(n)));
    for (int i = 0; i < n; i += 4) vst1q_f32(v + i, vmulq_f32(vld1q_f32(v + i), scale));
}

struct Prepared {
    std::vector<int8_t> q;
    std::vector<float> scales;
    Prepared(int padded, int groups): q(padded), scales(groups) {}
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
            norm_scales[i] = half_value(norms[i]) * centroid_scale;
    }

    void prepare(const float *input, Prepared &output) const {
        std::vector<float> rotated(padded, 0);
        std::copy(input, input + columns, rotated.begin());
        std::vector<int8_t> quantized(group);
        const auto lower = vdupq_n_f32(-127), upper = vdupq_n_f32(127);
        for (int g = 0; g < groups; ++g) {
            float *x = rotated.data() + g * group;
            walsh(x, group);
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
            int8_t *dst = output.q.data() + g * group;
            // Permute x once, making packed weight decoding simple bit planes.
            // SDOT sees identical permutations on both sides of each product.
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
        }
    }

    int32_t dot_group(const uint8_t *p, const int8_t *x) const {
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
    }

    void multiply(const Prepared &x, float *y) const {
        for (int row = 0; row < rows; ++row) {
            float value = 0;
            for (int g = 0; g < groups; ++g) {
                const auto *p = packed + size_t(row) * rowbytes + g * group * bits / 8;
                const auto dot = dot_group(p, x.q.data() + g * group);
                value += float(dot) * norm_scales[size_t(row) * groups + g] * x.scales[g];
            }
            y[row] = value;
        }
    }

    void linear(const float *x, float *y) const {
        Prepared prepared(padded, groups);
        prepare(x, prepared);
        multiply(prepared, y);
    }
};

extern "C" {
int needle2_sdot_supported() { return runtime_dotprod(); }
void *needle2_sdot_create(const uint8_t *p, const uint16_t *n, int out, int in, int bits, int group, const float *cb) {
    if (!runtime_dotprod() || out <= 0 || in <= 0 || (bits != 2 && bits != 4)
        || group < 64 || (group & (group - 1))) return nullptr;
    try { return new SdotCQ(p, n, out, in, bits, group, cb); }
    catch (...) { return nullptr; }
}
void needle2_sdot_destroy(void *handle) { delete static_cast<SdotCQ *>(handle); }
void needle2_sdot_linear(void *handle, const float *x, float *y) { static_cast<SdotCQ *>(handle)->linear(x, y); }
void *needle2_sdot_prepare(void *handle, const float *x) {
    auto *matrix = static_cast<SdotCQ *>(handle);
    auto *prepared = new Prepared(matrix->padded, matrix->groups);
    matrix->prepare(x, *prepared);
    return prepared;
}
void needle2_sdot_prepared_destroy(void *prepared) { delete static_cast<Prepared *>(prepared); }
void needle2_sdot_multiply(void *handle, void *prepared, float *y) {
    static_cast<SdotCQ *>(handle)->multiply(*static_cast<Prepared *>(prepared), y);
}
}
