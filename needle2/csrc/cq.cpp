// Open implementation of the public Needle CQ format. No vendor library required.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#ifdef __aarch64__
#include <arm_neon.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

static inline float half_float(uint16_t h) {
#ifdef __aarch64__
    __fp16 v; std::memcpy(&v, &h, 2); return float(v);
#else
    uint32_t sign = uint32_t(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 31, mant = h & 1023, result;
    if (!exp) {
        if (!mant) result = sign;
        else { int e = -14; while (!(mant & 1024)) { mant <<= 1; --e; }
               result = sign | uint32_t(e + 127) << 23 | (mant & 1023) << 13; }
    } else if (exp == 31) result = sign | 0x7f800000 | (mant << 13);
    else result = sign | ((exp + 112) << 23) | (mant << 13);
    float f; std::memcpy(&f, &result, 4); return f;
#endif
}

static void hadamard(float *v, int n) {
    for (int h = 1; h < n; h *= 2) {
        for (int base = 0; base < n; base += 2*h) {
            int j = 0;
#ifdef __aarch64__
            for (; j + 4 <= h; j += 4) {
                auto a=vld1q_f32(v+base+j), b=vld1q_f32(v+base+j+h);
                vst1q_f32(v+base+j,vaddq_f32(a,b));
                vst1q_f32(v+base+j+h,vsubq_f32(a,b));
            }
#endif
            for (; j<h; ++j) { float a=v[base+j], b=v[base+j+h]; v[base+j]=a+b; v[base+j+h]=a-b; }
        }
    }
    float scale=1/std::sqrt(float(n));
    for (int i=0; i<n; ++i) v[i]*=scale;
}

static inline unsigned index_at(const uint8_t *p, int i, int bits) {
    int bit=i*bits, byte=bit>>3, shift=bit&7;
    unsigned word=p[byte];
    if (shift+bits>8) word|=unsigned(p[byte+1])<<8;
    return (word>>shift)&((1u<<bits)-1);
}

struct CQ {
    const uint8_t *packed;
    const uint16_t *norms;
    int out, in, bits, group, padded, groups, rowbytes, storage_bits;
    std::vector<float> cb;
    alignas(64) float lut[256][4];
    CQ(const uint8_t *p,const uint16_t *n,int o,int i,int b,int g,const float *c):
        packed(p),norms(n),out(o),in(i),bits(b),group(g),padded((i+g-1)/g*g),
        groups(padded/g),storage_bits(b==5?2:b) {
        rowbytes=padded*storage_bits/8;
        cb.assign(c,c+(b==5?3:1<<b));
        for (int byte=0;byte<256;++byte) for(int j=0;j<4;++j) {
            int idx=(byte>>(2*j))&3;
            lut[byte][j]=b==4 ? cb[(byte>>(4*(j%2)))&15] : b==5 ? (idx==3?cb[0]:idx==0?cb[1]:cb[2]) : cb[idx];
        }
    }
    float weight(const uint8_t *p,int i) const {
        unsigned idx=index_at(p,i,storage_bits);
        return bits==5 ? cb[idx==3?0:idx+1] : cb[idx];
    }
    float group_dot(const uint8_t *p,const float *x) const {
        int i=0;
#ifdef __aarch64__
        float32x4_t a=vdupq_n_f32(0),b=vdupq_n_f32(0),c=vdupq_n_f32(0),d=vdupq_n_f32(0);
        if (storage_bits==2) {
            for (;i+16<=group;i+=16) {
                a=vfmaq_f32(a,vld1q_f32(lut[p[i/4]]),vld1q_f32(x+i));
                b=vfmaq_f32(b,vld1q_f32(lut[p[i/4+1]]),vld1q_f32(x+i+4));
                c=vfmaq_f32(c,vld1q_f32(lut[p[i/4+2]]),vld1q_f32(x+i+8));
                d=vfmaq_f32(d,vld1q_f32(lut[p[i/4+3]]),vld1q_f32(x+i+12));
            }
        } else if (storage_bits==4) {
            for (;i+16<=group;i+=16) {
                auto w0=vcombine_f32(vld1_f32(lut[p[i/2]]),vld1_f32(lut[p[i/2+1]]));
                auto w1=vcombine_f32(vld1_f32(lut[p[i/2+2]]),vld1_f32(lut[p[i/2+3]]));
                auto w2=vcombine_f32(vld1_f32(lut[p[i/2+4]]),vld1_f32(lut[p[i/2+5]]));
                auto w3=vcombine_f32(vld1_f32(lut[p[i/2+6]]),vld1_f32(lut[p[i/2+7]]));
                a=vfmaq_f32(a,w0,vld1q_f32(x+i));
                b=vfmaq_f32(b,w1,vld1q_f32(x+i+4));
                c=vfmaq_f32(c,w2,vld1q_f32(x+i+8));
                d=vfmaq_f32(d,w3,vld1q_f32(x+i+12));
            }
        } else {
            for (;i+4<=group;i+=4) {
                float w[4]={weight(p,i),weight(p,i+1),weight(p,i+2),weight(p,i+3)};
                a=vfmaq_f32(a,vld1q_f32(w),vld1q_f32(x+i));
            }
        }
        float sum=vaddvq_f32(vaddq_f32(vaddq_f32(a,b),vaddq_f32(c,d)));
#else
        float sum=0;
#endif
        for(;i<group;++i) sum+=weight(p,i)*x[i];
        return sum;
    }
    void transform(const float *x,float *rot) const {
        std::copy(x,x+in,rot);std::fill(rot+in,rot+padded,0);
        for(int g=0;g<groups;++g) hadamard(rot+g*group,group);
    }
    float dot_row(int row,const float *rot) const {
        float sum=0;
        const auto *p=packed+size_t(row)*rowbytes;
        const auto *s=norms+size_t(row)*groups;
        for(int g=0;g<groups;++g)
            sum+=group_dot(p+g*group*storage_bits/8,rot+g*group)*half_float(s[g]);
        return sum;
    }
    void linear(const float *x,float *y,int batch,int threads) const {
        std::vector<float> rot(size_t(batch)*padded);
        for(int b=0;b<batch;++b) transform(x+size_t(b)*in,rot.data()+size_t(b)*padded);
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) if(threads > 1) schedule(static)
#endif
        for(int r=0;r<batch*out;++r) y[r]=dot_row(r%out,rot.data()+size_t(r/out)*padded);
    }
    // Activation-dependent exact FP32 sums. Modes 1/3 use one 256-entry
    // table per packed byte; mode 2 uses 16-entry tables per 2 CQ2 values.
    int activation_table_stride(int mode)const{return bits==2&&mode==2?16:256;}
    int activation_table_count(int mode)const{return padded/(bits==2&&mode!=2?4:2);}
    void build_activation_table(const float *rot,float *table,int mode,int length=0)const {
        if(!length)length=padded;
        int stride=activation_table_stride(mode), width=bits==2&&mode!=2?4:2;
        for(int i=0;i<length;i+=width) {
            float *dst=table+size_t(i/width)*stride;
            if(bits==2&&width==4) {
                float lo[16],hi[16];
                for(int j=0;j<16;++j){lo[j]=cb[j&3]*rot[i]+cb[j>>2]*rot[i+1];hi[j]=cb[j&3]*rot[i+2]+cb[j>>2]*rot[i+3];}
                for(int j=0;j<16;++j)for(int k=0;k<16;++k)dst[(j<<4)|k]=lo[k]+hi[j];
            } else {
                int n=bits==2?4:16;
                float lo[16],hi[16];
                for(int j=0;j<n;++j){lo[j]=cb[j]*rot[i];hi[j]=cb[j]*rot[i+1];}
                for(int j=0;j<n;++j)for(int k=0;k<n;++k)dst[j*n+k]=lo[k]+hi[j];
            }
        }
    }
    float activation_group_dot(const uint8_t *p,const float *table,int mode)const {
        float a=0,b=0,c=0,d=0;
        int bytes=group*bits/8;
        if(bits==2&&mode==2) {
            for(int i=0;i<bytes;i+=4) {
                a+=table[(i*2)*16+(p[i]&15)]+table[(i*2+1)*16+(p[i]>>4)];
                b+=table[(i*2+2)*16+(p[i+1]&15)]+table[(i*2+3)*16+(p[i+1]>>4)];
                c+=table[(i*2+4)*16+(p[i+2]&15)]+table[(i*2+5)*16+(p[i+2]>>4)];
                d+=table[(i*2+6)*16+(p[i+3]&15)]+table[(i*2+7)*16+(p[i+3]>>4)];
            }
        } else {
            for(int i=0;i<bytes;i+=4) {
                a+=table[i*256+p[i]];b+=table[(i+1)*256+p[i+1]];
                c+=table[(i+2)*256+p[i+2]];d+=table[(i+3)*256+p[i+3]];
            }
        }
        return (a+b)+(c+d);
    }
    void activation_linear_prebuilt(const float *table,float*y,int threads,int mode,int first=0,int rows=-1)const {
        if(rows<0)rows=out;
        int group_table=activation_table_count(mode)/groups*activation_table_stride(mode);
#ifdef _OPENMP
#pragma omp parallel num_threads(threads) if(threads > 1)
#endif
        {
#ifdef _OPENMP
            int tid=omp_get_thread_num(),nt=omp_get_num_threads();
#else
            int tid=0,nt=1;
#endif
            int begin=rows*tid/nt,end=rows*(tid+1)/nt;
            if(mode==3) {
                std::fill(y+begin,y+end,0);
                for(int g=0;g<groups;++g)for(int r=begin;r<end;++r) {
                    const auto *p=packed+size_t(first+r)*rowbytes+g*group*bits/8;
                    y[r]+=activation_group_dot(p,table+g*group_table,mode)*half_float(norms[size_t(first+r)*groups+g]);
                }
            } else {
                for(int r=begin;r<end;++r) {
                    float val=0;
                    for(int g=0;g<groups;++g) {
                        const auto *p=packed+size_t(first+r)*rowbytes+g*group*bits/8;
                        val+=activation_group_dot(p,table+g*group_table,mode)*half_float(norms[size_t(first+r)*groups+g]);
                    }
                    y[r]=val;
                }
            }
        }
    }
    void activation_linear(const float*x,float*y,int batch,int threads,int mode)const {
        if((bits!=2&&bits!=4)||group<16){linear(x,y,batch,threads);return;}
        std::vector<float> rot(padded),table(size_t(activation_table_count(mode))*activation_table_stride(mode));
        for(int b=0;b<batch;++b) {
            transform(x+size_t(b)*in,rot.data());build_activation_table(rot.data(),table.data(),mode);
            activation_linear_prebuilt(table.data(),y+size_t(b)*out,threads,mode);
        }
    }
    void row(int r,float *dest) const {
        std::vector<float> tmp(padded);
        for(int g=0;g<groups;++g) {
            float norm=half_float(norms[size_t(r)*groups+g]);
            for(int i=0;i<group;++i) tmp[g*group+i]=weight(packed+size_t(r)*rowbytes,g*group+i)*norm;
            hadamard(tmp.data()+g*group,group);
        }
        std::copy(tmp.data(),tmp.data()+in,dest);
    }
};

static bool compatible_rotation(CQ*const*matrices,int count) {
    for(int i=1;i<count;++i)if(matrices[i]->in!=matrices[0]->in||matrices[i]->group!=matrices[0]->group)return false;
    return true;
}
static void cq_linear_many(CQ*const*matrices,int count,const float*x,float*const*outputs,int threads,int mode,float*rotation,float*table) {
    CQ *first=matrices[0];int total=0;for(int i=0;i<count;++i)total+=matrices[i]->out;
    first->transform(x,rotation);
    if(mode) {
        if((first->bits!=2&&first->bits!=4)||first->group<16)mode=0;
        for(int i=1;i<count&&mode;++i)if(matrices[i]->bits!=first->bits||matrices[i]->cb!=first->cb)mode=0;
    }
    if(mode&&mode!=4)first->build_activation_table(rotation,table,mode);
    int table_group=mode?first->activation_table_count(mode)/first->groups*first->activation_table_stride(mode):0;
#ifdef _OPENMP
#pragma omp parallel num_threads(threads) if(threads > 1)
#endif
    {
#ifdef _OPENMP
        int tid=omp_get_thread_num(),nt=omp_get_num_threads();
#else
        int tid=0,nt=1;
#endif
        int begin=total*tid/nt,end=total*(tid+1)/nt,offset=0;
        if(mode==4) {
            thread_local std::vector<float> local_table;local_table.resize(table_group);
            for(int m=0;m<count;++m){int b=std::max(0,begin-offset),e=std::min(matrices[m]->out,end-offset);offset+=matrices[m]->out;if(b<e)std::fill(outputs[m]+b,outputs[m]+e,0);}
            for(int g=0;g<first->groups;++g) {
                first->build_activation_table(rotation+g*first->group,local_table.data(),mode,first->group);offset=0;
                for(int m=0;m<count;++m) {
                    CQ*q=matrices[m];int b=std::max(0,begin-offset),e=std::min(q->out,end-offset);offset+=q->out;
                    for(int r=b;r<e;++r)outputs[m][r]+=q->activation_group_dot(q->packed+size_t(r)*q->rowbytes+g*q->group*q->bits/8,local_table.data(),mode)*half_float(q->norms[size_t(r)*q->groups+g]);
                }
            }
        } else for(int m=0;m<count;++m) {
            CQ *q=matrices[m];int b=std::max(0,begin-offset),e=std::min(q->out,end-offset);offset+=q->out;
            if(b>=e)continue;
            if(mode==3) {
                std::fill(outputs[m]+b,outputs[m]+e,0);
                for(int g=0;g<q->groups;++g)for(int r=b;r<e;++r) {
                    const auto *p=q->packed+size_t(r)*q->rowbytes+g*q->group*q->bits/8;
                    outputs[m][r]+=q->activation_group_dot(p,table+g*table_group,mode)*half_float(q->norms[size_t(r)*q->groups+g]);
                }
            } else if(mode) {
                for(int r=b;r<e;++r) {
                    float sum=0;
                    for(int g=0;g<q->groups;++g)sum+=q->activation_group_dot(q->packed+size_t(r)*q->rowbytes+g*q->group*q->bits/8,table+g*table_group,mode)*half_float(q->norms[size_t(r)*q->groups+g]);
                    outputs[m][r]=sum;
                }
            } else {
                for(int r=b;r<e;++r)outputs[m][r]=q->dot_row(r,rotation);
            }
        }
    }
}

extern "C" {
const char *needle2_native_features() {
#ifdef __aarch64__
    return "aarch64-neon;float32-accumulate;direct-packed-cq";
#else
    return "portable;float32-accumulate;direct-packed-cq";
#endif
}
void *needle2_cq_create(const uint8_t*p,const uint16_t*n,int out,int in,int bits,int group,const float*cb) {
    try {return new CQ(p,n,out,in,bits,group,cb);} catch(...) {return nullptr;}
}
void needle2_cq_destroy(void *handle) {delete static_cast<CQ*>(handle);}
void needle2_cq_linear(void *handle,const float*x,float*y,int batch,int threads) {
    static_cast<CQ*>(handle)->linear(x,y,batch,threads);
}
void needle2_cq_linear_lookup(void *handle,const float*x,float*y,int batch,int threads,int mode) {
    static_cast<CQ*>(handle)->activation_linear(x,y,batch,threads,mode);
}
void needle2_cq_linear_many(void *const*handles,int count,const float*x,float*const*ys,int threads,int mode) {
    std::vector<CQ*> matrices;for(int i=0;i<count;++i)matrices.push_back(static_cast<CQ*>(handles[i]));
    if(!compatible_rotation(matrices.data(),count)){for(int i=0;i<count;++i)matrices[i]->linear(x,ys[i],1,threads);return;}
    CQ*first=matrices[0];std::vector<float> rot(first->padded),table;
    if(mode&&mode!=4)table.resize(size_t(first->activation_table_count(mode))*first->activation_table_stride(mode));
    cq_linear_many(matrices.data(),count,x,ys,threads,mode,rot.data(),table.data());
}
void needle2_cq_rows(void *handle,const int64_t *ids,float*y,int count) {
    auto *q=static_cast<CQ*>(handle);
    for(int i=0;i<count;++i) q->row(int(ids[i]),y+size_t(i)*q->in);
}
}

#include "sdot.cpp"
#include "engine.cpp"
