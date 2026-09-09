// Standalone microbenchmark; compile with -O3 -std=c++17 -pthread -fopenmp.
#include "../needle2/csrc/cq.cpp"
#include <cstdio>
#include <random>

// Keep the runtime dimension unknown in the baseline, as in the original engine.
__attribute__((noinline, noclone)) static void score_generic(const float* q, const int8_t* keys,
        int stride, const int* slots, int length, int dim, float* out, int capacity) {
    for(int s=0;s<length;++s)dot_pair_i8_f32(q,q+dim,keys+slots[s]*stride,
        dim,out[s],out[capacity+s]);
}
__attribute__((noinline, noclone)) static void score_fixed64(const float* q, const int8_t* keys,
        int stride, const int* slots, int length, float* out, int capacity) {
    for(int s=0;s<length;++s)dot_pair_i8_f32(q,q+64,keys+slots[s]*stride,
        64,out[s],out[capacity+s]);
}

int main() {
#ifndef __aarch64__
    return 77;
#else
    std::mt19937 rng(712);
    std::uniform_real_distribution<float> floats(-3,3);
    constexpr int stride=256, capacity=768;
    std::vector<int8_t> keys(capacity*stride);
    std::vector<float> q(128), ref(2*capacity), out(2*capacity);
    std::vector<int> slots(capacity);
    for(auto&k:keys)k=int(rng()%256)-128;
    for(auto&v:q)v=floats(rng);
    volatile float checksum=0;
    std::puts("length,baseline_us,fixed64_us,speedup");
    for(int length: {1,3,4,5,17,177,256,418,674}) {
        // Pinned prefix followed by a wrapped recent-token ring, with head offset.
        for(int s=0;s<length;++s)slots[s]=s<9?s:9+(s-9+131)%(capacity-9);
        auto run=[&](bool fixed,float*result) {
            if(fixed)score_fixed64(q.data(),keys.data()+64,stride,slots.data(),length,result,capacity);
            else score_generic(q.data(),keys.data()+64,stride,slots.data(),length,64,result,capacity);
        };
        run(false,ref.data());run(true,out.data());
        for(int h=0;h<2;++h)for(int s=0;s<length;++s)
            if(std::memcmp(&ref[h*capacity+s],&out[h*capacity+s],sizeof(float)))return 1;
        std::vector<double> times[2];
        for(int rep=0;rep<23;++rep)for(int turn=0;turn<2;++turn) {
            int variant=(rep+turn)%2;
            auto start=std::chrono::steady_clock::now();
            for(int i=0;i<500;++i){run(variant,out.data());checksum=out[i%length];}
            double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/500;
            if(rep>=2)times[variant].push_back(us);
        }
        for(auto&t:times)std::sort(t.begin(),t.end());
        std::printf("%d,%.6f,%.6f,%.4f\n",length,times[0][10],times[1][10],times[0][10]/times[1][10]);
    }
    (void)checksum;
#endif
}
