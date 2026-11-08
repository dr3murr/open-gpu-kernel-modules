// Handoff round-trip latency between GPU0 (5080) and GPU1 (P100), three ways:
//  A  host-mapped flags, both sides persistent kernels, no CPU in the loop
//  B  P2P: each side's flag lives in the other card's VRAM (peer writes, local polls)
//  C  CPU in the loop, as Basalt's tier: GPU0 rings a host flag, the CPU sees it, launches a P100 kernel, waits
//     for it, then rings GPU0 (the CPU-side launch + completion + relay)
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <thread>
#include <atomic>
#include <cstring>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

#if __CUDA_ARCH__ >= 700
__device__ __forceinline__ unsigned ldcv(const unsigned* p) { unsigned v; asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void stv(unsigned* p, unsigned v) { asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
#else
__device__ __forceinline__ unsigned ldcv(const unsigned* p) { unsigned v; asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void stv(unsigned* p, unsigned v) { asm volatile("st.volatile.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
#endif
__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }

// GPU0 side: ring `out`, wait for `in` to reach i; returns ns per round trip in *res
__global__ void pinger(unsigned* out, const unsigned* in, int n, unsigned long long* res) {
    const unsigned long long t0 = gtime();
    for (int i = 1; i <= n; ++i) {
        stv(out, (unsigned) i);
        __threadfence_system();
        const unsigned long long w0 = gtime();
        while (ldcv(in) < (unsigned) i)
            if (gtime() - w0 > 2000000000ull) { res[1] = i; return; }   // 2 s: give up
    }
    res[0] = (gtime() - t0) / (unsigned long long) n;
}
// GPU1 side (persistent): wait for `in` to reach i, ring `out`
__global__ void ponger(const unsigned* in, unsigned* out, int n) {
    for (int i = 1; i <= n; ++i) {
        const unsigned long long w0 = gtime();
        while (ldcv(in) < (unsigned) i)
            if (gtime() - w0 > 2000000000ull) return;
        stv(out, (unsigned) i);
        __threadfence_system();
    }
}
// C: the P100 kernel the CPU launches per handoff (one thread, writes the reply word)
__global__ void p100_step(unsigned* out, unsigned v) { stv(out, v); __threadfence_system(); }

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const char* only = argc > 1 ? argv[1] : "ABC";
    const int N = 20000;
    unsigned long long* res;   // mapped host result words
    CK(cudaHostAlloc(&res, 16, cudaHostAllocMapped | cudaHostAllocPortable));

    // ---- A: host-mapped flags, no CPU in the loop
    if (strchr(only, 'A')) {
        unsigned *h; CK(cudaHostAlloc(&h, 64, cudaHostAllocMapped | cudaHostAllocPortable));
        h[0] = h[8] = 0; res[0] = res[1] = 0;
        cudaStream_t s0, s1;
        CK(cudaSetDevice(1)); CK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
        ponger<<<1, 1, 0, s1>>>(h, h + 8, N);
        CK(cudaSetDevice(0)); CK(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking));
        pinger<<<1, 1, 0, s0>>>(h, h + 8, N, res);
        CK(cudaStreamSynchronize(s0)); CK(cudaSetDevice(1)); CK(cudaStreamSynchronize(s1));
        printf("A host-mapped, GPU-driven       : %.2f us round trip (timeout at %llu)\n", res[0] / 1000.0, res[1]);
    }
    // ---- B: P2P flags
    if (strchr(only, 'B')) {
        int a = 0, b = 0;
        CK(cudaDeviceCanAccessPeer(&a, 0, 1)); CK(cudaDeviceCanAccessPeer(&b, 1, 0));
        printf("peer access possible: 0->1 %d, 1->0 %d\n", a, b);
        if (a && b) {
            CK(cudaSetDevice(0)); cudaDeviceEnablePeerAccess(1, 0); cudaGetLastError();
            CK(cudaSetDevice(1)); cudaDeviceEnablePeerAccess(0, 0); cudaGetLastError();
            unsigned *f0, *f1;   // f0 in the 5080's VRAM (the P100 writes it), f1 in the P100's (the 5080 writes it)
            CK(cudaSetDevice(0)); CK(cudaMalloc(&f0, 64)); CK(cudaMemset(f0, 0, 64));
            CK(cudaSetDevice(1)); CK(cudaMalloc(&f1, 64)); CK(cudaMemset(f1, 0, 64));
            res[0] = res[1] = 0;
            cudaStream_t s0, s1;
            CK(cudaSetDevice(1)); CK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
            ponger<<<1, 1, 0, s1>>>(f1, f0, N);
            CK(cudaSetDevice(0)); CK(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking));
            pinger<<<1, 1, 0, s0>>>(f1, f0, N, res);
            CK(cudaStreamSynchronize(s0)); CK(cudaSetDevice(1)); CK(cudaStreamSynchronize(s1));
            printf("B P2P flags, GPU-driven         : %.2f us round trip (timeout at %llu)\n", res[0] / 1000.0, res[1]);
        }
    }
    // ---- C: CPU in the loop (GPU0 rings host, CPU launches a P100 kernel and waits, the kernel rings GPU0)
    if (strchr(only, 'C')) {
        const int NC = 5000;
        unsigned *h; CK(cudaHostAlloc(&h, 64, cudaHostAllocMapped | cudaHostAllocPortable));
        h[0] = h[8] = 0; res[0] = res[1] = 0;
        cudaStream_t s0, s1;
        CK(cudaSetDevice(1)); CK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
        p100_step<<<1, 1, 0, s1>>>(h + 32, 0); CK(cudaStreamSynchronize(s1));   // warm
        CK(cudaSetDevice(0)); CK(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking));
        pinger<<<1, 1, 0, s0>>>(h, h + 8, NC, res);
        CK(cudaSetDevice(1));
        for (int i = 1; i <= NC; ++i) {
            const auto c0 = std::chrono::steady_clock::now();
            while (*(volatile unsigned*) h < (unsigned) i)
                if (std::chrono::steady_clock::now() - c0 > std::chrono::seconds(3)) { printf("C: GPU0 never rang %d\n", i); return 1; }
            p100_step<<<1, 1, 0, s1>>>(h + 8, (unsigned) i);   // the P100 writes the reply itself
            while (cudaStreamQuery(s1) == cudaErrorNotReady) {}   // the CPU waits for completion, as finish() does
        }
        CK(cudaSetDevice(0)); CK(cudaStreamSynchronize(s0));
        printf("C CPU launches + waits per hop  : %.2f us round trip (timeout at %llu)\n", res[0] / 1000.0, res[1]);
    }
    return 0;
}
