// SPDX-License-Identifier: MIT
/* Validate direct CUDA-kernel loads from and stores to peer GPU memory. */

#include <cuda.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK_CUDA(expr)                                                        \
    do {                                                                        \
        CUresult _status = (expr);                                              \
        if (_status != CUDA_SUCCESS) {                                          \
            const char *_name = NULL;                                           \
            const char *_message = NULL;                                        \
            cuGetErrorName(_status, &_name);                                    \
            cuGetErrorString(_status, &_message);                               \
            fprintf(stderr, "%s failed: %s (%s) at %s:%d\n", #expr,            \
                    _name ? _name : "unknown",                                 \
                    _message ? _message : "no description", __FILE__, __LINE__);\
            exit(2);                                                            \
        }                                                                       \
    } while (0)

/*
 * Hand-written baseline PTX keeps this probe independent of nvcc's host-side
 * architecture list. CUDA 13.3 no longer emits Pascal cubins, but the R580
 * driver's JIT still accepts PTX ISA 6.0 for both Pascal and Blackwell.
 */
static const char peer_ptx[] =
    ".version 6.0\n"
    ".target sm_60\n"
    ".address_size 64\n"
    "\n"
    ".visible .entry peer_xor(\n"
    "    .param .u64 dst_param,\n"
    "    .param .u64 src_param,\n"
    "    .param .u64 count_param,\n"
    "    .param .u32 mask_param\n"
    ")\n"
    "{\n"
    "    .reg .pred %p;\n"
    "    .reg .b32 %r<7>;\n"
    "    .reg .b64 %rd<8>;\n"
    "\n"
    "    ld.param.u64 %rd1, [dst_param];\n"
    "    ld.param.u64 %rd2, [src_param];\n"
    "    ld.param.u64 %rd3, [count_param];\n"
    "    ld.param.u32 %r4, [mask_param];\n"
    "    mov.u32 %r1, %ctaid.x;\n"
    "    mov.u32 %r2, %ntid.x;\n"
    "    mov.u32 %r3, %tid.x;\n"
    "    mad.lo.s32 %r5, %r1, %r2, %r3;\n"
    "    cvt.u64.u32 %rd4, %r5;\n"
    "    setp.ge.u64 %p, %rd4, %rd3;\n"
    "    @%p bra DONE;\n"
    "    shl.b64 %rd5, %rd4, 2;\n"
    "    add.u64 %rd6, %rd2, %rd5;\n"
    "    add.u64 %rd7, %rd1, %rd5;\n"
    "    ld.global.u32 %r6, [%rd6];\n"
    "    xor.b32 %r6, %r6, %r4;\n"
    "    st.global.u32 [%rd7], %r6;\n"
    "DONE:\n"
    "    ret;\n"
    "}\n";

static void set_current(CUcontext context)
{
    CHECK_CUDA(cuCtxSetCurrent(context));
}

static void fill_buffer(CUcontext context, CUdeviceptr buffer,
                        const uint32_t *input, size_t bytes)
{
    set_current(context);
    CHECK_CUDA(cuMemcpyHtoD(buffer, input, bytes));
    /* The peer context does not wait for this pageable-host transfer. */
    CHECK_CUDA(cuCtxSynchronize());
}

static void fill_sentinel(CUcontext context, CUdeviceptr buffer, size_t bytes)
{
    set_current(context);
    CHECK_CUDA(cuMemsetD8(buffer, 0xa5, bytes));
    CHECK_CUDA(cuCtxSynchronize());
}

static int validate_result(const char *label, const uint32_t *actual,
                           const uint32_t *input, uint32_t mask, size_t words)
{
    size_t mismatches = 0;

    for (size_t i = 0; i < words; ++i) {
        uint32_t expected = input[i] ^ mask;
        if (actual[i] != expected) {
            if (mismatches < 8) {
                fprintf(stderr,
                        "%s mismatch at word %zu: expected 0x%08" PRIx32
                        ", got 0x%08" PRIx32 "\n",
                        label, i, expected, actual[i]);
            }
            ++mismatches;
        }
    }

    if (mismatches != 0) {
        fprintf(stderr, "%s: FAIL (%zu corrupt words out of %zu)\n",
                label, mismatches, words);
        return 1;
    }

    printf("%s: PASS (%zu bytes checked)\n", label, words * sizeof(*actual));
    return 0;
}

static int launch_and_validate(const char *label, CUcontext execution_context,
                               CUfunction function, CUdeviceptr destination,
                               CUdeviceptr source, CUcontext result_context,
                               uint32_t *result, const uint32_t *input,
                               uint32_t mask, size_t words)
{
    uint64_t count = words;
    unsigned int block_x = 256;
    unsigned int grid_x = (unsigned int)((words + block_x - 1) / block_x);
    void *arguments[] = {&destination, &source, &count, &mask};

    set_current(execution_context);
    CHECK_CUDA(cuLaunchKernel(function, grid_x, 1, 1, block_x, 1, 1,
                              0, NULL, arguments, NULL));
    CHECK_CUDA(cuCtxSynchronize());

    set_current(result_context);
    CHECK_CUDA(cuMemcpyDtoH(result, destination, words * sizeof(*result)));
    return validate_result(label, result, input, mask, words);
}

int main(int argc, char **argv)
{
    size_t bytes = 16U * 1024U * 1024U;
    size_t words;
    CUdevice devices[2];
    CUcontext contexts[2];
    CUmodule modules[2];
    CUfunction functions[2];
    CUdeviceptr buffers[2];
    CUctxCreateParams context_params = {0};
    char names[2][128];
    uint32_t *input;
    uint32_t *result;
    int failures = 0;

    if (argc == 2) {
        char *end = NULL;
        unsigned long long requested = strtoull(argv[1], &end, 0);
        if (!end || *end != '\0' || requested == 0 || requested % 4 != 0) {
            fprintf(stderr, "Usage: %s [byte-count-multiple-of-4]\n", argv[0]);
            return 2;
        }
        bytes = (size_t)requested;
    } else if (argc != 1) {
        fprintf(stderr, "Usage: %s [byte-count-multiple-of-4]\n", argv[0]);
        return 2;
    }

    words = bytes / sizeof(uint32_t);
    input = malloc(bytes);
    result = malloc(bytes);
    if (!input || !result) {
        fprintf(stderr, "Host allocation failed for %zu bytes\n", bytes);
        return 2;
    }

    for (size_t i = 0; i < words; ++i)
        input[i] = (uint32_t)(0x9e3779b9U * (uint32_t)(i + 1)) ^ 0x51f15e5dU;

    CHECK_CUDA(cuInit(0));
    int device_count = 0;
    CHECK_CUDA(cuDeviceGetCount(&device_count));
    if (device_count < 2) {
        fprintf(stderr, "Need two CUDA devices; found %d\n", device_count);
        return 2;
    }

    for (int i = 0; i < 2; ++i) {
        CHECK_CUDA(cuDeviceGet(&devices[i], i));
        CHECK_CUDA(cuDeviceGetName(names[i], sizeof(names[i]), devices[i]));
        CHECK_CUDA(cuCtxCreate(&contexts[i], &context_params, 0, devices[i]));
        set_current(contexts[i]);

        char error_log[8192] = {0};
        unsigned int log_size = sizeof(error_log);
        CUjit_option options[] = {CU_JIT_ERROR_LOG_BUFFER,
                                  CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
        void *values[] = {error_log, (void *)(uintptr_t)log_size};
        CUresult load_status = cuModuleLoadDataEx(&modules[i], peer_ptx,
                                                  2, options, values);
        if (load_status != CUDA_SUCCESS) {
            const char *error_name = NULL;
            cuGetErrorName(load_status, &error_name);
            fprintf(stderr, "PTX JIT failed on GPU%d (%s): %s\n%s\n",
                    i, names[i], error_name ? error_name : "unknown", error_log);
            return 2;
        }
        CHECK_CUDA(cuModuleGetFunction(&functions[i], modules[i], "peer_xor"));
        CHECK_CUDA(cuMemAlloc(&buffers[i], bytes));
    }

    set_current(contexts[0]);
    CHECK_CUDA(cuCtxEnablePeerAccess(contexts[1], 0));
    set_current(contexts[1]);
    CHECK_CUDA(cuCtxEnablePeerAccess(contexts[0], 0));

    printf("Direct peer-kernel probe: GPU0=%s, GPU1=%s, bytes=%zu\n",
           names[0], names[1], bytes);

    for (int executing = 0; executing < 2; ++executing) {
        int remote = 1 - executing;
        uint32_t read_mask = 0x13579bdfU ^ (uint32_t)executing;
        uint32_t write_mask = 0x2468ace0U ^ (uint32_t)executing;
        char label[256];

        /* The executing GPU reads from its peer and writes to local memory. */
        fill_buffer(contexts[remote], buffers[remote], input, bytes);
        fill_sentinel(contexts[executing], buffers[executing], bytes);
        snprintf(label, sizeof(label),
                 "GPU%d kernel remote-read GPU%d -> local GPU%d",
                 executing, remote, executing);
        failures += launch_and_validate(label, contexts[executing],
                                        functions[executing], buffers[executing],
                                        buffers[remote], contexts[executing], result,
                                        input, read_mask, words);

        /* The executing GPU reads local memory and writes to its peer. */
        fill_buffer(contexts[executing], buffers[executing], input, bytes);
        fill_sentinel(contexts[remote], buffers[remote], bytes);
        snprintf(label, sizeof(label),
                 "GPU%d kernel remote-write local GPU%d -> GPU%d",
                 executing, executing, remote);
        failures += launch_and_validate(label, contexts[executing],
                                        functions[executing], buffers[remote],
                                        buffers[executing], contexts[remote], result,
                                        input, write_mask, words);
    }

    for (int i = 1; i >= 0; --i) {
        set_current(contexts[i]);
        CHECK_CUDA(cuMemFree(buffers[i]));
        CHECK_CUDA(cuModuleUnload(modules[i]));
        CHECK_CUDA(cuCtxDestroy(contexts[i]));
    }

    free(result);
    free(input);
    printf("DIRECT PEER KERNEL PROBE: %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
