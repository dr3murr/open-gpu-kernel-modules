// SPDX-License-Identifier: MIT
/* Probe CUDA V2 stream memory operations on local and peer GPU pointers. */

#include <cuda.h>

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GPU_COUNT 2

#define CHECK_CUDA(expr)                                                        \
    do {                                                                        \
        CUresult _status = (expr);                                              \
        if (_status != CUDA_SUCCESS) {                                          \
            const char *_name = NULL;                                           \
            const char *_message = NULL;                                        \
            cuGetErrorName(_status, &_name);                                    \
            cuGetErrorString(_status, &_message);                               \
            fprintf(stderr, "%s failed: %s (%s) at %s:%d\n", #expr,          \
                    _name ? _name : "unknown",                                \
                    _message ? _message : "no description", __FILE__,         \
                    __LINE__);                                                  \
            exit(2);                                                            \
        }                                                                       \
    } while (0)

typedef struct {
    CUdevice devices[GPU_COUNT];
    CUcontext contexts[GPU_COUNT];
    CUstream streams[GPU_COUNT];
    CUdeviceptr buffers[GPU_COUNT];
    char names[GPU_COUNT][128];
} state_t;

static void set_current(const state_t *state, int gpu)
{
    CHECK_CUDA(cuCtxSetCurrent(state->contexts[gpu]));
}

static const char *result_name(CUresult status)
{
    const char *name = NULL;
    cuGetErrorName(status, &name);
    return name ? name : "unknown";
}

static void set_u64(const state_t *state, int gpu, uint64_t value)
{
    set_current(state, gpu);
    CHECK_CUDA(cuMemcpyHtoD(state->buffers[gpu], &value, sizeof(value)));
    CHECK_CUDA(cuCtxSynchronize());
}

static uint64_t get_u64(const state_t *state, int gpu)
{
    uint64_t value = 0;
    set_current(state, gpu);
    CHECK_CUDA(cuMemcpyDtoH(&value, state->buffers[gpu], sizeof(value)));
    return value;
}

static void test_write32(const state_t *state, int executing, int target)
{
    const uint32_t expected = 0x51a7c0deU;
    set_u64(state, target, 0);
    set_current(state, executing);
    CUresult status = cuStreamWriteValue32(state->streams[executing],
                                           state->buffers[target], expected,
                                           CU_STREAM_WRITE_VALUE_DEFAULT);
    if (status == CUDA_SUCCESS)
        status = cuStreamSynchronize(state->streams[executing]);
    uint32_t actual = (uint32_t)get_u64(state, target);
    printf("GPU%d %-24s WriteValue32 -> GPU%d %-24s: %-28s value=%08" PRIx32
           " %s\n",
           executing, state->names[executing], target, state->names[target],
           result_name(status), actual,
           status == CUDA_SUCCESS && actual == expected ? "PASS" : "FAIL");
}

static void test_wait32(const state_t *state, int executing, int target)
{
    const uint32_t expected = 0x31415926U;
    set_u64(state, target, expected);
    set_current(state, executing);
    CUresult status = cuStreamWaitValue32(
        state->streams[executing], state->buffers[target], expected,
        CU_STREAM_WAIT_VALUE_EQ);
    if (status == CUDA_SUCCESS)
        status = cuStreamSynchronize(state->streams[executing]);
    printf("GPU%d %-24s WaitValue32  on GPU%d %-24s: %s %s\n", executing,
           state->names[executing], target, state->names[target],
           result_name(status), status == CUDA_SUCCESS ? "PASS" : "FAIL");
}

static void test_write64(const state_t *state, int executing, int target)
{
    const uint64_t expected = UINT64_C(0x0123456789abcdef);
    set_u64(state, target, 0);
    set_current(state, executing);
    CUresult status = cuStreamWriteValue64(state->streams[executing],
                                           state->buffers[target], expected,
                                           CU_STREAM_WRITE_VALUE_DEFAULT);
    if (status == CUDA_SUCCESS)
        status = cuStreamSynchronize(state->streams[executing]);
    uint64_t actual = get_u64(state, target);
    const char *verdict = status == CUDA_ERROR_NOT_SUPPORTED
                              ? "EXPECTED-UNSUPPORTED"
                              : (status == CUDA_SUCCESS && actual == expected
                                     ? "PASS"
                                     : "FAIL");
    printf("GPU%d %-24s WriteValue64 -> GPU%d %-24s: %-28s value=%016" PRIx64
           " %s\n",
           executing, state->names[executing], target, state->names[target],
           result_name(status), actual, verdict);
}

static void test_wait64(const state_t *state, int executing, int target)
{
    const uint64_t expected = UINT64_C(0xfedcba9876543210);
    set_u64(state, target, expected);
    set_current(state, executing);
    CUresult status = cuStreamWaitValue64(
        state->streams[executing], state->buffers[target], expected,
        CU_STREAM_WAIT_VALUE_EQ);
    if (status == CUDA_SUCCESS)
        status = cuStreamSynchronize(state->streams[executing]);
    const char *verdict = status == CUDA_ERROR_NOT_SUPPORTED
                              ? "EXPECTED-UNSUPPORTED"
                              : (status == CUDA_SUCCESS ? "PASS" : "FAIL");
    printf("GPU%d %-24s WaitValue64  on GPU%d %-24s: %s %s\n", executing,
           state->names[executing], target, state->names[target],
           result_name(status), verdict);
}

static void test_flush_memop(const state_t *state, int gpu)
{
    CUstreamBatchMemOpParams parameter;
    memset(&parameter, 0, sizeof(parameter));
    parameter.flushRemoteWrites.operation =
        CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES;

    set_current(state, gpu);
    CUresult status = cuStreamBatchMemOp(state->streams[gpu], 1, &parameter, 0);
    if (status == CUDA_SUCCESS)
        status = cuStreamSynchronize(state->streams[gpu]);
    printf("GPU%d %-24s stream FLUSH_REMOTE_WRITES: %s %s\n", gpu,
           state->names[gpu], result_name(status),
           status == CUDA_ERROR_NOT_SUPPORTED ? "EXPECTED-UNSUPPORTED"
                                              : "UNEXPECTED");
}

static void test_host_flush(const state_t *state, int gpu)
{
    set_current(state, gpu);
    CUresult status = cuFlushGPUDirectRDMAWrites(
        CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX,
        CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER);
    printf("GPU%d %-24s host cuFlushGPUDirectRDMAWrites: %s %s\n", gpu,
           state->names[gpu], result_name(status),
           status == CUDA_SUCCESS ? "PASS" : "FAIL");
}

static void print_attributes(const state_t *state, int gpu)
{
    int v1 = 0;
    int v1_64 = 0;
    int current_64 = 0;
    int flush = 0;
    int rdma_options = 0;
    int rdma_ordering = 0;
    CHECK_CUDA(cuDeviceGetAttribute(
        &v1, CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1,
        state->devices[gpu]));
    CHECK_CUDA(cuDeviceGetAttribute(
        &v1_64, CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS_V1,
        state->devices[gpu]));
    CHECK_CUDA(cuDeviceGetAttribute(
        &current_64, CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS,
        state->devices[gpu]));
    CHECK_CUDA(cuDeviceGetAttribute(
        &flush, CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES,
        state->devices[gpu]));
    CHECK_CUDA(cuDeviceGetAttribute(
        &rdma_options,
        CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS,
        state->devices[gpu]));
    CHECK_CUDA(cuDeviceGetAttribute(
        &rdma_ordering,
        CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING,
        state->devices[gpu]));
    printf("GPU%d %s\n", gpu, state->names[gpu]);
    printf("  CAN_USE_STREAM_MEM_OPS_V1=%d\n", v1);
    printf("  CAN_USE_64_BIT_STREAM_MEM_OPS_V1=%d\n", v1_64);
    printf("  CAN_USE_64_BIT_STREAM_MEM_OPS=%d\n", current_64);
    printf("  CAN_FLUSH_REMOTE_WRITES=%d\n", flush);
    printf("  GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS=0x%x\n", rdma_options);
    printf("  GPU_DIRECT_RDMA_WRITES_ORDERING=%d\n", rdma_ordering);
}

static void initialize(state_t *state)
{
    memset(state, 0, sizeof(*state));
    CHECK_CUDA(cuInit(0));
    int count = 0;
    CHECK_CUDA(cuDeviceGetCount(&count));
    if (count != GPU_COUNT) {
        fprintf(stderr, "Expected exactly two visible CUDA GPUs; found %d\n",
                count);
        exit(2);
    }

    CUctxCreateParams context_params = {0};
    for (int gpu = 0; gpu < GPU_COUNT; ++gpu) {
        CHECK_CUDA(cuDeviceGet(&state->devices[gpu], gpu));
        CHECK_CUDA(cuDeviceGetName(state->names[gpu],
                                   (int)sizeof(state->names[gpu]),
                                   state->devices[gpu]));
        CHECK_CUDA(cuCtxCreate(&state->contexts[gpu], &context_params,
                               CU_CTX_SCHED_SPIN, state->devices[gpu]));
        set_current(state, gpu);
        CHECK_CUDA(cuStreamCreate(&state->streams[gpu], CU_STREAM_NON_BLOCKING));
        CHECK_CUDA(cuMemAlloc(&state->buffers[gpu], 8));
    }

    set_current(state, 0);
    CHECK_CUDA(cuCtxEnablePeerAccess(state->contexts[1], 0));
    set_current(state, 1);
    CHECK_CUDA(cuCtxEnablePeerAccess(state->contexts[0], 0));
}

static void cleanup(state_t *state)
{
    for (int gpu = GPU_COUNT - 1; gpu >= 0; --gpu) {
        set_current(state, gpu);
        CHECK_CUDA(cuMemFree(state->buffers[gpu]));
        CHECK_CUDA(cuStreamDestroy(state->streams[gpu]));
        CHECK_CUDA(cuCtxDestroy(state->contexts[gpu]));
    }
}

int main(void)
{
    state_t state;
    initialize(&state);

    for (int gpu = 0; gpu < GPU_COUNT; ++gpu)
        print_attributes(&state, gpu);

    printf("\n32-bit V2 MemOps\n");
    for (int executing = 0; executing < GPU_COUNT; ++executing) {
        test_write32(&state, executing, executing);
        test_wait32(&state, executing, executing);
        test_write32(&state, executing, 1 - executing);
        test_wait32(&state, executing, 1 - executing);
    }

    printf("\n64-bit V2 MemOps\n");
    for (int executing = 0; executing < GPU_COUNT; ++executing) {
        test_write64(&state, executing, executing);
        test_wait64(&state, executing, executing);
        test_write64(&state, executing, 1 - executing);
        test_wait64(&state, executing, 1 - executing);
    }

    printf("\nRemote-write flush mechanisms\n");
    for (int gpu = 0; gpu < GPU_COUNT; ++gpu) {
        test_flush_memop(&state, gpu);
        test_host_flush(&state, gpu);
    }

    cleanup(&state);
    return 0;
}
