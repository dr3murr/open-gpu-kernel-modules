// SPDX-License-Identifier: MIT
/*
 * Measure steady-state device completion latency for RTX 5080 <-> P100 P2P.
 *
 * CUDA events bracket a serial batch in one stream. The reported value is the
 * event elapsed time divided by the number of operations, so it excludes host
 * synchronization/API round-trip time and does not convert the result into
 * bandwidth. Multiple independently synchronized batches provide p05/p50/p95
 * repeatability statistics.
 *
 * Kernel paths operate on half-sized (16-bit) elements. In particular, the
 * 10 KiB case performs exactly 5,120 active 16-bit remote loads or stores.
 */

#define _POSIX_C_SOURCE 200809L
#include <cuda.h>

#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GPU_COUNT 2
#define FUNCTION_READ 0
#define FUNCTION_WRITE 1
#define FUNCTION_COUNT 2
#define MAX_BATCH_ITERATIONS 8192U
#define TARGET_BATCH_MS 10.0f
#define DEFAULT_TRIALS 21

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

/*
 * One active thread handles exactly one 16-bit element.
 *
 * The remote-read kernel uses ld.global.cv so each timed invocation refetches
 * peer memory instead of intentionally retaining a stale matching L2 line.
 * The remote-write kernel uses st.global.wt so stores propagate through the
 * peer mapping instead of measuring only a local write-back cache insertion.
 */
static const char latency_ptx[] =
    ".version 6.0\n"
    ".target sm_60\n"
    ".address_size 64\n"
    "\n"
    ".visible .entry remote_read_half(\n"
    "    .param .u64 dst_local_param,\n"
    "    .param .u64 src_remote_param,\n"
    "    .param .u32 count_param\n"
    ")\n"
    "{\n"
    "    .reg .pred %p;\n"
    "    .reg .b32 %r<8>;\n"
    "    .reg .b64 %rd<8>;\n"
    "    ld.param.u64 %rd1, [dst_local_param];\n"
    "    ld.param.u64 %rd2, [src_remote_param];\n"
    "    ld.param.u32 %r1, [count_param];\n"
    "    mov.u32 %r2, %ctaid.x;\n"
    "    mov.u32 %r3, %ntid.x;\n"
    "    mov.u32 %r4, %tid.x;\n"
    "    mad.lo.s32 %r5, %r2, %r3, %r4;\n"
    "    setp.ge.u32 %p, %r5, %r1;\n"
    "    @%p bra READ_DONE;\n"
    "    cvt.u64.u32 %rd3, %r5;\n"
    "    shl.b64 %rd4, %rd3, 1;\n"
    "    add.u64 %rd5, %rd2, %rd4;\n"
    "    add.u64 %rd6, %rd1, %rd4;\n"
    "    ld.global.cv.u16 %r6, [%rd5];\n"
    "    st.global.u16 [%rd6], %r6;\n"
    "READ_DONE:\n"
    "    ret;\n"
    "}\n"
    "\n"
    ".visible .entry remote_write_half(\n"
    "    .param .u64 dst_remote_param,\n"
    "    .param .u64 src_local_param,\n"
    "    .param .u32 count_param\n"
    ")\n"
    "{\n"
    "    .reg .pred %p;\n"
    "    .reg .b32 %r<8>;\n"
    "    .reg .b64 %rd<8>;\n"
    "    ld.param.u64 %rd1, [dst_remote_param];\n"
    "    ld.param.u64 %rd2, [src_local_param];\n"
    "    ld.param.u32 %r1, [count_param];\n"
    "    mov.u32 %r2, %ctaid.x;\n"
    "    mov.u32 %r3, %ntid.x;\n"
    "    mov.u32 %r4, %tid.x;\n"
    "    mad.lo.s32 %r5, %r2, %r3, %r4;\n"
    "    setp.ge.u32 %p, %r5, %r1;\n"
    "    @%p bra WRITE_DONE;\n"
    "    cvt.u64.u32 %rd3, %r5;\n"
    "    shl.b64 %rd4, %rd3, 1;\n"
    "    add.u64 %rd5, %rd2, %rd4;\n"
    "    add.u64 %rd6, %rd1, %rd4;\n"
    "    ld.global.u16 %r6, [%rd5];\n"
    "    st.global.wt.u16 [%rd6], %r6;\n"
    "WRITE_DONE:\n"
    "    ret;\n"
    "}\n";

typedef struct {
    size_t bytes;
    const char *label;
} payload_t;

static const payload_t payloads[] = {
    {4, "4 B"},
    {64, "64 B"},
    {256, "256 B"},
    {1024, "1 KiB"},
    {4096, "4 KiB"},
    {10 * 1024, "10 KiB / 5120 halfs"},
    {16 * 1024, "16 KiB"},
    {64 * 1024, "64 KiB"},
    {256 * 1024, "256 KiB"},
    {1024 * 1024, "1 MiB"},
};

typedef enum {
    OP_PEER_COPY,
    OP_REMOTE_READ,
    OP_REMOTE_WRITE,
} operation_kind_t;

typedef struct {
    const char *label;
    operation_kind_t kind;
    int executing;
    int source;
    int destination;
} operation_t;

typedef struct {
    CUdevice devices[GPU_COUNT];
    CUcontext contexts[GPU_COUNT];
    CUstream streams[GPU_COUNT];
    CUmodule modules[GPU_COUNT];
    CUfunction functions[GPU_COUNT][FUNCTION_COUNT];
    CUdeviceptr buffers[GPU_COUNT];
    char names[GPU_COUNT][128];
} state_t;

typedef struct {
    double p05_us;
    double p50_us;
    double p95_us;
    unsigned int batch_iterations;
} result_t;

static const operation_t operations[] = {
    {"cuMemcpyPeerAsync P100 -> RTX 5080", OP_PEER_COPY, -1, 1, 0},
    {"cuMemcpyPeerAsync RTX 5080 -> P100", OP_PEER_COPY, -1, 0, 1},
    {"RTX 5080 kernel remote-read P100 -> local", OP_REMOTE_READ, 0, 1, 0},
    {"RTX 5080 kernel remote-write -> P100", OP_REMOTE_WRITE, 0, 0, 1},
    {"P100 kernel remote-read RTX 5080 -> local", OP_REMOTE_READ, 1, 0, 1},
    {"P100 kernel remote-write -> RTX 5080", OP_REMOTE_WRITE, 1, 1, 0},
};

static void set_current(const state_t *state, int gpu)
{
    CHECK_CUDA(cuCtxSetCurrent(state->contexts[gpu]));
}

static int operation_gpu(const operation_t *operation)
{
    if (operation->kind == OP_PEER_COPY)
        return operation->destination;
    return operation->executing;
}

static void enqueue_operation(const state_t *state,
                              const operation_t *operation, size_t bytes)
{
    if (operation->kind == OP_PEER_COPY) {
        CHECK_CUDA(cuMemcpyPeerAsync(
            state->buffers[operation->destination],
            state->contexts[operation->destination],
            state->buffers[operation->source],
            state->contexts[operation->source], bytes,
            state->streams[operation->destination]));
        return;
    }

    uint32_t half_count = (uint32_t)(bytes / sizeof(uint16_t));
    CUdeviceptr destination = state->buffers[operation->destination];
    CUdeviceptr source = state->buffers[operation->source];
    unsigned int block_x = 256;
    unsigned int grid_x = (half_count + block_x - 1) / block_x;
    int function_index = operation->kind == OP_REMOTE_READ ? FUNCTION_READ
                                                            : FUNCTION_WRITE;
    void *arguments[] = {&destination, &source, &half_count};

    CHECK_CUDA(cuLaunchKernel(
        state->functions[operation->executing][function_index], grid_x, 1, 1,
        block_x, 1, 1, 0, state->streams[operation->executing], arguments,
        NULL));
}

static void prepare_and_validate(const state_t *state,
                                 const operation_t *operation,
                                 const uint16_t *input, uint16_t *actual,
                                 size_t bytes)
{
    set_current(state, operation->source);
    CHECK_CUDA(cuMemcpyHtoD(state->buffers[operation->source], input, bytes));
    CHECK_CUDA(cuCtxSynchronize());
    set_current(state, operation->destination);
    CHECK_CUDA(cuMemsetD8(state->buffers[operation->destination], 0xa5, bytes));
    CHECK_CUDA(cuCtxSynchronize());

    set_current(state, operation_gpu(operation));
    enqueue_operation(state, operation, bytes);
    CHECK_CUDA(cuStreamSynchronize(state->streams[operation_gpu(operation)]));

    set_current(state, operation->destination);
    CHECK_CUDA(cuMemcpyDtoH(actual, state->buffers[operation->destination],
                           bytes));
    size_t count = bytes / sizeof(*input);
    for (size_t i = 0; i < count; ++i) {
        if (actual[i] != input[i]) {
            fprintf(stderr,
                    "%s correctness failure at %zu bytes, half %zu: "
                    "expected 0x%04" PRIx16 ", got 0x%04" PRIx16 "\n",
                    operation->label, bytes, i, input[i], actual[i]);
            exit(1);
        }
    }
}

static float time_batch(const state_t *state, const operation_t *operation,
                        size_t bytes, unsigned int iterations, CUevent start,
                        CUevent stop)
{
    int gpu = operation_gpu(operation);
    CUstream stream = state->streams[gpu];

    set_current(state, gpu);
    CHECK_CUDA(cuEventRecord(start, stream));
    for (unsigned int i = 0; i < iterations; ++i)
        enqueue_operation(state, operation, bytes);
    CHECK_CUDA(cuEventRecord(stop, stream));
    CHECK_CUDA(cuEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cuEventElapsedTime(&elapsed_ms, start, stop));
    return elapsed_ms;
}

static int compare_double(const void *left, const void *right)
{
    double a = *(const double *)left;
    double b = *(const double *)right;
    return (a > b) - (a < b);
}

static double percentile(const double *sorted, int count, double p)
{
    double position = p * (double)(count - 1);
    int low = (int)position;
    int high = low + 1 < count ? low + 1 : low;
    double fraction = position - (double)low;
    return sorted[low] + (sorted[high] - sorted[low]) * fraction;
}

static result_t measure_operation(const state_t *state,
                                  const operation_t *operation, size_t bytes,
                                  int trials)
{
    int gpu = operation_gpu(operation);
    CUevent start;
    CUevent stop;
    double *samples = calloc((size_t)trials, sizeof(*samples));
    if (!samples) {
        fprintf(stderr, "Unable to allocate timing samples\n");
        exit(2);
    }

    set_current(state, gpu);
    CHECK_CUDA(cuEventCreate(&start, CU_EVENT_DEFAULT));
    CHECK_CUDA(cuEventCreate(&stop, CU_EVENT_DEFAULT));

    for (unsigned int i = 0; i < 32; ++i)
        enqueue_operation(state, operation, bytes);
    CHECK_CUDA(cuStreamSynchronize(state->streams[gpu]));

    unsigned int iterations = 1;
    while (iterations < MAX_BATCH_ITERATIONS) {
        float elapsed_ms = time_batch(state, operation, bytes, iterations,
                                      start, stop);
        if (elapsed_ms >= TARGET_BATCH_MS)
            break;
        iterations *= 2;
    }

    for (int trial = 0; trial < trials; ++trial) {
        float elapsed_ms = time_batch(state, operation, bytes, iterations,
                                      start, stop);
        samples[trial] = (double)elapsed_ms * 1000.0 / (double)iterations;
    }

    qsort(samples, (size_t)trials, sizeof(*samples), compare_double);
    result_t result = {
        .p05_us = percentile(samples, trials, 0.05),
        .p50_us = percentile(samples, trials, 0.50),
        .p95_us = percentile(samples, trials, 0.95),
        .batch_iterations = iterations,
    };

    CHECK_CUDA(cuEventDestroy(stop));
    CHECK_CUDA(cuEventDestroy(start));
    free(samples);
    return result;
}

static void load_module_for_gpu(state_t *state, int gpu)
{
    char error_log[8192] = {0};
    unsigned int log_size = sizeof(error_log);
    CUjit_option options[] = {CU_JIT_ERROR_LOG_BUFFER,
                              CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
    void *values[] = {error_log, (void *)(uintptr_t)log_size};

    set_current(state, gpu);
    CUresult status = cuModuleLoadDataEx(&state->modules[gpu], latency_ptx, 2,
                                         options, values);
    if (status != CUDA_SUCCESS) {
        const char *name = NULL;
        cuGetErrorName(status, &name);
        fprintf(stderr, "PTX JIT failed on GPU%d (%s): %s\n%s\n", gpu,
                state->names[gpu], name ? name : "unknown", error_log);
        exit(2);
    }
    CHECK_CUDA(cuModuleGetFunction(&state->functions[gpu][FUNCTION_READ],
                                   state->modules[gpu], "remote_read_half"));
    CHECK_CUDA(cuModuleGetFunction(&state->functions[gpu][FUNCTION_WRITE],
                                   state->modules[gpu], "remote_write_half"));
}

static void initialize(state_t *state, size_t allocation_bytes)
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
        CHECK_CUDA(cuMemAlloc(&state->buffers[gpu], allocation_bytes));
        load_module_for_gpu(state, gpu);
    }

    if (!strstr(state->names[0], "RTX 5080") ||
        !strstr(state->names[1], "P100")) {
        fprintf(stderr,
                "This version expects GPU0=RTX 5080 and GPU1=P100; got "
                "GPU0=%s, GPU1=%s\n",
                state->names[0], state->names[1]);
        exit(2);
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
        CHECK_CUDA(cuModuleUnload(state->modules[gpu]));
        CHECK_CUDA(cuStreamDestroy(state->streams[gpu]));
        CHECK_CUDA(cuCtxDestroy(state->contexts[gpu]));
    }
}

int main(int argc, char **argv)
{
    int trials = DEFAULT_TRIALS;
    if (argc == 3 && strcmp(argv[1], "--trials") == 0) {
        char *end = NULL;
        long value = strtol(argv[2], &end, 10);
        if (!end || *end != '\0' || value < 3 || value > 101) {
            fprintf(stderr, "--trials must be between 3 and 101\n");
            return 2;
        }
        trials = (int)value;
    } else if (argc != 1) {
        fprintf(stderr, "Usage: %s [--trials 3..101]\n", argv[0]);
        return 2;
    }

    const size_t payload_count = sizeof(payloads) / sizeof(payloads[0]);
    const size_t operation_count = sizeof(operations) / sizeof(operations[0]);
    const size_t max_bytes = payloads[payload_count - 1].bytes;
    state_t state;
    result_t *results = calloc(payload_count * operation_count,
                               sizeof(*results));
    uint16_t *input = malloc(max_bytes);
    uint16_t *actual = malloc(max_bytes);
    if (!results || !input || !actual) {
        fprintf(stderr, "Host allocation failed\n");
        return 2;
    }

    for (size_t i = 0; i < max_bytes / sizeof(*input); ++i)
        input[i] = (uint16_t)(0x5a5aU ^ (uint16_t)(i * 40503U));

    initialize(&state, max_bytes);
    printf("P2P latency benchmark: GPU0=%s, GPU1=%s\n", state.names[0],
           state.names[1]);
    printf("Timing: CUDA-event elapsed time per serialized operation; "
           "%d trial batches; no bandwidth conversion.\n",
           trials);
    printf("Kernel payload unit: 16-bit half-sized element; 10 KiB is exactly "
           "5120 active elements.\n\n");

    for (size_t payload = 0; payload < payload_count; ++payload) {
        for (size_t operation = 0; operation < operation_count; ++operation) {
            prepare_and_validate(&state, &operations[operation], input, actual,
                                 payloads[payload].bytes);
            result_t result = measure_operation(
                &state, &operations[operation], payloads[payload].bytes, trials);
            results[payload * operation_count + operation] = result;
            printf("%-47s %20s: p50=%9.3f us  p05=%9.3f  p95=%9.3f  "
                   "batch=%u  integrity=PASS\n",
                   operations[operation].label, payloads[payload].label,
                   result.p50_us, result.p05_us, result.p95_us,
                   result.batch_iterations);
        }
    }

    printf("\nP50 DEVICE LATENCY (microseconds per completed operation)\n");
    printf("size_bytes,size_label");
    for (size_t operation = 0; operation < operation_count; ++operation)
        printf(",\"%s\"", operations[operation].label);
    printf("\n");
    for (size_t payload = 0; payload < payload_count; ++payload) {
        printf("%zu,\"%s\"", payloads[payload].bytes,
               payloads[payload].label);
        for (size_t operation = 0; operation < operation_count; ++operation) {
            result_t result = results[payload * operation_count + operation];
            printf(",%.3f", result.p50_us);
        }
        printf("\n");
    }

    cleanup(&state);
    free(actual);
    free(input);
    free(results);
    return 0;
}
