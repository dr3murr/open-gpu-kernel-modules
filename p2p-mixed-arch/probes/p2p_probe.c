#define _POSIX_C_SOURCE 200809L
#include <cuda.h>

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define CHECK_CU(call)                                                        \
    do {                                                                      \
        CUresult _status = (call);                                            \
        if (_status != CUDA_SUCCESS) {                                        \
            const char *_name = "unknown";                                  \
            const char *_description = "unknown";                           \
            cuGetErrorName(_status, &_name);                                  \
            cuGetErrorString(_status, &_description);                         \
            fprintf(stderr, "%s failed: %s (%d): %s\n", #call, _name,      \
                    (int)_status, _description);                              \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

static double seconds_now(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        perror("clock_gettime");
        exit(EXIT_FAILURE);
    }
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1.0e9;
}

static void fill_pattern(unsigned char *data, size_t size, unsigned int tag)
{
    uint32_t state = 0x9e3779b9U ^ (tag * 0x85ebca6bU);
    for (size_t i = 0; i < size; ++i) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        data[i] = (unsigned char)(state ^ (uint32_t)i ^ (uint32_t)(i >> 8));
    }
}

static bool verify_pattern(const unsigned char *expected,
                           const unsigned char *actual, size_t size,
                           const char *direction)
{
    size_t mismatches = 0;
    for (size_t i = 0; i < size; ++i) {
        if (expected[i] != actual[i]) {
            if (mismatches < 8) {
                fprintf(stderr,
                        "%s mismatch at byte %zu: expected 0x%02x, got 0x%02x\n",
                        direction, i, expected[i], actual[i]);
            }
            ++mismatches;
        }
    }
    if (mismatches != 0) {
        fprintf(stderr, "%s FAILED: %zu corrupt bytes out of %zu\n",
                direction, mismatches, size);
        return false;
    }
    printf("%s integrity: PASS (%zu bytes checked)\n", direction, size);
    return true;
}

static void print_device(CUdevice device, int ordinal)
{
    char name[128] = {0};
    char bus_id[32] = {0};
    int major = 0, minor = 0;
    size_t memory = 0;
    int stream_mem_ops_v1 = 0;
    int stream_mem_ops_64_v1 = 0;
    int stream_mem_ops_64 = 0;
    int flush_remote_writes = 0;
    int rdma_flush_options = 0;
    int rdma_write_ordering = 0;

    CHECK_CU(cuDeviceGetName(name, (int)sizeof(name), device));
    CHECK_CU(cuDeviceGetPCIBusId(bus_id, (int)sizeof(bus_id), device));
    CHECK_CU(cuDeviceComputeCapability(&major, &minor, device));
    CHECK_CU(cuDeviceTotalMem(&memory, device));
    CHECK_CU(cuDeviceGetAttribute(
        &stream_mem_ops_v1, CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1,
        device));
    CHECK_CU(cuDeviceGetAttribute(
        &stream_mem_ops_64_v1,
        CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS_V1, device));
    CHECK_CU(cuDeviceGetAttribute(
        &stream_mem_ops_64,
        CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS, device));
    CHECK_CU(cuDeviceGetAttribute(
        &flush_remote_writes, CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES,
        device));
    CHECK_CU(cuDeviceGetAttribute(
        &rdma_flush_options,
        CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS, device));
    CHECK_CU(cuDeviceGetAttribute(
        &rdma_write_ordering,
        CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING, device));
    printf("GPU%d: %s, PCI %s, compute %d.%d, %.0f MiB\n", ordinal, name,
           bus_id, major, minor, (double)memory / (1024.0 * 1024.0));
    printf("  streamMemOpsV1=%d streamMemOps64V1=%d streamMemOps64=%d "
           "flushRemoteWrites=%d rdmaFlushOptions=0x%x "
           "rdmaWriteOrdering=%d\n",
           stream_mem_ops_v1, stream_mem_ops_64_v1, stream_mem_ops_64,
           flush_remote_writes, rdma_flush_options, rdma_write_ordering);
}

static void print_peer_attributes(CUdevice src, CUdevice dst,
                                  int src_ordinal, int dst_ordinal)
{
    int can_access = 0;
    int access_supported = 0;
    int native_atomic = 0;
    int partial_native_atomic = 0;
    int performance_rank = 0;
    int array_access = 0;
    CUatomicOperation atomic_operations[CU_ATOMIC_OPERATION_MAX];
    unsigned int atomic_capabilities[CU_ATOMIC_OPERATION_MAX] = {0};
    unsigned int atomic_capability_union = 0;

    for (int operation = 0; operation < CU_ATOMIC_OPERATION_MAX; ++operation)
        atomic_operations[operation] = (CUatomicOperation)operation;

    CHECK_CU(cuDeviceCanAccessPeer(&can_access, src, dst));
    CHECK_CU(cuDeviceGetP2PAttribute(
        &access_supported, CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED, src, dst));
    CHECK_CU(cuDeviceGetP2PAttribute(
        &native_atomic, CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED, src, dst));
    CHECK_CU(cuDeviceGetP2PAttribute(
        &partial_native_atomic,
        CU_DEVICE_P2P_ATTRIBUTE_ONLY_PARTIAL_NATIVE_ATOMIC_SUPPORTED,
        src, dst));
    CHECK_CU(cuDeviceGetP2PAttribute(
        &performance_rank, CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK, src, dst));
    CHECK_CU(cuDeviceGetP2PAttribute(
        &array_access, CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED, src, dst));
    CHECK_CU(cuDeviceGetP2PAtomicCapabilities(
        atomic_capabilities, atomic_operations, CU_ATOMIC_OPERATION_MAX, src,
        dst));
    for (int operation = 0; operation < CU_ATOMIC_OPERATION_MAX; ++operation)
        atomic_capability_union |= atomic_capabilities[operation];

    printf("GPU%d -> GPU%d: canAccessPeer=%d access=%d nativeAtomic=%d "
           "partialNativeAtomic=%d atomicCapabilityUnion=0x%x "
           "arrayAccess=%d performanceRank=%d\n",
           src_ordinal, dst_ordinal, can_access, access_supported,
           native_atomic, partial_native_atomic, atomic_capability_union,
           array_access, performance_rank);

    if (!can_access || !access_supported)
        fprintf(stderr, "Driver API does not advertise GPU%d -> GPU%d peer access\n",
                src_ordinal, dst_ordinal);
}

static void enable_peer(CUcontext local, CUcontext peer,
                        int local_ordinal, int peer_ordinal)
{
    CHECK_CU(cuCtxSetCurrent(local));
    CUresult status = cuCtxEnablePeerAccess(peer, 0);
    if (status != CUDA_SUCCESS && status != CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED) {
        const char *name = "unknown";
        const char *description = "unknown";
        cuGetErrorName(status, &name);
        cuGetErrorString(status, &description);
        fprintf(stderr, "cuCtxEnablePeerAccess(GPU%d -> GPU%d) failed: %s (%d): %s\n",
                local_ordinal, peer_ordinal, name, (int)status, description);
        exit(EXIT_FAILURE);
    }
    printf("GPU%d -> GPU%d peer mapping: ENABLED\n",
           local_ordinal, peer_ordinal);
}

static bool verify_local_dtod(CUcontext context, CUdeviceptr source,
                              CUdeviceptr destination,
                              unsigned char *expected, unsigned char *actual,
                              size_t bytes, int ordinal, unsigned int tag)
{
    char label[40];
    snprintf(label, sizeof(label), "GPU%d local DtoD", ordinal);

    fill_pattern(expected, bytes, tag);
    memset(actual, 0, bytes);
    CHECK_CU(cuCtxSetCurrent(context));
    CHECK_CU(cuMemcpyHtoD(source, expected, bytes));
    CHECK_CU(cuMemsetD8(destination, 0xa5, bytes));
    CHECK_CU(cuMemcpyDtoD(destination, source, bytes));
    CHECK_CU(cuCtxSynchronize());
    CHECK_CU(cuMemcpyDtoH(actual, destination, bytes));
    return verify_pattern(expected, actual, bytes, label);
}

static bool run_direction(CUcontext src_context, CUdeviceptr src_pointer,
                          CUcontext dst_context, CUdeviceptr dst_pointer,
                          CUstream dst_stream, unsigned char *expected,
                          unsigned char *actual, size_t bytes, int iterations,
                          int src_ordinal, int dst_ordinal, unsigned int tag,
                          bool use_direct_uva)
{
    char direction[32];
    char local_source[32];
    snprintf(direction, sizeof(direction), "GPU%d -> GPU%d", src_ordinal,
             dst_ordinal);
    snprintf(local_source, sizeof(local_source), "GPU%d local source",
             src_ordinal);

    fill_pattern(expected, bytes, tag);
    memset(actual, 0, bytes);

    CHECK_CU(cuCtxSetCurrent(src_context));
    CHECK_CU(cuMemcpyHtoD(src_pointer, expected, bytes));
    CHECK_CU(cuMemcpyDtoH(actual, src_pointer, bytes));
    if (!verify_pattern(expected, actual, bytes, local_source))
        return false;

    CHECK_CU(cuCtxSetCurrent(dst_context));
    CHECK_CU(cuMemsetD8(dst_pointer, 0xa5, bytes));
    if (use_direct_uva)
        CHECK_CU(cuMemcpyDtoD(dst_pointer, src_pointer, bytes));
    else
        CHECK_CU(cuMemcpyPeer(dst_pointer, dst_context, src_pointer, src_context,
                              bytes));
    CHECK_CU(cuCtxSynchronize());
    CHECK_CU(cuMemcpyDtoH(actual, dst_pointer, bytes));
    if (!verify_pattern(expected, actual, bytes, direction))
        return false;

    /* Time peer-copy traffic only; host transfers and verification are outside. */
    if (use_direct_uva)
        CHECK_CU(cuMemcpyDtoDAsync(dst_pointer, src_pointer, bytes, dst_stream));
    else
        CHECK_CU(cuMemcpyPeerAsync(dst_pointer, dst_context, src_pointer,
                                   src_context, bytes, dst_stream));
    CHECK_CU(cuStreamSynchronize(dst_stream));
    double start = seconds_now();
    for (int i = 0; i < iterations; ++i) {
        if (use_direct_uva)
            CHECK_CU(cuMemcpyDtoDAsync(dst_pointer, src_pointer, bytes,
                                       dst_stream));
        else
            CHECK_CU(cuMemcpyPeerAsync(dst_pointer, dst_context, src_pointer,
                                       src_context, bytes, dst_stream));
    }
    CHECK_CU(cuStreamSynchronize(dst_stream));
    double elapsed = seconds_now() - start;
    double gib = ((double)bytes * (double)iterations) /
                 (1024.0 * 1024.0 * 1024.0);
    printf("%s bandwidth: %.3f GiB/s (%.3f GiB in %.3f s)\n",
           direction, gib / elapsed, gib, elapsed);

    memset(actual, 0, bytes);
    CHECK_CU(cuMemcpyDtoH(actual, dst_pointer, bytes));
    return verify_pattern(expected, actual, bytes, direction);
}

int main(int argc, char **argv)
{
    size_t bytes = 64U * 1024U * 1024U;
    int iterations = 32;
    unsigned int hold_seconds = 0;
    unsigned int hold_after_seconds = 0;
    bool query_only = false;
    bool use_direct_uva = false;
    int driver_version = 0;
    int count = 0;
    CUdevice devices[2];
    CUcontext contexts[2] = {NULL, NULL};
    CUdeviceptr pointers[2] = {0, 0};
    CUdeviceptr scratch_pointers[2] = {0, 0};
    CUstream streams[2] = {NULL, NULL};

    setvbuf(stdout, NULL, _IOLBF, 0);

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--query-only") == 0) {
            query_only = true;
        }
        else if (strcmp(argv[i], "--copy-method") == 0 && i + 1 < argc) {
            const char *method = argv[++i];
            if (strcmp(method, "peer") == 0)
                use_direct_uva = false;
            else if (strcmp(method, "dtod") == 0)
                use_direct_uva = true;
            else {
                fprintf(stderr, "Invalid --copy-method value: %s\n", method);
                return EXIT_FAILURE;
            }
        }
        else if (strcmp(argv[i], "--bytes") == 0 && i + 1 < argc) {
            char *end = NULL;
            errno = 0;
            unsigned long long value = strtoull(argv[++i], &end, 0);
            if (errno != 0 || end == argv[i] || *end != '\0' || value == 0 ||
                value > SIZE_MAX) {
                fprintf(stderr, "Invalid --bytes value: %s\n", argv[i]);
                return EXIT_FAILURE;
            }
            bytes = (size_t)value;
        }
        else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            char *end = NULL;
            errno = 0;
            long value = strtol(argv[++i], &end, 0);
            if (errno != 0 || end == argv[i] || *end != '\0' || value < 1 ||
                value > INT_MAX) {
                fprintf(stderr, "Invalid --iterations value: %s\n", argv[i]);
                return EXIT_FAILURE;
            }
            iterations = (int)value;
        }
        else if (strcmp(argv[i], "--hold-seconds") == 0 && i + 1 < argc) {
            char *end = NULL;
            errno = 0;
            unsigned long value = strtoul(argv[++i], &end, 0);
            if (errno != 0 || end == argv[i] || *end != '\0' || value > 300) {
                fprintf(stderr, "Invalid --hold-seconds value: %s\n", argv[i]);
                return EXIT_FAILURE;
            }
            hold_seconds = (unsigned int)value;
        }
        else if (strcmp(argv[i], "--hold-after-seconds") == 0 && i + 1 < argc) {
            char *end = NULL;
            errno = 0;
            unsigned long value = strtoul(argv[++i], &end, 0);
            if (errno != 0 || end == argv[i] || *end != '\0' || value > 300) {
                fprintf(stderr, "Invalid --hold-after-seconds value: %s\n",
                        argv[i]);
                return EXIT_FAILURE;
            }
            hold_after_seconds = (unsigned int)value;
        }
        else {
            fprintf(stderr,
                    "Usage: %s [--query-only] [--copy-method peer|dtod] "
                    "[--bytes N] [--iterations N] [--hold-seconds N] "
                    "[--hold-after-seconds N]\n",
                    argv[0]);
            return EXIT_FAILURE;
        }
    }

    CHECK_CU(cuInit(0));
    CHECK_CU(cuDriverGetVersion(&driver_version));
    CHECK_CU(cuDeviceGetCount(&count));
    printf("CUDA driver API version: %d, visible GPUs: %d\n", driver_version,
           count);
    if (count != 2) {
        fprintf(stderr, "Expected exactly 2 GPUs, found %d\n", count);
        return EXIT_FAILURE;
    }

    for (int i = 0; i < 2; ++i) {
        CHECK_CU(cuDeviceGet(&devices[i], i));
        print_device(devices[i], i);
    }
    print_peer_attributes(devices[0], devices[1], 0, 1);
    print_peer_attributes(devices[1], devices[0], 1, 0);
    if (query_only) {
        puts("P2P QUERY: COMPLETE");
        return EXIT_SUCCESS;
    }

    printf("Transfer size: %zu bytes; timed iterations: %d\n", bytes,
           iterations);
    printf("Copy method: %s\n",
           use_direct_uva ? "cuMemcpyDtoD (peer UVA)" : "cuMemcpyPeer");

    for (int i = 0; i < 2; ++i) {
        CHECK_CU(cuDevicePrimaryCtxRetain(&contexts[i], devices[i]));
        CHECK_CU(cuCtxSetCurrent(contexts[i]));
        CHECK_CU(cuMemAlloc(&pointers[i], bytes));
        CHECK_CU(cuMemAlloc(&scratch_pointers[i], bytes));
        CHECK_CU(cuStreamCreate(&streams[i], CU_STREAM_NON_BLOCKING));
    }

    enable_peer(contexts[0], contexts[1], 0, 1);
    enable_peer(contexts[1], contexts[0], 1, 0);

    if (hold_seconds != 0) {
        printf("Holding live peer mappings for %u seconds (PID %ld)\n",
               hold_seconds, (long)getpid());
        sleep(hold_seconds);
    }

    unsigned char *expected = malloc(bytes);
    unsigned char *actual = malloc(bytes);
    if (expected == NULL || actual == NULL) {
        fprintf(stderr, "Unable to allocate host verification buffers\n");
        return EXIT_FAILURE;
    }

    bool all_ok = true;
    all_ok &= verify_local_dtod(contexts[0], pointers[0], scratch_pointers[0],
                                expected, actual, bytes, 0, 0x101U);
    all_ok &= verify_local_dtod(contexts[1], pointers[1], scratch_pointers[1],
                                expected, actual, bytes, 1, 0x102U);
    all_ok &= run_direction(contexts[0], pointers[0], contexts[1], pointers[1],
                            streams[1], expected, actual, bytes, iterations,
                            0, 1, 1, use_direct_uva);
    all_ok &= run_direction(contexts[1], pointers[1], contexts[0], pointers[0],
                            streams[0], expected, actual, bytes, iterations,
                            1, 0, 2, use_direct_uva);

    if (hold_after_seconds != 0) {
        printf("Holding after peer-copy tests for %u seconds (PID %ld)\n",
               hold_after_seconds, (long)getpid());
        sleep(hold_after_seconds);
    }

    free(actual);
    free(expected);
    for (int i = 0; i < 2; ++i) {
        CHECK_CU(cuCtxSetCurrent(contexts[i]));
        CHECK_CU(cuStreamDestroy(streams[i]));
        CHECK_CU(cuMemFree(scratch_pointers[i]));
        CHECK_CU(cuMemFree(pointers[i]));
    }
    CHECK_CU(cuCtxSetCurrent(NULL));
    for (int i = 0; i < 2; ++i) {
        CHECK_CU(cuDevicePrimaryCtxRelease(devices[i]));
    }

    if (all_ok) {
        puts("P2P PROBE: PASS");
        return EXIT_SUCCESS;
    }

    puts("P2P PROBE: FAIL");
    return EXIT_FAILURE;
}
