# Test programs for the RTX 5080 + Tesla P100 P2P setup (see P2P_P100_RTX50.md).
#
#   make                 the four C probes; they use only the CUDA driver API (cuda.h,
#                        libcuda) and load hand-written sm_60 PTX at run time, so the
#                        headers of any recent CUDA toolkit work
#   make p2p_lat         the handoff-latency test; needs an nvcc that still targets
#                        sm_60 (CUDA 12.x; CUDA 13 dropped Pascal), for example
#                        make p2p_lat NVCC=/usr/local/cuda-12.9/bin/nvcc
#
# The probes expect exactly two visible GPUs, ordinal 0 = RTX 5080, ordinal 1 = P100
# (CUDA's default fastest-first order on the test machine).

CUDA_ROOT ?= /usr/local/cuda
CC ?= gcc
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -Wpedantic
NVCC ?= $(CUDA_ROOT)/bin/nvcc
NVCCFLAGS ?= -O2 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_120,code=sm_120

PROBES := p2p_probe p2p_kernel_probe p2p_latency p2p_memops_probe

.PHONY: all clean

all: $(PROBES)

$(PROBES): %: %.c
	$(CC) $(CFLAGS) -I$(CUDA_ROOT)/include $< -L$(CUDA_ROOT)/lib64/stubs -lcuda -o $@

p2p_lat: p2p_lat.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -f $(PROBES) p2p_lat
