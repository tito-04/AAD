#
# Arquiteturas de Alto Desempenho 2025/2026
#
# makefile for the first practical assignment (A1)
#
# makefile automatic variables:
# $@ is the name of the target
# $< is the name of the first prerequisite
# $^ is the list of names of all prerequisites (without duplicates)
#
#
# CUDA installation directory --- /usr/local/cuda or $(CUDA_HOME)
#
CUDA_DIR = /usr/local/cuda
#
# OpenCL installation directory (for a NVidia graphics card, same as CUDA)
#
OPENCL_DIR = $(CUDA_DIR)
#
# CUDA device architecture
#
# GeForce GTX 1660 Ti --- sm_75
# RTX A2000 Ada --------- sm_86
# RTX A6000 Ada --------- sm_86
# RTX 4070 -------------- sm_89
#
CUDA_ARCH = sm_75
#
# default target
#
all: sha1_tests sha1_cuda_test sha1_cuda_kernel.cubin deti_coin_no_simd avx avx2 avx512f
#
# clean up
#
clean:
	rm -f sha1_tests
	rm -f sha1_cuda_test sha1_cuda_kernel.cubin
	rm -f deti_coin_no_simd
	rm -f avx avx2 avx512f
	rm -f a.out
#
# test the CUSTOM_SHA1_CODE macro
#
sha1_tests: aad_sha1_cpu_tests.c aad_sha1.h aad_data_types.h aad_utilities.h makefile
	cc -march=native -Wall -Wshadow -Werror -O3 $< -o $@

sha1_cuda_test: aad_sha1_cuda_test.c sha1_cuda_kernel.cubin aad_sha1.h aad_data_types.h aad_utilities.h aad_cuda_utilities.h makefile
	cc -march=native -Wall -Wshadow -Werror -O3 $< -o $@ -lcuda
#
# compile the CUDA kernels
#
sha1_cuda_kernel.cubin: aad_sha1_cuda_kernel.cu aad_sha1.h makefile
	nvcc -arch=$(CUDA_ARCH) --compiler-options -O2,-Wall -I$(CUDA_DIR)/include --cubin $< -o $@
#
# CPU-only miner (no SIMD - V3 unified version)
#
deti_coin_no_simd: deti_coin_no_simd.c aad_sha1_cpu.h aad_sha1.h aad_data_types.h aad_utilities.h aad_vault.h makefile
	cc -march=native -Wall -Wshadow -Werror -O3 -ffast-math -funroll-loops -flto $< -o $@
#
# SIMD versions of temp.c
#
# AVX version (SIMD width = 4)
avx: deti_coin_miner_simd.c aad_sha1_cpu.h aad_sha1.h aad_data_types.h aad_utilities.h aad_vault.h makefile
	cc -mavx -Wall -Wshadow -Werror -O3 -ffast-math -funroll-loops -flto -fopenmp $< -o $@

# AVX2 version (SIMD width = 8)
avx2: deti_coin_miner_simd.c aad_sha1_cpu.h aad_sha1.h aad_data_types.h aad_utilities.h aad_vault.h makefile
	cc -mavx2 -Wall -Wshadow -Werror -O3 -ffast-math -funroll-loops -flto -fopenmp $< -o $@

# AVX-512F version (SIMD width = 16)
avx512f: deti_coin_miner_simd.c aad_sha1_cpu.h aad_sha1.h aad_data_types.h aad_utilities.h aad_vault.h makefile
	cc -mavx512f -Wall -Wshadow -Werror -O3 -ffast-math -funroll-loops -flto -fopenmp $< -o $@
#
# Helper targets for testing different SIMD versions
#
test_avx: avx
	@echo "========================================="
	@echo "Testing AVX version (SIMD width = 4)"
	@echo "========================================="
	./avx "AVX Test" 10000000

test_avx2: avx2
	@echo "========================================="
	@echo "Testing AVX2 version (SIMD width = 8)"
	@echo "========================================="
	./avx2 "AVX2 Test" 10000000

test_avx512f: avx512f
	@echo "========================================="
	@echo "Testing AVX-512F version (SIMD width = 16)"
	@echo "========================================="
	./avx512f "AVX512F Test" 10000000
#
# Test the no-SIMD version
#
test_no_simd: deti_coin_no_simd
	@echo "========================================="
	@echo "Testing CPU No-SIMD version (V3 Unified)"
	@echo "========================================="
	./deti_coin_no_simd "NoSIMD Test" 10000000
#
# Run all SIMD tests in sequence
#
test_all_simd: test_avx test_avx2 test_avx512f
	@echo "========================================="
	@echo "All SIMD tests completed!"
	@echo "========================================="
#
# Compare all implementations (no-SIMD + all SIMD versions)
#
test_compare: test_no_simd test_all_simd
	@echo "========================================="
	@echo "All implementation tests completed!"
	@echo "Compare the performance results above"
	@echo "========================================="
#
# Check CPU features
#
check_cpu:
	@echo "Checking CPU SIMD support..."
	@grep -o 'avx[^ ]*' /proc/cpuinfo | sort -u || echo "AVX not found in /proc/cpuinfo"
	@echo ""
	@lscpu | grep -i flags | grep -o 'avx[^ ]*' || echo "Use 'lscpu' to check manually"