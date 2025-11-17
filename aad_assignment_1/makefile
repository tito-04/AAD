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
# MUDE ESTA LINHA para a sua grÃ¡fica (ex: sm_89 para uma RTX 4070)
CUDA_ARCH = sm_89
#
# default target
#
all: sha1_tests deti_coin_no_simd avx avx2 avx512f sha1_cuda_test cuda_miner
#
# clean up
#
clean:
	rm -f sha1_tests
	rm -f sha1_cuda_test sha1_cuda_kernel.cubin
	rm -f deti_coin_no_simd
	rm -f avx avx2 avx512f
	rm -f temp_avx temp_avx2 temp_avx512f
	rm -f a.out
	rm -f cuda_miner miner_kernel.cubin
#
# test the CUSTOM_SHA1_CODE macro
#
sha1_tests: aad_sha1_cpu_tests.c aad_sha1.h aad_data_types.h aad_utilities.h makefile
	cc -march=native -Wall -Wshadow -Werror -O3 $< -o $@

sha1_cuda_test: aad_sha1_cuda_test.c sha1_cuda_kernel.cubin aad_sha1.h aad_data_types.h aad_utilities.h aad_cuda_utilities.h makefile
	cc -march=native -Wall -Wshadow -Werror -O3 -I$(CUDA_DIR)/include $< -o $@ -L$(CUDA_DIR)/lib64 -lcuda
#
# compile the CUDA kernels
#
sha1_cuda_kernel.cubin: aad_sha1_cuda_kernel.cu aad_sha1.h makefile
	nvcc -arch=$(CUDA_ARCH) --compiler-options -O2,-Wall -I$(CUDA_DIR)/include --cubin $< -o $@

miner_kernel.cubin: miner_kernel.cu aad_sha1.h aad_data_types.h makefile
	nvcc -arch=$(CUDA_ARCH) --compiler-options -O2,-Wall -I$(CUDA_DIR)/include --cubin $< -o $@
#
# CPU-only miner (no SIMD - V3 unified version)
#
deti_coin_no_simd: deti_coin_no_simd.c aad_sha1_cpu.h aad_sha1.h aad_data_types.h aad_utilities.h aad_vault.h makefile
	cc -march=native -Wall -Wshadow -Werror -O3 -funroll-loops -flto $< -o $@
#
# SIMD versions of deti_coin_miner_simd.c (nome de ficheiro assumido)
#
# AVX version (SIMD width = 4)
avx: deti_coin_miner_simd.c aad_sha1_cpu.h aad_sha1.h aad_data_types.h aad_utilities.h aad_vault.h makefile
	cc -mavx -Wall -Wshadow -Werror -O3 -funroll-loops -flto $< -o $@

# AVX2 version (SIMD width = 8)
avx2: deti_coin_miner_simd.c aad_sha1_cpu.h aad_sha1.h aad_data_types.h aad_utilities.h aad_vault.h makefile
	cc -mavx2 -Wall -Wshadow -Werror -O3 -funroll-loops -flto $< -o $@

# AVX-512F version (SIMD width = 16)
avx512f: deti_coin_miner_simd.c aad_sha1_cpu.h aad_sha1.h aad_data_types.h aad_utilities.h aad_vault.h makefile
	cc -mavx512f -Wall -Wshadow -Werror -O3 -funroll-loops -flto $< -o $@

#
# SIMD versions of temp.c
#
temp_avx: temp.c aad_sha1_cpu.h aad_sha1.h aad_data_types.h aad_utilities.h aad_vault.h makefile
	cc -mavx -Wall -Wshadow -Werror -O3 -funroll-loops -flto -fopenmp $< -o $@

temp_avx2: temp.c aad_sha1_cpu.h aad_sha1.h aad_data_types.h aad_utilities.h aad_vault.h makefile
	cc -mavx2 -Wall -Wshadow -Werror -O3 -funroll-loops -flto -fopenmp $< -o $@

temp_avx512f: temp.c aad_sha1_cpu.h aad_sha1.h aad_data_types.h aad_utilities.h aad_vault.h makefile
	cc -mavx512f -Wall -Wshadow -Werror -O3 -funroll-loops -flto -fopenmp $< -o $@
#
# CUDA miner (High Performance)
#
cuda_miner: cuda_miner.c miner_kernel.cubin aad_sha1.h aad_data_types.h aad_utilities.h aad_cuda_utilities.h aad_sha1_cpu.h aad_vault.h makefile
	cc -march=native -Wall -Wshadow -O3 -flto -ffast-math -I$(CUDA_DIR)/include $< -o $@ -L$(CUDA_DIR)/lib64 -lcuda

miner_kernel.cubin: miner_kernel.cu aad_sha1.h aad_data_types.h
	nvcc -O3 -arch=sm_89 --use_fast_math -maxrregcount=64 --ptxas-options=-v -cubin miner_kernel.cu -o miner_kernel.cubin