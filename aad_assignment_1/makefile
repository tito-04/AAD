#
# Arquiteturas de Alto Desempenho 2025/2026
# REVISED: Fixed Linking and Compilation Flags
#
# CUDA directory
CUDA_DIR = /usr/local/cuda
OPENCL_DIR = $(CUDA_DIR)

# CUDA architecture
CUDA_ARCH = sm_89

# Compilers
CC = cc
NVCC = nvcc

# --- Global CFLAGS ---
CFLAGS = -Wall -Wshadow -O3 -pthread

# --- NVCC Flags ---
# Note: We include the -arch flag here.
NVCCFLAGS = -arch=$(CUDA_ARCH) -O3

#
# Default target
#
all: sha1_tests deti_coin_no_simd avx avx2 avx512f sha1_cuda_test cuda_miner opencl_miner server client_scalar client_avx client_avx2 client_avx512f client_cuda client_opencl miner_wasm oneapi

#
# Clean up
#
clean:
	rm -f sha1_tests
	rm -f sha1_cuda_test sha1_cuda_kernel.cubin
	rm -f deti_coin_no_simd
	rm -f avx avx2 avx512f
	rm -f MP_avx MP_avx2 MP_avx512f
	rm -f a.out
	rm -f cuda_miner miner_kernel.cubin
	rm -f opencl_miner
	rm -f server
	rm -f client_scalar client_avx client_avx2 client_avx512f client_cuda client_opencl
	rm -f client_scalar.o client_cuda.o client_opencl.o deti_coin_cuda_worker.o
	rm -f miner.wasm.js miner_simd_wasm.wasm.js miner.wasm.wasm
	rm -f miner_oneapi miner_oneapi_cpu

#
# Tests & Kernels
#
sha1_tests: aad_sha1_cpu_tests.c aad_sha1.h aad_data_types.h aad_utilities.h makefile
	$(CC) -march=native -Wall -Wshadow -Werror -O3 $< -o $@

sha1_cuda_test: aad_sha1_cuda_test.c sha1_cuda_kernel.cubin aad_sha1.h aad_data_types.h aad_utilities.h aad_cuda_utilities.h makefile
	$(CC) -march=native -Wall -Wshadow -Werror -O3 -I$(CUDA_DIR)/include $< -o $@ -L$(CUDA_DIR)/lib64 -lcuda

sha1_cuda_kernel.cubin: aad_sha1_cuda_kernel.cu aad_sha1.h makefile
	$(NVCC) $(NVCCFLAGS) --compiler-options -O2,-Wall -I$(CUDA_DIR)/include --cubin $< -o $@

miner_kernel.cubin: miner_kernel.cu aad_sha1.h aad_data_types.h makefile
	$(NVCC) $(NVCCFLAGS) --compiler-options -O2,-Wall -I$(CUDA_DIR)/include --cubin $< -o $@

#
# Miners (Standalone)
#
deti_coin_no_simd: deti_coin_miner_no_simd.c aad_sha1_cpu.h aad_sha1.h aad_data_types.h aad_utilities.h aad_vault.h makefile
	$(CC) -march=native -Wall -Wshadow -Werror -O3 -funroll-loops -flto $< -o $@

avx: deti_coin_miner_simd.c aad_sha1_cpu.h aad_sha1.h aad_data_types.h aad_utilities.h aad_vault.h makefile
	$(CC) -mavx -Wall -Wshadow -Werror -O3 -funroll-loops -flto $< -o $@

avx2: deti_coin_miner_simd.c aad_sha1_cpu.h aad_sha1.h aad_data_types.h aad_utilities.h aad_vault.h makefile
	$(CC) -mavx2 -Wall -Wshadow -Werror -O3 -funroll-loops -flto $< -o $@

avx512f: deti_coin_miner_simd.c aad_sha1_cpu.h aad_sha1.h aad_data_types.h aad_utilities.h aad_vault.h makefile
	$(CC) -mavx512f -Wall -Wshadow -Werror -O3 -funroll-loops -flto $< -o $@

# SIMD versions of deti_coin_miner_OpenMP.c
#
MP_avx: deti_coin_miner_omp_simd.c aad_sha1_cpu.h aad_sha1.h aad_data_types.h aad_utilities.h aad_vault.h makefile
	cc -mavx -Wall -Wshadow -Werror -O3 -funroll-loops -flto -fopenmp $< -o $@

MP_avx2: deti_coin_miner_omp_simd.c aad_sha1_cpu.h aad_sha1.h aad_data_types.h aad_utilities.h aad_vault.h makefile
	cc -mavx2 -Wall -Wshadow -Werror -O3 -funroll-loops -flto -fopenmp $< -o $@

MP_avx512f: deti_coin_miner_omp_simd.c aad_sha1_cpu.h aad_sha1.h aad_data_types.h aad_utilities.h aad_vault.h makefile
	cc -mavx512f -Wall -Wshadow -Werror -O3 -funroll-loops -flto -fopenmp $< -o $@

cuda_miner: cuda_miner.c miner_kernel.cubin aad_sha1.h aad_data_types.h aad_utilities.h aad_cuda_utilities.h aad_sha1_cpu.h aad_vault.h makefile
	$(CC) -march=native -Wall -Wshadow -O3 -flto -ffast-math -I$(CUDA_DIR)/include cuda_miner.c aad_histogram.c -o $@ -L$(CUDA_DIR)/lib64 -lcuda

opencl_miner: deti_coin_miner_opencl.c aad_data_types.h aad_sha1_cpu.h aad_vault.h open_cl_util.h makefile
	$(CC) -march=native -Wall -Wshadow -Werror -O3 -I$(OPENCL_DIR)/include $< -o $@ -L$(OPENCL_DIR)/lib64 -lOpenCL

server: server.c aad_vault.h aad_sha1_cpu.h aad_data_types.h
	$(CC) -march=native -Wall -Wshadow -O3 -pthread server.c -o server

# ---------------------------------------------------------
# CLIENTS
# ---------------------------------------------------------

# Compile CUDA worker host code
deti_coin_cuda_worker.o: deti_coin_cuda_worker.cu aad_cuda_utilities.h
	$(NVCC) $(NVCCFLAGS) -I$(CUDA_DIR)/include -c $< -o $@

# 1. Scalar Client
# We compile client.c directly here to avoid object file sharing issues
client_scalar: client.c deti_coin_no_simd_worker.c
	$(CC) $(CFLAGS) client.c deti_coin_no_simd_worker.c -o $@

# 2. AVX Client
client_avx: client.c deti_coin_simd_worker.c
	$(CC) $(CFLAGS) -mavx client.c deti_coin_simd_worker.c -o $@

# 3. AVX2 Client
client_avx2: client.c deti_coin_simd_worker.c
	$(CC) $(CFLAGS) -mavx2 client.c deti_coin_simd_worker.c -o $@

# 4. AVX-512F Client
client_avx512f: client.c deti_coin_simd_worker.c
	$(CC) $(CFLAGS) -mavx512f client.c deti_coin_simd_worker.c -o $@

# 5. CUDA Client
# 1) Compile client.c with -DUSE_CUDA_WORKER so the cleanup code is included.
# 2) Link everything using NVCC to handle C++ and CUDA Runtime libraries.
client_cuda: client.c deti_coin_cuda_worker.o miner_kernel.cubin
	$(CC) $(CFLAGS) -DUSE_CUDA_WORKER -c client.c -o client_cuda.o
	$(NVCC) $(NVCCFLAGS) -o $@ client_cuda.o deti_coin_cuda_worker.o -lcuda

# 6. OpenCL Client
# Compile client.c with -DUSE_OPENCL_WORKER
client_opencl: client.c deti_coin_opencl_worker.c
	$(CC) $(CFLAGS) -DUSE_OPENCL_WORKER -I$(OPENCL_DIR)/include -o $@ client.c deti_coin_opencl_worker.c -L$(OPENCL_DIR)/lib64 -lOpenCL

# WebAssembly Miner Target (Scalar)
#
miner_wasm: miner_wasm.c aad_sha1.h makefile
	emcc $< -o miner.wasm.js -O3 -flto \
	-s WASM=1 \
	-s MODULARIZE=1 \
	-s EXPORT_ES6=1 \
	-s EXPORTED_RUNTIME_METHODS="['stringToNewUTF8', 'allocateUTF8', 'HEAPU8', 'HEAPU32']" \
	-s EXPORTED_FUNCTIONS="['_search_chunk', '_get_coin_buffer_ptr', '_get_hash_buffer_ptr', '_get_lcg_state_ptr', '_malloc', '_free']" \
	-s ALLOW_MEMORY_GROWTH=1
#
# WebAssembly Miner Target (SIMD 128-bit)
#
miner_simd_wasm: miner_simd_wasm.c aad_sha1.h makefile
	emcc $< -o miner.wasm.js -O3 -flto -msimd128 \
	-s WASM=1 \
	-s MODULARIZE=1 \
	-s EXPORT_ES6=1 \
	-s EXPORTED_RUNTIME_METHODS="['stringToNewUTF8', 'allocateUTF8', 'HEAPU8', 'HEAPU32']" \
	-s EXPORTED_FUNCTIONS="['_search_chunk', '_get_coin_buffer_ptr', '_get_hash_buffer_ptr', '_get_lcg_state_ptr', '_malloc', '_free']" \
	-s ALLOW_MEMORY_GROWTH=1

#
# oneAPI / SYCL Implementation (Intel GPU / CPU)
#
oneapi: deti_coin_miner_oneapi.cpp aad_sha1.h aad_data_types.h aad_vault.h makefile
	icpx -fsycl -O3 -std=c++20 -Wno-c99-designator -Wno-writable-strings deti_coin_miner_oneapi.cpp -o miner_oneapi


oneapi_cpu: deti_coin_miner_oneapi.cpp aad_sha1.h aad_data_types.h aad_vault.h makefile
	icpx -fsycl -fsycl-targets=spir64_x86_64 -O3 -std=c++20 deti_coin_miner_oneapi.cpp -o miner_oneapi_cpu