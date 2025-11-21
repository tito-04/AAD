//
// deti_coin_cuda_worker.cu
// Optimized CUDA Worker with Dynamic Occupancy
//

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <cuda.h>

#include "aad_data_types.h"
#include "aad_sha1.h" 
#include "aad_cuda_utilities.h" 

#define COIN_HEX_STRLEN (55 * 2 + 1) 
#define MAX_CUSTOM_LEN 34
#define N_STREAMS 4
#define DISPLAY_INTERVAL_ATTEMPTS 50000000 
#define MAX_COINS_PER_ROUND 1024
#define LOOP_SIZE 95 

extern "C" volatile int keep_running;

// ============================================================================
// HOST HELPERS
// ============================================================================

static u64_t host_lcg_state = 0;

static void generate_host_template(u32_t *buffer, const char *custom_text, int custom_len) {
    u08_t *bytes = (u08_t *)buffer;
    
    const char header[] = "DETI coin 2 ";
    for(int i = 0; i < 12; i++) bytes[i ^ 3] = (u08_t)header[i];

    int current_idx = 12;
    const int end_idx = 45;
    int text_pos = 0;

    if (custom_text != NULL) {
        while(text_pos < custom_len && current_idx <= end_idx) {
            char c = custom_text[text_pos++];
            if(c < 32 || c > 126) c = ' '; 
            bytes[current_idx ^ 3] = (u08_t)c;
            current_idx++;
        }
    }

    while (current_idx <= end_idx) {
        host_lcg_state = 6364136223846793005ul * host_lcg_state + 1442695040888963407ul;
        u08_t random_raw = (u08_t)(host_lcg_state >> 56);
        u08_t ascii_char = 32 + (u08_t)((random_raw * 95) >> 8);
        bytes[current_idx ^ 3] = ascii_char;
        current_idx++;
    }

    bytes[54 ^ 3] = (u08_t)'\n';
    bytes[55 ^ 3] = 0x80;
}

// ============================================================================
// PERSISTENT STATE AND CLEANUP
// ============================================================================

static int is_cuda_init = 0;
static cuda_data_t cd;
static CUstream streams[N_STREAMS];
static u32_t* h_vaults[N_STREAMS];      
static CUdeviceptr d_vaults[N_STREAMS]; 
static u32_t* h_templates[N_STREAMS];     
static CUdeviceptr d_templates[N_STREAMS]; 
static u64_t base_counters[N_STREAMS];  
static void* kernel_args[N_STREAMS][3];

// Assume this declaration comes from aad_cuda_utilities.h
extern "C" void terminate_cuda(cuda_data_t *cd); 

extern "C" void cleanup_cuda_worker() {
    if (is_cuda_init) {
        printf("[CUDA Worker] Shutting down CUDA resources...\n");

        for(int s = 0; s < N_STREAMS; s++) {
            cuStreamSynchronize(streams[s]); 
        }

        for(int s = 1; s < N_STREAMS; s++) {
            if (d_vaults[s])    CU_CALL( cuMemFree, (d_vaults[s]) );
            if (h_vaults[s])    CU_CALL( cuMemFreeHost, (h_vaults[s]) );
            if (d_templates[s]) CU_CALL( cuMemFree, (d_templates[s]) );
            if (h_templates[s]) CU_CALL( cuMemFreeHost, (h_templates[s]) );
            if (streams[s])     CU_CALL( cuStreamDestroy, (streams[s]) );
        }

        terminate_cuda(&cd); 
        is_cuda_init = 0;
        printf("[CUDA Worker] Shutdown complete.\n");
    }
}

// ============================================================================
// MAIN WORKER
// ============================================================================

extern "C" void run_mining_round(long work_id,
                                 long *attempts_out,
                                 int *coins_found_out,
                                 double *mhs_out, 
                                 char coins_out[][COIN_HEX_STRLEN],
                                 const char *custom_text)
{
    if (!is_cuda_init) {
        memset(&cd, 0, sizeof(cuda_data_t));
        cd.device_number = 0; 
        
        cd.cubin_file_name = (char*)"miner_kernel.cubin"; 
        cd.kernel_name = (char*)"miner_kernel";
        
        cd.data_size[0] = 14 * sizeof(u32_t);
        cd.data_size[1] = 1024 * sizeof(u32_t); 

        initialize_cuda(&cd); 

        streams[0] = cd.cu_stream;
        h_templates[0] = (u32_t*)cd.host_data[0];
        d_templates[0] = cd.device_data[0];
        h_vaults[0] = (u32_t*)cd.host_data[1];
        d_vaults[0] = cd.device_data[1];

        for(int i = 1; i < N_STREAMS; i++) {
            CU_CALL( cuStreamCreate, (&streams[i], CU_STREAM_NON_BLOCKING) );
            CU_CALL( cuMemAllocHost, ((void **)&h_vaults[i], (size_t)cd.data_size[1]) );
            CU_CALL( cuMemAlloc, (&d_vaults[i], (size_t)cd.data_size[1]) );
            CU_CALL( cuMemAllocHost, ((void **)&h_templates[i], (size_t)cd.data_size[0]) );
            CU_CALL( cuMemAlloc, (&d_templates[i], (size_t)cd.data_size[0]) );
        }

        for(int i = 0; i < N_STREAMS; i++) {
            kernel_args[i][0] = &base_counters[i];
            kernel_args[i][1] = &d_vaults[i];     
            kernel_args[i][2] = &d_templates[i]; 
        }
        
        // --- OCCUPANCY OPTIMIZATION START ---
        int minGridSize = 0;
        int blockSize = 0;
        
        // Calculate optimal block size dynamically
        cudaError_t occupancy_status = cudaOccupancyMaxPotentialBlockSize(
            &minGridSize, 
            &blockSize, 
            (const void*)cd.cu_kernel, // Cast may depend on your cuda_data_t struct; if this is CUfunction, careful. 
            // NOTE: cudaOccupancyMaxPotentialBlockSize requires a __global__ function pointer (C++ API).
            // Since we are using Driver API (CUfunction) in 'cd.cu_kernel', we cannot use the Runtime API occupancy calculator easily
            // UNLESS we also link the kernel code here or use cuOccupancyMaxPotentialBlockSize (Driver API equivalent).
            //
            // SAFE FIX: Use Driver API occupancy check
            0, 0
        );
        
        // Fallback to Driver API approach since we are loading a CUBIN:
        CU_CALL( cuOccupancyMaxPotentialBlockSize, (&minGridSize, &blockSize, cd.cu_kernel, NULL, 0, 0) );
        
        cd.block_dim_x = blockSize;
        int target_threads = 262144; // Target ~250k in flight
        int calc_grid = (target_threads + blockSize - 1) / blockSize;
        if (calc_grid < minGridSize) calc_grid = minGridSize;
        cd.grid_dim_x = calc_grid;

        printf("[CUDA] Optimization: BlockSize=%d | GridSize=%d\n", cd.block_dim_x, cd.grid_dim_x);
        // --- OCCUPANCY OPTIMIZATION END ---
        
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        host_lcg_state = ((u64_t)ts.tv_nsec) ^ ((u64_t)getpid() << 32);

        is_cuda_init = 1;
        printf("[CUDA Worker] Initialized Device: %s\n", cd.device_name);
    }

    u64_t num_threads = (u64_t)cd.block_dim_x * (u64_t)cd.grid_dim_x;
    u64_t hashes_per_launch = num_threads * LOOP_SIZE; 
    u64_t total_hashes = 0;
    int coins_found = 0;
    u64_t current_base_counter = (u64_t)work_id * 100000000000ULL;
    
    int custom_len = 0;
    if (custom_text != NULL) {
        size_t len = strlen(custom_text);
        custom_len = (len > MAX_CUSTOM_LEN) ? MAX_CUSTOM_LEN : (int)len;
    }
    
    struct timespec t_start, t_curr;
    clock_gettime(CLOCK_MONOTONIC, &t_start);
    const double MAX_SECONDS = 60.0;
    u64_t last_report = 0;
    
    printf("--- [CUDA] Mining Round (WorkID: %ld) ---\n", work_id);
    if(custom_text) printf("--- Using Custom Text: \"%s\" ---\n", custom_text);

    for(int s = 0; s < N_STREAMS; s++) {
        h_vaults[s][0] = 1u; 
        generate_host_template(h_templates[s], custom_text, custom_len);
        CU_CALL( cuMemcpyHtoDAsync, (d_templates[s], h_templates[s], (size_t)cd.data_size[0], streams[s]) );
        CU_CALL( cuMemcpyHtoDAsync, (d_vaults[s], h_vaults[s], (size_t)cd.data_size[1], streams[s]) );
        base_counters[s] = current_base_counter;
        current_base_counter += num_threads; 
        CU_CALL( cuLaunchKernel, (cd.cu_kernel, cd.grid_dim_x, 1u, 1u, cd.block_dim_x, 1u, 1u, 0u, streams[s], &kernel_args[s][0], NULL) );
    }

    int keep_looping = 1;
    int stream_idx = 0;

    while(keep_looping) {
        if (!keep_running) break;
        int s = stream_idx;
        
        CU_CALL( cuStreamSynchronize, (streams[s]) );
        total_hashes += hashes_per_launch; 

        CU_CALL( cuMemcpyDtoHAsync, ((void *)&h_vaults[s][0], d_vaults[s], sizeof(u32_t), streams[s]) );
        CU_CALL( cuStreamSynchronize, (streams[s]) ); 
        u32_t num_words = h_vaults[s][0];

        if(num_words > 1u) {
             if (num_words > 1024u) num_words = 1024u;
             CU_CALL( cuMemcpyDtoHAsync, ((void *)h_vaults[s], d_vaults[s], num_words * sizeof(u32_t), streams[s]) );
             CU_CALL( cuStreamSynchronize, (streams[s]) ); 

             for(u32_t i = 1; i < num_words; i += 14) {
                 if (coins_found < MAX_COINS_PER_ROUND) { 
                     u08_t *coin_bytes = (u08_t *)&h_vaults[s][i];
                     char *out_hex = coins_out[coins_found];
                     int pos = 0;
                     
                     printf("\n[!] CUDA COIN FOUND! Nonce: %c%c%c%c%c%c%c%c\n",
                        coin_bytes[46^3], coin_bytes[47^3], coin_bytes[48^3], coin_bytes[49^3], 
                        coin_bytes[50^3], coin_bytes[51^3], coin_bytes[52^3], coin_bytes[53^3]);
                     
                     for (int b = 0; b < 55; ++b) {
                         pos += snprintf(out_hex + pos, COIN_HEX_STRLEN - pos, "%02x", coin_bytes[b ^ 3]);
                     }
                     out_hex[COIN_HEX_STRLEN-1] = '\0';
                     coins_found++;
                 }
             }
        }

        generate_host_template(h_templates[s], custom_text, custom_len);
        base_counters[s] = current_base_counter;
        current_base_counter += num_threads;
        h_vaults[s][0] = 1u; 

        CU_CALL( cuMemcpyHtoDAsync, (d_templates[s], h_templates[s], (size_t)cd.data_size[0], streams[s]) );
        CU_CALL( cuMemcpyHtoDAsync, (d_vaults[s], h_vaults[s], (size_t)cd.data_size[1], streams[s]) );
        CU_CALL( cuLaunchKernel, (cd.cu_kernel, cd.grid_dim_x, 1u, 1u, cd.block_dim_x, 1u, 1u, 0u, streams[s], &kernel_args[s][0], NULL) );

        if ((total_hashes - last_report) >= DISPLAY_INTERVAL_ATTEMPTS) {
            clock_gettime(CLOCK_MONOTONIC, &t_curr);
            double cur_elapsed = (t_curr.tv_sec - t_start.tv_sec) + (t_curr.tv_nsec - t_start.tv_nsec) * 1e-9;
            if(cur_elapsed > 0) {
                double cur_mhs = (double)total_hashes / cur_elapsed / 1000000.0;
                printf("\r[Status] Speed: %.2f MH/s | Hashes: %lu | Coins: %d", 
                       cur_mhs, (unsigned long)total_hashes, coins_found);
                fflush(stdout);
            }
            last_report = total_hashes;

            if (cur_elapsed >= MAX_SECONDS) keep_looping = 0;
        }

        stream_idx = (stream_idx + 1) % N_STREAMS;
    }

    for(int s = 0; s < N_STREAMS; s++) CU_CALL( cuStreamSynchronize, (streams[s]) );

    printf("\n");
    clock_gettime(CLOCK_MONOTONIC, &t_curr);
    double elapsed = (t_curr.tv_sec - t_start.tv_sec) + (t_curr.tv_nsec - t_start.tv_nsec) * 1e-9;

    *attempts_out = (long)total_hashes;
    *coins_found_out = coins_found;
    *mhs_out = (elapsed > 0) ? ((double)total_hashes / elapsed / 1000000.0) : 0.0;
}