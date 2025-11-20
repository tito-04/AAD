//
// deti_coin_cuda_worker.cu
// Optimized CUDA Worker for Distributed Miner (Nonce Grinding Strategy)
//
// Compile with: 
// 1. nvcc -O3 -arch=sm_89 -cubin deti_coin_cuda_worker.cu -o miner_kernel.cubin
// 2. nvcc -O3 -arch=sm_89 -o client_cuda client.c deti_coin_cuda_worker.cu -lcuda
//

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <cuda.h>

// Includes AAD 
#include "aad_data_types.h"
#include "aad_sha1.h" 
#include "aad_cuda_utilities.h" 

#define COIN_HEX_STRLEN (55 * 2 + 1) 
#define MAX_CUSTOM_LEN 34
#define N_STREAMS 4
#define DISPLAY_INTERVAL_ATTEMPTS 50000000 
#define MAX_COINS_PER_ROUND 1024

// Optimization Constants
#define RECOMENDED_CUDA_BLOCK_SIZE 128
#define LOOP_SIZE 95 // Characters 32 to 126

// ============================================================================
// KERNEL CUDA (Optimized Nonce Grinding v2)
// ============================================================================

static __device__ __forceinline__ u64_t lcg_rand(u64_t state) {
    return 6364136223846793005ul * state + 1442695040888963407ul;
}

extern "C" __global__ __launch_bounds__(RECOMENDED_CUDA_BLOCK_SIZE, 2) 
void miner_kernel(
    u64_t base_counter,           
    u32_t *coins_storage_area,    
    const u32_t *__restrict__ template_msg 
)
{
    // Unique ID for RNG seeding
    u64_t thread_id = base_counter + (u64_t)threadIdx.x + (u64_t)blockDim.x * (u64_t)blockIdx.x;

    // Local buffer for the message words
    u32_t data[14];
    u32_t hash[5];

    // 1. STATIC LOAD (Bytes 0-43)
    #pragma unroll
    for(int i = 0; i < 11; i++) {
        data[i] = template_msg[i];
    }

    // 2. RANDOM GENERATION (Bytes 46-52)
    u64_t rng = lcg_rand(thread_id);

    // Map random bits to ASCII (32-126)
    u32_t b46 = (u32_t)(rng & 0xFF);         b46 = 0x20 + ((b46 * 95) >> 8);
    u32_t b47 = (u32_t)((rng >> 8) & 0xFF);  b47 = 0x20 + ((b47 * 95) >> 8);
    u32_t b48 = (u32_t)((rng >> 16) & 0xFF); b48 = 0x20 + ((b48 * 95) >> 8);
    u32_t b49 = (u32_t)((rng >> 24) & 0xFF); b49 = 0x20 + ((b49 * 95) >> 8);
    u32_t b50 = (u32_t)((rng >> 32) & 0xFF); b50 = 0x20 + ((b50 * 95) >> 8);
    u32_t b51 = (u32_t)((rng >> 40) & 0xFF); b51 = 0x20 + ((b51 * 95) >> 8);
    u32_t b52 = (u32_t)((rng >> 48) & 0xFF); b52 = 0x20 + ((b52 * 95) >> 8);
    
    // 3. WORD ASSEMBLY
    // Word 11: Bytes 44-45 (from Host) | Bytes 46-47 (Random)
    u32_t w11_static = template_msg[11] & 0xFFFF0000u;
    data[11] = w11_static | (b46 << 8) | b47;

    // Word 12: Bytes 48-51 (Random)
    data[12] = (b48 << 24) | (b49 << 16) | (b50 << 8) | b51;

    // Word 13 Base: Byte 52 (Random) | Byte 53 (Placeholder) | Bytes 54-55 (Host Footer)
    u32_t w13_base = (b52 << 24) | (template_msg[13] & 0x0000FFFFu);

    // 4. THE GRINDING LOOP (Byte 53)
    #define T            u32_t
    #define C(c)         (c)
    #define ROTATE(x,n)  (((x) << (n)) | ((x) >> (32 - (n))))
    #define DATA(idx)    data[idx]
    #define HASH(idx)    hash[idx]

    // Loop from ' ' (32) to '~' (126)
    #pragma unroll 4 
    for(u32_t c53 = 32; c53 <= 126; c53++)
    {
        // Inject current loop character into Word 13 (Bits 16-23)
        data[13] = w13_base | (c53 << 16);

        CUSTOM_SHA1_CODE();

        // Check Signature (aad20250)
        if(hash[0] == 0xAAD20250u)
        {
            u32_t idx = atomicAdd(&coins_storage_area[0], 14u);
            if(idx < (1024u - 14u)) 
            {
                #pragma unroll
                for(int i = 0; i < 14; i++) {
                    coins_storage_area[idx + i] = data[i];
                }
            }
        }
    }

    #undef T
    #undef C
    #undef ROTATE
    #undef DATA
    #undef HASH
}

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
// WORKER INTERFACE
// ============================================================================

// Persistent CUDA State
static int is_cuda_init = 0;
static cuda_data_t cd;
static CUstream streams[N_STREAMS];
static u32_t* h_vaults[N_STREAMS];      
static CUdeviceptr d_vaults[N_STREAMS]; 
static u32_t* h_templates[N_STREAMS];     
static CUdeviceptr d_templates[N_STREAMS]; 
static u64_t base_counters[N_STREAMS];  
static void* kernel_args[N_STREAMS][3];

extern "C" void run_mining_round(long work_id,
                                 long *attempts_out,
                                 int *coins_found_out,
                                 double *mhs_out, 
                                 char coins_out[][COIN_HEX_STRLEN],
                                 const char *custom_text)
{
    // 1. Initialize CUDA (Only Once)
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
        
        // Grid Config (Optimized for Loop Strategy)
        // Smaller grid, because each thread does 95 hashes.
        cd.block_dim_x = RECOMENDED_CUDA_BLOCK_SIZE; // 128
        cd.grid_dim_x = 4096; 
        
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        host_lcg_state = ((u64_t)ts.tv_nsec) ^ ((u64_t)getpid() << 32);

        is_cuda_init = 1;
        printf("[CUDA Worker] Initialized Device: %s (Nonce Grinding Optimized)\n", cd.device_name);
    }

    // 2. Round Setup
    u64_t num_threads = (u64_t)cd.block_dim_x * (u64_t)cd.grid_dim_x;
    
    // Calculate actual hashes per launch (Threads * 95 iterations)
    u64_t hashes_per_launch = num_threads * LOOP_SIZE; 

    u64_t total_hashes = 0;
    int coins_found = 0;
    
    u64_t current_base_counter = (u64_t)work_id * 100000000000ULL;

    // Custom Text Handling
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

    // 3. Pipeline Priming
    for(int s = 0; s < N_STREAMS; s++) {
        h_vaults[s][0] = 1u; // Reset count
        generate_host_template(h_templates[s], custom_text, custom_len);
        
        CU_CALL( cuMemcpyHtoDAsync, (d_templates[s], h_templates[s], (size_t)cd.data_size[0], streams[s]) );
        CU_CALL( cuMemcpyHtoDAsync, (d_vaults[s], h_vaults[s], (size_t)cd.data_size[1], streams[s]) );

        base_counters[s] = current_base_counter;
        current_base_counter += num_threads; // Increment ID by threads (not hashes)

        CU_CALL( cuLaunchKernel, (cd.cu_kernel, cd.grid_dim_x, 1u, 1u, cd.block_dim_x, 1u, 1u, 0u, streams[s], &kernel_args[s][0], NULL) );
    }

    // 4. Main Loop
    int keep_looping = 1;
    int stream_idx = 0;

    while(keep_looping) {
        int s = stream_idx;
        
        CU_CALL( cuStreamSynchronize, (streams[s]) );
        total_hashes += hashes_per_launch; // Add 95 * threads

        // Check Vault
        CU_CALL( cuMemcpyDtoHAsync, ((void *)&h_vaults[s][0], d_vaults[s], sizeof(u32_t), streams[s]) );
        CU_CALL( cuStreamSynchronize, (streams[s]) ); 
        
        u32_t num_words = h_vaults[s][0];

        if(num_words > 1u) {
             if (num_words > 1024u) num_words = 1024u;
             CU_CALL( cuMemcpyDtoHAsync, ((void *)h_vaults[s], d_vaults[s], num_words * sizeof(u32_t), streams[s]) );
             CU_CALL( cuStreamSynchronize, (streams[s]) ); 

             for(u32_t i = 1; i < num_words; i += 14) {
                 if (coins_found < MAX_COINS_PER_ROUND) { 
                     // Convert to Bytes
                     u08_t *coin_bytes = (u08_t *)&h_vaults[s][i];
                     char *out_hex = coins_out[coins_found];
                     int pos = 0;
                     
                     // Print found nonce for debug
                     printf("\n[!] CUDA COIN FOUND! Nonce: %c%c%c%c%c%c%c%c\n",
                        coin_bytes[46^3], coin_bytes[47^3], coin_bytes[48^3], coin_bytes[49^3], 
                        coin_bytes[50^3], coin_bytes[51^3], coin_bytes[52^3], coin_bytes[53^3]);
                     
                     // Fill Hex String for Server
                     for (int b = 0; b < 55; ++b) {
                         pos += snprintf(out_hex + pos, COIN_HEX_STRLEN - pos, "%02x", coin_bytes[b ^ 3]);
                     }
                     out_hex[COIN_HEX_STRLEN-1] = '\0';
                     coins_found++;
                 }
             }
        }

        // Relaunch
        generate_host_template(h_templates[s], custom_text, custom_len);
        base_counters[s] = current_base_counter;
        current_base_counter += num_threads;
        h_vaults[s][0] = 1u; // Reset

        CU_CALL( cuMemcpyHtoDAsync, (d_templates[s], h_templates[s], (size_t)cd.data_size[0], streams[s]) );
        CU_CALL( cuMemcpyHtoDAsync, (d_vaults[s], h_vaults[s], (size_t)cd.data_size[1], streams[s]) );
        CU_CALL( cuLaunchKernel, (cd.cu_kernel, cd.grid_dim_x, 1u, 1u, cd.block_dim_x, 1u, 1u, 0u, streams[s], &kernel_args[s][0], NULL) );

        // Stats & Time
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

    // Final Sync
    for(int s = 0; s < N_STREAMS; s++) CU_CALL( cuStreamSynchronize, (streams[s]) );

    printf("\n");
    clock_gettime(CLOCK_MONOTONIC, &t_curr);
    double elapsed = (t_curr.tv_sec - t_start.tv_sec) + (t_curr.tv_nsec - t_start.tv_nsec) * 1e-9;

    *attempts_out = (long)total_hashes;
    *coins_found_out = coins_found;
    *mhs_out = (elapsed > 0) ? ((double)total_hashes / elapsed / 1000000.0) : 0.0;
}