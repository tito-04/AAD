#include <time.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <math.h>

#include "aad_data_types.h"
#include "aad_utilities.h"
#include "aad_sha1_cpu.h"
#include "aad_vault.h"
#include "aad_cuda_utilities.h"

// Include the new histogram module
#include "aad_histogram.h"

#define DEFAULT_DEVICE_ID 0
#define LOOP_SIZE 95

// --- STRATEGY CONFIGURATION (MATCHING CPU) ---
#define SALT_UPDATE_INTERVAL 50000ULL 
#define MAX_CUSTOM_LEN 34

static volatile int keep_running = 1;
static u64_t total_attempts = 0;
static u64_t total_coins_found = 0;
static u64_t host_lcg_state = 0;

void signal_handler(int signum) {
    (void)signum;
    keep_running = 0;
    printf("\n\nShutting down...\n");
}

// Helper to generate random 64-bit numbers (Same LCG as CPU)
static inline u64_t get_random_u64() {
    host_lcg_state = 6364136223846793005ul * host_lcg_state + 1442695040888963407ul;
    return host_lcg_state;
}

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

    // Fill remaining salt with random bytes
    while (current_idx <= end_idx) {
        u64_t rnd = get_random_u64();
        u08_t ascii_char = 32 + (u08_t)(( (u08_t)(rnd >> 56) * 95) >> 8);
        bytes[current_idx ^ 3] = ascii_char;
        current_idx++;
    }

    bytes[54 ^ 3] = (u08_t)'\n';
    bytes[55 ^ 3] = 0x80;
}

void run_cuda_miner(const char *custom_text, u64_t max_attempts, int gpu_device_id)
{
    #define N_STREAMS 4
    cuda_data_t cd;
    
    int custom_len = 0;
    if (custom_text != NULL) {
        size_t len = strlen(custom_text);
        custom_len = (len > MAX_CUSTOM_LEN) ? MAX_CUSTOM_LEN : (int)len;
    }

    CUstream streams[N_STREAMS];
    CUevent start_events[N_STREAMS]; 
    CUevent stop_events[N_STREAMS];  
    
    u32_t* h_vaults[N_STREAMS];
    CUdeviceptr d_vaults[N_STREAMS];
    u32_t* h_templates[N_STREAMS];
    CUdeviceptr d_templates[N_STREAMS];
    
    u64_t base_counters[N_STREAMS];
    // REMOVED: u64_t stream_salt_ages[N_STREAMS]; -- No longer needed

    void* kernel_args[N_STREAMS][3];
    
    memset(&cd, 0, sizeof(cuda_data_t));
    cd.device_number = gpu_device_id;
    cd.cubin_file_name = "miner_kernel.cubin";
    cd.kernel_name = "miner_kernel";
    
    cd.data_size[0] = 14 * sizeof(u32_t);
    cd.data_size[1] = 1024 * sizeof(u32_t);
    
    initialize_cuda(&cd);

    streams[0] = cd.cu_stream;
    CU_CALL( cuEventCreate, (&start_events[0], CU_EVENT_DEFAULT) );
    CU_CALL( cuEventCreate, (&stop_events[0], CU_EVENT_DEFAULT) );

    h_templates[0] = (u32_t*)cd.host_data[0];
    d_templates[0] = cd.device_data[0];
    h_vaults[0] = (u32_t*)cd.host_data[1];
    d_vaults[0] = cd.device_data[1];

    for(int i = 1; i < N_STREAMS; i++) {
        CU_CALL( cuStreamCreate, (&streams[i], CU_STREAM_NON_BLOCKING) );
        CU_CALL( cuEventCreate, (&start_events[i], CU_EVENT_DEFAULT) ); 
        CU_CALL( cuEventCreate, (&stop_events[i], CU_EVENT_DEFAULT) ); 
        
        CU_CALL( cuMemAllocHost, ((void **)&h_vaults[i], (size_t)cd.data_size[1]) );
        CU_CALL( cuMemAlloc, (&d_vaults[i], (size_t)cd.data_size[1]) );
        CU_CALL( cuMemAllocHost, ((void **)&h_templates[i], (size_t)cd.data_size[0]) );
        CU_CALL( cuMemAlloc, (&d_templates[i], (size_t)cd.data_size[0]) );
    }

    // --- OCCUPANCY OPTIMIZATION ---
    int minGridSize = 0;
    int blockSize = 0;
    CU_CALL( cuOccupancyMaxPotentialBlockSize, (&minGridSize, &blockSize, cd.cu_kernel, NULL, 0, 0) );
    cd.block_dim_x = blockSize;
    int target_threads = 262144;
    int calc_grid = (target_threads + blockSize - 1) / blockSize;
    if (calc_grid < minGridSize) calc_grid = minGridSize;
    cd.grid_dim_x = calc_grid;

    printf("[CUDA] Optimization: BlockSize=%d | GridSize=%d\n", cd.block_dim_x, cd.grid_dim_x);
    
    u64_t hashes_per_launch = (u64_t)cd.block_dim_x * (u64_t)cd.grid_dim_x * LOOP_SIZE;
    
    cd.n_kernel_arguments = 3;
    for(int i = 0; i < N_STREAMS; i++) {
        kernel_args[i][0] = &base_counters[i];
        kernel_args[i][1] = &d_vaults[i];
        kernel_args[i][2] = &d_templates[i];
    }

    printf("========================================\n");
    printf("DETI COIN MINER v2 (CUDA)\n");
    printf("Strategy: High Variance (New Salt Every Launch)\n");
    printf("Device: %s | Streams: %d\n", cd.device_name, N_STREAMS);
    printf("========================================\n");

    time_measurement();
    double start_time = measured_wall_time[1].tv_sec + measured_wall_time[1].tv_nsec * 1e-9;
    double last_report_time = start_time;

    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    host_lcg_state = ((u64_t)ts.tv_nsec) ^ ((u64_t)getpid() << 32);
    for(int k=0;k<10;k++) get_random_u64(); // Warmup

    // Prime Pipeline
    for(int s = 0; s < N_STREAMS; s++) {
        h_vaults[s][0] = 1u;
        generate_host_template(h_templates[s], custom_text, custom_len);
        
        CU_CALL( cuMemcpyHtoDAsync, (d_templates[s], h_templates[s], (size_t)cd.data_size[0], streams[s]) );
        CU_CALL( cuMemcpyHtoDAsync, (d_vaults[s], h_vaults[s], (size_t)cd.data_size[1], streams[s]) );

        base_counters[s] = get_random_u64();

        CU_CALL( cuEventRecord, (start_events[s], streams[s]) );
        CU_CALL( cuLaunchKernel , (cd.cu_kernel, cd.grid_dim_x, 1u, 1u, cd.block_dim_x, 1u, 1u, 0u, streams[s], &kernel_args[s][0], NULL) );
        CU_CALL( cuEventRecord, (stop_events[s], streams[s]) );
    }

    alarm(60); 

    // Main Loop
    for(int s = 0; keep_running; s = (s + 1) % N_STREAMS)
    {
        CU_CALL( cuStreamSynchronize, (streams[s]) );
        
        float elapsed_ms = 0.0f;
        CU_CALL( cuEventElapsedTime, (&elapsed_ms, start_events[s], stop_events[s]) );
        update_time_histogram(elapsed_ms);

        total_attempts += hashes_per_launch;

        // Check Vault
        CU_CALL( cuMemcpyDtoHAsync , ((void *)&h_vaults[s][0], d_vaults[s], sizeof(u32_t), streams[s]) );
        CU_CALL( cuStreamSynchronize, (streams[s]) ); 
        
        u32_t num_words = h_vaults[s][0];
        
        int coins_found_this_run = 0;
        if (num_words > 1) {
            coins_found_this_run = (int)((num_words - 1) / 14);
        }
        update_coin_histogram(coins_found_this_run);

        if(num_words > 1u)
        {
            if (num_words > 1024u) num_words = 1024u;
            CU_CALL( cuMemcpyDtoHAsync , ((void *)h_vaults[s], d_vaults[s], num_words * sizeof(u32_t), streams[s]) );
            CU_CALL( cuStreamSynchronize, (streams[s]) );

            for(u32_t i = 1; i < num_words; i += 14) {
                save_coin(&h_vaults[s][i]);
                total_coins_found++;
                u08_t *cb = (u08_t *)&h_vaults[s][i];
                printf("\n[Stream %d] FOUND! Nonce: %c\n", s,cb[53^3]);
            }
        }

        if(max_attempts != 0 && total_attempts >= max_attempts) keep_running = 0;

        if(keep_running) {
            // =========================================================
            // STRATEGY UPDATE: Always Refresh Salt (Like CPU No-SIMD)
            // =========================================================
            
            // 1. Generate NEW random template (Bytes 12-45)
            generate_host_template(h_templates[s], custom_text, custom_len);
            
            // 2. Upload template to GPU immediately
            CU_CALL( cuMemcpyHtoDAsync, (d_templates[s], h_templates[s], (size_t)cd.data_size[0], streams[s]) );

            // 3. Generate NEW random nonce base (Bytes 46-52)
            base_counters[s] = get_random_u64();
            
            // 4. Reset Vault
            h_vaults[s][0] = 1u;
            CU_CALL( cuMemcpyHtoDAsync, (d_vaults[s], h_vaults[s], (size_t)cd.data_size[1], streams[s]) );
            
            // 5. Launch Kernel
            CU_CALL( cuEventRecord, (start_events[s], streams[s]) );
            CU_CALL( cuLaunchKernel , (cd.cu_kernel, cd.grid_dim_x, 1u, 1u, cd.block_dim_x, 1u, 1u, 0u, streams[s], &kernel_args[s][0], NULL) );
            CU_CALL( cuEventRecord, (stop_events[s], streams[s]) );
        }

        // Reporting
        time_measurement();
        double current_time = measured_wall_time[1].tv_sec + measured_wall_time[1].tv_nsec * 1e-9;
        if(current_time - last_report_time >= 1.0)
        {
            double elapsed = current_time - start_time;
            printf("\r[%llu MH] [%llu coins] [%.2f MH/s] [Last Kernel: %.3f ms]   ",
                   (unsigned long long)(total_attempts/1000000),
                   (unsigned long long)total_coins_found,
                   (double)total_attempts / elapsed / 1e6,
                   elapsed_ms);
            fflush(stdout);
            last_report_time = current_time;
        }
    }
    
    // Cleanup logic (Same as before)
    for(int s = 1; s < N_STREAMS; s++) {
        if (d_vaults[s])    CU_CALL( cuMemFree, (d_vaults[s]) );
        if (d_templates[s]) CU_CALL( cuMemFree, (d_templates[s]) );
        if (h_vaults[s])    CU_CALL( cuMemFreeHost, (h_vaults[s]) );
        if (h_templates[s]) CU_CALL( cuMemFreeHost, (h_templates[s]) );
        if (streams[s])     CU_CALL( cuStreamDestroy, (streams[s]) );
        if (start_events[s]) CU_CALL( cuEventDestroy, (start_events[s]) );
        if (stop_events[s])  CU_CALL( cuEventDestroy, (stop_events[s]) );
    }
    CU_CALL( cuEventDestroy, (start_events[0]) );
    CU_CALL( cuEventDestroy, (stop_events[0]) );

    terminate_cuda(&cd);
    save_coin(NULL);
    save_gnuplot_data();

    time_measurement();
    double total_time = (measured_wall_time[1].tv_sec + measured_wall_time[1].tv_nsec * 1e-9) - start_time;
    printf("\n\n--- DONE ---\nAttempts: %llu Time: %.2fs\nAvg: %.2f MH/s\n", (unsigned long long)total_attempts, total_time, (double)total_attempts / total_time / 1e6);
}

int main(int argc, char *argv[])
{
    signal(SIGINT, signal_handler);
    const char *custom_text = NULL;
    u64_t max_attempts = 0;
    int gpu_device = DEFAULT_DEVICE_ID;
    
    if(argc > 1) custom_text = argv[1];
    if(argc > 2) max_attempts = strtoull(argv[2], NULL, 10);
    
    run_cuda_miner(custom_text, max_attempts, gpu_device);
    return 0;
}