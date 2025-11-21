#include <time.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h> 

#include "aad_data_types.h"
#include "aad_utilities.h" 
#include "aad_sha1_cpu.h"   
#include "aad_vault.h"      
#include "aad_cuda_utilities.h"

#define DEFAULT_DEVICE_ID 0
#define LOOP_SIZE 95  // Characters 32 to 126

static volatile int keep_running = 1;
static u64_t total_attempts = 0;
static u64_t total_coins_found = 0;
static u64_t host_lcg_state = 0;

void signal_handler(int signum) {
    (void)signum;
    keep_running = 0;
    printf("\n\nShutting down...\n");
}

// Gera Template (Bytes 0-45 e 54-55)
static void generate_host_template(u32_t *buffer, const char *custom_text, int custom_len) {
    u08_t *bytes = (u08_t *)buffer;
    
    // Header
    const char header[] = "DETI coin 2 ";
    for(int i = 0; i < 12; i++) bytes[i ^ 3] = (u08_t)header[i];

    // Custom Text + Salt (12-45)
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

    // Footer
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
        custom_len = (len > 34) ? 34 : (int)len;
    }

    CUstream streams[N_STREAMS];
    u32_t* h_vaults[N_STREAMS];      
    CUdeviceptr d_vaults[N_STREAMS]; 
    u32_t* h_templates[N_STREAMS];     
    CUdeviceptr d_templates[N_STREAMS]; 
    u64_t base_counters[N_STREAMS];  
    void* kernel_args[N_STREAMS][3]; 
    
    memset(&cd, 0, sizeof(cuda_data_t));
    cd.device_number = gpu_device_id;
    cd.cubin_file_name = "miner_kernel.cubin"; 
    cd.kernel_name = "miner_kernel";
    
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

    // --- OCCUPANCY OPTIMIZATION START ---
    int minGridSize = 0;
    int blockSize = 0;
    
    // Calculate optimal block size dynamically using Driver API
    // NOTE: We MUST separate the function name and arguments with a comma for the CU_CALL macro
    CU_CALL( cuOccupancyMaxPotentialBlockSize, (&minGridSize, &blockSize, cd.cu_kernel, NULL, 0, 0) );
    
    cd.block_dim_x = blockSize;
    
    // Target roughly ~250k threads in flight to hide memory latency and instruction stalls
    int target_threads = 262144; 
    int calc_grid = (target_threads + blockSize - 1) / blockSize;
    
    // Ensure we meet the minimum grid size suggested by CUDA
    if (calc_grid < minGridSize) calc_grid = minGridSize;
    
    cd.grid_dim_x = calc_grid;

    printf("[CUDA] Optimization: BlockSize=%d | GridSize=%d\n", cd.block_dim_x, cd.grid_dim_x);
    // --- OCCUPANCY OPTIMIZATION END ---
    
    u64_t threads_per_launch = (u64_t)cd.block_dim_x * (u64_t)cd.grid_dim_x;
    u64_t hashes_per_launch = threads_per_launch * LOOP_SIZE;
    
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    u64_t current_base_counter = (u64_t)ts.tv_nsec ^ ((u64_t)getpid() << 32);
    
    cd.n_kernel_arguments = 3;
    for(int i = 0; i < N_STREAMS; i++) {
        kernel_args[i][0] = &base_counters[i];
        kernel_args[i][1] = &d_vaults[i];     
        kernel_args[i][2] = &d_templates[i]; 
    }

    printf("========================================\n");
    printf("DETI COIN MINER v2 (NONCE GRINDING)\n");
    printf("Device: %s | Streams: %d\n", cd.device_name, N_STREAMS);
    printf("Grid: %d blocks | Loop: %d iter/thread\n", cd.grid_dim_x, LOOP_SIZE);
    printf("Launch Throughput: %.2f MHashes\n", (double)hashes_per_launch / 1e6);
    printf("========================================\n");

    time_measurement(); 
    double start_time = measured_wall_time[1].tv_sec + measured_wall_time[1].tv_nsec * 1e-9;
    double last_report_time = start_time;

    // Prime Pipeline
    for(int s = 0; s < N_STREAMS; s++) {
        h_vaults[s][0] = 1u; 
        generate_host_template(h_templates[s], custom_text, custom_len);

        CU_CALL( cuMemcpyHtoDAsync, (d_templates[s], h_templates[s], (size_t)cd.data_size[0], streams[s]) );
        CU_CALL( cuMemcpyHtoDAsync, (d_vaults[s], h_vaults[s], (size_t)cd.data_size[1], streams[s]) );

        base_counters[s] = current_base_counter;
        current_base_counter += threads_per_launch; 

        CU_CALL( cuLaunchKernel , (cd.cu_kernel, cd.grid_dim_x, 1u, 1u, cd.block_dim_x, 1u, 1u, 0u, streams[s], &kernel_args[s][0], NULL) );
    }

    // Main Loop
    for(int s = 0; keep_running; s = (s + 1) % N_STREAMS)
    {
        CU_CALL( cuStreamSynchronize, (streams[s]) );
        total_attempts += hashes_per_launch; // Update count with Loop Factor

        // Check Vault
        CU_CALL( cuMemcpyDtoHAsync , ((void *)&h_vaults[s][0], d_vaults[s], sizeof(u32_t), streams[s]) );
        CU_CALL( cuStreamSynchronize, (streams[s]) ); 
        
        u32_t num_words = h_vaults[s][0];
        if(num_words > 1u)
        {
            if (num_words > 1024u) num_words = 1024u;
            CU_CALL( cuMemcpyDtoHAsync , ((void *)h_vaults[s], d_vaults[s], num_words * sizeof(u32_t), streams[s]) );
            CU_CALL( cuStreamSynchronize, (streams[s]) ); 

            for(u32_t i = 1; i < num_words; i += 14) {
                save_coin(&h_vaults[s][i]); 
                total_coins_found++;
                
                // Print found nonce for debug
                u08_t *cb = (u08_t *)&h_vaults[s][i];
                printf("\n[Stream %d] FOUND! Nonce: %c%c%c%c%c%c%c%c\n", s,
                    cb[46^3], cb[47^3], cb[48^3], cb[49^3], cb[50^3], cb[51^3], cb[52^3], cb[53^3]);
            }
        }

        if(max_attempts != 0 && total_attempts >= max_attempts) keep_running = 0;

        if(keep_running) {
            generate_host_template(h_templates[s], custom_text, custom_len);
            base_counters[s] = current_base_counter;
            current_base_counter += threads_per_launch; // Note: Counter increments by THREADS, not Hashes

            h_vaults[s][0] = 1u; 
            
            CU_CALL( cuMemcpyHtoDAsync, (d_templates[s], h_templates[s], (size_t)cd.data_size[0], streams[s]) );
            CU_CALL( cuMemcpyHtoDAsync, (d_vaults[s], h_vaults[s], (size_t)cd.data_size[1], streams[s]) );
            
            CU_CALL( cuLaunchKernel , (cd.cu_kernel, cd.grid_dim_x, 1u, 1u, cd.block_dim_x, 1u, 1u, 0u, streams[s], &kernel_args[s][0], NULL) );
        }

        // Reporting
        time_measurement();
        double current_time = measured_wall_time[1].tv_sec + measured_wall_time[1].tv_nsec * 1e-9;
        if(current_time - last_report_time >= 1.0) 
        {
            double elapsed = current_time - start_time;
            printf("\r[%llu MH] [%llu coins] [%.2f MH/s]    ",
                   (unsigned long long)(total_attempts/1000000),
                   (unsigned long long)total_coins_found,
                   (double)total_attempts / elapsed / 1e6);
            fflush(stdout);
            last_report_time = current_time;
        }
    }
    
    for(int s = 1; s < N_STREAMS; s++) {
        // Free Device Memory
        if (d_vaults[s])    CU_CALL( cuMemFree, (d_vaults[s]) );
        if (d_templates[s]) CU_CALL( cuMemFree, (d_templates[s]) );
        
        // Free Host (Pinned) Memory - CRITICAL to prevent RAM leaks
        if (h_vaults[s])    CU_CALL( cuMemFreeHost, (h_vaults[s]) );
        if (h_templates[s]) CU_CALL( cuMemFreeHost, (h_templates[s]) );
        
        // Destroy Stream
        if (streams[s])     CU_CALL( cuStreamDestroy, (streams[s]) );
    }

    terminate_cuda(&cd); 
    save_coin(NULL); 

    time_measurement();
    double total_time = (measured_wall_time[1].tv_sec + measured_wall_time[1].tv_nsec * 1e-9) - start_time;
    printf("\n\n--- DONE ---\nTime: %.2fs\nAvg: %.2f MH/s\n", total_time, (double)total_attempts / total_time / 1e6);
}

int main(int argc, char *argv[])
{
    signal(SIGINT, signal_handler);
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    unsigned int seed = (unsigned int)(time(NULL) ^ (uintptr_t)&ts ^ (unsigned int)getpid());
    host_lcg_state = ((u64_t)seed << 32) | (u64_t)ts.tv_nsec;
    for(int i=0; i<5; i++) host_lcg_state = 6364136223846793005ul * host_lcg_state + 1442695040888963407ul;

    const char *custom_text = NULL;
    u64_t max_attempts = 0;
    int gpu_device = DEFAULT_DEVICE_ID;
    
    if(argc > 1) custom_text = argv[1];
    if(argc > 2) max_attempts = strtoull(argv[2], NULL, 10);
    
    run_cuda_miner(custom_text, max_attempts, gpu_device);
    return 0;
}