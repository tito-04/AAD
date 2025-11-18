//
// Ficheiro: cuda_miner.c
//
// Host otimizado para 4 Streams e suporte completo a Custom Text
// Estratégia:
// 1. Host gera Bytes 00-45 (Texto + Salt Lento) e Bytes 54-55 (Footer)
// 2. GPU gera Bytes 46-53 (Fast Nonce)
//
// Compile: nvcc -O3 -arch=sm_XX cuda_miner.c aad_cuda_utilities.c -o miner_cuda
//

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

// Defaults
#define DEFAULT_DEVICE_ID 0
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

// --- GERAÇÃO DO TEMPLATE (Igual ao SIMD) ---
// Preenche bytes 0-45 e 54-55. A GPU sobrescreve 46-53.
static void generate_host_template(u32_t *buffer, const char *custom_text, int custom_len) {
    u08_t *bytes = (u08_t *)buffer;
    
    // 1. Header (0-11)
    const char header[] = "DETI coin 2 ";
    for(int i = 0; i < 12; i++) { bytes[i ^ 3] = (u08_t)header[i]; }

    // 2. Custom Text + Salt (12-45)
    int current_idx = 12;
    const int end_idx = 45;
    int text_pos = 0;

    // A. Inserir Texto Customizado (se existir)
    if (custom_text != NULL) {
        while(text_pos < custom_len && current_idx <= end_idx) {
            char c = custom_text[text_pos++];
            if(c < 32 || c > 126) c = ' '; // Sanitização
            bytes[current_idx ^ 3] = (u08_t)c;
            current_idx++;
        }
    }

    // B. Preencher o resto com Random Salt (LCG Host)
    while (current_idx <= end_idx) {
        host_lcg_state = 6364136223846793005ul * host_lcg_state + 1442695040888963407ul;
        u08_t random_raw = (u08_t)(host_lcg_state >> 56);
        u08_t ascii_char = 32 + (u08_t)((random_raw * 95) >> 8);
        bytes[current_idx ^ 3] = ascii_char;
        current_idx++;
    }

    // 3. Footer (54-55)
    bytes[54 ^ 3] = (u08_t)'\n';
    bytes[55 ^ 3] = 0x80;
}

void run_cuda_miner(const char *custom_text, u64_t max_attempts, int gpu_device_id)
{
    #define N_STREAMS 4

    cuda_data_t cd;
    
    // Calcular tamanho do texto customizado com verificação
    int custom_len = 0;
    if (custom_text != NULL) {
        size_t len = strlen(custom_text);
        if (len > MAX_CUSTOM_LEN) {
            printf("!!! WARNING !!!\n");
            printf("Custom text (%zu bytes) exceeds the variable template size (%d bytes).\n", len, MAX_CUSTOM_LEN);
            printf("Text will be truncated to fit.\n");
            printf("!!! WARNING !!!\n\n");
            custom_len = MAX_CUSTOM_LEN;
        } else {
            custom_len = (int)len;
        }
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
    
    cd.data_size[0] = 14 * sizeof(u32_t); // Template Size
    cd.data_size[1] = 1024 * sizeof(u32_t); // Vault Size
    
    initialize_cuda(&cd);

    // Stream 0 setup (usando alocações da struct base)
    streams[0] = cd.cu_stream;
    h_templates[0] = (u32_t*)cd.host_data[0];
    d_templates[0] = cd.device_data[0];
    h_vaults[0] = (u32_t*)cd.host_data[1];
    d_vaults[0] = cd.device_data[1];

    // Streams 1..3 setup
    for(int i = 1; i < N_STREAMS; i++)
    {
        CU_CALL( cuStreamCreate, (&streams[i], CU_STREAM_NON_BLOCKING) );
        CU_CALL( cuMemAllocHost, ((void **)&h_vaults[i], (size_t)cd.data_size[1]) );
        CU_CALL( cuMemAlloc, (&d_vaults[i], (size_t)cd.data_size[1]) );
        CU_CALL( cuMemAllocHost, ((void **)&h_templates[i], (size_t)cd.data_size[0]) );
        CU_CALL( cuMemAlloc, (&d_templates[i], (size_t)cd.data_size[0]) );
    }

    cd.block_dim_x = (unsigned int)RECOMENDED_CUDA_BLOCK_SIZE;
    cd.grid_dim_x = 32768; // Ajustável conforme GPU
    
    u64_t num_threads_per_launch = (u64_t)cd.block_dim_x * (u64_t)cd.grid_dim_x;
    
    // Inicializa LCG base_counter
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    u64_t current_base_counter = (u64_t)ts.tv_nsec ^ ((u64_t)getpid() << 32);
    
    cd.n_kernel_arguments = 3;
    for(int i = 0; i < N_STREAMS; i++)
    {
        kernel_args[i][0] = &base_counters[i];
        kernel_args[i][1] = &d_vaults[i];     
        kernel_args[i][2] = &d_templates[i]; 
    }

    printf("========================================\n");
    printf("DETI COIN MINER (CUDA - 8 BYTE FAST NONCE)\n");
    printf("Device %d: %s\n", gpu_device_id, cd.device_name);
    printf("Streams: %d\n", N_STREAMS);
    printf("Bytes 00-11: Header\n");
    printf("Bytes 12-45: Custom + Slow Random (Max %d Bytes)\n", MAX_CUSTOM_LEN);
    printf("Bytes 46-53: Fast Random Nonce (Update every Kernel)\n");
    printf("Bytes 54-55: Footer (\\n, 0x80)\n");
    if (custom_text != NULL && custom_len > 0) {
        printf("Custom Text Length: %d bytes\n", custom_len);
    }
    printf("========================================\n");

    time_measurement(); 
    double start_time = measured_wall_time[1].tv_sec + measured_wall_time[1].tv_nsec * 1e-9;
    double last_report_time = start_time;

    // Pipeline Priming
    for(int s = 0; s < N_STREAMS; s++)
    {
        h_vaults[s][0] = 1u; // Reset vault count
        
        // Atualiza Salt Lento no Host
        generate_host_template(h_templates[s], custom_text, custom_len);

        CU_CALL( cuMemcpyHtoDAsync, (d_templates[s], h_templates[s], (size_t)cd.data_size[0], streams[s]) );
        CU_CALL( cuMemcpyHtoDAsync, (d_vaults[s], h_vaults[s], (size_t)cd.data_size[1], streams[s]) );

        base_counters[s] = current_base_counter;
        current_base_counter += num_threads_per_launch;

        CU_CALL( cuLaunchKernel , (cd.cu_kernel, cd.grid_dim_x, 1u, 1u, cd.block_dim_x, 1u, 1u, 0u, streams[s], &kernel_args[s][0], NULL) );
    }

    // Main Loop
    for(int s = 0; keep_running; s = (s + 1) % N_STREAMS)
    {
        CU_CALL( cuStreamSynchronize, (streams[s]) );
        total_attempts += num_threads_per_launch;

        // Download Vault header only
        CU_CALL( cuMemcpyDtoHAsync , ((void *)&h_vaults[s][0], d_vaults[s], sizeof(u32_t), streams[s]) );
        CU_CALL( cuStreamSynchronize, (streams[s]) ); 
        
        u32_t num_words_in_vault = h_vaults[s][0];

        if(num_words_in_vault > 1u)
        {
            // Download found coins
            if (num_words_in_vault > 1024u) num_words_in_vault = 1024u; // Safety cap
            CU_CALL( cuMemcpyDtoHAsync , ((void *)h_vaults[s], d_vaults[s], num_words_in_vault * sizeof(u32_t), streams[s]) );
            CU_CALL( cuStreamSynchronize, (streams[s]) ); 

            for(u32_t i = 1; i < num_words_in_vault; i += 14)
            {
                save_coin(&h_vaults[s][i]); 
                total_coins_found++;
                
                // Print Nonce (Bytes 46-53)
                // Usamos ^3 para ler bytes correctamente de um buffer u32 formatado para SHA1 BigEndian
                u08_t *coin_bytes = (u08_t *)&h_vaults[s][i];
                char nonce_str[9];
                for(int k=0; k<8; k++) nonce_str[k] = coin_bytes[(46+k)^3];
                nonce_str[8] = '\0';
                
                printf("\n[Stream %d] Coin Found! Nonce: %s\n", s, nonce_str);
            }
        }

        if(max_attempts != 0 && total_attempts >= max_attempts) keep_running = 0;

        if(keep_running) {
            // REGENERA O TEMPLATE (Salt Lento muda a cada Kernel Launch)
            generate_host_template(h_templates[s], custom_text, custom_len);
            
            base_counters[s] = current_base_counter;
            current_base_counter += num_threads_per_launch;

            h_vaults[s][0] = 1u; // Reset vault
            
            CU_CALL( cuMemcpyHtoDAsync, (d_templates[s], h_templates[s], (size_t)cd.data_size[0], streams[s]) );
            CU_CALL( cuMemcpyHtoDAsync, (d_vaults[s], h_vaults[s], (size_t)cd.data_size[1], streams[s]) );
            
            CU_CALL( cuLaunchKernel , (cd.cu_kernel, cd.grid_dim_x, 1u, 1u, cd.block_dim_x, 1u, 1u, 0u, streams[s], &kernel_args[s][0], NULL) );
        }

        // Stats
        time_measurement();
        double current_time = measured_wall_time[1].tv_sec + measured_wall_time[1].tv_nsec * 1e-9;
        if(current_time - last_report_time >= 1.0) 
        {
            double elapsed = current_time - start_time;
            printf("\r[%llu attempts] [%llu coins] [%.2f MH/s]    ",
                   (unsigned long long)total_attempts,
                   (unsigned long long)total_coins_found,
                   (double)total_attempts / elapsed / 1e6);
            fflush(stdout);
            last_report_time = current_time;
        }
    }
    
    // Cleanup
    for(int s = 0; s < N_STREAMS; s++) CU_CALL( cuStreamSynchronize, (streams[s]) );

    for(int i = 1; i < N_STREAMS; i++)
    {
        CU_CALL( cuStreamDestroy, (streams[i]) );
        CU_CALL( cuMemFreeHost, (h_vaults[i]) );
        CU_CALL( cuMemFree, (d_vaults[i]) );
        CU_CALL( cuMemFreeHost, (h_templates[i]) );
        CU_CALL( cuMemFree, (d_templates[i]) );
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
    
    // RNG Init
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    unsigned int seed = (unsigned int)(time(NULL) ^ (uintptr_t)&ts ^ (unsigned int)getpid());
    host_lcg_state = ((u64_t)seed << 32) | (u64_t)ts.tv_nsec;
    // Warmup
    for(int i=0; i<5; i++) host_lcg_state = 6364136223846793005ul * host_lcg_state + 1442695040888963407ul;

    const char *custom_text = NULL;
    u64_t max_attempts = 0;
    int gpu_device = DEFAULT_DEVICE_ID;
    
    // Parse Arguments Simplificado
    if(argc > 1) custom_text = argv[1];
    if(argc > 2) max_attempts = strtoull(argv[2], NULL, 10);
    // Opcional: se quiser forçar device, pode ser via var de ambiente ou hardcoded, 
    // mas aqui removemos a obrigatoriedade do argumento.
    
    run_cuda_miner(custom_text, max_attempts, gpu_device);
    
    return 0;
}