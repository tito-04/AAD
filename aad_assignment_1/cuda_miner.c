//
// Ficheiro: cuda_miner.c
//
// Anfitrião (Host) para o minerador CUDA de DETI coins
// (Versão Otimizada com Pipeline Assíncrono)
//
// Arquitetura:
// 1. Usa N_STREAMS (ex: 2) para "ping-pong".
// 2. Enquanto a GPU corre o kernel no stream[0], a CPU processa resultados do stream[1].
// 3. A sincronização é feita manualmente no host com cuStreamSynchronize.
//
#include <time.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h> 

// Headers do projeto
#include "aad_data_types.h"
#include "aad_utilities.h" 
#include "aad_sha1_cpu.h"   
#include "aad_vault.h"      
#include "aad_cuda_utilities.h"

// --- Globais (para estatísticas e sinal) ---
static volatile int keep_running = 1;
static u64_t total_attempts = 0;
static u64_t total_coins_found = 0;
static unsigned int global_seed = 0;
static u64_t global_counter_offset = 0;

// --- Funções de Sinal (para Ctrl+C) ---
void signal_handler(int signum) {
    (void)signum;
    keep_running = 0;
    printf("\n\nShutting down gracefully...\n");
}

// --- Funções de Template (copiadas do seu código AVX) ---
static u32_t template_message[14];
static u32_t template_with_custom[14];
static int template_initialized = 0;
static int custom_template_initialized = 0;
static int custom_text_length = 0;

static void init_template(void) {
    if(template_initialized) return;
    memset(template_message, 0, 14 * sizeof(u32_t));
    u08_t *bytes = (u08_t *)template_message;
    const char header[] = "DETI coin 2 ";
    for(int i = 0; i < 12; i++) { bytes[i ^ 3] = (u08_t)header[i]; }
    bytes[54 ^ 3] = (u08_t)'\n';
    bytes[55 ^ 3] = 0x80;
    template_initialized = 1;
}

static void init_custom_template(const char *custom_text) {
    if(custom_template_initialized) return;
    if(!template_initialized) init_template();
    memcpy(template_with_custom, template_message, 14 * sizeof(u32_t));
    u08_t *bytes = (u08_t *)template_with_custom;
    int pos = 12;
    if(custom_text != NULL) {
        for(size_t i = 0; custom_text[i] != '\0' && pos < 54; i++, pos++) {
            char c = custom_text[i];
            if(c == '\n') c = '\b';
            bytes[pos ^ 3] = (u08_t)c;
        }
    }
    custom_text_length = pos - 12;
    custom_template_initialized = 1;
}

// --- O Minerador CUDA (Função Principal do Host) ---
void run_cuda_miner(const char *custom_text, u64_t max_attempts, int gpu_device_id)
{
    // --- OPTIMIZATION 1 ---
    // Usar 2 streams ("slots") para o pipeline "ping-pong"
    #define N_STREAMS 2 

    cuda_data_t cd;
    int start_pos = 12; 

    // --- OPTIMIZATION 1 ---
    // Arrays para gerir os "slots" do pipeline
    CUstream streams[N_STREAMS];
    u32_t* h_vaults[N_STREAMS];     // Buffers do Host (pinned)
    CUdeviceptr d_vaults[N_STREAMS]; // Buffers do Device
    u64_t base_counters[N_STREAMS];  // Contadores para cada slot
    void* kernel_args[N_STREAMS][3]; // Argumentos do kernel para cada slot
    
    // 1. Preparar Templates no Host
    init_template();
    if(custom_text != NULL) {
        init_custom_template(custom_text);
        start_pos += custom_text_length;
    }

    // 2. Inicializar CUDA (Configura o SLOT 0)
    memset(&cd, 0, sizeof(cuda_data_t));
    cd.device_number = gpu_device_id;
    cd.cubin_file_name = "miner_kernel.cubin"; 
    cd.kernel_name = "miner_kernel";
    cd.data_size[0] = 0;     // Buffer 0: Template
    cd.data_size[1] = 1024 * sizeof(u32_t); // Buffer 1: Vault (Slot 0)
    
    initialize_cuda(&cd);

    // --- OPTIMIZATION 1 ---
    // Guardar os recursos do SLOT 0 (criados por initialize_cuda)
    streams[0] = cd.cu_stream;
    h_vaults[0] = (u32_t*)cd.host_data[1];
    d_vaults[0] = cd.device_data[1];

    // Criar recursos para os restantes SLOTS (apenas SLOT 1 neste caso)
    for(int i = 1; i < N_STREAMS; i++)
    {
        CU_CALL( cuStreamCreate, (&streams[i], CU_STREAM_NON_BLOCKING) );
        // Alocar memória "pinned" (Host) para cópias assíncronas
        CU_CALL( cuMemAllocHost, ((void **)&h_vaults[i], (size_t)cd.data_size[1]) );
        // Alocar memória no Device
        CU_CALL( cuMemAlloc, (&d_vaults[i], (size_t)cd.data_size[1]) );
    }

    // 3. Copiar Template para a __constant__ memory
    // Esta lógica substitui a antiga secção "3. Copiar Template para a GPU"
    
    // 3a. Preparar o template num buffer local do host
    u32_t h_template_local[14]; 
    if(custom_text != NULL) {
        memcpy(h_template_local, template_with_custom, 14 * sizeof(u32_t));
    } else {
        memcpy(h_template_local, template_message, 14 * sizeof(u32_t));
    }

    // 3b. Obter o ponteiro de device para o símbolo "c_template_message"
    CUdeviceptr d_template_symbol;
    size_t symbol_size;
    CU_CALL( cuModuleGetGlobal , (&d_template_symbol, &symbol_size, cd.cu_module, "c_template_message") );

    // 3c. Copiar o template do host para o símbolo no device (síncrono)
    CU_CALL( cuMemcpyHtoD , (d_template_symbol, h_template_local, 14 * sizeof(u32_t)) );
    synchronize_cuda(&cd); // Garantir que a cópia única está completa

    // 4. Configurar Lançamento do Kernel
    cd.block_dim_x = (unsigned int)RECOMENDED_CUDA_BLOCK_SIZE;
    cd.grid_dim_x = 32768; 
    
    u64_t num_threads_per_launch = (u64_t)cd.block_dim_x * (u64_t)cd.grid_dim_x;
    u64_t current_base_counter = global_counter_offset;
    
    // --- OPTIMIZATION 1 ---
    // Configurar os argumentos para cada stream/slot
    cd.n_kernel_arguments = 3;
    for(int i = 0; i < N_STREAMS; i++)
    {
        base_counters[i] = current_base_counter;
        current_base_counter += num_threads_per_launch;

        kernel_args[i][0] = &base_counters[i];
        kernel_args[i][1] = &d_vaults[i];      // Vault *específico* do stream
        kernel_args[i][2] = &start_pos;  // Template (comum a todos)
    }


    printf("========================================\n");
    printf("DETI COIN MINER (CUDA Kernel - Device %d)\n", gpu_device_id);
    printf("GPU: %s\n", cd.device_name);
    printf("Grid Size: %u blocks, Block Size: %u threads\n", cd.grid_dim_x, cd.block_dim_x);
    printf("Hashes per launch: %llu\n", (unsigned long long)num_threads_per_launch);
    printf("Pipeline Slots: %d\n", N_STREAMS);
    printf("========================================\n\n");


    time_measurement(); 
    double start_time = measured_wall_time[1].tv_sec + measured_wall_time[1].tv_nsec * 1e-9;
    double last_report_time = start_time;

    // --- OPTIMIZATION 1 ---
    // 5. "Priming" do Pipeline: Lançar um kernel em CADA stream
    for(int s = 0; s < N_STREAMS; s++)
    {
        // a. Resetar o contador do vault
        h_vaults[s][0] = 1u;
        
        // b. Copiar o reset (assíncrono)
        CU_CALL( cuMemcpyHtoDAsync, (d_vaults[s], h_vaults[s], (size_t)cd.data_size[1], streams[s]) );

        // c. Lançar o Kernel (assíncrono)
        CU_CALL( cuLaunchKernel , (cd.cu_kernel, cd.grid_dim_x, 1u, 1u, cd.block_dim_x, 1u, 1u, 0u, streams[s], &kernel_args[s][0], NULL) );
    }

    // 6. Loop Principal de Mineração (Pipeline)
    for(int s = 0; keep_running; s = (s + 1) % N_STREAMS)
    {
        // a. Sincronizar (ESPERAR) pelo slot 's' (o mais antigo)
        CU_CALL( cuStreamSynchronize, (streams[s]) );

        // Neste ponto, o kernel [s] terminou. A CPU pode processar os seus resultados.
        total_attempts += num_threads_per_launch;

        // b. Processar moedas encontradas no slot 's'
        // (A cópia de DtoH foi síncrona, por isso h_vaults[s][0] não é válido)
        // Precisamos de copiar o contador de volta primeiro.
        CU_CALL( cuMemcpyDtoHAsync , ((void *)&h_vaults[s][0], d_vaults[s], sizeof(u32_t), streams[s]) );
        // Esperar *apenas* por esta pequena cópia
        CU_CALL( cuStreamSynchronize, (streams[s]) ); 
        
        u32_t num_words_in_vault = h_vaults[s][0];

        if(num_words_in_vault > 1u && num_words_in_vault < 1024u)
        {
            // Copiar o resto do vault (assíncrono)
            CU_CALL( cuMemcpyDtoHAsync , ((void *)h_vaults[s], d_vaults[s], num_words_in_vault * sizeof(u32_t), streams[s]) );
            // Esperar pela cópia
            CU_CALL( cuStreamSynchronize, (streams[s]) ); 

            for(u32_t i = 1; i < num_words_in_vault; i += 14)
            {
                save_coin(&h_vaults[s][i]); 
                total_coins_found++;
                printf("\n>>> COIN FOUND! (CUDA Device %d) Counter ~%llu\n", gpu_device_id, (unsigned long long)base_counters[s]);
            }
        }

        // c. Verificar se paramos
        if(max_attempts != 0 && total_attempts >= max_attempts) {
            keep_running = 0;
        }

        // d. RELANÇAR o slot 's' com novo trabalho
        // (Enquanto a CPU faz isto, a GPU está a trabalhar nos outros N-1 slots)
        base_counters[s] = current_base_counter;
        current_base_counter += num_threads_per_launch;

        h_vaults[s][0] = 1u;
        
        // Copiar o reset (assíncrono)
        CU_CALL( cuMemcpyHtoDAsync, (d_vaults[s], h_vaults[s], (size_t)cd.data_size[1], streams[s]) );
        
        // Lançar o Kernel (assíncrono)
        CU_CALL( cuLaunchKernel , (cd.cu_kernel, cd.grid_dim_x, 1u, 1u, cd.block_dim_x, 1u, 1u, 0u, streams[s], &kernel_args[s][0], NULL) );

        // e. Reportar
        time_measurement();
        double current_time = measured_wall_time[1].tv_sec + measured_wall_time[1].tv_nsec * 1e-9;
        if(current_time - last_report_time >= 1.0) 
        {
            double elapsed = current_time - start_time;
            double rate = (double)total_attempts / elapsed;
            printf("\r[Device %d] [%llu attempts] [%llu coins] [%.2f MH/s]          ",
                   gpu_device_id,
                   (unsigned long long)total_attempts,
                   (unsigned long long)total_coins_found,
                   rate / 1e6);
            fflush(stdout);
            last_report_time = current_time;
        }
    }
    
    // Sincronizar tudo antes de sair
    for(int s = 0; s < N_STREAMS; s++) {
        CU_CALL( cuStreamSynchronize, (streams[s]) );
    }

    // 7. Limpeza
    
    // --- OPTIMIZATION 1 ---
    // Limpar os recursos extra que criámos
    for(int i = 1; i < N_STREAMS; i++)
    {
        CU_CALL( cuStreamDestroy, (streams[i]) );
        CU_CALL( cuMemFreeHost, (h_vaults[i]) );
        CU_CALL( cuMemFree, (d_vaults[i]) );
    }
    
    terminate_cuda(&cd); // Limpa o stream[0] e os buffers[0]
    save_coin(NULL); 

    // Estatísticas Finais
    time_measurement();
    double end_time = measured_wall_time[1].tv_sec + measured_wall_time[1].tv_nsec * 1e-9;
    double total_time = end_time - start_time;
    
    printf("\n\n========================================\n");
    printf("FINAL STATISTICS (CUDA Device %d)\n", gpu_device_id);
    printf("========================================\n");
    printf("Total attempts: %llu\n", (unsigned long long)total_attempts);
    printf("Total coins found: %llu\n", (unsigned long long)total_coins_found);
    printf("Total time: %.2f seconds\n", total_time);
    
    if (total_time > 0) {
        printf("Average rate: %.2f million hashes/second\n", (double)total_attempts / total_time / 1e6);
    }
    printf("========================================\n");
}

// --- Ponto de Entrada Principal ---
int main(int argc, char *argv[])
{
    signal(SIGINT, signal_handler);
    
    // Inicializar sementes (copiado do seu código AVX)
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    global_seed = (unsigned int)(time(NULL) ^ (uintptr_t)&global_seed ^ (unsigned int)getpid() ^ (unsigned int)ts.tv_nsec);
    for(int i = 0; i < 10; i++) { rand_r(&global_seed); }
    global_counter_offset = ((u64_t)rand_r(&global_seed) << 32) | (u64_t)rand_r(&global_seed);
    clock_gettime(CLOCK_MONOTONIC, &ts);
    global_counter_offset ^= ((u64_t)ts.tv_nsec << 32) | (u64_t)ts.tv_sec;

    // --- Nova Lógica de Argumentos ---
    const char *custom_text = NULL;
    u64_t max_attempts = 0;
    int gpu_device_id = 0; 
    
    if(argc > 1) {
        gpu_device_id = (int)strtol(argv[1], NULL, 10);
    }
    if(argc > 2)
        custom_text = argv[2];
    if(argc > 3)
        max_attempts = strtoull(argv[3], NULL, 10);
    
    // Sanitizar texto
    char sanitized_text[64];
    if(custom_text != NULL) {
        size_t si = 0;
        for(size_t i = 0; custom_text[i] != '\0' && si + 1 < sizeof(sanitized_text); i++) {
            if(custom_text[i] == '\n') sanitized_text[si++] = '\b';
            else sanitized_text[si++] = custom_text[i];
        }
        sanitized_text[si] = '\0';
        run_cuda_miner(sanitized_text, max_attempts, gpu_device_id);
    } else {
        run_cuda_miner(NULL, max_attempts, gpu_device_id);
    }
    
    return 0;
}
