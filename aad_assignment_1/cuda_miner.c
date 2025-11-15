//
// Ficheiro: cuda_miner.c
//
// Anfitrião (Host) para o minerador CUDA de DETI coins.
//
// Arquitetura:
// 1. Prepara os templates de mensagem (base e custom) no CPU.
// 2. Inicializa o CUDA e aloca dois buffers na GPU:
//    - data[0]: 14 * u32_t (para guardar o template)
//    - data[1]: 1024 * u32_t (para o vault de resultados)
// 3. Copia o template escolhido do CPU para o data[0] da GPU.
// 4. Inicia um loop (while keep_running):
//    a. Define o contador do vault (h_vault[0]) para 1.
//    b. Copia h_vault[0] para o data[1] da GPU.
//    c. Lança o 'miner_kernel' com o 'base_counter' atualizado.
//    d. Copia o data[1] (vault) da GPU de volta para o h_vault.
//    e. Verifica se h_vault[0] > 1. Se sim, processa as moedas
//       encontradas e guarda-as no vault do disco.
//    f. Atualiza 'base_counter' e reporta o progresso.
// 5. Limpa e termina.
//

#include <time.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h> // Para getpid()

// Headers do projeto
#include "aad_data_types.h"
#include "aad_utilities.hh" // O seu ficheiro tem .hh, assumindo que é .h
#include "aad_sha1_cpu.h"   // Necessário para o save_coin()
#include "aad_vault.h"      // Para save_coin()
#include "aad_cuda_utilities.h" // Para as funções CUDA

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
// Estas são necessárias para preparar o template que será enviado para a GPU.
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
void run_cuda_miner(const char *custom_text, u64_t max_attempts)
{
    cuda_data_t cd;
    u32_t *h_template, *h_vault;
    int start_pos = 12; // Posição default de início do PRNG

    // 1. Preparar Templates no Host
    init_template();
    if(custom_text != NULL) {
        init_custom_template(custom_text);
        start_pos += custom_text_length;
    }

    // 2. Inicializar CUDA
    memset(&cd, 0, sizeof(cuda_data_t));
    cd.device_number = 0;
    cd.cubin_file_name = "miner_kernel.cubin"; // Nome do seu .cubin compilado
    cd.kernel_name = "miner_kernel";
    // Buffer 0: 14 * u32_t (para o template da mensagem)
    cd.data_size[0] = 14 * sizeof(u32_t);
    // Buffer 1: 1024 * u32_t (para o vault de resultados)
    cd.data_size[1] = 1024 * sizeof(u32_t);
    
    initialize_cuda(&cd);

    // Obter ponteiros de host para os buffers
    h_template = (u32_t *)cd.host_data[0];
    h_vault    = (u32_t *)cd.host_data[1];

    // 3. Copiar Template para a GPU (só uma vez)
    if(custom_text != NULL) {
        memcpy(h_template, template_with_custom, 14 * sizeof(u32_t));
    } else {
        memcpy(h_template, template_message, 14 * sizeof(u32_t));
    }
    host_to_device_copy(&cd, 0); // Copia h_template para cd.device_data[0]

    // 4. Configurar Lançamento do Kernel
    cd.block_dim_x = (unsigned int)RECOMENDED_CUDA_BLOCK_SIZE;
    
    // Configurar o Grid (ex: 8192 blocos)
    // Um número maior mantém a GPU ocupada por mais tempo.
    cd.grid_dim_x = 8192; 
    
    u64_t num_threads_per_launch = (u64_t)cd.block_dim_x * (u64_t)cd.grid_dim_x;
    u64_t base_counter = global_counter_offset;
    
    cd.n_kernel_arguments = 4;
    cd.arg[0] = &base_counter;        // u64_t base_counter
    cd.arg[1] = &cd.device_data[1];   // u32_t *coins_storage_area
    cd.arg[2] = &cd.device_data[0];   // const u32_t *d_template_message
    cd.arg[3] = &start_pos;           // int start_pos

    printf("========================================\n");
    printf("DETI COIN MINER (CUDA Kernel)\n");
    printf("GPU: %s\n", cd.device_name);
    printf("========================================\n");
    printf("Grid Size: %u blocks, Block Size: %u threads\n", cd.grid_dim_x, cd.block_dim_x);
    printf("Hashes per launch: %llu\n", (unsigned long long)num_threads_per_launch);
    if(custom_text) printf("Custom text: %s\n", custom_text);
    printf("Press Ctrl+C to stop\n");
    printf("========================================\n\n");

    time_measurement(); // Iniciar relógio principal
    double start_time = measured_wall_time[1].tv_sec + measured_wall_time[1].tv_nsec * 1e-9;
    double last_report_time = start_time;

    // 5. Loop Principal de Mineração
    while(keep_running)
    {
        // a. Resetar o contador do vault da GPU para 1
        h_vault[0] = 1u;
        
        // b. Copiar o contador resetado para a GPU
        // (Nota: host_to_device_copy copia o buffer todo, mas é pequeno)
        host_to_device_copy(&cd, 1); 

        // c. Lançar o Kernel! (arg[0] é atualizado automaticamente)
        lauch_kernel(&cd); // Esta função bloqueia até o kernel terminar

        // d. Copiar resultados de volta do vault da GPU
        device_to_host_copy(&cd, 1);

        // e. Processar moedas encontradas
        u32_t num_words_in_vault = h_vault[0];
        if(num_words_in_vault > 1u)
        {
            // O contador é > 1, encontrámos moedas!
            // Iterar pelas moedas (cada uma tem 14 palavras)
            for(u32_t i = 1; i < num_words_in_vault; i += 14)
            {
                // h_vault[i] é o início de uma moeda encontrada
                save_coin(&h_vault[i]); // save_coin verifica o hash
                total_coins_found++;
                
                // (Opcional) Imprimir notificação
                printf("\n>>> COIN FOUND! (CUDA) Counter ~%llu\n", (unsigned long long)base_counter);
            }
        }

        // f. Atualizar contadores e reportar
        base_counter += num_threads_per_launch;
        total_attempts += num_threads_per_launch;

        if(max_attempts != 0 && total_attempts >= max_attempts) {
            keep_running = 0;
        }

        time_measurement();
        double current_time = measured_wall_time[1].tv_sec + measured_wall_time[1].tv_nsec * 1e-9;
        if(current_time - last_report_time >= 1.0) // Reportar a cada segundo
        {
            double elapsed = current_time - start_time;
            double rate = (double)total_attempts / elapsed;
            printf("\r[%llu attempts] [%llu coins] [%.2f MH/s]          ",
                   (unsigned long long)total_attempts,
                   (unsigned long long)total_coins_found,
                   rate / 1e6);
            fflush(stdout);
            last_report_time = current_time;
        }
    }

    // 6. Limpeza
    terminate_cuda(&cd);
    save_coin(NULL); // Salvar moedas restantes no buffer do vault

    // Estatísticas Finais
    time_measurement();
    double end_time = measured_wall_time[1].tv_sec + measured_wall_time[1].tv_nsec * 1e-9;
    double total_time = end_time - start_time;
    
    printf("\n\n========================================\n");
    printf("FINAL STATISTICS (CUDA)\n");
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
    
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    global_seed = (unsigned int)(time(NULL) ^ (uintptr_t)&global_seed ^ (unsigned int)getpid() ^ (unsigned int)ts.tv_nsec);
    for(int i = 0; i < 10; i++) { rand_r(&global_seed); }
    global_counter_offset = ((u64_t)rand_r(&global_seed) << 32) | (u64_t)rand_r(&global_seed);
    clock_gettime(CLOCK_MONOTONIC, &ts);
    global_counter_offset ^= ((u64_t)ts.tv_nsec << 32) | (u64_t)ts.tv_sec;

    // Obter argumentos
    const char *custom_text = NULL;
    u64_t max_attempts = 0;
    if(argc > 1) custom_text = argv[1];
    if(argc > 2) max_attempts = strtoull(argv[2], NULL, 10);

    // Sanitizar texto
    char sanitized_text[64];
    if(custom_text != NULL) {
        size_t si = 0;
        for(size_t i = 0; custom_text[i] != '\0' && si + 1 < sizeof(sanitized_text); i++) {
            if(custom_text[i] == '\n') sanitized_text[si++] = '\b';
            else sanitized_text[si++] = custom_text[i];
        }
        sanitized_text[si] = '\0';
        run_cuda_miner(sanitized_text, max_attempts);
    } else {
        run_cuda_miner(NULL, max_attempts);
    }
    
    return 0;
}