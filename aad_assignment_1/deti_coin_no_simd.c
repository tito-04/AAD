//
// DETI Coin Miner - CPU Implementation (No SIMD) - V3 (Counter-based)
// Lógica de geração de mensagens alinhada com a versão SIMD.
// Arquiteturas de Alto Desempenho 2025/2026
//

#include <time.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>           // getpid()
#include "aad_data_types.h"
#include "aad_utilities.h"
#include "aad_sha1_cpu.h"
#include "aad_vault.h"

// Global variables for statistics
static volatile int keep_running = 1;
static u64_t total_attempts = 0;
static u64_t total_coins_found = 0;

// Global PRNG seed (initialized once)
static unsigned int global_seed = 0;
// Global counter offset (initialized once)
static u64_t global_counter_offset = 0;

// Signal handler for graceful shutdown
void signal_handler(int signum) {
    (void)signum;
    keep_running = 0;
    printf("\n\nShutting down gracefully...\n");
}

// Check if hash starts with 0xAAD20250
int is_valid_deti_coin(u32_t *hash) {
    return (hash[0] == 0xAAD20250u);
}

// Count leading zero bits after the signature
u32_t count_coin_value(u32_t *hash) {
    u32_t n = 0;
    for(n = 0; n < 128; n++) {
        if(((hash[1 + n / 32] >> (31 - (n % 32))) & 1u) != 0u)
            break;
    }
    return (n > 99) ? 99 : n;
}

// PRNG based on MMIX by Donald Knuth (from aad_utilities.h)
// Thread-safe version with explicit state parameter
static inline u08_t random_byte_seeded(u64_t *state) {
    *state = 6364136223846793005ul * (*state) + 1442695040888963407ul;
    return (u08_t)((*state) >> 43);
}


// Generate candidate message with random content
// Lógica alinhada com a versão SIMD:
// O conteúdo aleatório é determinístico com base em (global_seed ^ (global_offset + counter))
void generate_message(u32_t data[14], u64_t counter, const char *custom_text) {
    // data is 14 * 4 = 56 bytes buffer (we use bytes 0..55, file is bytes 0..54)
    memset(data, 0, 14 * sizeof(u32_t));
    
    u08_t *bytes = (u08_t *)data;
    /* Fixed prefix (12 bytes) --- includes trailing space */
    const char header[] = "DETI coin 2 ";
    for(int i = 0; i < 12; ++i)
        bytes[i ^ 3] = (u08_t)header[i];

    int pos = 12; // next writable byte in the 0..54 file range

    /* Se custom_text for fornecido, copia a versão sanitizada primeiro. */
    if(custom_text != NULL) {
        for(size_t i = 0; custom_text[i] != '\0' && pos < 54; ++i) {
            char c = custom_text[i];
            if(c == '\n') c = '\b'; // Troca newline por backspace
            bytes[pos ^ 3] = (u08_t)c;
            pos++;
        }
    }

    /* PRNG por-mensagem, alinhado com a versão SIMD */
    // Combina seed global, o contador único, e o endereço para entropia
    u64_t unique_id = global_counter_offset + counter;
    u64_t rng_state = global_seed ^ unique_id ^ (u64_t)(uintptr_t)data;
    
    // Mistura adicional para garantir um bom estado inicial
    rng_state = 6364136223846793005ul * rng_state + 1442695040888963407ul;

    /* Preenche as posições restantes com caracteres ASCII imprimíveis aleatórios */
    while(pos < 54) {
        u08_t b = random_byte_seeded(&rng_state);
        
        // Mapeia para o range ASCII imprimível 0x20-0x7E (espaço até til)
        b = 0x21 + (b % 0x5E);  // 0x5F = 0x7F - 0x20
        
        // Não é preciso verificar por '\n' (0x0A) porque o range 0x20-0x7E não o contém
        bytes[pos ^ 3] = b;
        pos++;
    }


    /* Final required newline at byte index 54 (last byte of the 55-byte file) */
    bytes[54 ^ 3] = (u08_t)'\n';

    /* Internal SHA1 padding byte at position 55 (not part of the file) */
    bytes[55 ^ 3] = 0x80;
}

// Print coin information
void print_coin(u32_t data[14], u32_t hash[5], u64_t counter, u32_t value) {
    printf("\n========================================\n");
    printf("DETI COIN FOUND!\n");
    printf("========================================\n");
    // Imprime o contador e o offset para contexto
    printf("Counter: %llu (offset: %llu)\n", 
           (unsigned long long)counter,
           (unsigned long long)global_counter_offset);
    printf("Value: %u\n", value);
    printf("SHA1: ");
    for(int i = 0; i < 5; i++)
        printf("%08x", hash[i]);
    printf("\n");

    printf("Content: ");
    u08_t *bytes = (u08_t *)data;
    for(int i = 0; i < 55; i++) {
        u08_t c = bytes[i ^ 3];
        if(c >= 32 && c <= 126)
            printf("%c", c);
        else if(c == '\n')
            printf("\\n");
        else if(c == '\b')
            printf("\\b");
        else
            printf("\\x%02x", c);
    }
    printf("\n========================================\n");
}


// Main search function
void search_deti_coins(const char *custom_text, u64_t max_attempts) {
    u32_t data[14];
    u32_t hash[5];
    u64_t counter = 0; // Este é o contador local da execução
    u64_t last_report = 0;
    const u64_t report_interval = 1000000; // Report every 1M attempts

    printf("========================================\n");
    printf("DETI COIN MINER - CPU (No SIMD) v3\n");
    printf("========================================\n");
    printf("Global seed: 0x%08x\n", global_seed);
    printf("Counter offset: %llu\n", (unsigned long long)global_counter_offset);
    if(custom_text)
        printf("Custom text: %s\n", custom_text);
    printf("Press Ctrl+C to stop\n");
    printf("========================================\n\n");
    
    time_measurement();
    double start_time = measured_wall_time[1].tv_sec + measured_wall_time[1].tv_nsec * 1e-9;
    
    while(keep_running && (max_attempts == 0 || counter < max_attempts)) {
        // Gera a mensagem usando o contador local. O offset global está embutido
        generate_message(data, counter, custom_text);
        sha1(data, hash);

        if(is_valid_deti_coin(hash)) {
            u32_t value = count_coin_value(hash);
            print_coin(data, hash, counter, value);
            save_coin(data);
            total_coins_found++;
        }
        
        counter++;
        total_attempts++;
        
        // Periodic status
        if(counter - last_report >= report_interval) {
            time_measurement();
            double elapsed = measured_wall_time[1].tv_sec + measured_wall_time[1].tv_nsec * 1e-9 - start_time;
            double rate = (double)counter / elapsed;
            
            printf("\r[%llu attempts] [%llu coins] [%.2f MH/s]          ",
                   (unsigned long long)counter,
                   (unsigned long long)total_coins_found,
                   rate / 1e6);
            fflush(stdout);
            
            last_report = counter;
        }
    }
    
    // Final stats
    time_measurement();
    double end_time = measured_wall_time[1].tv_sec + measured_wall_time[1].tv_nsec * 1e-9;
    double total_time = end_time - start_time;
    
    printf("\n\n========================================\n");
    printf("FINAL STATISTICS\n");
    printf("========================================\n");
    printf("Total attempts: %llu\n", (unsigned long long)total_attempts);
    printf("Total coins found: %llu\n", (unsigned long long)total_coins_found);
    printf("Total time: %.2f seconds\n", total_time);
    printf("Average rate: %.2f million hashes/second\n", (double)total_attempts / total_time / 1e6);
    if(total_coins_found > 0)
        printf("Average time per coin: %.2f seconds\n", total_time / total_coins_found);
    printf("========================================\n");
    
    save_coin(NULL);
}

// Main program
int main(int argc, char *argv[]) {
    // Install signal handler
    signal(SIGINT, signal_handler);

    // Inicializa a seed global do PRNG com máxima entropia
    // Combina: tempo atual, PID, endereço da stack, e nanossegundos
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    
    global_seed = (unsigned int)(time(NULL) ^ 
                                 (uintptr_t)&global_seed ^ 
                                 (unsigned int)getpid() ^
                                 (unsigned int)ts.tv_nsec);
    
    // Mistura bem a seed
    for(int i = 0; i < 10; i++) {
        rand_r(&global_seed);
    }
    
    // Gera um offset de contador aleatório para cada execução
    global_counter_offset = ((u64_t)rand_r(&global_seed) << 32) | 
                           (u64_t)rand_r(&global_seed);
    
    // Adiciona mais entropia do timer de alta resolução
    clock_gettime(CLOCK_MONOTONIC, &ts);
    global_counter_offset ^= ((u64_t)ts.tv_nsec << 32) | (u64_t)ts.tv_sec;
    
    
    const char *custom_text = NULL;
    u64_t max_attempts = 0;
    
    if(argc > 1)
        custom_text = argv[1];
    if(argc > 2)
        max_attempts = strtoull(argv[2], NULL, 10);
    
    // Sanitiza o custom_text (remove '\n')
    char sanitized_text[64];
    if(custom_text != NULL) {
        size_t si = 0;
        for(size_t i = 0; custom_text[i] != '\0' && si + 1 < sizeof(sanitized_text); ++i) {
            if(custom_text[i] == '\n')
            {
                sanitized_text[si++] = '\b'; // troca por backspace
            }
            else
            {
                sanitized_text[si++] = custom_text[i];
            }
        }
        sanitized_text[si] = '\0';
        search_deti_coins(sanitized_text, max_attempts);
    } else {
        search_deti_coins(NULL, max_attempts);
    }
    return 0;
}