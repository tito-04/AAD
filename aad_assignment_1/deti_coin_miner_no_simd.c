//
// DETI Coin Miner
// Strategy: Single Thread, Scalar Nonce Grinding (Byte 53 loop)
// Layout:
//   Bytes 00-11: Header
//   Bytes 12-45: Custom Text + Slow Random Salt
//   Bytes 46-52: Fast Prefix (Random per batch)
//   Byte     53: Counter (Loop 32-126)
//   Bytes 54-55: Footer
//
// Arquiteturas de Alto Desempenho 2025/2026
//

#define _GNU_SOURCE
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

#define SALT_UPDATE_INTERVAL 5000ULL 
#define SALT_START_IDX 12
#define SLOW_SALT_END 45
#define FAST_NONCE_START 46
#define MAX_CUSTOM_LEN 34

static volatile int keep_running = 1;
static u64_t total_attempts = 0;
static u64_t total_coins_found = 0;

void signal_handler(int signum) {
    (void)signum;
    keep_running = 0;
    printf("\n\nShutting down...\n");
}

static u32_t count_coin_value(u32_t *hash) {
    u32_t n;
    for(n = 0; n < 128; n++) {
        if (((hash[1 + n / 32] >> (31 - (n % 32))) & 1u) != 0u) break;
    }
    return (n > 99) ? 99 : n;
}

static void generate_safe_salt(u08_t *full_buffer, int start_idx, int end_idx, const char *custom_prefix, int prefix_len, u64_t *state) {
    int current_logical = start_idx;
    int prefix_pos = 0;

    if (start_idx == SALT_START_IDX && custom_prefix != NULL) {
        while (prefix_pos < prefix_len && current_logical <= end_idx) {
            char c = custom_prefix[prefix_pos++];
            if (c < 32 || c > 126) c = ' '; 
            full_buffer[current_logical ^ 3] = (u08_t)c; 
            current_logical++;
        }
    }

    while (current_logical <= end_idx) {
        *state = 6364136223846793005ULL * (*state) + 1442695040888963407ULL;
        u08_t random_raw = (u08_t)((*state) >> 56);
        u08_t ascii_char = 32 + (u08_t)((random_raw * 95) >> 8);
        full_buffer[current_logical ^ 3] = ascii_char;
        current_logical++;
    }
}

static void generate_random_bytes(u08_t *buffer, int start_idx, int count, u64_t *state) {
    for(int i=0; i<count; i++) {
        *state = 6364136223846793005ULL * (*state) + 1442695040888963407ULL;
        u08_t random_raw = (u08_t)((*state) >> 56);
        buffer[(start_idx + i) ^ 3] = 32 + (u08_t)((random_raw * 95) >> 8);
    }
}

// --- MAIN SEARCH LOOP (SCALAR) ---
void search_deti_coins_cpu(const char *custom_text, u64_t max_attempts) {
    
    // Validação do tamanho do texto
    int custom_len = 0;
    if (custom_text != NULL) {
        size_t len = strlen(custom_text);
        if (len > MAX_CUSTOM_LEN) {
            printf("!!! WARNING: Text truncated to %d bytes !!!\n", MAX_CUSTOM_LEN);
            custom_len = MAX_CUSTOM_LEN;
        } else {
            custom_len = (int)len;
        }
    }

    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    u64_t lcg_state = (u64_t)time(NULL) ^ ((u64_t)getpid() << 32) ^ (u64_t)ts.tv_nsec;
    for(int i=0; i<20; i++) lcg_state = 6364136223846793005ULL * lcg_state + 1442695040888963407ULL;

    u32_t coin_data[14];          
    u32_t hash_result[5];         
    u08_t *coin_bytes = (u08_t *)coin_data;
    
    memset(coin_data, 0, sizeof(coin_data));

    const char header[] = "DETI coin 2 ";
    for(int i = 0; i < 12; i++) coin_bytes[i ^ 3] = (u08_t)header[i];
    coin_bytes[54 ^ 3] = '\n';
    coin_bytes[55 ^ 3] = 0x80;

    generate_safe_salt(coin_bytes, SALT_START_IDX, SLOW_SALT_END, custom_text, custom_len, &lcg_state);

    u64_t salt_counter = 0;
    u64_t last_report = 0;

    printf("========================================\n");
    printf("DETI COIN MINER\n");
    printf("Method: Nonce Grinding (Byte 53 Loop)\n");
    printf("Custom Text: %s\n", custom_text ? custom_text : "(None)");
    printf("========================================\n");

    time_measurement();
    double start_time = measured_wall_time[1].tv_sec + measured_wall_time[1].tv_nsec * 1e-9;

    while(keep_running) {
        
        if (salt_counter >= SALT_UPDATE_INTERVAL) {
            generate_safe_salt(coin_bytes, SALT_START_IDX, SLOW_SALT_END, custom_text, custom_len, &lcg_state);
            salt_counter = 0;
        }

        generate_random_bytes(coin_bytes, FAST_NONCE_START, 7, &lcg_state);

        for(int c = 32; c <= 126; c++) 
        {
            coin_bytes[53 ^ 3] = (u08_t)c;

            sha1(coin_data, hash_result);

            if (hash_result[0] == 0xAAD20250u) {
                save_coin(coin_data);
                total_coins_found++;
                
                u32_t val = count_coin_value(hash_result);
                printf("\n[HIT] Val: %u | NonceEnd: '%c'\n", val, (char)c);
            }
        }

        u64_t batch_size = 95;
        total_attempts += batch_size;
        salt_counter += 1;

        if ((total_attempts - last_report) >= 500000000) { 
            time_measurement();
            double elapsed = measured_wall_time[1].tv_sec + measured_wall_time[1].tv_nsec * 1e-9 - start_time;
            double mhash = (double)total_attempts / elapsed / 1000000.0;
            
            printf("\rAttempts: %llu | Found: %llu | Speed: %.2f MH/s   ", 
                   (unsigned long long)total_attempts, 
                   (unsigned long long)total_coins_found, 
                   mhash);
            fflush(stdout);
            last_report = total_attempts;
        }

        if(max_attempts > 0 && total_attempts >= max_attempts) break;
    }

    time_measurement();
    double total_time = (measured_wall_time[1].tv_sec + measured_wall_time[1].tv_nsec * 1e-9) - start_time;
    printf("\n\n--- Finished ---\nTotal Attempts: %llu\nTotal Time: %.2fs\nAvg Speed: %.2f MH/s\n", 
           (unsigned long long)total_attempts, total_time, (double)total_attempts/total_time/1e6);
    
    save_coin(NULL); 
}

int main(int argc, char *argv[]) {
    signal(SIGINT, signal_handler);

    const char *custom_text = NULL;
    u64_t max_attempts = 0;
    
    if(argc > 1) custom_text = argv[1];
    if(argc > 2) max_attempts = strtoull(argv[2], NULL, 10);
    
    search_deti_coins_cpu(custom_text, max_attempts);
    return 0;
}