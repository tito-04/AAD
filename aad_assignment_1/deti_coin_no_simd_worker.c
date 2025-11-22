// deti_coin_worker.c
// Scalar (No-SIMD) worker for Distributed Miner
// Strategy: Single Thread, Scalar Nonce Grinding (Byte 53 loop)

#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "aad_data_types.h"
#include "aad_sha1_cpu.h"

#define COIN_HEX_STRLEN (55 * 2 + 1) 
#define MAX_COINS_PER_ROUND 1024
#define SALT_UPDATE_INTERVAL 5000ULL 
#define SALT_START_IDX 12
#define SLOW_SALT_END 45
#define FAST_NONCE_START 46
#define MAX_CUSTOM_LEN 34

extern volatile int keep_running;

// --- HELPER FUNCTIONS (Adapted from deti_coin_miner_no_simd.c) ---

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

// --- MAIN WORKER FUNCTION ---
void run_mining_round(long work_id,
                      long *attempts_out,
                      int *coins_found_out,
                      double *mhs_out, 
                      char coins_out[][COIN_HEX_STRLEN],
                      const char *custom_text) 
{
    // Initialize LCG State using work_id to ensure uniqueness across clients
    u64_t lcg_state = (u64_t)work_id ^ 0xCAFEBABECAFEBABEULL;
    // Warm up RNG
    for(int i=0; i<10; i++) lcg_state = 6364136223846793005ULL * lcg_state + 1442695040888963407ULL;

    // Validate Custom Text
    int custom_len = 0;
    if (custom_text != NULL) {
        size_t len = strlen(custom_text);
        custom_len = (len > MAX_CUSTOM_LEN) ? MAX_CUSTOM_LEN : (int)len;
    }

    u32_t coin_data[14];          
    u32_t hash_result[5];         
    u08_t *coin_bytes = (u08_t *)coin_data;
    
    memset(coin_data, 0, sizeof(coin_data));

    // Header Setup
    const char header[] = "DETI coin 2 ";
    for(int i = 0; i < 12; i++) coin_bytes[i ^ 3] = (u08_t)header[i];
    coin_bytes[54 ^ 3] = '\n';
    coin_bytes[55 ^ 3] = 0x80;

    // Initial Slow Salt
    generate_safe_salt(coin_bytes, SALT_START_IDX, SLOW_SALT_END, custom_text, custom_len, &lcg_state);

    u64_t salt_counter = 0;
    u64_t total_attempts = 0;
    u64_t last_report = 0;
    int coins_found = 0;

    struct timespec t_start, t_curr;
    clock_gettime(CLOCK_MONOTONIC, &t_start);
    const double MAX_SECONDS = 60.0;

    printf("--- [Scalar] Mining Round (WorkID: %ld) ---\n", work_id);
    if (custom_text) printf("--- Using Custom Text: \"%s\" ---\n", custom_text);

    while(keep_running) {
        
        // 1. Update Slow Salt (Bytes 12-45)
        if (salt_counter >= SALT_UPDATE_INTERVAL) {
            generate_safe_salt(coin_bytes, SALT_START_IDX, SLOW_SALT_END, custom_text, custom_len, &lcg_state);
            salt_counter = 0;
        }

        // 2. Update Fast Prefix (Bytes 46-52)
        generate_random_bytes(coin_bytes, FAST_NONCE_START, 7, &lcg_state);

        // 3. Grinding Loop (Byte 53)
        for(int c = 32; c <= 126; c++) 
        {
            coin_bytes[53 ^ 3] = (u08_t)c;

            sha1(coin_data, hash_result);

            if (hash_result[0] == 0xAAD20250u) {
                char *out = coins_out[coins_found];
                int pos = 0;
                for (int b = 0; b < 55; ++b) {
                    pos += snprintf(out + pos, COIN_HEX_STRLEN - pos, "%02x", coin_bytes[b ^ 3]);
                }
                out[COIN_HEX_STRLEN-1] = '\0';

                u32_t val = count_coin_value(hash_result);
                printf("\n[HIT] Val: %u | NonceEnd: '%c'\n", val, (char)c);

                coins_found++;
                if (coins_found >= MAX_COINS_PER_ROUND) goto end_mining; 
            }
        }

        // 4. Accounting
        u64_t batch_size = 95;
        total_attempts += batch_size;
        salt_counter += 1;

        // 5. Reporting & Time Check (Every ~2M attempts)
        if ((total_attempts - last_report) >= 2000000) { 
            clock_gettime(CLOCK_MONOTONIC, &t_curr);
            double elapsed = (t_curr.tv_sec - t_start.tv_sec) + (t_curr.tv_nsec - t_start.tv_nsec) * 1e-9;
            
            if (elapsed >= MAX_SECONDS) goto end_mining;

            if (elapsed > 0) {
                double mhash = (double)total_attempts / elapsed / 1000000.0;
                printf("\r[Status] Speed: %.2f MH/s | Attempts: %lu | Coins: %d", 
                       mhash, (unsigned long)total_attempts, coins_found);
                fflush(stdout);
            }
            last_report = total_attempts;
        }
    }

end_mining:
    printf("\n");
    clock_gettime(CLOCK_MONOTONIC, &t_curr);
    double total_time = (t_curr.tv_sec - t_start.tv_sec) + (t_curr.tv_nsec - t_start.tv_nsec) * 1e-9;

    *attempts_out = (long)total_attempts;
    *coins_found_out = coins_found;
    *mhs_out = (total_time > 0) ? ((double)total_attempts/total_time/1e6) : 0.0;
}