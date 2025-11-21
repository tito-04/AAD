// deti_coin_worker.c
// Scalar (No-SIMD) worker for Distributed Miner

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "aad_data_types.h"
#include "aad_sha1_cpu.h"

#define COIN_HEX_STRLEN (55 * 2 + 1) 
#define DISPLAY_INTERVAL_HASHES 10000000
#define MAX_COINS_PER_ROUND 1024

extern volatile int keep_running;

unsigned int global_seed = 0;
u64_t global_counter_offset = 0;

static inline u08_t random_byte_seeded(u64_t *state) {
    *state = 6364136223846793005ul * (*state) + 1442695040888963407ul;
    return (u08_t)((*state) >> 43);
}

static int is_valid_deti_coin(u32_t *hash) {
    return (hash[0] == 0xAAD20250u);
}

static u32_t count_coin_value(u32_t *hash) {
    u32_t n = 0;
    for (n = 0; n < 128; n++) {
        u32_t word = hash[1 + n / 32];
        u32_t bit = (word >> (31 - (n % 32))) & 1u;
        if (bit != 0u) break;
    }
    return (n > 99) ? 99 : n;
}

static void generate_message(u32_t data[14], u64_t counter, const char *custom_text) {
    memset(data, 0, 14 * sizeof(u32_t));
    u08_t *bytes = (u08_t *)data;

    const char header[] = "DETI coin 2 ";
    for (int i = 0; i < 12; ++i) bytes[i ^ 3] = (u08_t)header[i];

    int pos = 12;
    if (custom_text != NULL) {
        for (size_t i = 0; custom_text[i] != '\0' && pos < 54; ++i) {
            char c = custom_text[i];
            if (c == '\n') c = '\b'; // Sanitização básica
            bytes[pos ^ 3] = (u08_t)c;
            pos++;
        }
    }

    u64_t unique_id = global_counter_offset + counter;
    u64_t rng_state = (u64_t)global_seed ^ unique_id ^ (u64_t)(uintptr_t)data;
    rng_state = 6364136223846793005ul * rng_state + 1442695040888963407ul;

    while (pos < 54) {
        u08_t b = random_byte_seeded(&rng_state);
        b = 0x20 + (u08_t)((b * 95) >> 8);
        bytes[pos ^ 3] = b;
        pos++;
    }
    bytes[54 ^ 3] = (u08_t)'\n';
    bytes[55 ^ 3] = 0x80;
}

// --- MAIN WORKER FUNCTION ---
void run_mining_round(long work_id,
                      long *attempts_out,
                      int *coins_found_out,
                      double *mhs_out, 
                      char coins_out[][COIN_HEX_STRLEN],
                      const char *custom_text) // <--- Novo argumento
{
    u32_t data[14];
    u32_t hash[5];
    u64_t attempts = 0;
    int coins_found = 0;
    
    // Scalar offset logic
    global_counter_offset = (u64_t)work_id * 100000000000ULL;
    global_seed = (unsigned int)time(NULL);

    u64_t local_counter = 0;
    const double MAX_SECONDS = 60.0;
    
    struct timespec t_start, t_curr;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    printf("--- [Scalar] Mining Round (WorkID: %ld) ---\n", work_id);
    if (custom_text) printf("--- Using Custom Text: \"%s\" ---\n", custom_text);

    while (1) {
        if (!keep_running) break;
        generate_message(data, local_counter, custom_text);
        sha1(data, hash);

        if (is_valid_deti_coin(hash)) {
            uint8_t *bytes = (uint8_t *)data;
            char *out = coins_out[coins_found];
            int pos = 0;
            for (int i = 0; i < 55; ++i) {
                pos += snprintf(out + pos, COIN_HEX_STRLEN - pos, "%02x", bytes[i ^ 3]);
            }
            out[COIN_HEX_STRLEN-1] = '\0';
            
            printf("\n\n[!] COIN FOUND! (Value: %u)\nContent: ", count_coin_value(hash));
            for (int i = 0; i < 55; ++i) putchar(bytes[i ^ 3]);
            printf("\n");

            coins_found++;
            if (coins_found >= MAX_COINS_PER_ROUND) break;
        }

        local_counter++;
        attempts++;

        if (attempts % DISPLAY_INTERVAL_HASHES == 0) {
            clock_gettime(CLOCK_MONOTONIC, &t_curr);
            double cur_elapsed = (t_curr.tv_sec - t_start.tv_sec) + 
                                 (t_curr.tv_nsec - t_start.tv_nsec) * 1e-9;
            if (cur_elapsed > 0) {
                double cur_mhs = (double)attempts / cur_elapsed / 1000000.0;
                printf("\r[Status] Speed: %.2f MH/s | Attempts: %lu | Coins: %d", 
                       cur_mhs, (unsigned long)attempts, coins_found);
                fflush(stdout);
            }
        }

        if ((attempts & 0xFFFF) == 0) { 
            clock_gettime(CLOCK_MONOTONIC, &t_curr);
            double elapsed = (t_curr.tv_sec - t_start.tv_sec) + (t_curr.tv_nsec - t_start.tv_nsec) * 1e-9;
            if (elapsed >= MAX_SECONDS) break;
        }
    }

    printf("\n");
    clock_gettime(CLOCK_MONOTONIC, &t_curr);
    double elapsed = (t_curr.tv_sec - t_start.tv_sec) + (t_curr.tv_nsec - t_start.tv_nsec) * 1e-9;
    
    *attempts_out = (long)attempts;
    *coins_found_out = coins_found;
    *mhs_out = (elapsed > 0) ? ((double)attempts / elapsed / 1000000.0) : 0.0;
}