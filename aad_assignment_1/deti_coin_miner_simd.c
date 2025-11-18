//
// DETI Coin Miner - CPU SIMD Implementation
// Strategy: 64-bit LCG Random Generator
// Layout:
//   Bytes 00-11: Header ("DETI coin 2 ")
//   Bytes 12-45: Custom Text + Slow Random Salt (Total 34 Bytes)
//   Bytes 46-53: FAST RANDOM NONCE (8 Bytes, Updates every loop)
//   Bytes 54-55: Footer (\n, 0x80)
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

// --- SIMD Setup ---
#if defined(__AVX512F__)
    #include <immintrin.h>
    #define SIMD_WIDTH 16
    #define SIMD_TYPE v16si
    #define SHA1_SIMD sha1_avx512f
    #define SIMD_NAME "AVX-512F"
#elif defined(__AVX2__)
    #include <immintrin.h>
    #define SIMD_WIDTH 8
    #define SIMD_TYPE v8si
    #define SHA1_SIMD sha1_avx2
    #define SIMD_NAME "AVX2"
#elif defined(__AVX__)
    #include <immintrin.h>
    #define SIMD_WIDTH 4
    #define SIMD_TYPE v4si
    #define SHA1_SIMD sha1_avx
    #define SIMD_NAME "AVX"
#else
    #error "No compatible AVX support detected. Compile with -mavx512f, -mavx2, or -mavx."
#endif

// --- Headers AAD ---
#include "aad_data_types.h"
#include "aad_utilities.h"
#include "aad_sha1_cpu.h"
#include "aad_vault.h"

// --- SIMD Macros ---
#if defined(__AVX512F__)
    #define TARGET_HASH_TYPE    v16si
    #define TARGET_HASH_SET1(v) (v16si)_mm512_set1_epi32(v)
    #define TARGET_HASH_CMP(a,b) _mm512_cmpeq_epi32_mask((__m512i)a, (__m512i)b)
    #define TARGET_MASK_TYPE    __mmask16
    #define MASK_IS_ZERO(m)     (m == 0)
    #define MASK_TEST_LANE(m,i) (m & (1 << i))
    #define SIMD_LOAD(ptr)      _mm512_load_si512((void*)ptr)
#elif defined(__AVX2__)
    #define TARGET_HASH_TYPE    v8si
    #define TARGET_HASH_SET1(v) (v8si)_mm256_set1_epi32(v)
    #define TARGET_HASH_CMP(a,b) _mm256_movemask_ps((__m256)_mm256_cmpeq_epi32((__m256i)a, (__m256i)b))
    #define TARGET_MASK_TYPE    int
    #define MASK_IS_ZERO(m)     (m == 0)
    #define MASK_TEST_LANE(m,i) (m & (1 << i))
    #define SIMD_LOAD(ptr)      _mm256_load_si256((__m256i*)ptr)
#elif defined(__AVX__)
    #define TARGET_HASH_TYPE    v4si
    #define TARGET_HASH_SET1(v) (v4si)_mm_set1_epi32(v)
    #define TARGET_HASH_CMP(a,b) _mm_movemask_ps((__m128)_mm_cmpeq_epi32((__m128i)a, (__m128i)b))
    #define TARGET_MASK_TYPE    int
    #define MASK_IS_ZERO(m)     (m == 0)
    #define MASK_TEST_LANE(m,i) (m & (1 << i))
    #define SIMD_LOAD(ptr)      _mm_load_si128((__m128i*)ptr)
#endif

// --- Configuration ---
#define SALT_UPDATE_INTERVAL 1000ULL 
#define SALT_START_IDX 12
#define SLOW_SALT_END 45
#define FAST_NONCE_START 46

// CHECK: O tamanho máximo do texto customizado é (45 - 12 + 1) = 34 Bytes
#define MAX_CUSTOM_LEN 34

// --- Globals ---
static volatile int keep_running = 1;
static u64_t total_attempts = 0;
static u64_t total_coins_found = 0;

static u64_t lcg_state = 0; 

void signal_handler(int signum) {
    (void)signum;
    keep_running = 0;
    printf("\n\nShutting down...\n");
}

static inline u32_t count_coin_value(u32_t *hash) {
    u32_t n;
    for(n = 0; n < 128; n++) {
        if (((hash[1 + n / 32] >> (31 - (n % 32))) & 1u) != 0u) break;
    }
    return (n > 99) ? 99 : n;
}

static void deinterleave_hashes(u32_t hashes[][5], SIMD_TYPE *interleaved_hash) {
    for(int word = 0; word < 5; word++) {
        u32_t *temp = (u32_t *)&interleaved_hash[word];
        for(int lane = 0; lane < SIMD_WIDTH; lane++) {
            hashes[lane][word] = temp[lane];
        }
    }
}

static void update_static_simd_data(SIMD_TYPE *interleaved_data, u32_t *template_msg) {
    u32_t temp_lane_buffer[SIMD_WIDTH] __attribute__((aligned(64)));
    int static_words[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int num_static = sizeof(static_words) / sizeof(int);

    for (int k = 0; k < num_static; k++) {
        int w = static_words[k];
        for (int lane = 0; lane < SIMD_WIDTH; lane++) {
            temp_lane_buffer[lane] = template_msg[w];
        }
        interleaved_data[w] = (SIMD_TYPE)SIMD_LOAD(temp_lane_buffer);
    }
}

// --- RANDOM GENERATOR ---
// Alterado: Agora aceita 'prefix_len' explicitamente para evitar strlen repetido e respeitar limites
static void generate_safe_salt(u08_t *full_buffer, int start_idx, int end_idx, const char *custom_prefix, int prefix_len) {
    int current_logical = start_idx;
    int prefix_pos = 0;

    // 1. Custom Prefix (Only applies if start_idx matches prefix start logic)
    if (start_idx == SALT_START_IDX && custom_prefix != NULL) {
        while (prefix_pos < prefix_len && current_logical <= end_idx) {
            char c = custom_prefix[prefix_pos++];
            if (c < 32 || c > 126) c = ' '; 
            full_buffer[current_logical ^ 3] = (u08_t)c; 
            current_logical++;
        }
    }

    // 2. Random Fill
    while (current_logical <= end_idx) {
        lcg_state = 6364136223846793005ULL * lcg_state + 1442695040888963407ULL;
        u08_t random_raw = (u08_t)(lcg_state >> 56);
        u08_t ascii_char = 32 + (u08_t)((random_raw * 95) >> 8);
        full_buffer[current_logical ^ 3] = ascii_char;
        current_logical++;
    }
}

// --- MINER MAIN ---
void search_deti_coins_simd(const char *custom_text, u64_t max_attempts) {
    // Buffers
    SIMD_TYPE interleaved_data[14] __attribute__((aligned(64)));
    SIMD_TYPE hash_result[5] __attribute__((aligned(64)));
    u32_t hashes_deinterleaved[SIMD_WIDTH][5] __attribute__((aligned(64)));
    
    u32_t w11_buffer[SIMD_WIDTH] __attribute__((aligned(64)));
    u32_t w12_buffer[SIMD_WIDTH] __attribute__((aligned(64)));
    u32_t w13_buffer[SIMD_WIDTH] __attribute__((aligned(64)));

    u32_t master_template[14];
    u08_t *template_bytes = (u08_t *)master_template;
    memset(master_template, 0, sizeof(master_template));

    // Header
    const char header[] = "DETI coin 2 ";
    for(int i = 0; i < 12; i++) template_bytes[i ^ 3] = (u08_t)header[i];
    
    // Footer
    template_bytes[54 ^ 3] = '\n';
    template_bytes[55 ^ 3] = 0x80;

    // --- CHECK CUSTOM TEXT SIZE ---
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

    // Initial Random Generation
    generate_safe_salt(template_bytes, SALT_START_IDX, SLOW_SALT_END, custom_text, custom_len);
    
    update_static_simd_data(interleaved_data, master_template);

    u64_t salt_counter = 0;
    u64_t local_counter = 0;
    u64_t last_report = 0;
    u08_t base_fast_nonce[8];

    printf("========================================\n");
    printf("DETI COIN MINER (END-OF-MESSAGE NONCE)\n");
    printf("SIMD: %s (x%d Lanes)\n", SIMD_NAME, SIMD_WIDTH);
    printf("Bytes 00-11: Header\n");
    printf("Bytes 12-45: Custom + Slow Random (Max %d Bytes)\n", MAX_CUSTOM_LEN);
    printf("Bytes 46-53: Fast Random Nonce (Update every Loop)\n");
    printf("Bytes 54-55: Footer (\\n, 0x80)\n");
    printf("========================================\n");

    time_measurement();
    double start_time = measured_wall_time[1].tv_sec + measured_wall_time[1].tv_nsec * 1e-9;
    const TARGET_HASH_TYPE target_hash = TARGET_HASH_SET1(0xAAD20250u);

    u08_t static_byte_44 = template_bytes[44 ^ 3];
    u08_t static_byte_45 = template_bytes[45 ^ 3];
    u08_t static_byte_54 = template_bytes[54 ^ 3]; // '\n'
    u08_t static_byte_55 = template_bytes[55 ^ 3]; // 0x80

    while(keep_running) {
        
        // A. Update Slow Salt Periodically
        if (salt_counter >= SALT_UPDATE_INTERVAL) {
            generate_safe_salt(template_bytes, SALT_START_IDX, SLOW_SALT_END, custom_text, custom_len);
            update_static_simd_data(interleaved_data, master_template);
            
            static_byte_44 = template_bytes[44 ^ 3];
            static_byte_45 = template_bytes[45 ^ 3];
            
            salt_counter = 0;
        }

        // B. Update Fast Nonce (Bytes 46-53)
        // Pass NULL and 0 len since this is purely random generation
        generate_safe_salt(template_bytes, FAST_NONCE_START, FAST_NONCE_START + 7, NULL, 0);
        for(int k=0; k<8; k++) base_fast_nonce[k] = template_bytes[(FAST_NONCE_START+k)^3];

        // C. Distribute to Lanes
        for (int i = 0; i < SIMD_WIDTH; i++) {
            u08_t lane_nonce[8];
            memcpy(lane_nonce, base_fast_nonce, 8);
            
            // Lane Variation
            lane_nonce[7] = 32 + ((lane_nonce[7] - 32 + i) % 95);

            // Word 11 (44, 45 Static | 46, 47 Nonce)
            w11_buffer[i] = ((u32_t)static_byte_44 << 24) | 
                            ((u32_t)static_byte_45 << 16) | 
                            ((u32_t)lane_nonce[0]  << 8)  | 
                            (u32_t)lane_nonce[1];

            // Word 12 (48-51 Nonce)
            w12_buffer[i] = ((u32_t)lane_nonce[2] << 24) | 
                            ((u32_t)lane_nonce[3] << 16) | 
                            ((u32_t)lane_nonce[4] << 8)  | 
                            (u32_t)lane_nonce[5];

            // Word 13 (52, 53 Nonce | 54, 55 Static)
            w13_buffer[i] = ((u32_t)lane_nonce[6]  << 24) | 
                            ((u32_t)lane_nonce[7]  << 16) | 
                            ((u32_t)static_byte_54 << 8)  | 
                            (u32_t)static_byte_55;
        }

        interleaved_data[11] = (SIMD_TYPE)SIMD_LOAD(w11_buffer);
        interleaved_data[12] = (SIMD_TYPE)SIMD_LOAD(w12_buffer);
        interleaved_data[13] = (SIMD_TYPE)SIMD_LOAD(w13_buffer);

        SHA1_SIMD(interleaved_data, hash_result);

        TARGET_MASK_TYPE mask = TARGET_HASH_CMP(hash_result[0], target_hash);

        if (!MASK_IS_ZERO(mask)) {
            deinterleave_hashes(hashes_deinterleaved, hash_result);

            for(int i = 0; i < SIMD_WIDTH; i++) {
                if (MASK_TEST_LANE(mask, i)) {
                    u32_t found_coin[14];
                    memcpy(found_coin, master_template, sizeof(master_template));
                    u08_t *coin_bytes = (u08_t *)found_coin;

                    u08_t lane_nonce[8];
                    memcpy(lane_nonce, base_fast_nonce, 8);
                    lane_nonce[7] = 32 + ((lane_nonce[7] - 32 + i) % 95);

                    for(int k=0; k<8; k++) coin_bytes[(FAST_NONCE_START+k)^3] = lane_nonce[k];

                    save_coin(found_coin);
                    total_coins_found++;
                    
                    u32_t val = count_coin_value(hashes_deinterleaved[i]);
                    char ns[9]; memcpy(ns, lane_nonce, 8); ns[8]=0;
                    printf("\n[HIT] Val: %u | Nonce(46-53): \"%s\"\n", val, ns);
                }
            }
        }

        local_counter += SIMD_WIDTH;
        salt_counter += SIMD_WIDTH;
        total_attempts += SIMD_WIDTH;

        if ((local_counter - last_report) >= 20000000) {
            time_measurement();
            double elapsed = measured_wall_time[1].tv_sec + measured_wall_time[1].tv_nsec * 1e-9 - start_time;
            double mhash = (double)total_attempts / elapsed / 1000000.0;
            
            printf("\rAttempts: %llu | Found: %llu | Speed: %.2f MH/s   ", 
                   (unsigned long long)total_attempts, 
                   (unsigned long long)total_coins_found, 
                   mhash);
            fflush(stdout);
            last_report = local_counter;
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
    
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    u64_t seed_part1 = (u64_t)time(NULL);
    u64_t seed_part2 = (u64_t)ts.tv_nsec ^ (u64_t)getpid();
    lcg_state = (seed_part1 << 32) | seed_part2;

    for(int i=0; i<10; i++) {
        lcg_state = 6364136223846793005ULL * lcg_state + 1442695040888963407ULL;
    }

    const char *custom_text = NULL;
    u64_t max_attempts = 0;
    
    if(argc > 1) custom_text = argv[1];
    if(argc > 2) max_attempts = strtoull(argv[2], NULL, 10);
    
    search_deti_coins_simd(custom_text, max_attempts);
    return 0;
}