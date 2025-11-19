// deti_coin_simd_worker.c
// High-performance SIMD worker with CUSTOM TEXT support
// Supports: AVX-512, AVX2, AVX (Intel/AMD) and NEON (ARM)

#define _GNU_SOURCE
#include <time.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// --- SIMD Setup (Igual ao anterior) ---
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
#elif defined(__ARM_NEON) || defined(__aarch64__)
    #include <arm_neon.h>
    #define SIMD_WIDTH 4
    #define SIMD_TYPE uint32x4_t
    #define SHA1_SIMD sha1_neon
    #define SIMD_NAME "ARM NEON"
#else
    #error "No compatible SIMD support. Compile with AVX flags or on ARM."
#endif

#include "aad_data_types.h"
#include "aad_sha1_cpu.h"

// --- Configuration ---
#define COIN_HEX_STRLEN (55 * 2 + 1)
#define MAX_COINS_PER_ROUND 1024
#define SALT_UPDATE_INTERVAL 1000ULL 
#define SALT_START_IDX 12
#define SLOW_SALT_END 45
#define FAST_NONCE_START 46
#define MAX_CUSTOM_LEN 34  // (45 - 12 + 1)
#define DISPLAY_INTERVAL_ATTEMPTS 20000000

// --- Macros Definition (Igual ao anterior) ---
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
#elif defined(__ARM_NEON) || defined(__aarch64__)
    #define TARGET_HASH_TYPE    uint32x4_t
    #define TARGET_HASH_SET1(v) vdupq_n_u32(v)
    static inline int neon_cmp_mask(uint32x4_t a, uint32x4_t b) {
        uint32x4_t res = vceqq_u32(a, b);
        return (vgetq_lane_u32(res, 0) ? 1 : 0) | 
               (vgetq_lane_u32(res, 1) ? 2 : 0) | 
               (vgetq_lane_u32(res, 2) ? 4 : 0) | 
               (vgetq_lane_u32(res, 3) ? 8 : 0);
    }
    #define TARGET_HASH_CMP(a,b) neon_cmp_mask(a, b)
    #define TARGET_MASK_TYPE    int
    #define MASK_IS_ZERO(m)     (m == 0)
    #define MASK_TEST_LANE(m,i) (m & (1 << i))
    #define SIMD_LOAD(ptr)      vld1q_u32((const uint32_t *)ptr)
#endif

static u64_t lcg_state = 0;

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

// --- LÓGICA DE CUSTOM TEXT / SALT (Baseada no original) ---
static void generate_safe_salt(u08_t *full_buffer, int start_idx, int end_idx, const char *custom_prefix, int prefix_len) {
    int current_logical = start_idx;
    int prefix_pos = 0;

    // 1. Inserir Custom Prefix (Apenas se estivermos na zona inicial de salt)
    if (start_idx == SALT_START_IDX && custom_prefix != NULL) {
        while (prefix_pos < prefix_len && current_logical <= end_idx) {
            char c = custom_prefix[prefix_pos++];
            if (c < 32 || c > 126) c = ' '; // Sanitizar
            full_buffer[current_logical ^ 3] = (u08_t)c; 
            current_logical++;
        }
    }

    // 2. Preencher o resto com Random
    while (current_logical <= end_idx) {
        lcg_state = 6364136223846793005ULL * lcg_state + 1442695040888963407ULL;
        u08_t random_raw = (u08_t)(lcg_state >> 56);
        u08_t ascii_char = 32 + (u08_t)((random_raw * 95) >> 8);
        full_buffer[current_logical ^ 3] = ascii_char;
        current_logical++;
    }
}

// --- MAIN WORKER FUNCTION ---
void run_mining_round(long work_id,
                      long *attempts_out,
                      int *coins_found_out,
                      double *mhs_out, 
                      char coins_out[][COIN_HEX_STRLEN],
                      const char *custom_text) // <--- Recebe o texto
{
    // 1. Setup SIMD Data
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
    template_bytes[54 ^ 3] = '\n';
    template_bytes[55 ^ 3] = 0x80;

    // Tratar o Custom Text
    int custom_len = 0;
    if (custom_text != NULL) {
        size_t len = strlen(custom_text);
        custom_len = (len > MAX_CUSTOM_LEN) ? MAX_CUSTOM_LEN : (int)len;
    }

    lcg_state = (u64_t)work_id ^ 0xCAFEBABECAFEBABE;
    
    // Initial Generation (Passamos o custom text aqui)
    generate_safe_salt(template_bytes, SALT_START_IDX, SLOW_SALT_END, custom_text, custom_len);
    update_static_simd_data(interleaved_data, master_template);

    const TARGET_HASH_TYPE target_hash = TARGET_HASH_SET1(0xAAD20250u);

    u08_t static_byte_44 = template_bytes[44 ^ 3];
    u08_t static_byte_45 = template_bytes[45 ^ 3];
    u08_t static_byte_54 = template_bytes[54 ^ 3];
    u08_t static_byte_55 = template_bytes[55 ^ 3];

    u64_t salt_counter = 0;
    u64_t total_attempts = 0;
    int coins_found = 0;
    u64_t last_report = 0;
    u08_t base_fast_nonce[8];

    // Timing
    struct timespec t_start, t_curr;
    clock_gettime(CLOCK_MONOTONIC, &t_start);
    const double MAX_SECONDS = 60.0;

    printf("--- [SIMD %s] Mining Round (WorkID: %ld) ---\n", SIMD_NAME, work_id);
    if(custom_text) printf("--- Using Custom Text: \"%s\" ---\n", custom_text);

    while (1) {
        // A. Update Slow Salt
        if (salt_counter >= SALT_UPDATE_INTERVAL) {
            // Regenerar salt mantendo o prefixo fixo
            generate_safe_salt(template_bytes, SALT_START_IDX, SLOW_SALT_END, custom_text, custom_len);
            update_static_simd_data(interleaved_data, master_template);
            static_byte_44 = template_bytes[44 ^ 3];
            static_byte_45 = template_bytes[45 ^ 3];
            salt_counter = 0;
        }

        // B. Update Fast Nonce (Bytes 46-53)
        // AQUI passamos NULL, 0 porque o Nonce é puramente aleatório
        generate_safe_salt(template_bytes, FAST_NONCE_START, FAST_NONCE_START + 7, NULL, 0);
        for(int k=0; k<8; k++) base_fast_nonce[k] = template_bytes[(FAST_NONCE_START+k)^3];

        // C. Distribute to Lanes
        for (int i = 0; i < SIMD_WIDTH; i++) {
            u08_t lane_nonce[8];
            memcpy(lane_nonce, base_fast_nonce, 8);
            lane_nonce[7] = 32 + ((lane_nonce[7] - 32 + i) % 95);

            w11_buffer[i] = ((u32_t)static_byte_44 << 24) | ((u32_t)static_byte_45 << 16) | 
                            ((u32_t)lane_nonce[0]  << 8)  | (u32_t)lane_nonce[1];
            w12_buffer[i] = ((u32_t)lane_nonce[2] << 24) | ((u32_t)lane_nonce[3] << 16) | 
                            ((u32_t)lane_nonce[4] << 8)  | (u32_t)lane_nonce[5];
            w13_buffer[i] = ((u32_t)lane_nonce[6]  << 24) | ((u32_t)lane_nonce[7]  << 16) | 
                            ((u32_t)static_byte_54 << 8)  | (u32_t)static_byte_55;
        }

        interleaved_data[11] = (SIMD_TYPE)SIMD_LOAD(w11_buffer);
        interleaved_data[12] = (SIMD_TYPE)SIMD_LOAD(w12_buffer);
        interleaved_data[13] = (SIMD_TYPE)SIMD_LOAD(w13_buffer);

        // D. Hash
        SHA1_SIMD(interleaved_data, hash_result);

        // E. Check
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

                    char *out = coins_out[coins_found];
                    int pos = 0;
                    for (int b = 0; b < 55; ++b) {
                        pos += snprintf(out + pos, COIN_HEX_STRLEN - pos, "%02x", coin_bytes[b ^ 3]);
                    }
                    out[COIN_HEX_STRLEN-1] = '\0';

                    u32_t val = count_coin_value(hashes_deinterleaved[i]);
                    printf("\n\n[!] SIMD COIN FOUND! (Value: %u)\nContent: ", val);
                    for (int b = 0; b < 55; ++b) putchar(coin_bytes[b ^ 3]);
                    printf("\n");

                    coins_found++;
                }
            }
            if (coins_found >= MAX_COINS_PER_ROUND) break;
        }

        total_attempts += SIMD_WIDTH;
        salt_counter += SIMD_WIDTH;

        // F. Live Reporting
        if ((total_attempts - last_report) >= DISPLAY_INTERVAL_ATTEMPTS) {
            clock_gettime(CLOCK_MONOTONIC, &t_curr);
            double cur_elapsed = (t_curr.tv_sec - t_start.tv_sec) + 
                                 (t_curr.tv_nsec - t_start.tv_nsec) * 1e-9;
            if (cur_elapsed > 0) {
                double cur_mhs = (double)total_attempts / cur_elapsed / 1000000.0;
                printf("\r[Status] Speed: %.2f MH/s | Attempts: %lu | Coins: %d", 
                       cur_mhs, (unsigned long)total_attempts, coins_found);
                fflush(stdout);
            }
            last_report = total_attempts;
        }

        // G. Time Check
        if ((total_attempts & 0xFFFF) == 0) {
            clock_gettime(CLOCK_MONOTONIC, &t_curr);
            double elapsed = (t_curr.tv_sec - t_start.tv_sec) + 
                             (t_curr.tv_nsec - t_start.tv_nsec) * 1e-9;
            if (elapsed >= MAX_SECONDS) break;
        }
    }

    printf("\n");
    clock_gettime(CLOCK_MONOTONIC, &t_curr);
    double elapsed = (t_curr.tv_sec - t_start.tv_sec) + (t_curr.tv_nsec - t_start.tv_nsec) * 1e-9;
    
    *attempts_out = (long)total_attempts;
    *coins_found_out = coins_found;
    *mhs_out = (elapsed > 0) ? ((double)total_attempts / elapsed / 1000000.0) : 0.0;
}