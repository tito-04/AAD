//
// DETI Coin Miner
//
// Strategy:
//   1. Multi-threading: OpenMP uses all CPU cores.
//   2. Vectorization: AVX2/AVX-512 processes 8/16 hashes per core.
//   3. Nonce Grinding: Inner loop (Byte 53) runs 32-126 without memory reloads.
//   4. Divergent Search: Each thread and lane searches a unique space.
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
#include <omp.h> 

#if defined(__AVX512F__)
    #include <immintrin.h>
    #define SIMD_WIDTH 16
    #define SIMD_TYPE v16si
    #define SHA1_SIMD sha1_avx512f
    #define SIMD_NAME "AVX-512F"
    #define SIMD_OR(a, b) (v16si)_mm512_or_si512((__m512i)a, (__m512i)b)
    #define TARGET_HASH_TYPE    v16si
    #define TARGET_HASH_SET1(v) (v16si)_mm512_set1_epi32(v)
    #define TARGET_HASH_CMP(a,b) _mm512_cmpeq_epi32_mask((__m512i)a, (__m512i)b)
    #define TARGET_MASK_TYPE    __mmask16
    #define MASK_IS_ZERO(m)     (m == 0)
    #define MASK_TEST_LANE(m,i) (m & (1 << i))
    #define SIMD_LOAD(ptr)      _mm512_load_si512((void*)ptr)
#elif defined(__AVX2__)
    #include <immintrin.h>
    #define SIMD_WIDTH 8
    #define SIMD_TYPE v8si
    #define SHA1_SIMD sha1_avx2
    #define SIMD_NAME "AVX2"
    #define SIMD_OR(a, b) (v8si)_mm256_or_si256((__m256i)a, (__m256i)b)
    #define TARGET_HASH_TYPE    v8si
    #define TARGET_HASH_SET1(v) (v8si)_mm256_set1_epi32(v)
    #define TARGET_HASH_CMP(a,b) _mm256_movemask_ps((__m256)_mm256_cmpeq_epi32((__m256i)a, (__m256i)b))
    #define TARGET_MASK_TYPE    int
    #define MASK_IS_ZERO(m)     (m == 0)
    #define MASK_TEST_LANE(m,i) (m & (1 << i))
    #define SIMD_LOAD(ptr)      _mm256_load_si256((__m256i*)ptr)
#elif defined(__AVX__)
    #include <immintrin.h>
    #define SIMD_WIDTH 4
    #define SIMD_TYPE v4si
    #define SHA1_SIMD sha1_avx
    #define SIMD_NAME "AVX"
    #define SIMD_OR(a, b) (v4si)_mm_or_si128((__m128i)a, (__m128i)b)
    #define TARGET_HASH_TYPE    v4si
    #define TARGET_HASH_SET1(v) (v4si)_mm_set1_epi32(v)
    #define TARGET_HASH_CMP(a,b) _mm_movemask_ps((__m128)_mm_cmpeq_epi32((__m128i)a, (__m128i)b))
    #define TARGET_MASK_TYPE    int
    #define MASK_IS_ZERO(m)     (m == 0)
    #define MASK_TEST_LANE(m,i) (m & (1 << i))
    #define SIMD_LOAD(ptr)      _mm_load_si128((__m128i*)ptr)
#else
    #error "Compile with -mavx2 or -mavx512f"
#endif

#include "aad_data_types.h"
#include "aad_utilities.h"
#include "aad_sha1_cpu.h"
#include "aad_vault.h"

#define SALT_UPDATE_INTERVAL 5000ULL 
#define SALT_START_IDX 12
#define SLOW_SALT_END 45
#define MAX_CUSTOM_LEN 34

static volatile int keep_running = 1;
static u64_t global_total_attempts = 0;
static u64_t total_coins_found = 0;

void signal_handler(int signum) {
    (void)signum;
    keep_running = 0;
    #pragma omp flush(keep_running)
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

static void generate_random_bytes(u08_t *buffer, int count, u64_t *state) {
    for (int i=0; i<count; i++) {
        *state = 6364136223846793005ULL * (*state) + 1442695040888963407ULL;
        u08_t random_raw = (u08_t)(*state >> 56);
        buffer[i] = 32 + (u08_t)((random_raw * 95) >> 8);
    }
}

static void generate_slow_salt(u08_t *full_buffer, const char *custom_prefix, int prefix_len, u64_t *state) {
    int current_logical = SALT_START_IDX;
    int prefix_pos = 0;

    if (custom_prefix != NULL) {
        while (prefix_pos < prefix_len && current_logical <= SLOW_SALT_END) {
            char c = custom_prefix[prefix_pos++];
            if (c < 32 || c > 126) c = ' '; 
            full_buffer[current_logical ^ 3] = (u08_t)c; 
            current_logical++;
        }
    }
    while (current_logical <= SLOW_SALT_END) {
        *state = 6364136223846793005ULL * (*state) + 1442695040888963407ULL;
        u08_t random_raw = (u08_t)(*state >> 56);
        full_buffer[current_logical ^ 3] = 32 + (u08_t)((random_raw * 95) >> 8);
        current_logical++;
    }
}

void search_deti_coins_openmp(const char *custom_text, u64_t max_attempts) {
    
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

    printf("========================================\n");
    printf("DETI COIN MINER (OPENMP + SIMD GRINDING)\n");
    printf("SIMD: %s (x%d Lanes)\n", SIMD_NAME, SIMD_WIDTH);
    printf("Threads: %d | Nonce: ASCII 32-126\n", omp_get_max_threads());
    printf("Custom Text: %s\n", custom_text ? custom_text : "(None)");
    printf("========================================\n");

    time_measurement();
    double start_time = measured_wall_time[1].tv_sec + measured_wall_time[1].tv_nsec * 1e-9;
    const TARGET_HASH_TYPE target_hash = TARGET_HASH_SET1(0xAAD20250u);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        
        u64_t thread_state = (u64_t)time(NULL) ^ ((u64_t)tid << 32) ^ (u64_t)clock();
        for(int k=0; k<10; k++) thread_state = 6364136223846793005ULL * thread_state + 1442695040888963407ULL;

        SIMD_TYPE interleaved_data[14] __attribute__((aligned(64)));
        SIMD_TYPE hash_result[5] __attribute__((aligned(64)));
        u32_t hashes_deinterleaved[SIMD_WIDTH][5] __attribute__((aligned(64)));
        
        u32_t w11_buffer[SIMD_WIDTH] __attribute__((aligned(64)));
        u32_t w12_buffer[SIMD_WIDTH] __attribute__((aligned(64)));
        u32_t w13_buffer[SIMD_WIDTH] __attribute__((aligned(64)));
        u08_t lane_randoms[SIMD_WIDTH][8]; 

        u32_t master_template[14];
        u08_t *template_bytes = (u08_t *)master_template;
        memset(master_template, 0, sizeof(master_template));

        const char header[] = "DETI coin 2 ";
        for(int i = 0; i < 12; i++) template_bytes[i ^ 3] = (u08_t)header[i];
        template_bytes[54 ^ 3] = '\n';
        template_bytes[55 ^ 3] = 0x80;

        generate_slow_salt(template_bytes, custom_text, custom_len, &thread_state);
        update_static_simd_data(interleaved_data, master_template);

        u08_t static_byte_44 = template_bytes[44 ^ 3];
        u08_t static_byte_45 = template_bytes[45 ^ 3];
        u08_t static_byte_54 = template_bytes[54 ^ 3]; 
        u08_t static_byte_55 = template_bytes[55 ^ 3]; 

        u64_t salt_counter = 0;
        u64_t thread_attempts = 0;
        u64_t last_report_local = 0;

        while(keep_running) {
            if (max_attempts > 0 && global_total_attempts >= max_attempts) break;

            if (salt_counter >= SALT_UPDATE_INTERVAL) {
                generate_slow_salt(template_bytes, custom_text, custom_len, &thread_state);
                update_static_simd_data(interleaved_data, master_template);
                static_byte_44 = template_bytes[44 ^ 3];
                static_byte_45 = template_bytes[45 ^ 3];
                salt_counter = 0;
            }

            for (int i = 0; i < SIMD_WIDTH; i++) {
                generate_random_bytes(lane_randoms[i], 7, &thread_state);

                w11_buffer[i] = ((u32_t)static_byte_44 << 24) | ((u32_t)static_byte_45 << 16) | 
                                ((u32_t)lane_randoms[i][0] << 8)  | (u32_t)lane_randoms[i][1];

                w12_buffer[i] = ((u32_t)lane_randoms[i][2] << 24) | ((u32_t)lane_randoms[i][3] << 16) | 
                                ((u32_t)lane_randoms[i][4] << 8)  | (u32_t)lane_randoms[i][5];

                w13_buffer[i] = ((u32_t)lane_randoms[i][6] << 24) | 
                                (0u) | ((u32_t)static_byte_54 << 8)  | (u32_t)static_byte_55;
            }

            interleaved_data[11] = (SIMD_TYPE)SIMD_LOAD(w11_buffer);
            interleaved_data[12] = (SIMD_TYPE)SIMD_LOAD(w12_buffer);
            SIMD_TYPE w13_base = (SIMD_TYPE)SIMD_LOAD(w13_buffer);

            for(u32_t c = 32; c <= 126; c++) 
            {
                SIMD_TYPE increment = TARGET_HASH_SET1(c << 16);
                interleaved_data[13] = SIMD_OR(w13_base, increment);

                SHA1_SIMD(interleaved_data, hash_result);

                TARGET_MASK_TYPE mask = TARGET_HASH_CMP(hash_result[0], target_hash);

                if (!MASK_IS_ZERO(mask)) {
                    deinterleave_hashes(hashes_deinterleaved, hash_result);

                    for(int i = 0; i < SIMD_WIDTH; i++) {
                        if (MASK_TEST_LANE(mask, i)) {
                            u32_t found_coin[14];
                            memcpy(found_coin, master_template, sizeof(master_template));
                            u08_t *coin_bytes = (u08_t *)found_coin;

                            for(int k=0; k<7; k++) coin_bytes[(46+k)^3] = lane_randoms[i][k];
                            coin_bytes[53^3] = (u08_t)c;

                            #pragma omp critical
                            {
                                save_coin(found_coin);
                                total_coins_found++;
                                u32_t val = count_coin_value(hashes_deinterleaved[i]);
                                printf("\n[T%d][HIT] Val:%u | Lane:%d | NonceEnd:'%c'\n", tid, val, i, (char)c);
                            }
                        }
                    }
                }
            }

            u64_t batch_attempts = (u64_t)SIMD_WIDTH * 95;
            thread_attempts += batch_attempts;
            salt_counter += 1;

            if ((thread_attempts - last_report_local) >= 500000000) {
                #pragma omp atomic
                global_total_attempts += (thread_attempts - last_report_local);
                last_report_local = thread_attempts;

                if (tid == 0) {
                    time_measurement();
                    double elapsed = measured_wall_time[1].tv_sec + measured_wall_time[1].tv_nsec * 1e-9 - start_time;
                    double mhash = (double)global_total_attempts / elapsed / 1000000.0;
                    printf("\rAttempts: %llu | Found: %llu | Speed: %.2f MH/s   ", 
                        (unsigned long long)global_total_attempts, (unsigned long long)total_coins_found, mhash);
                    fflush(stdout);
                }
            }
        }

        #pragma omp atomic
        global_total_attempts += (thread_attempts - last_report_local);

    }

    time_measurement();
    double total_time = (measured_wall_time[1].tv_sec + measured_wall_time[1].tv_nsec * 1e-9) - start_time;
    printf("\n\n--- DONE ---\nAttempts: %llu Time: %.2fs\nAvg Speed: %.2f MH/s\n", (unsigned long long)global_total_attempts, total_time, (double)global_total_attempts/total_time/1e6);
    save_coin(NULL); 
}

int main(int argc, char *argv[]) {
    signal(SIGINT, signal_handler);
    const char *custom_text = NULL;
    u64_t max_attempts = 0;
    if(argc > 1) custom_text = argv[1];
    if(argc > 2) max_attempts = strtoull(argv[2], NULL, 10);
    search_deti_coins_openmp(custom_text, max_attempts);
    return 0;
}