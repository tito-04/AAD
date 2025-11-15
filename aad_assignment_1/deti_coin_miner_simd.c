//
// DETI Coin Miner - CPU SIMD Implementation
// Arquiteturas de Alto Desempenho 2025/2026
//
#include <time.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

#include "aad_data_types.h"
#include "aad_utilities.h"
#include "aad_sha1_cpu.h"
#include "aad_vault.h"

#if defined(__AVX512F__)
    #define SIMD_WIDTH 16
    #define SIMD_TYPE v16si
    #define SHA1_SIMD sha1_avx512f
    #define SIMD_NAME "AVX-512F"
#elif defined(__AVX2__)
    #define SIMD_WIDTH 8
    #define SIMD_TYPE v8si
    #define SHA1_SIMD sha1_avx2
    #define SIMD_NAME "AVX2"
#elif defined(__AVX__)
    #define SIMD_WIDTH 4
    #define SIMD_TYPE v4si
    #define SHA1_SIMD sha1_avx
    #define SIMD_NAME "AVX"
#else
    #error "No compatible AVX support detected. Compile with -mavx512f, -mavx2, or -mavx."
#endif

#if defined(__AVX512F__)
    #define TARGET_HASH_TYPE    v16si
    #define TARGET_HASH_SET1(v) (v16si)_mm512_set1_epi32(v)
    #define TARGET_HASH_CMP(a,b) _mm512_cmpeq_epi32_mask((__m512i)a, (__m512i)b)
    #define TARGET_MASK_TYPE    __mmask16
    #define MASK_IS_ZERO(m)     (m == 0)
    #define MASK_TEST_LANE(m,i) (m & (1 << i))
#elif defined(__AVX2__)
    #define TARGET_HASH_TYPE    v8si
    #define TARGET_HASH_SET1(v) (v8si)_mm256_set1_epi32(v)
    #define TARGET_HASH_CMP(a,b) _mm256_movemask_ps((__m256)_mm256_cmpeq_epi32((__m256i)a, (__m256i)b))
    #define TARGET_MASK_TYPE    int
    #define MASK_IS_ZERO(m)     (m == 0)
    #define MASK_TEST_LANE(m,i) (m & (1 << i))
#elif defined(__AVX__)
    #define TARGET_HASH_TYPE    v4si
    #define TARGET_HASH_SET1(v) (v4si)_mm_set1_epi32(v)
    #define TARGET_HASH_CMP(a,b) _mm_movemask_ps((__m128)_mm_cmpeq_epi32((__m128i)a, (__m128i)b))
    #define TARGET_MASK_TYPE    int
    #define MASK_IS_ZERO(m)     (m == 0)
    #define MASK_TEST_LANE(m,i) (m & (1 << i))
#endif

static volatile int keep_running = 1;
static u64_t total_attempts = 0;
static u64_t total_coins_found = 0;
static unsigned int global_seed = 0;
static u64_t global_counter_offset = 0;

static u32_t template_message[14];
static u32_t template_with_custom[14];
static int template_initialized = 0;
static int custom_template_initialized = 0;
static int custom_text_length = 0;

void signal_handler(int signum) {
    (void)signum;
    keep_running = 0;
    printf("\n\nShutting down gracefully...\n");
}

static inline int is_valid_deti_coin(u32_t *hash) {
    return (hash[0] == 0xAAD20250u);
}

static inline u32_t count_coin_value(u32_t *hash) {
    u32_t n;
    for(n = 0; n < 128; n++) {
        if (((hash[1 + n / 32] >> (31 - (n % 32))) & 1u) != 0u)
            break;
    }
    return (n > 99) ? 99 : n;
}

static void init_template(void) {
    if(template_initialized) return;
    
    memset(template_message, 0, 14 * sizeof(u32_t));
    u08_t *bytes = (u08_t *)template_message;
    const char header[] = "DETI coin 2 ";
    
    for(int i = 0; i < 12; i++) {
        bytes[i ^ 3] = (u08_t)header[i];
    }
    
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

static inline void generate_single_message_optimized(u32_t data[14], u64_t counter, int has_custom) {
    if(has_custom) {
        memcpy(data, template_with_custom, 14 * sizeof(u32_t));
    } else {
        memcpy(data, template_message, 14 * sizeof(u32_t));
    }
    
    u08_t *bytes = (u08_t *)data;
    int pos = 12 + (has_custom ? custom_text_length : 0);
    
    u64_t rng_state = global_seed ^ counter;
    
    rng_state = 6364136223846793005ul * rng_state + 1442695040888963407ul;
    
    while(pos < 54) {
        rng_state = 6364136223846793005ul * rng_state + 1442695040888963407ul;
        
        u64_t temp = rng_state;
        for(int j = 0; j < 8 && pos < 54; j++, pos++) {
            u08_t b = (u08_t)((temp >> (j * 8)) & 0xFF);
            b = 0x20 + (u08_t)((b * 95) >> 8);
            bytes[pos ^ 3] = b;
        }
    }
}

static void deinterleave_hashes(u32_t hashes[][5], SIMD_TYPE *interleaved_hash, int count) {
    for(int word = 0; word < 5; word++) {
        u32_t *temp = (u32_t *)&interleaved_hash[word];
        for(int lane = 0; lane < count; lane++) {
            hashes[lane][word] = temp[lane];
        }
    }
}

void search_deti_coins_simd_optimized(const char *custom_text, u64_t max_attempts) {
    SIMD_TYPE interleaved_data[14] __attribute__((aligned(64)));
    SIMD_TYPE hash_result[5] __attribute__((aligned(64)));

    u32_t hashes_deinterleaved[SIMD_WIDTH][5] __attribute__((aligned(64)));
    u32_t message_to_save[14];

    
    u64_t counter = 0;
    u64_t last_report = 0;
    
    const u64_t report_interval = 1000000;
    
    init_template();
    int has_custom = 0;
    if(custom_text != NULL) {
        init_custom_template(custom_text);
        has_custom = 1;
    }
    
    printf("========================================\n");
    printf("DETI COIN MINER (Loop-Fused Pipeline Refactor)\n");
    printf("CPU %s (SIMD x%d)\n", SIMD_NAME, SIMD_WIDTH);
    printf("========================================\n");
    printf("Global seed: 0x%08x\n", global_seed);
    printf("Counter offset: %llu\n", (unsigned long long)global_counter_offset);
    if(custom_text)
        printf("Custom text: %s\n", custom_text);
    printf("Report interval: %llu attempts\n", (unsigned long long)report_interval);
    printf("Press Ctrl+C to stop\n");
    printf("========================================\n\n");
    
    time_measurement();
    double start_time = measured_wall_time[1].tv_sec + measured_wall_time[1].tv_nsec * 1e-9;
    
    const TARGET_HASH_TYPE target_hash = TARGET_HASH_SET1(0xAAD20250u);
    
    while(keep_running) {
        
        u64_t remaining_attempts = (max_attempts == 0) ? (u64_t)SIMD_WIDTH : (max_attempts - counter);
        if(remaining_attempts == 0 && max_attempts != 0) break;
        
        u64_t iter_to_run_this_cycle = 1024 * (u64_t)SIMD_WIDTH;
        if (remaining_attempts < iter_to_run_this_cycle) {
            iter_to_run_this_cycle = remaining_attempts;
        }

        int batch_iterations = iter_to_run_this_cycle / SIMD_WIDTH;
        if(iter_to_run_this_cycle % SIMD_WIDTH != 0) batch_iterations++; 
        
        u64_t batch_start_counter = global_counter_offset + counter;

        for (int b = 0; b < batch_iterations; b++) {
            u64_t iter_counter = batch_start_counter + (b * (u64_t)SIMD_WIDTH);

            for(int i = 0; i < SIMD_WIDTH; i++) {
                
                generate_single_message_optimized(message_to_save, 
                                                 iter_counter + (u64_t)i,
                                                 has_custom);
                
                for(int word = 0; word < 14; word++) {
                    ((u32_t*)&interleaved_data[word])[i] = message_to_save[word];
                }
            }
            
            SHA1_SIMD(interleaved_data, hash_result);
            
            TARGET_MASK_TYPE mask = TARGET_HASH_CMP(hash_result[0], target_hash);
            
            if (!MASK_IS_ZERO(mask)) {

                deinterleave_hashes(hashes_deinterleaved, hash_result, SIMD_WIDTH);
                
                for(int i = 0; i < SIMD_WIDTH; i++) {
                    if (MASK_TEST_LANE(mask, i)) {
                        
                        u64_t found_counter = iter_counter + (u64_t)i;
                        
                        if(max_attempts != 0 && (found_counter >= (global_counter_offset + max_attempts))) {
                            continue;
                        }


                        generate_single_message_optimized(message_to_save, 
                                                         found_counter, 
                                                         has_custom);
                        
                        save_coin(message_to_save);
                        total_coins_found++;
                        
                        // Usar o hash de-intercalado para contar o valor
                        u32_t value = count_coin_value(hashes_deinterleaved[i]);
                        printf("\n>>> COIN FOUND! Value=%u Counter=%llu\n", 
                               value, (unsigned long long)found_counter);
                    }
                }
            }
        } 
 

        u64_t attempts_in_batch = (u64_t)batch_iterations * SIMD_WIDTH;
        counter += attempts_in_batch;
        total_attempts += attempts_in_batch;
        
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

        if(max_attempts != 0 && counter >= max_attempts) break;
    }
    
    time_measurement();
    double end_time = measured_wall_time[1].tv_sec + measured_wall_time[1].tv_nsec * 1e-9;
    double total_time = end_time - start_time;
    
    printf("\n\n========================================\n");
    printf("FINAL STATISTICS\n");
    printf("========================================\n");
    if(max_attempts != 0 && total_attempts > max_attempts) total_attempts = max_attempts;
    printf("Total attempts: %llu\n", (unsigned long long)total_attempts);
    
    printf("Total coins found: %llu\n", (unsigned long long)total_coins_found);
    printf("Total time: %.2f seconds\n", total_time);
    
    if (total_time > 0) {
        printf("Average rate: %.2f million hashes/second\n", (double)total_attempts / total_time / 1e6);
        if(total_coins_found > 0)
            printf("Average time per coin: %.2f seconds\n", total_time / total_coins_found);
    }
    
    printf("SIMD width: %d (%s)\n", SIMD_WIDTH, SIMD_NAME);
    printf("========================================\n");
    
    save_coin(NULL);
}

// Main program
int main(int argc, char *argv[]) {
    signal(SIGINT, signal_handler);
    
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    
    global_seed = (unsigned int)(time(NULL) ^ 
                                 (uintptr_t)&global_seed ^ 
                                 (unsigned int)getpid() ^
                                 (unsigned int)ts.tv_nsec);
    
    for(int i = 0; i < 10; i++) {
        rand_r(&global_seed);
    }
    
    global_counter_offset = ((u64_t)rand_r(&global_seed) << 32) | 
                           (u64_t)rand_r(&global_seed);
    
    clock_gettime(CLOCK_MONOTONIC, &ts);
    global_counter_offset ^= ((u64_t)ts.tv_nsec << 32) | (u64_t)ts.tv_sec;
    
    const char *custom_text = NULL;
    u64_t max_attempts = 0;
    
    if(argc > 1)
        custom_text = argv[1];
    if(argc > 2)
        max_attempts = strtoull(argv[2], NULL, 10);
    
    char sanitized_text[64];
    if(custom_text != NULL) {
        size_t si = 0;
        for(size_t i = 0; custom_text[i] != '\0' && si + 1 < sizeof(sanitized_text); i++) {
            if(custom_text[i] == '\n')
                sanitized_text[si++] = '\b';
            else
                sanitized_text[si++] = custom_text[i];
        }
        sanitized_text[si] = '\0';
        search_deti_coins_simd_optimized(sanitized_text, max_attempts);
    } else {
        search_deti_coins_simd_optimized(NULL, max_attempts);
    }
    
    return 0;
}