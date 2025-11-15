//
// DETI Coin Miner - CPU SIMD Implementation (AVX-512/AVX2/AVX/NEON)
// Otimização Nível 2: OpenMP + Templates + Fast Check
// Arquiteturas de Alto Desempenho 2025/2026
//
#include <time.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <omp.h> // <--- ADICIONADO PARA OPENMP

// Adiciona o header para todos os intrinsics Intel (AVX, AVX2, AVX512)
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

#include "aad_data_types.h"
#include "aad_utilities.h"
#include "aad_sha1_cpu.h"
#include "aad_vault.h"

// Determine SIMD width at compile time
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
    #error "No SIMD support detected. Compile with -mavx512f, -mavx2, -mavx"
#endif


// Macros para a otimização "Fast Check" (com casts corrigidos)
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
#else
    // NEON ou outro: Usa a verificação original (sem otimização de máscara)
    // (Ainda não suportado neste OMP, mas a estrutura está aqui)
    #define USE_FALLBACK_CHECK
#endif


// Global variables for statistics and randomization
static volatile int keep_running = 1;
static u64_t total_attempts = 0;
static u64_t total_coins_found = 0;
static unsigned int global_seed = 0;
static u64_t global_counter_offset = 0;

// --- NOVOS GLOBAIS PARA TEMPLATES ---
static u32_t template_message[14];
static u32_t template_with_custom[14];
static int template_initialized = 0;
static int custom_template_initialized = 0;
static int custom_text_length = 0;
// --- FIM NOVOS GLOBAIS ---


// Signal handler for graceful shutdown
void signal_handler(int signum) {
    (void)signum;
    keep_running = 0;
    printf("\n\nShutting down gracefully...\n");
}

// Check if hash starts with 0xAAD20250
static inline int is_valid_deti_coin(u32_t *hash) {
    return (hash[0] == 0xAAD20250u);
}

// Count leading zero bits after the signature
static inline u32_t count_coin_value(u32_t *hash) {
    u32_t n;
    for(n = 0; n < 128; n++) {
        if (((hash[1 + n / 32] >> (31 - (n % 32))) & 1u) != 0u)
            break;
    }
    return (n > 99) ? 99 : n;
}


// --- INÍCIO FUNÇÕES DE TEMPLATE (THREAD-SAFE PARA LEITURA) ---

// Initialize base template (call once at start)
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

// Initialize template with custom text (call once if custom text exists)
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

// Generate message by copying template + filling random bytes
// (Esta função é 100% thread-safe)
static inline void generate_single_message_optimized(u32_t data[14], u64_t counter, int has_custom) {
    // Copy appropriate template (with or without custom text)
    if(has_custom) {
        memcpy(data, template_with_custom, 14 * sizeof(u32_t));
    } else {
        memcpy(data, template_message, 14 * sizeof(u32_t));
    }
    
    u08_t *bytes = (u08_t *)data;
    int pos = 12 + (has_custom ? custom_text_length : 0);
    
    // Optimization: use counter as part of seed to vary messages
    // (global_seed é lido de forma atómica aqui, o que é OK)
    u64_t rng_state = global_seed ^ counter;
    
    // Single mixing step (sufficient for good randomness)
    rng_state = 6364136223846793005ul * rng_state + 1442695040888963407ul;
    
    // Fill remaining positions with random bytes
    while(pos < 54) {
        rng_state = 6364136223846793005ul * rng_state + 1442695040888963407ul;
        
        u64_t temp = rng_state;
        for(int j = 0; j < 8 && pos < 54; j++, pos++) {
            u08_t b = (u08_t)((temp >> (j * 8)) & 0xFF);
            b = 0x20 + (b % 0x5F); // Map to printable ASCII
            bytes[pos ^ 3] = b;
        }
    }
}
// --- FIM FUNÇÕES DE TEMPLATE ---


// Interleave SIMD_WIDTH messages into SIMD format
static void interleave_messages(SIMD_TYPE *interleaved_data, u32_t messages[][14], int count) {
    for(int word = 0; word < 14; word++) {
        u32_t temp[SIMD_WIDTH];
        
        for(int lane = 0; lane < count; lane++) {
            temp[lane] = messages[lane][word];
        }
        for(int lane = count; lane < SIMD_WIDTH; lane++) {
            temp[lane] = 0;
        }
        
#if defined(__AVX512F__)
        interleaved_data[word] = (v16si){ temp[0],  temp[1],  temp[2],  temp[3],
                                          temp[4],  temp[5],  temp[6],  temp[7],
                                          temp[8],  temp[9],  temp[10], temp[11],
                                          temp[12], temp[13], temp[14], temp[15] };
#elif defined(__AVX2__)
        interleaved_data[word] = (v8si){ temp[0], temp[1], temp[2], temp[3],
                                         temp[4], temp[5], temp[6], temp[7] };
#elif defined(__AVX__)
        interleaved_data[word] = (v4si){ temp[0], temp[1], temp[2], temp[3] };
#elif defined(__ARM_NEON)
        interleaved_data[word] = (uint32x4_t){ temp[0], temp[1], temp[2], temp[3] };
#endif
    }
}

// Deinterleave SIMD results back to individual hashes
static void deinterleave_hashes(u32_t hashes[][5], SIMD_TYPE *interleaved_hash, int count) {
    for(int word = 0; word < 5; word++) {
        u32_t *temp = (u32_t *)&interleaved_hash[word];
        for(int lane = 0; lane < count; lane++) {
            hashes[lane][word] = temp[lane];
        }
    }
}


// Main SIMD search function
// <--- FUNÇÃO PRINCIPAL MODIFICADA COM TEMPLATES E FAST-CHECK
void search_deti_coins_simd(const char *custom_text, u64_t max_attempts) {

    u64_t last_report = 0;
    const u64_t report_interval = 1000000;
    
    // --- NOVO: Inicializa os templates UMA VEZ ---
    init_template();
    int has_custom = 0;
    if(custom_text != NULL) {
        init_custom_template(custom_text);
        has_custom = 1;
    }
    // --- FIM NOVO ---
    
    printf("========================================\n");
    printf("DETI COIN MINER - CPU %s (SIMD x%d) - OpenMP (Optimized)\n", SIMD_NAME, SIMD_WIDTH);
    printf("========================================\n");
    printf("Global seed: 0x%08x\n", global_seed);
    printf("Counter offset: %llu\n", (unsigned long long)global_counter_offset);
    if(custom_text)
        printf("Custom text: %s\n", custom_text);
    printf("Press Ctrl+C to stop\n");
    printf("========================================\n\n");
    
    time_measurement();
    double start_time = measured_wall_time[1].tv_sec + measured_wall_time[1].tv_nsec * 1e-9;
    
    // <--- INÍCIO DA REGIÃO PARALELA ---
    #pragma omp parallel
    {
        // --- Variáveis Privadas da Thread ---
        // Alinhamento é importante para performance SIMD
        u32_t messages[SIMD_WIDTH][14] __attribute__((aligned(64)));
        u32_t hashes[SIMD_WIDTH][5]   __attribute__((aligned(64)));
        SIMD_TYPE interleaved_data[14] __attribute__((aligned(64)));
        SIMD_TYPE interleaved_hash[5]  __attribute__((aligned(64)));
        
        // Obter informação da thread
        int tid = omp_get_thread_num();
        int n_threads = omp_get_num_threads();
        
        u64_t block_counter = (u64_t)tid;

        // --- NOVO: Constante Fast-Check (privada para cada thread) ---
        #ifndef USE_FALLBACK_CHECK
        const TARGET_HASH_TYPE target_hash = TARGET_HASH_SET1(0xAAD20250u);
        #endif
        // --- FIM NOVO ---

        // --- Loop Principal da Thread ---
        while(keep_running) {
            
            // 1. Condição de paragem (otimizada)
            // (Apenas verifica o contador global periodicamente para reduzir o tráfego)
            if (max_attempts > 0 && (block_counter % 1000 == 0)) {
                u64_t current_total;
                #pragma omp atomic read
                current_total = total_attempts;
                
                if (current_total >= max_attempts) {
                    keep_running = 0; 
                    break;
                }
            }
            
            // 2. Gerar ID de trabalho
            u64_t counter_id_base = global_counter_offset + (block_counter * SIMD_WIDTH);
            
            // 3. --- MODIFICADO: Gerar mensagens com a função otimizada ---
            for(int i = 0; i < SIMD_WIDTH; i++) {
                generate_single_message_optimized(messages[i], 
                                                 counter_id_base + (u64_t)i, 
                                                 has_custom); // 'has_custom' é lida (read-only)
            }
            
            // 4. Processar (local)
            interleave_messages(interleaved_data, messages, SIMD_WIDTH);
            SHA1_SIMD(interleaved_data, interleaved_hash);
            
            // 5. --- MODIFICADO: Verificar com "Fast Check" ---
            
            #if defined(USE_FALLBACK_CHECK)
            // --- CAMINHO FALLBACK (NEON) ---
            deinterleave_hashes(hashes, interleaved_hash, SIMD_WIDTH);
            for(int i = 0; i < SIMD_WIDTH; i++) {
                if(is_valid_deti_coin(hashes[i])) {
                    u32_t value = count_coin_value(hashes[i]);
                    #pragma omp critical (print_and_save)
                    {
                        total_coins_found++;
                        printf("\n>>> [T%d] COIN FOUND! Value=%u Counter=%llu\n", 
                               tid, value, (unsigned long long)(counter_id_base + i));
                        save_coin(messages[i]);
                    }
                }
            }
            
            #else
            // --- CAMINHO OTIMIZADO (AVX/AVX2/AVX512) ---
            TARGET_MASK_TYPE mask = TARGET_HASH_CMP(interleaved_hash[0], target_hash);
            
            if (!MASK_IS_ZERO(mask)) {
                // CAMINHO LENTO (RARO): Moeda(s) encontrada(s)!
                deinterleave_hashes(hashes, interleaved_hash, SIMD_WIDTH);
                
                for(int i = 0; i < SIMD_WIDTH; i++) {
                    if (MASK_TEST_LANE(mask, i)) {
                        u32_t value = count_coin_value(hashes[i]);
                        
                        // --- Secção Crítica ---
                        #pragma omp critical (print_and_save)
                        {
                            total_coins_found++; 
                            // Impressão minimalista para reduzir contenção
                            printf("\n>>> [T%d] COIN FOUND! Value=%u Counter=%llu\n", 
                                   tid, value, (unsigned long long)(counter_id_base + i));
                            save_coin(messages[i]);
                        }
                    }
                }
            }
            // (Caminho rápido, 99.999% das vezes, não faz nada)
            #endif
            
            // 6. Atualizar contadores globais
            #pragma omp atomic update
            total_attempts += SIMD_WIDTH;
            
            // Avança para o próximo bloco de trabalho desta thread
            block_counter += (u64_t)n_threads;
            
            
            // 7. Reportar (apenas a thread master)
            #pragma omp master
            {
                // Lê o total global (não precisa ser 100% exato,
                // por isso podemos ler fora de um 'atomic read' para performance,
                // mas vamos manter o 'atomic read' por segurança)
                u64_t current_total;
                #pragma omp atomic read
                current_total = total_attempts;

                if(current_total - last_report >= report_interval) {
                    last_report += report_interval; 
                    
                    u64_t current_coins;
                    #pragma omp atomic read
                    current_coins = total_coins_found;
                    
                    time_measurement();
                    double elapsed = measured_wall_time[1].tv_sec + measured_wall_time[1].tv_nsec * 1e-9 - start_time;
                    double rate = (double)current_total / elapsed;
                    
                    printf("\r[%llu attempts] [%llu coins] [%.2f MH/s]          ",
                           (unsigned long long)current_total,
                           (unsigned long long)current_coins,
                           rate / 1e6);
                    fflush(stdout);
                }
            }
            
        } // --- Fim do while(keep_running) ---
        
    } // --- Fim da região paralela ---
    
    
    // Final statistics
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
    
    printf("SIMD width: %d (%s) x %d Threads (OpenMP)\n", SIMD_WIDTH, SIMD_NAME, omp_get_max_threads());
    printf("========================================\n");
    
    save_coin(NULL);
}

// Main program (sem alterações)
int main(int argc, char *argv[]) {
    signal(SIGINT, signal_handler);
    
    // Initialize global PRNG seed with maximum entropy
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    
    global_seed = (unsigned int)(time(NULL) ^ 
                                 (uintptr_t)&global_seed ^ 
                                 (unsigned int)getpid() ^
                                 (unsigned int)ts.tv_nsec);
    
    // Mix the seed thoroughly
    for(int i = 0; i < 10; i++) {
        rand_r(&global_seed);
    }
    
    // Generate random counter offset so each run explores different space
    global_counter_offset = ((u64_t)rand_r(&global_seed) << 32) | 
                           (u64_t)rand_r(&global_seed);
    
    // Add more entropy from high-resolution timer
    clock_gettime(CLOCK_MONOTONIC, &ts);
    global_counter_offset ^= ((u64_t)ts.tv_nsec << 32) | (u64_t)ts.tv_sec;
    
    const char *custom_text = NULL;
    u64_t max_attempts = 0;
    
    // Parse command line arguments
    if(argc > 1)
        custom_text = argv[1];
    if(argc > 2)
        max_attempts = strtoull(argv[2], NULL, 10);
    
    // Sanitize custom text (replace newlines with backspace)
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
        search_deti_coins_simd(sanitized_text, max_attempts);
    } else {
        search_deti_coins_simd(NULL, max_attempts);
    }
    
    return 0;
}