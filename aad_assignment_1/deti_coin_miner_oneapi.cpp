//
// Ficheiro: deti_coin_miner_oneapi.cpp
// Implementação oneAPI / SYCL (Fixed SYCL 2020 Syntax)
// CORRIGIDO: seed incremental idêntica ao CUDA
//
#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <ctime>
#include <unistd.h>
#include <signal.h>

// Headers do projeto
#include "aad_data_types.h"
#include "aad_sha1.h"

// IMPORTANTE: Definir sha1() ANTES de incluir aad_vault.h
void sha1(u32_t *coin, u32_t *hash) {
    #define T u32_t
    #define C(c) (c)
    #define ROTATE(x,n) (((x) << (n)) | ((x) >> (32 - (n))))
    #define DATA(idx) coin[idx]
    #define HASH(idx) hash[idx]
    CUSTOM_SHA1_CODE(); 
    #undef T
    #undef C
    #undef ROTATE
    #undef DATA
    #undef HASH
}

#include "aad_vault.h"

using namespace sycl;

// --- Configuração ---
#define GLOBAL_RANGE  (1024 * 1024) 
#define LOCAL_RANGE   256           
#define MAX_COINS     1024
#define MAX_CUSTOM_LEN 34
#define SLOW_SALT_END  45  // Último byte do "slow salt" (bytes 12-45)
#define REPORT_INTERVAL 500000000ULL

static volatile int keep_running = 1;
static u64_t total_attempts = 0;
static u64_t total_coins_found = 0;

void signal_handler(int signum) {
    (void)signum;
    keep_running = 0;
    printf("\n\nShutting down...\n");
}

// --- LCG Random ---
static inline uint64_t lcg_rand(uint64_t state) {
    return 6364136223846793005ULL * state + 1442695040888963407ULL;
}

// --- Count Coin Value ---
static inline u32_t count_coin_value(u32_t *hash) {
    u32_t n;
    for(n = 0; n < 128; n++) {
        if (((hash[1 + n / 32] >> (31 - (n % 32))) & 1u) != 0u) break;
    }
    return (n > 99) ? 99 : n;
}

// --- Wrapper SHA1 para device ---
void run_sha1_sycl(u32_t* data, u32_t* hash) {
    #define T u32_t
    #define C(c) (c)
    #define ROTATE(x,n) (((x) << (n)) | ((x) >> (32 - (n))))
    #define DATA(idx) data[idx]
    #define HASH(idx) hash[idx]
    CUSTOM_SHA1_CODE(); 
    #undef T
    #undef C
    #undef ROTATE
    #undef DATA
    #undef HASH
}

int main(int argc, char* argv[]) {
    signal(SIGINT, signal_handler);

    const char* custom_text = (argc > 1) ? argv[1] : NULL;
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

    // Queue
    queue q(default_selector_v);
    
    // Header igual ao SIMD
    printf("========================================\n");
    printf("DETI COIN MINER\n");
    printf("SIMD: oneAPI/SYCL\n");
    printf("Strategy: Vectorized Loop ASCII(32-126) on Byte 53\n");
    printf("Custom Text: %s\n", custom_text ? custom_text : "(None)");
    printf("Device: %s\n", q.get_device().get_info<info::device::name>().c_str());
    printf("========================================\n");

    // USM Allocation
    u32_t* shared_template = malloc_shared<u32_t>(14, q);
    u32_t* shared_vault    = malloc_shared<u32_t>(MAX_COINS * 14, q);
    u32_t* shared_counter  = malloc_shared<u32_t>(1, q);
    u32_t* shared_hashes   = malloc_shared<u32_t>(MAX_COINS * 5, q);

    u08_t* template_bytes = (u08_t*)shared_template;
    memset(shared_template, 0, 14 * sizeof(u32_t));  // Zerar tudo primeiro!
    
    const char header[] = "DETI coin 2 ";
    for(int i=0; i<12; i++) template_bytes[i^3] = (u08_t)header[i];
    template_bytes[54^3] = '\n';
    template_bytes[55^3] = 0x80;

    // --- CORREÇÃO: Lógica IDÊNTICA ao CUDA para o base_counter inicial ---
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    // Seed inicial combinando tempo e PID (como no CUDA)
    u64_t current_base_counter = (u64_t)ts.tv_nsec ^ ((u64_t)getpid() << 32);
    
    // O host_lcg continua a servir apenas para gerar o "salt" (padding) no CPU
    uint64_t seed_part1 = (uint64_t)time(NULL);
    uint64_t host_lcg = (seed_part1 << 32) | ((u64_t)ts.tv_nsec);
    
    for(int i=0; i<10; i++) {
        host_lcg = lcg_rand(host_lcg);
    }

    auto start_time = std::chrono::steady_clock::now();
    u64_t last_report = 0;

    while(keep_running) {
        // Host Salt Gen (bytes 12-45, igual ao SIMD)
        int current_idx = 12;  // SALT_START_IDX
        int text_pos = 0;
        
        // Copiar custom text (até custom_len)
        if (custom_text && custom_len > 0) {
            while(text_pos < custom_len && current_idx <= SLOW_SALT_END) {
                char c = custom_text[text_pos++];
                if (c < 32 || c > 126) c = ' ';
                template_bytes[current_idx^3] = (u08_t)c;
                current_idx++;
            }
        }
        
        // Preencher resto com random (de current_idx até SLOW_SALT_END=45)
        while (current_idx <= SLOW_SALT_END) {
            host_lcg = lcg_rand(host_lcg);
            template_bytes[current_idx^3] = 32 + ((host_lcg >> 56) * 95) / 256;
            current_idx++;
        }
        
        // IMPORTANTE: Bytes 46-52 ficam em ZERO no template
        // O kernel vai gerar rnd[0-6] para esses bytes!

        *shared_counter = 0;
        
        // --- CORREÇÃO: NÃO usar host_lcg. Usar o counter que incrementa sempre ---
        uint64_t batch_seed = current_base_counter;
        
        // Incrementa o contador para a próxima execução (IDÊNTICO AO CUDA)
        current_base_counter += GLOBAL_RANGE;

        // Kernel Submission
        q.submit([&](handler& h) {
            h.parallel_for(nd_range<1>(range<1>(GLOBAL_RANGE), range<1>(LOCAL_RANGE)), 
                          [=](nd_item<1> item) {
                
                size_t global_id = item.get_global_id(0);
                // A seed agora vem do contador incremental
                uint64_t thread_id = batch_seed + global_id;

                u32_t data[14];
                u32_t hash[5];

                // Copiar palavras estáticas (0-10) do template
                for(int i=0; i<11; i++) data[i] = shared_template[i];
                
                // Extrair bytes estáticos que vêm do host (bytes 44-45 e 54-55)
                u32_t w13_footer = shared_template[13] & 0x0000FFFFu;  // bytes 54-55
                u32_t w11_static = shared_template[11] & 0xFFFF0000u;  // bytes 44-45

                // --- CORREÇÃO DA GERAÇÃO ALEATÓRIA (IDÊNTICO AO CUDA) ---
                // 1. Gera APENAS UM número aleatório de 64 bits
                u64_t rng = lcg_rand(thread_id);
                
                // 2. Extrai os bytes usando shifts (Tal como no miner_kernel.cu)
                u32_t b46 = (u32_t)(rng & 0xFF);         b46 = 0x20 + ((b46 * 95) >> 8);
                u32_t b47 = (u32_t)((rng >> 8) & 0xFF);  b47 = 0x20 + ((b47 * 95) >> 8);
                u32_t b48 = (u32_t)((rng >> 16) & 0xFF); b48 = 0x20 + ((b48 * 95) >> 8);
                u32_t b49 = (u32_t)((rng >> 24) & 0xFF); b49 = 0x20 + ((b49 * 95) >> 8);
                u32_t b50 = (u32_t)((rng >> 32) & 0xFF); b50 = 0x20 + ((b50 * 95) >> 8);
                u32_t b51 = (u32_t)((rng >> 40) & 0xFF); b51 = 0x20 + ((b51 * 95) >> 8);
                u32_t b52 = (u32_t)((rng >> 48) & 0xFF); b52 = 0x20 + ((b52 * 95) >> 8);

                // 3. Constrói as palavras (Word 11, 12, 13)
                // Word 11: [44][45] fixos | [46][47] random
                data[11] = w11_static | (b46 << 8) | b47;

                // Word 12: [48][49][50][51] random
                data[12] = (b48 << 24) | (b49 << 16) | (b50 << 8) | b51;

                // Word 13: [52] random | [53] loop | [54][55] footer
                u32_t w13_base = (b52 << 24) | w13_footer;

                // Loop for c=32 to 126 (igual)
                for(u32_t c = 32; c <= 126; c++) {
                    data[13] = w13_base | (c << 16);  // Byte 53 = c
                    run_sha1_sycl(data, hash);

                    if(hash[0] == 0xAAD20250u) {
                        auto ref = atomic_ref<u32_t, 
                                             memory_order::relaxed, 
                                             memory_scope::device, 
                                             access::address_space::global_space>(*shared_counter);
                        u32_t idx = ref.fetch_add(1);
                        if(idx < MAX_COINS) {
                            for(int k=0; k<14; k++) {
                                shared_vault[idx * 14 + k] = data[k];
                            }
                            for(int k=0; k<5; k++) {
                                shared_hashes[idx * 5 + k] = hash[k];
                            }
                        }
                    }
                }
            });
        }).wait();

        // Process Results
        u32_t found_count = *shared_counter;
        if (found_count > 0) {
            if (found_count > MAX_COINS) found_count = MAX_COINS;
            for(u32_t i=0; i<found_count; i++) {
                u32_t* coin_data = &shared_vault[i * 14];
                u32_t* hash_data = &shared_hashes[i * 5];
                
                save_coin(coin_data);
                total_coins_found++;
                
                u32_t val = count_coin_value(hash_data);
                u08_t* coin_bytes = (u08_t*)coin_data;
                char nonce_char = (char)coin_bytes[53^3];
                
                printf("\n[HIT] Val:%u | NonceEnd: '%c'\n", val, nonce_char);
            }
        }

        u64_t attempts_this_loop = (u64_t)GLOBAL_RANGE * 95;
        total_attempts += attempts_this_loop;

        // Progress report (igual ao SIMD)
        if ((total_attempts - last_report) >= REPORT_INTERVAL) {
            auto now = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed = now - start_time;
            double mhash = (double)total_attempts / elapsed.count() / 1000000.0;
            
            printf("\rAttempts: %llu | Found: %llu | Speed: %.2f MH/s   ", 
                   (unsigned long long)total_attempts, 
                   (unsigned long long)total_coins_found, 
                   mhash);
            fflush(stdout);
            last_report = total_attempts;
        }
    }

    // Final report (igual ao SIMD)
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> total_time = end_time - start_time;
    double avg_speed = (double)total_attempts / total_time.count() / 1e6;
    
    printf("\n\n--- Finished ---\nTotal Attempts: %llu\nTotal Time: %.2fs\nAvg Speed: %.2f MH/s\n", 
           (unsigned long long)total_attempts, 
           total_time.count(), 
           avg_speed);

    free(shared_template, q);
    free(shared_vault, q);
    free(shared_counter, q);
    free(shared_hashes, q);
    
    save_coin(NULL);
    return 0;
}