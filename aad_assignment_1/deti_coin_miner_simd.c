//
// DETI Coin Miner - CPU SIMD Implementation (AVX-512/AVX2/AVX/NEON)
// Arquiteturas de Alto Desempenho 2025/2026
//
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
#elif defined(__ARM_NEON)
    #define SIMD_WIDTH 4
    #define SIMD_TYPE uint32x4_t
    #define SHA1_SIMD sha1_neon
    #define SIMD_NAME "NEON"
#else
    #error "No SIMD support detected. Compile with -mavx512f, -mavx2, -mavx, or ensure ARM NEON is available"
#endif

// Global variables for statistics and randomization
static volatile int keep_running = 1;
static u64_t total_attempts = 0;
static u64_t total_coins_found = 0;
static unsigned int global_seed = 0;
static u64_t global_counter_offset = 0;

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

// PRNG based on MMIX by Donald Knuth (from aad_utilities.h)
// Thread-safe version with explicit state parameter
static inline u08_t random_byte_seeded(u64_t *state) {
    *state = 6364136223846793005ul * (*state) + 1442695040888963407ul;
    return (u08_t)((*state) >> 43);
}

// Generate a single DETI coin candidate message
static void generate_single_message(u32_t data[14], u64_t counter, const char *custom_text) {
    memset(data, 0, 14 * sizeof(u32_t));
    
    u08_t *bytes = (u08_t *)data;
    const char header[] = "DETI coin 2 ";
    
    // Copy header with correct endianness (big-endian for SHA1)
    for(int i = 0; i < 12; i++) {
        bytes[i ^ 3] = (u08_t)header[i];
    }
    
    int pos = 12;
    
    // Add custom text if provided
    if(custom_text != NULL) {
        for(size_t i = 0; custom_text[i] != '\0' && pos < 54; i++, pos++) {
            char c = custom_text[i];
            if(c == '\n') c = '\b';  // Replace newlines with backspace
            bytes[pos ^ 3] = (u08_t)c;
        }
    }
    
    // Initialize PRNG state uniquely for this message
    // Combines global_seed, counter, and memory address for maximum entropy
    u64_t rng_state = global_seed ^ counter ^ (u64_t)(uintptr_t)data;
    
    // Additional mixing to ensure good initial state
    rng_state = 6364136223846793005ul * rng_state + 1442695040888963407ul;
    
    // Fill remaining positions with random printable ASCII characters
    while(pos < 54) {
        u08_t b = random_byte_seeded(&rng_state);
        
        // Map to printable ASCII range 0x20-0x7E (space to tilde)
        b = 0x20 + (b % 0x5F);  // 0x5F = 0x7F - 0x20
        
        // Skip newline character (0x0A would never occur here, but be safe)
        if(b != (u08_t)'\n') {
            bytes[pos ^ 3] = b;
            pos++;
        }
    }
    
    // Add mandatory newline at position 54
    bytes[54 ^ 3] = (u08_t)'\n';
    
    // Add SHA1 padding byte at position 55
    bytes[55 ^ 3] = 0x80;
}

// Interleave SIMD_WIDTH messages into SIMD format
static void interleave_messages(SIMD_TYPE *interleaved_data, u32_t messages[][14], int count) {
    for(int word = 0; word < 14; word++) {
        u32_t temp[SIMD_WIDTH];
        
        // Copy message data
        for(int lane = 0; lane < count; lane++) {
            temp[lane] = messages[lane][word];
        }
        
        // Fill remaining lanes with zeros if count < SIMD_WIDTH
        for(int lane = count; lane < SIMD_WIDTH; lane++) {
            temp[lane] = 0;
        }
        
        // Pack into SIMD register based on architecture
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

// Print coin information
static void print_coin(u32_t data[14], u32_t hash[5], u64_t counter, u32_t value) {
    printf("\n========================================\n");
    printf("DETI COIN FOUND!\n");
    printf("========================================\n");
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

// Main SIMD search function
void search_deti_coins_simd(const char *custom_text, u64_t max_attempts) {
    u32_t messages[SIMD_WIDTH][14];
    u32_t hashes[SIMD_WIDTH][5];
    SIMD_TYPE interleaved_data[14];
    SIMD_TYPE interleaved_hash[5];
    
    u64_t counter = 0;
    u64_t last_report = 0;
    const u64_t report_interval = 1000000;
    
    printf("========================================\n");
    printf("DETI COIN MINER - CPU %s (SIMD x%d)\n", SIMD_NAME, SIMD_WIDTH);
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
        // Generate SIMD_WIDTH messages with unique global counter offset
        for(int i = 0; i < SIMD_WIDTH; i++) {
            generate_single_message(messages[i], 
                                   global_counter_offset + counter + (u64_t)i, 
                                   custom_text);
        }
        
        // Interleave messages for SIMD processing
        interleave_messages(interleaved_data, messages, SIMD_WIDTH);
        
        // Compute SIMD_WIDTH hashes in parallel
        SHA1_SIMD(interleaved_data, interleaved_hash);
        
        // Deinterleave results
        deinterleave_hashes(hashes, interleaved_hash, SIMD_WIDTH);
        
        // Check each hash for valid coins
        for(int i = 0; i < SIMD_WIDTH; i++) {
            if(is_valid_deti_coin(hashes[i])) {
                u32_t value = count_coin_value(hashes[i]);
                print_coin(messages[i], hashes[i], counter + (u64_t)i, value);
                save_coin(messages[i]);
                total_coins_found++;
            }
        }
        
        counter += SIMD_WIDTH;
        total_attempts += SIMD_WIDTH;
        
        // Periodic status report
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
        
        // Break early if max_attempts reached
        if (max_attempts != 0 && counter >= max_attempts) 
            break;
    }
    
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
    printf("SIMD width: %d (%s)\n", SIMD_WIDTH, SIMD_NAME);
    printf("========================================\n");
    
    save_coin(NULL);
}

// Main program
int main(int argc, char *argv[]) {
    signal(SIGINT, signal_handler);
    
    // Initialize global PRNG seed with maximum entropy
    // Combines: current time, process ID, stack address, and nanoseconds
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
    // This ensures different coins are found in different runs
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
