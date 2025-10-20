//
// DETI Coin Miner - CPU Implementation (No SIMD)
// Arquiteturas de Alto Desempenho 2025/2026
//

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include "aad_data_types.h"
#include "aad_utilities.h"
#include "aad_sha1_cpu.h"
#include "aad_vault.h"

// Global variables for statistics
static volatile int keep_running = 1;
static u64_t total_attempts = 0;
static u64_t total_coins_found = 0;

// Signal handler for graceful shutdown
void signal_handler(int signum) {
    keep_running = 0;
    printf("\n\nShutting down gracefully...\n");
}

// Check if hash starts with 0xAAD20250
int is_valid_deti_coin(u32_t *hash) {
    return (hash[0] == 0xAAD20250u);
}

// Count leading zero bits after the signature
u32_t count_coin_value(u32_t *hash) {
    u32_t n = 0;
    for(n = 0; n < 128; n++) {
        if((hash[1 + n / 32] >> (31 - n % 32)) % 2 != 0)
            break;
    }
    return (n > 99) ? 99 : n;
}

// Generate candidate message with counter
void generate_message(u32_t data[14], u64_t counter, const char *custom_text) {
    // Clear data
    memset(data, 0, 14 * sizeof(u32_t));
    
    // Fixed header: "DETI coin 2 " (12 bytes)
    u08_t *bytes = (u08_t *)data;
    const char header[] = "DETI coin 2 ";
    for(int i = 0; i < 12; i++)
        bytes[i ^ 3] = (u08_t)header[i];
    
    // Variable content (bytes 12-53): 42 bytes
    // Use counter and custom text to generate unique content
    if(custom_text != NULL && strlen(custom_text) > 0) {
        int text_len = strlen(custom_text);
        int pos = 12;
        
        // Add custom text
        for(int i = 0; i < text_len && pos < 54; i++, pos++)
            bytes[pos ^ 3] = (u08_t)custom_text[i];
        
        // Add counter representation
        char counter_str[32];
        snprintf(counter_str, sizeof(counter_str), "%016llx", (unsigned long long)counter);
        for(int i = 0; i < (int)strlen(counter_str) && pos < 54; i++, pos++)
            bytes[pos ^ 3] = (u08_t)counter_str[i];
        
        // Fill remaining with spaces or pattern
        while(pos < 54) {
            bytes[pos ^ 3] = (u08_t)(' ' + (counter + pos) % 94);
            pos++;
        }
    } else {
        // Use counter-based pattern
        for(int i = 12; i < 54; i++) {
            u64_t val = counter + i;
            bytes[i ^ 3] = (u08_t)(32 + (val % 94)); // Printable ASCII
        }
    }
    
    // Byte 54: newline
    bytes[54 ^ 3] = (u08_t)'\n';
    
    // Byte 55: SHA1 padding
    bytes[55 ^ 3] = 0x80;
}

// Print coin information
void print_coin(u32_t data[14], u32_t hash[5], u64_t counter, u32_t value) {
    printf("\n========================================\n");
    printf("DETI COIN FOUND!\n");
    printf("========================================\n");
    printf("Counter: %llu\n", (unsigned long long)counter);
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
        else
            printf("\\x%02x", c);
    }
    printf("\n========================================\n");
}

// Main search function
void search_deti_coins(void) {
    u32_t data[14];
    u32_t hash[5];
    u64_t counter = 0;
    u64_t last_report = 0;
    const u64_t report_interval = 1000000; // Report every 1M attempts
    
    printf("========================================\n");
    printf("DETI COIN MINER - CPU (No SIMD)\n");
    printf("========================================\n");
    printf("Press Ctrl+C to stop\n");
    printf("========================================\n\n");
    
    time_measurement();
    double start_time = measured_wall_time[1].tv_sec + measured_wall_time[1].tv_nsec * 1e-9;
    
    while(keep_running) {
        // Generate candidate message
        generate_message(data, counter, NULL);
        
        // Compute SHA1 hash
        sha1(data, hash);
        
        // Check if it's a valid DETI coin
        if(is_valid_deti_coin(hash)) {
            u32_t value = count_coin_value(hash);
            print_coin(data, hash, counter, value);
            
            // Save to vault
            save_coin(data);
            total_coins_found++;
        }
        
        counter++;
        total_attempts++;
        
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
    printf("========================================\n");
    
    // Save any remaining coins
    save_coin(NULL);
}

// Main program
int main(void) {
    // Set up signal handler
    signal(SIGINT, signal_handler);
    
    // Run search
    search_deti_coins();
    
    return 0;
}