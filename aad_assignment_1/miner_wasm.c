//
// Ficheiro: miner_wasm.c
// Estratégia: Nonce Grinding + Tipos Seguros (uint64_t)
//
#include <string.h>
#include <stdint.h>     // OBRIGATÓRIO para corrigir a matemática
#include <emscripten.h>

// --- Definições Manuais para evitar erros de 32-bits ---
typedef uint8_t  u08_t;
typedef uint32_t u32_t;
typedef uint64_t u64_t;

// --- Incluir SHA1 (Certifica-te que aad_sha1.h está na pasta) ---
#include "aad_sha1.h"

// --- LCG PRNG (Matemática 64-bits Forçada) ---
#define LCG_MULT 6364136223846793005ULL
#define LCG_INCR 1442695040888963407ULL

static u64_t lcg_rand(u64_t state) {
  return LCG_MULT * state + LCG_INCR;
}

// --- Wrapper SHA1 ---
static void sha1(u32_t *data, u32_t *hash) {
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

// --- Utilitários ---
static u32_t count_coin_value(u32_t *hash) {
    u32_t n;
    for(n = 0u; n < 128u; n++) {
        if (((hash[1u + n / 32u] >> (31u - (n % 32u))) & 1u) != 0u) break;
    }
    return (n > 99u) ? 99u : n;
}

static u64_t generate_safe_salt_wasm(u08_t *full_buffer, int start_idx, int end_idx, const char *custom_prefix, u32_t prefix_len, u64_t state) {
    int current_logical = start_idx;
    u32_t prefix_pos = 0u;
    if (start_idx == 12 && custom_prefix != NULL) {
        while (prefix_pos < prefix_len && current_logical <= end_idx) {
            char c = custom_prefix[prefix_pos++];
            if (c < 32 || c > 126) c = ' '; 
            full_buffer[current_logical ^ 3] = (u08_t)c; 
            current_logical++;
        }
    }
    while (current_logical <= end_idx) {
        state = lcg_rand(state);
        u08_t random_raw = (u08_t)(state >> 56);
        u08_t ascii_char = 32 + (u08_t)((random_raw * 95u) >> 8); 
        full_buffer[current_logical ^ 3] = ascii_char;
        current_logical++;
    }
    return state;
}

static u64_t generate_fast_prefix(u08_t *full_buffer, u64_t state) {
    for(int i = 0; i < 7; i++) {
        state = lcg_rand(state);
        u08_t random_raw = (u08_t)(state >> 56);
        u08_t ascii_char = 32 + (u08_t)((random_raw * 95u) >> 8);
        full_buffer[(46 + i) ^ 3] = ascii_char;
    }
    return state;
}

// --- Buffers ---
u32_t g_found_coin[14];
u32_t g_hash_result[5];
u64_t g_lcg_state; 

// --- Exports ---
EMSCRIPTEN_KEEPALIVE u32_t* get_coin_buffer_ptr() { return g_found_coin; }
EMSCRIPTEN_KEEPALIVE u32_t* get_hash_buffer_ptr() { return g_hash_result; }
EMSCRIPTEN_KEEPALIVE u64_t* get_lcg_state_ptr()   { return &g_lcg_state; }

// --- Main Search ---
EMSCRIPTEN_KEEPALIVE
int search_chunk(u32_t initial_lcg_state_low, u32_t initial_lcg_state_high, u32_t initial_salt_low, u32_t initial_salt_high, const char* custom_text, u32_t custom_len, u32_t chunk_size)
{
    u32_t data[14];
    u32_t hash[5];
    u08_t *bytes = (u08_t *)data;
    
    // Reconstrói 64-bits manualmente (Seguro)
    u64_t lcg_state = ((u64_t)initial_lcg_state_high << 32) | (u64_t)initial_lcg_state_low;
    u64_t initial_salt_state = ((u64_t)initial_salt_high << 32) | (u64_t)initial_salt_low;

    memset(data, 0, 14 * sizeof(u32_t));
    const char header[] = "DETI coin 2 ";
    for(int i = 0; i < 12; i++) bytes[i ^ 3] = (u08_t)header[i];
    bytes[54 ^ 3] = (u08_t)'\n';
    bytes[55 ^ 3] = 0x80;

    // Slow Salt
    (void)generate_safe_salt_wasm(bytes, 12, 45, custom_text, custom_len, initial_salt_state);
    
    u32_t hashes_computed = 0;

    while (hashes_computed < chunk_size) 
    {
        // Fast Prefix (46-52)
        lcg_state = generate_fast_prefix(bytes, lcg_state);

        // Nonce Grinding (53)
        for (u32_t c = 32; c <= 126; c++) 
        {
            bytes[53 ^ 3] = (u08_t)c;
            sha1(data, hash);

            if(hash[0] == 0xAAD20250u)
            {
                memcpy(g_found_coin, data, 14 * 4);
                memcpy(g_hash_result, hash, 5 * 4);
                g_lcg_state = lcg_state;
                return (int)count_coin_value(hash);
            }
        }
        hashes_computed += 95;
    }
    
    g_lcg_state = lcg_state;
    return -1;
}