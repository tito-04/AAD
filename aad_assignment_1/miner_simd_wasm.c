//
// Ficheiro: miner_simd_wasm.c (Versão Limpa & Otimizada)
// Estratégia: SIMD 128-bit (4 Lanes) + Nonce Grinding
//
#include <string.h>
#include <stdint.h>        // Tipos 64-bit reais
#include <emscripten.h>
#include <wasm_simd128.h>  // Intrínsecos Wasm

// --- DEFINIÇÃO DE TIPOS ---
// Definimos manualmente para evitar conflitos de 32-bit do aad_data_types.h original
typedef uint8_t  u08_t;
typedef uint32_t u32_t;
typedef uint64_t u64_t;

// Incluir a lógica matemática (Certifica-te que aad_sha1.h está na pasta)
#include "aad_sha1.h"

// --- União para extração de dados ---
typedef union {
    v128_t v;
    u32_t s[4];
} v128_scalar_union;

// --- Wrapper SIMD para a Macro SHA1 ---
static void sha1_simd(v128_t *data, v128_t *hash) 
{
    // 1. Definimos o Tipo como vetor de 128 bits
    #define T v128_t
    
    // 2. Definimos como criar constantes (Broadcast/Splat)
    #define C(c) wasm_i32x4_splat((int32_t)c)
    
    // 3. Definimos a Rotação (Usando intrínsecos para ser eficiente)
    // Nota: O resto das operações (+, ^, &, |) funcionam automaticamente com v128_t!
    #define ROTATE(x,n) wasm_v128_or(wasm_i32x4_shl((x), (n)), wasm_u32x4_shr((x), (32 - (n))))
    
    // 4. Definimos o acesso aos dados
    #define DATA(idx) data[idx]
    #define HASH(idx) hash[idx]

    // 5. Chamamos a macro "feia" do ficheiro header
    CUSTOM_SHA1_CODE(); 

    // 6. Limpeza
    #undef T
    #undef C
    #undef ROTATE
    #undef DATA
    #undef HASH
}

// --- LCG PRNG 64-bit ---
#define LCG_MULT 6364136223846793005ULL
#define LCG_INCR 1442695040888963407ULL
static u64_t lcg_rand(u64_t state) { return LCG_MULT * state + LCG_INCR; }

// --- Configuração ---
#define SIMD_WIDTH 4
#define SALT_START_IDX 12
#define SLOW_SALT_END 45
#define FAST_NONCE_START 46

// Utility: Contar zeros
static u32_t count_coin_value(u32_t *hash) {
    u32_t n;
    for(n = 0u; n < 128u; n++) {
        if (((hash[1u + n / 32u] >> (31u - (n % 32u))) & 1u) != 0u) break;
    }
    return (n > 99u) ? 99u : n;
}

// Utility: Gerador de Salt Lento (Global)
static u64_t generate_safe_salt(u08_t *full_buffer, int start_idx, int end_idx, const char *custom_prefix, u32_t prefix_len, u64_t state) {
    int current_logical = start_idx;
    u32_t prefix_pos = 0u;
    if (start_idx == SALT_START_IDX && custom_prefix != NULL) {
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

// Utility: Gerador de Prefixo Rápido (Lane Unique)
static u64_t generate_lane_randoms(u08_t *buffer, u64_t state) {
    for (int i = 0; i < 7; i++) {
        state = lcg_rand(state);
        u08_t random_raw = (u08_t)(state >> 56);
        u08_t ascii_char = 32 + (u08_t)((random_raw * 95u) >> 8);
        buffer[i] = ascii_char;
    }
    return state;
}

// --- Buffers Globais ---
u32_t g_found_coin[14];
u32_t g_hash_result[5];
u64_t g_lcg_state;

EMSCRIPTEN_KEEPALIVE u32_t* get_coin_buffer_ptr() { return g_found_coin; }
EMSCRIPTEN_KEEPALIVE u32_t* get_hash_buffer_ptr() { return g_hash_result; }
EMSCRIPTEN_KEEPALIVE u64_t* get_lcg_state_ptr()   { return &g_lcg_state; }

// --- FUNÇÃO PRINCIPAL SIMD ---
EMSCRIPTEN_KEEPALIVE
int search_chunk(u32_t initial_lcg_state_low, u32_t initial_lcg_state_high, u32_t initial_salt_low, u32_t initial_salt_high, const char *custom_text, u32_t custom_len, u32_t chunk_size)
{
    u64_t lcg_state = ((u64_t)initial_lcg_state_high << 32) | initial_lcg_state_low;
    u64_t initial_salt_state = ((u64_t)initial_salt_high << 32) | initial_salt_low;

    // 1. Estruturas
    u32_t master_template[14];
    u32_t lane_data[SIMD_WIDTH][14] __attribute__((aligned(16)));
    
    v128_t interleaved_data[14];
    v128_t interleaved_hash[5];
    
    u08_t *master_bytes = (u08_t *)master_template;
    memset(master_template, 0, sizeof(master_template));

    // 2. Setup Header/Footer
    const char header[] = "DETI coin 2 ";
    for(int i = 0; i < 12; i++) master_bytes[i ^ 3] = (u08_t)header[i];
    master_bytes[54 ^ 3] = (u08_t)'\n';
    master_bytes[55 ^ 3] = 0x80;

    // 3. Gerar Salt Lento
    (void)generate_safe_salt(master_bytes, SALT_START_IDX, SLOW_SALT_END, custom_text, custom_len, initial_salt_state);

    // 4. Carregar estáticos para Vetores (Broadcast)
    for(int i=0; i<=10; i++) {
        interleaved_data[i] = wasm_i32x4_splat(master_template[i]);
    }
    
    // Pré-cálculos para montagem rápida
    u32_t w13_static_footer = master_template[13] & 0x0000FFFFu;
    u32_t static_w11_prefix = master_template[11] & 0xFFFF0000u;
    const v128_t target_hash = wasm_i32x4_splat(0xAAD20250u);
    
    u08_t lane_rnd[SIMD_WIDTH][8]; 

    // 5. Loop de Chunks
    for (u32_t count = 0; count < chunk_size; count += (SIMD_WIDTH * 95))
    {
        // A. Preparar Lanes com Prefixos Únicos
        u32_t lane_w11[4], lane_w12[4], lane_w13_base[4];

        for (int i = 0; i < SIMD_WIDTH; i++) {
            lcg_state = generate_lane_randoms(lane_rnd[i], lcg_state);

            // Montar Words 11, 12 e 13 Base
            lane_w11[i] = static_w11_prefix | ((u32_t)lane_rnd[i][0] << 8) | (u32_t)lane_rnd[i][1];
            lane_w12[i] = ((u32_t)lane_rnd[i][2] << 24) | ((u32_t)lane_rnd[i][3] << 16) |
                          ((u32_t)lane_rnd[i][4] << 8)  | (u32_t)lane_rnd[i][5];
            lane_w13_base[i] = ((u32_t)lane_rnd[i][6] << 24) | w13_static_footer;
            
            // Atualizar lane_data para recuperação posterior
            lane_data[i][11] = lane_w11[i];
            lane_data[i][12] = lane_w12[i];
            // (As outras words já são cópias do master_template, só precisamos de atualizar estas)
            // Mas para segurança, copiamos o master template completo na inicialização se necessário.
            // Otimização: Apenas guardamos o necessário se encontrarmos a moeda.
        }

        interleaved_data[11] = wasm_i32x4_make(lane_w11[0], lane_w11[1], lane_w11[2], lane_w11[3]);
        interleaved_data[12] = wasm_i32x4_make(lane_w12[0], lane_w12[1], lane_w12[2], lane_w12[3]);
        v128_t w13_base = wasm_i32x4_make(lane_w13_base[0], lane_w13_base[1], lane_w13_base[2], lane_w13_base[3]);

        // B. Grinding Loop (Byte 53: 32-126)
        for (u32_t c = 32; c <= 126; c++) 
        {
            // Injetar 'c' em todas as lanes
            v128_t increment = wasm_i32x4_splat(c << 16);
            interleaved_data[13] = wasm_v128_or(w13_base, increment);

            sha1_simd(interleaved_data, interleaved_hash);

            v128_t cmp = wasm_i32x4_eq(interleaved_hash[0], target_hash);
            if (wasm_i32x4_bitmask(cmp) != 0) 
            {
                // Sucesso!
                v128_scalar_union u_cmp; u_cmp.v = cmp;
                
                for (int i = 0; i < SIMD_WIDTH; i++) {
                    if (u_cmp.s[i] != 0) { 
                        // Reconstruir moeda escalar vencedora
                        u32_t final_coin[14];
                        memcpy(final_coin, master_template, sizeof(master_template));
                        
                        // Preencher partes dinâmicas
                        final_coin[11] = lane_w11[i];
                        final_coin[12] = lane_w12[i];
                        final_coin[13] = lane_w13_base[i] | (c << 16);

                        // Extrair hash
                        v128_scalar_union h0, h1, h2, h3, h4;
                        h0.v = interleaved_hash[0]; h1.v = interleaved_hash[1];
                        h2.v = interleaved_hash[2]; h3.v = interleaved_hash[3];
                        h4.v = interleaved_hash[4];
                        u32_t final_hash[5] = { h0.s[i], h1.s[i], h2.s[i], h3.s[i], h4.s[i] };

                        memcpy(g_found_coin, final_coin, 14 * 4);
                        memcpy(g_hash_result, final_hash, 5 * 4);
                        g_lcg_state = lcg_state;
                        return (int)count_coin_value(final_hash);
                    }
                }
            }
        }
    }
    
    g_lcg_state = lcg_state;
    return -1;
}