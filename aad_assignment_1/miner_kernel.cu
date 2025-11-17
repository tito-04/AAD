//
// Ficheiro: miner_kernel.cu
//
// Kernel CUDA de mineração de DETI coins de alto desempenho.
//
// Arquitetura:
// 1. O host (CPU) envia um "template" da mensagem (com ou sem texto custom)
//    para a memória da GPU (buffer 0).
// 2. O host (CPU) envia um contador "base_counter" (u64_t).
// 3. Cada thread da GPU calcula um ID global único (n).
// 4. Cada thread combina o ID (n) com o base_counter para criar um contador único.
// 5. Cada thread copia o template para a sua memória local (registos).
// 6. Cada thread preenche o resto da sua mensagem local usando um PRNG
//    semeado com o seu contador único.
// 7. Cada thread calcula o hash SHA-1 (usando a macro).
// 8. Se (hash[0] == 0xAAD20250u), a thread usa atomicAdd() no
//    buffer de resultados (buffer 1) para reservar espaço e
//    guardar a moeda encontrada.
//

#include "aad_sha1.h"       // Para CUSTOM_SHA1_CODE
#include "aad_data_types.h" // Para u32_t, u64_t

__constant__ u32_t c_template_message[14]; // Template da mensagem (constante na GPU)

// PRNG (LCG) adaptado de aad_utilities.h para correr na GPU.
// É uma função __device__ para ser chamada por cada thread.
static __device__ inline u64_t lcg_rand(u64_t state)
{
  return 6364136223846793005ul * state + 1442695040888963407ul;
}

// O Kernel de Mineração
extern "C" __global__ __launch_bounds__(RECOMENDED_CUDA_BLOCK_SIZE,1)
void miner_kernel(
    u64_t base_counter,           // Contador base para este lançamento
    u32_t *coins_storage_area,    // Buffer de resultados (o "vault" da GPU)
    int start_pos                 // Posição onde o PRNG deve começar a escrever
)
{
    // --- 1. Calcular ID Único ---
    u64_t n = (u64_t)threadIdx.x + (u64_t)blockDim.x * (u64_t)blockIdx.x;
    u64_t thread_counter = base_counter + n;

    // --- 2. Declarar Buffers Locais (irão para os registos) ---
    u32_t data[14]; // A mensagem (AoS)
    u32_t hash[5];  // O hash (AoS)

    // --- 3. Gerar Mensagem "On The Fly" ---
    
    // Copiar o template (header, custom text, \n, 0x80)
    #pragma unroll
    for(int i = 0; i < 14; i++)
    {
        data[i] = c_template_message[i];
    }

    // Obter um ponteiro de bytes para preencher a parte aleatória
    u08_t *bytes = (u08_t *)data;
    int pos = start_pos;

    // Iniciar o LCG com o contador único
    u64_t rng_state = lcg_rand(thread_counter);

    // Preencher a parte aleatória
    while(pos < 54)
    {
        rng_state = lcg_rand(rng_state);
        u64_t temp = rng_state;

        for(int j = 0; j < 8 && pos < 54; j++, pos++)
        {
            u08_t b = (u08_t)((temp >> (j * 8)) & 0xFF);
            b = 0x20 + (u08_t)((b * 95) >> 8); // Converter para ASCII imprimível
            bytes[pos ^ 3] = b; // `^ 3` para a correção de endianness
        }
    }

    // --- 4. Calcular o Hash SHA-1 ---
    // Definir as macros que CUSTOM_SHA1_CODE espera
    #define T            u32_t
    #define C(c)         (c)
    #define ROTATE(x,n)  (((x) << (n)) | ((x) >> (32 - (n))))
    #define DATA(idx)    data[idx] // Aceder ao array local
    #define HASH(idx)    hash[idx] // Aceder ao array local

    CUSTOM_SHA1_CODE(); // O "motor" de hash

    // Limpar as macros
    #undef T
    #undef C
    #undef ROTATE
    #undef DATA
    #undef HASH

    // --- 5. Verificar e Guardar a Moeda ---
    if(hash[0] == 0xAAD20250u)
    {
        // Encontrámos uma moeda!
        u32_t idx = atomicAdd(&coins_storage_area[0], 14u);
        if(idx < 1010u) // 1024 - 14 = 1010
        {
            #pragma unroll
            for(int i = 0; i < 14; i++)
            {
                coins_storage_area[idx + i] = data[i];
            }
        }
    }
}