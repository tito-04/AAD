//
// Ficheiro: miner_kernel.cu (CORRIGIDO: CÃ¡lculo SHA1 Funcional)
//
// Optimized DETI Coin Miner Kernel
//

#include "aad_sha1.h"       
#include "aad_data_types.h" 

static __device__ __forceinline__ u64_t lcg_rand(u64_t state)
{
  return 6364136223846793005ul * state + 1442695040888963407ul;
}

extern "C" __global__ __launch_bounds__(RECOMENDED_CUDA_BLOCK_SIZE, 4)
void miner_kernel(
    u64_t base_counter,           
    u32_t *coins_storage_area,    
    const u32_t *__restrict__ template_msg 
)
{
    u64_t thread_counter = base_counter + (u64_t)threadIdx.x + (u64_t)blockDim.x * (u64_t)blockIdx.x;

    u32_t data[14];
    u32_t hash[5];

    #pragma unroll
    for(int i = 0; i < 11; i++) {
        data[i] = template_msg[i];
    }

    u64_t rng = lcg_rand(thread_counter);

    u32_t b46 = (u32_t)(rng & 0xFF);         b46 = 0x20 + ((b46 * 95) >> 8);
    u32_t b47 = (u32_t)((rng >> 8) & 0xFF);  b47 = 0x20 + ((b47 * 95) >> 8);
    u32_t b48 = (u32_t)((rng >> 16) & 0xFF); b48 = 0x20 + ((b48 * 95) >> 8);
    u32_t b49 = (u32_t)((rng >> 24) & 0xFF); b49 = 0x20 + ((b49 * 95) >> 8);
    u32_t b50 = (u32_t)((rng >> 32) & 0xFF); b50 = 0x20 + ((b50 * 95) >> 8);
    u32_t b51 = (u32_t)((rng >> 40) & 0xFF); b51 = 0x20 + ((b51 * 95) >> 8);
    u32_t b52 = (u32_t)((rng >> 48) & 0xFF); b52 = 0x20 + ((b52 * 95) >> 8);
    u32_t b53 = (u32_t)((rng >> 56) & 0xFF); b53 = 0x20 + ((b53 * 95) >> 8);

    u32_t w11_static = template_msg[11] & 0xFFFF0000u;
    data[11] = w11_static | (b46 << 8) | b47;

    data[12] = (b48 << 24) | (b49 << 16) | (b50 << 8) | b51;

    u32_t w13_static = template_msg[13] & 0x0000FFFFu;
    data[13] = (b52 << 24) | (b53 << 16) | w13_static;

    #define T            u32_t
    #define C(c)         (c)
    #define ROTATE(x,n)  (((x) << (n)) | ((x) >> (32 - (n))))
    #define DATA(idx)    data[idx]
    #define HASH(idx)    hash[idx]

    CUSTOM_SHA1_CODE();

    #undef T
    #undef C
    #undef ROTATE
    #undef DATA
    #undef HASH


    if(hash[0] == 0xAAD20250u)
    {
        u32_t idx = atomicAdd(&coins_storage_area[0], 14u);
        
        if(idx < (1024u - 14u)) 
        {
            #pragma unroll
            for(int i = 0; i < 14; i++)
            {
                coins_storage_area[idx + i] = data[i];
            }
        }
    }
}