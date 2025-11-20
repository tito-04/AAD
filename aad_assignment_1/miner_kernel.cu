//
// Ficheiro: miner_kernel.cu
//
// Optimized DETI Coin Miner Kernel v2 (Nonce Grinding)
// Strategy:
// 1. Host provides Bytes 0-45 (Template)
// 2. GPU generates Bytes 46-52 (Random per thread)
// 3. GPU loops Byte 53 from 32 to 126 (Nonce Grinding)
//

#include "aad_sha1.h"       
#include "aad_data_types.h" 

// LCG Random Generator
static __device__ __forceinline__ u64_t lcg_rand(u64_t state)
{
  return 6364136223846793005ul * state + 1442695040888963407ul;
}

extern "C" __global__ __launch_bounds__(RECOMENDED_CUDA_BLOCK_SIZE, 2)
void miner_kernel(
    u64_t base_counter,           
    u32_t *coins_storage_area,    
    const u32_t *__restrict__ template_msg 
)
{
    // Unique ID for RNG seeding
    u64_t thread_id = base_counter + (u64_t)threadIdx.x + (u64_t)blockDim.x * (u64_t)blockIdx.x;

    // Local buffer for the message words
    u32_t data[14];
    u32_t hash[5];

    // ----------------------------------------------------------------
    // 1. STATIC LOAD (Bytes 0-43)
    // Load words 0-10 directly from global memory (cached via __restrict__)
    // ----------------------------------------------------------------
    #pragma unroll
    for(int i = 0; i < 11; i++) {
        data[i] = template_msg[i];
    }

    // ----------------------------------------------------------------
    // 2. RANDOM GENERATION (Bytes 46-52)
    // Generate 7 random bytes using 1 LCG call
    // ----------------------------------------------------------------
    u64_t rng = lcg_rand(thread_id);

    // Map random bits to ASCII (32-126)
    u32_t b46 = (u32_t)(rng & 0xFF);         b46 = 0x20 + ((b46 * 95) >> 8);
    u32_t b47 = (u32_t)((rng >> 8) & 0xFF);  b47 = 0x20 + ((b47 * 95) >> 8);
    u32_t b48 = (u32_t)((rng >> 16) & 0xFF); b48 = 0x20 + ((b48 * 95) >> 8);
    u32_t b49 = (u32_t)((rng >> 24) & 0xFF); b49 = 0x20 + ((b49 * 95) >> 8);
    u32_t b50 = (u32_t)((rng >> 32) & 0xFF); b50 = 0x20 + ((b50 * 95) >> 8);
    u32_t b51 = (u32_t)((rng >> 40) & 0xFF); b51 = 0x20 + ((b51 * 95) >> 8);
    u32_t b52 = (u32_t)((rng >> 48) & 0xFF); b52 = 0x20 + ((b52 * 95) >> 8);
    
    // ----------------------------------------------------------------
    // 3. WORD ASSEMBLY (Static parts for the loop)
    // ----------------------------------------------------------------
    
    // Word 11: Bytes 44-45 (from Host) | Bytes 46-47 (Random)
    u32_t w11_static = template_msg[11] & 0xFFFF0000u;
    data[11] = w11_static | (b46 << 8) | b47;

    // Word 12: Bytes 48-51 (Random)
    data[12] = (b48 << 24) | (b49 << 16) | (b50 << 8) | b51;

    // Word 13 Base: Byte 52 (Random) | Byte 53 (Placeholder) | Bytes 54-55 (Host Footer)
    // Note: Byte 52 is at bits 24-31. Bytes 54-55 are at bits 0-15.
    // Byte 53 will be injected at bits 16-23 inside the loop.
    u32_t w13_base = (b52 << 24) | (template_msg[13] & 0x0000FFFFu);

    // ----------------------------------------------------------------
    // 4. THE GRINDING LOOP (Byte 53)
    // Iterate through all valid ASCII characters for Byte 53
    // ----------------------------------------------------------------
    
    // Prepare SHA1 macros referencing local 'data' array
    #define T            u32_t
    #define C(c)         (c)
    #define ROTATE(x,n)  (((x) << (n)) | ((x) >> (32 - (n))))
    #define DATA(idx)    data[idx]
    #define HASH(idx)    hash[idx]

    // Loop from ' ' (32) to '~' (126)
    // Unrolling helps pipeline the SHA1 instructions
    #pragma unroll 4 
    for(u32_t c53 = 32; c53 <= 126; c53++)
    {
        // Inject current loop character into Word 13 (Bits 16-23)
        data[13] = w13_base | (c53 << 16);

        // Compute Hash
        CUSTOM_SHA1_CODE();

        // Check Signature (aad20250)
        if(hash[0] == 0xAAD20250u)
        {
            // Found a candidate! Atomic Save.
            u32_t idx = atomicAdd(&coins_storage_area[0], 14u);
            
            if(idx < (1024u - 14u)) 
            {
                #pragma unroll
                for(int i = 0; i < 14; i++) {
                    coins_storage_area[idx + i] = data[i];
                }
            }
        }
    }

    #undef T
    #undef C
    #undef ROTATE
    #undef DATA
    #undef HASH
}