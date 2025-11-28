/* deti_coin_universal_kernel.cl
 * Universal Kernel for GPU (Scalar) and CPU (AVX Vectors)
 * FIXED: Added missing int_v typedefs for CPU compilation
 */

// --- ADAPTIVE TYPES ---
#if defined(USE_AVX512)
    typedef uint16 uint_v;
    typedef int16 int_v;       // Added for CPU
    typedef ulong16 ulong_v;
    #define VEC_WIDTH 16
    #define TO_UINT_V convert_uint16
    #define GET_LANE_OFFSETS (ulong16)(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15)

#elif defined(USE_AVX2) || defined(USE_AVX)
    typedef uint8 uint_v;
    typedef int8 int_v;        // Added for CPU
    typedef ulong8 ulong_v;
    #define VEC_WIDTH 8
    #define TO_UINT_V convert_uint8
    #define GET_LANE_OFFSETS (ulong8)(0,1,2,3,4,5,6,7)

#elif defined(USE_SCALAR)
    // GPU MODE
    typedef uint uint_v;
    typedef int int_v;         // Added for GPU logic consistency
    typedef ulong ulong_v;
    #define VEC_WIDTH 1
    #define TO_UINT_V (uint)
    #define GET_LANE_OFFSETS (ulong)(0)
#endif

// --- SHA-1 CONSTANTS ---
#define K1 (uint_v)(0x5A827999u)
#define K2 (uint_v)(0x6ED9EBA1u)
#define K3 (uint_v)(0x8F1BBCDCu)
#define K4 (uint_v)(0xCA62C1D6u)

// --- MACROS ---
#define ROTL(x, n) rotate(x, (uint_v)(n))

#define F1(b,c,d) ((b & c) | (~b & d))
#define F2(b,c,d) (b ^ c ^ d)
#define F3(b,c,d) ((b & c) | (b & d) | (c & d))
#define F4(b,c,d) (b ^ c ^ d)

#define EXPAND(t) do { \
    uint_v tmp = w[(t-3)&15] ^ w[(t-8)&15] ^ w[(t-14)&15] ^ w[(t-16)&15]; \
    w[t&15] = ROTL(tmp, 1); \
} while(0)

#define STEP(F, K, t) do { \
    uint_v temp = ROTL(a, 5) + F(b,c,d) + e + w[t&15] + K; \
    e = d; d = c; c = ROTL(b, 30); b = a; a = temp; \
} while(0)

__kernel void miner_kernel(
    __global uint* storage,     
    ulong seed,
    ulong offset,
    ulong base_counter,
    uint max_ints,
    uint debug,
    __constant uint* template_data
) {
    size_t gid = get_global_id(0);
    
    // 1. UNIQUE ID & RNG
    ulong_v thread_id = (ulong_v)base_counter + (ulong_v)(gid * VEC_WIDTH) + GET_LANE_OFFSETS;

    // LCG Random
    ulong_v rng = thread_id * (ulong_v)6364136223846793005ul + (ulong_v)1442695040888963407ul;

    // 2. GENERATE RANDOM BYTES (46-52)
    uint_v r_lo = TO_UINT_V(rng);
    uint_v r_hi = TO_UINT_V(rng >> 32);

    uint_v b46 = r_lo & 0xFF;         b46 = 0x20 + ((b46 * 95) >> 8);
    uint_v b47 = (r_lo >> 8) & 0xFF;  b47 = 0x20 + ((b47 * 95) >> 8);
    uint_v b48 = (r_lo >> 16) & 0xFF; b48 = 0x20 + ((b48 * 95) >> 8);
    uint_v b49 = (r_lo >> 24) & 0xFF; b49 = 0x20 + ((b49 * 95) >> 8);
    uint_v b50 = r_hi & 0xFF;         b50 = 0x20 + ((b50 * 95) >> 8);
    uint_v b51 = (r_hi >> 8) & 0xFF;  b51 = 0x20 + ((b51 * 95) >> 8);
    uint_v b52 = (r_hi >> 16) & 0xFF; b52 = 0x20 + ((b52 * 95) >> 8);

    // 3. CONSTRUCT WORDS
    uint_v w_master[16];

    #pragma unroll
    for(int i = 0; i < 11; i++) w_master[i] = (uint_v)(template_data[i]);

    // Word 11: Host(44,45) | Random(46,47)
    uint_v w11_static = (uint_v)(template_data[11] & 0xFFFF0000u);
    w_master[11] = w11_static | (b46 << 8) | b47;

    // Word 12: Random(48-51)
    w_master[12] = (b48 << 24) | (b49 << 16) | (b50 << 8) | b51;

    // Word 13 Base: Random(52) | ... | Host Footer
    uint_v w13_base = (b52 << 24) | (uint_v)(template_data[13] & 0x0000FFFFu);

    w_master[14] = (uint_v)0;
    w_master[15] = (uint_v)440;

    // 4. NONCE GRINDING LOOP
    uint_v w[16];

    for (uint nonce = 32; nonce <= 126; nonce++) 
    {
        #pragma unroll
        for(int i=0; i<16; i++) w[i] = w_master[i];

        // Inject Nonce (Byte 53)
        w[13] = w13_base | (uint_v)(nonce << 16);

        // SHA-1 Core
        uint_v a = (uint_v)0x67452301u;
        uint_v b = (uint_v)0xEFCDAB89u;
        uint_v c = (uint_v)0x98BADCFEu;
        uint_v d = (uint_v)0x10325476u;
        uint_v e = (uint_v)0xC3D2E1F0u;

        #pragma unroll
        for (int t = 0; t < 16; t++) STEP(F1, K1, t);
        #pragma unroll
        for (int t = 16; t < 20; t++) { EXPAND(t); STEP(F1, K1, t); }
        #pragma unroll
        for (int t = 20; t < 40; t++) { EXPAND(t); STEP(F2, K2, t); }
        #pragma unroll
        for (int t = 40; t < 60; t++) { EXPAND(t); STEP(F3, K3, t); }
        #pragma unroll
        for (int t = 60; t < 80; t++) { EXPAND(t); STEP(F4, K4, t); }

        uint_v h0 = a + (uint_v)0x67452301u;
        
        // --- RESULT CHECK ---
        #if defined(USE_SCALAR)
            // GPU Path
            if (h0 == 0xAAD20250u) {
                 uint idx = atomic_add(&storage[0], 15u);
                 if (idx + 15u <= max_ints) {
                     storage[idx] = 99;
                     __global uint* out = &storage[idx+1];
                     for(int k=0; k<11; k++) out[k] = template_data[k];
                     out[11] = w_master[11];
                     out[12] = w_master[12];
                     out[13] = w13_base | (nonce << 16);
                     out[14] = 0; 
                     out[15] = 440;
                 }
            }
        #else
            // CPU Path (Vector)
            int_v mask = (h0 == (uint_v)0xAAD20250u);
            
            // Check if ANY lane succeeded
            if (any(mask)) {
                union { int_v v; int s[16]; } mask_u; mask_u.v = mask;
                union { uint_v v; uint s[16]; } w11_u; w11_u.v = w_master[11];
                union { uint_v v; uint s[16]; } w12_u; w12_u.v = w_master[12];
                union { uint_v v; uint s[16]; } w13_base_u; w13_base_u.v = w13_base;

                for(int lane=0; lane<VEC_WIDTH; lane++) {
                    if (mask_u.s[lane]) {
                        uint idx = atomic_add(&storage[0], 15u);
                        if (idx + 15u <= max_ints) {
                            storage[idx] = 99;
                            __global uint* out = &storage[idx+1];
                            for(int k=0; k<11; k++) out[k] = template_data[k];
                            out[11] = w11_u.s[lane];
                            out[12] = w12_u.s[lane];
                            out[13] = w13_base_u.s[lane] | (nonce << 16);
                            out[14] = 0; 
                            out[15] = 440;
                        }
                    }
                }
            }
        #endif
    }
}