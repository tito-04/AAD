// deti_coin_opencl_kernel.cl
// V6: NONCE GRINDING STRATEGY (Matches CUDA approach)
// Features:
// 1. Generates random prefix ONCE.
// 2. Grinds byte 53 (ASCII 32-126) inside the kernel.
// 3. Performs ~95 hashes per thread.

#define ROTL(x,n) (((x) << (n)) | ((x) >> (32 - (n))))

#define F1(b,c,d) ((b & c) | (~b & d))
#define F2(b,c,d) (b ^ c ^ d)
#define F3(b,c,d) ((b & c) | (b & d) | (c & d))
#define F4(b,c,d) (b ^ c ^ d)

#define K1 0x5A827999u
#define K2 0x6ED9EBA1u
#define K3 0x8F1BBCDCu
#define K4 0xCA62C1D6u

#define STEP(f,k,t) do { \
    uint temp = ROTL(a,5) + f(b,c,d) + e + w[t&15] + k; \
    e = d; d = c; c = ROTL(b,30); b = a; a = temp; \
} while(0)

#define EXPAND(t) do { \
    w[t&15] = ROTL(w[(t-3)&15] ^ w[(t-8)&15] ^ w[(t-14)&15] ^ w[(t-16)&15], 1); \
} while(0)

ulong rand_step(ulong* s) {
    *s = 6364136223846793005UL * (*s) + 1442695040888963407UL;
    return *s >> 43;
}

__kernel void search_deti_coins(
    __global uint* storage,
    ulong seed,
    ulong offset,
    ulong base_counter,
    uint max_ints,
    uint debug,
    __constant uint* template_data, 
    uint fixed_len
) {
    ulong gid = get_global_id(0);
    ulong counter = base_counter + gid;

    // 1. Initialize RNG
    ulong rng_start = seed ^ (offset + counter);
    rng_start = 6364136223846793005UL * rng_start + 1442695040888963407UL;
    ulong rng = rng_start;

    // 2. Prepare SHA-1 Input Block Template (w_master)
    // We need a master copy because SHA-1 destroys the 'w' array during processing.
    uint w_master[16];

    // Load template
    #pragma unroll
    for(int i = 0; i < 14; i++) w_master[i] = template_data[i];
    
    // Fill Randomness (Up to byte 52 only)
    uchar* ptr = (uchar*)w_master;
    
    #pragma unroll
    for (int i = 12; i < 53; i++) { // Stop at 53 (exclusive), so we fill up to 52
        uchar r = (uchar)rand_step(&rng);
        uchar c = 0x20 + ((r * 95) >> 8);
        if (i >= fixed_len) {
            ptr[i ^ 3] = c;
        }
    }

    // Set footer/padding
    ptr[54 ^ 3] = '\n';
    ptr[55 ^ 3] = 0x80;
    w_master[14] = 0;
    w_master[15] = 440; 

    // 3. Nonce Grinding Loop (32 to 126)
    // We declare w inside the loop or reset it from master
    uint w[16]; 
    uchar* w_ptr = (uchar*)w;

    // Note: Constant loop bounds for unrolling potential
    for (int nonce = 32; nonce <= 126; nonce++) 
    {
        // A. Restore State
        #pragma unroll
        for(int i=0; i<16; i++) w[i] = w_master[i];

        // B. Inject Nonce at Byte 53
        w_ptr[53 ^ 3] = (uchar)nonce;

        // C. Run SHA-1
        uint a = 0x67452301u;
        uint b = 0xEFCDAB89u;
        uint c = 0x98BADCFE;
        uint d = 0x10325476u;
        uint e = 0xC3D2E1F0u;

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

        // D. Check Result
        uint h0 = 0x67452301u + a;
        if (h0 == 0xAAD20250u) {
            uint h[4] = { 0xEFCDAB89u + b, 0x98BADCFE + c, 0x10325476u + d, 0xC3D2E1F0u + e };

            uint value = 0;
            for (int i = 0; i < 4; i++) {
                if (h[i] == 0) value += 32;
                else { value += clz(h[i]); break; }
            }
            if (value > 99) value = 99;

            uint idx = atomic_add(&storage[0], 15u);
            if (idx + 15u <= max_ints) {
                storage[idx] = value;
                // Save the winning message currently in w
                // (Since we just ran SHA-1, w is garbage, so we must reconstruct from master + nonce)
                uchar* saved_ptr = (uchar*)&storage[idx + 1];
                uchar* master_ptr = (uchar*)w_master;
                
                for(int k=0; k<56; k++) saved_ptr[k] = master_ptr[k];
                // Apply the specific nonce that won
                saved_ptr[53 ^ 3] = (uchar)nonce; 
            }
        }
    }
}