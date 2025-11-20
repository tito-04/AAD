// deti_coin_opencl_kernel.cl - Optimized V3
// Features:
// 1. Correct Coin Value calculation (clz logic fixed)
// 2. Low register usage (generates directly into W)
// 3. Hardcoded constants for "DETI coin 2 "
// 4. Loop unrolling

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
    uint debug
) {
    ulong gid = get_global_id(0);
    ulong counter = base_counter + gid;

    // 1. Initialize RNG
    // We save the start state so we can regenerate the message if we find a coin
    ulong rng_start = seed ^ (offset + counter);
    rng_start = 6364136223846793005UL * rng_start + 1442695040888963407UL;
    ulong rng = rng_start; 

    // 2. Prepare SHA-1 Input Block (w)
    // We write directly to w to save register space (removing 'data' array)
    uint w[16];

    // Hardcoded "DETI coin 2 " (Little Endian packed)
    // w[0]: "DETI" -> 0x44, 0x45, 0x54, 0x49 -> 0x44455449
    // w[1]: " coi" -> 0x20, 0x63, 0x6F, 0x69 -> 0x20636F69
    // w[2]: "n 2 " -> 0x6E, 0x20, 0x32, 0x20 -> 0x6E203220
    w[0] = 0x44455449;
    w[1] = 0x20636F69;
    w[2] = 0x6E203220;

    // Fill w[3]..w[13] with random bytes (bytes 12 to 53)
    // Accessing w as uchar* to match the specific XOR indexing of the original code
    uchar* ptr = (uchar*)w;
    
    // Note: We rely on the compiler to unroll/optimize this fill
    for (int i = 12; i < 54; i++) {
        uchar r = (uchar)rand_step(&rng);
        uchar c = 0x20 + ((r * 95) >> 8);
        ptr[i ^ 3] = c;
    }

    // Padding and Length
    ptr[54 ^ 3] = '\n'; // Byte 54
    ptr[55 ^ 3] = 0x80; // 0x80 Padding
    
    // w[14] is implicitly handled because the loop doesn't touch high bytes of w[13] or low of w[14]
    // but to be safe and strict about the 0s:
    w[14] = 0; 
    w[15] = 440; // Length: 55 bytes * 8 = 440 bits

    // 3. Initialize SHA-1 State
    uint a = 0x67452301u;
    uint b = 0xEFCDAB89u;
    uint c = 0x98BADCFE;
    uint d = 0x10325476u;
    uint e = 0xC3D2E1F0u;

    // 4. Run SHA-1 Rounds (Unrolled)
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

    // 5. Check Result
    uint h0 = 0x67452301u + a;

    if (h0 == 0xAAD20250u) {
        // Reconstruct full hash for value calculation
        uint h[4] = {
            0xEFCDAB89u + b,
            0x98BADCFE + c,
            0x10325476u + d,
            0xC3D2E1F0u + e
        };

        // --- CORRECTED VALUE CALCULATION ---
        uint value = 0; // Start at 0
        for (int i = 0; i < 4; i++) {
            if (h[i] == 0) {
                value += 32;
            } else {
                value += clz(h[i]); // Count leading zeros exactly
                break;
            }
        }
        if (value > 99) value = 99; // Cap at 99

        // Save to Global Memory
        uint idx = atomic_add(&storage[0], 15u);
        if (idx + 15u <= max_ints) {
            storage[idx] = value;
            
            // REGENERATE THE MESSAGE
            // We must regenerate the exact same text to save it.
            // Reset RNG to the state it was before the message generation loop
            rng = rng_start; 
            
            // Temporary buffer to reconstruct the message for saving
            uint saved_data[14];
            uchar* saved_ptr = (uchar*)saved_data;
            
            // Re-write header
            saved_ptr[0^3] = 'D'; saved_ptr[1^3] = 'E'; saved_ptr[2^3] = 'T'; saved_ptr[3^3] = 'I';
            saved_ptr[4^3] = ' '; saved_ptr[5^3] = 'c'; saved_ptr[6^3] = 'o'; saved_ptr[7^3] = 'i';
            saved_ptr[8^3] = 'n'; saved_ptr[9^3] = ' '; saved_ptr[10^3] = '2'; saved_ptr[11^3] = ' ';

            // Re-run RNG loop
            for (int i = 12; i < 54; i++) {
                uchar r = (uchar)rand_step(&rng);
                uchar c = 0x20 + ((r * 95) >> 8);
                saved_ptr[i ^ 3] = c;
            }
            saved_ptr[54^3] = '\n';
            saved_ptr[55^3] = 0x80;

            // Copy to global storage
            for (int i = 0; i < 14; i++) {
                storage[idx + 1 + i] = saved_data[i];
            }
        }
    }
}