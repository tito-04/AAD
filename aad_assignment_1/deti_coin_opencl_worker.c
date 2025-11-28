/* deti_coin_opencl_worker.c
 * FIXED: 'source' variable declaration error
 * FEATURE: Supports -DFORCE_CPU flag
 */

#define CL_TARGET_OPENCL_VERSION 120
#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <signal.h>

#include "open_cl_util.h" 
#include "aad_data_types.h"
#include "aad_sha1_cpu.h" 

// --- CONFIGURATION ---
#define N_STREAMS 4
#define STORAGE_INTS (1 << 16)
#define LOOP_SIZE 95
#define MAX_SECONDS 60.0 
#define MAX_COINS_PER_ROUND 1024
#define COIN_HEX_STRLEN (55 * 2 + 1) 
#define MAX_CUSTOM_LEN 34

extern volatile int keep_running;

// --- GLOBAL STATE ---
static int is_cl_init = 0;
static cl_context ctx;
static cl_program prog_handle = NULL;
static cl_kernel kernel;

// Multi-stream arrays
static cl_command_queue queues[N_STREAMS];
static cl_mem d_stor[N_STREAMS];
static cl_mem d_tpl[N_STREAMS];
static u32_t* h_tpl[N_STREAMS]; 

static u64_t host_lcg_state = 0;
static char device_name[128] = {0};

static inline u64_t get_random_u64() {
    host_lcg_state = 6364136223846793005ul * host_lcg_state + 1442695040888963407ul;
    return host_lcg_state;
}

void check_cl_error(cl_int err, const char* op) {
    if (err != CL_SUCCESS) { fprintf(stderr, "OpenCL Error %s: %d\n", op, err); exit(1); }
}

char* load_kernel(const char* filename) {
    FILE* f = fopen(filename, "rb"); 
    if (!f) { perror("fopen"); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); rewind(f);
    char* buf = malloc(sz + 1);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, sz, f) != (size_t)sz) { free(buf); fclose(f); return NULL; }
    buf[sz] = 0; fclose(f); return buf;
}

static void generate_host_template(u32_t *buffer, const char *custom_text, int custom_len) {
    u08_t *bytes = (u08_t *)buffer;
    memset(buffer, 0, 16 * sizeof(u32_t)); 
    
    const char header[] = "DETI coin 2 ";
    for(int i = 0; i < 12; i++) bytes[i ^ 3] = (u08_t)header[i];

    int current_idx = 12;
    const int end_idx = 45; 
    int text_pos = 0;

    if (custom_text != NULL) {
        while(text_pos < custom_len && current_idx <= end_idx) {
            char c = custom_text[text_pos++];
            if(c < 32 || c > 126) c = ' '; 
            bytes[current_idx ^ 3] = (u08_t)c;
            current_idx++;
        }
    }
    while (current_idx <= end_idx) {
        u64_t rnd = get_random_u64();
        u08_t ascii_char = 32 + (u08_t)(((u08_t)(rnd >> 56) * 95) >> 8);
        bytes[current_idx ^ 3] = ascii_char;
        current_idx++;
    }
    bytes[54 ^ 3] = '\n';
    bytes[55 ^ 3] = 0x80;
}

extern void cleanup_cl_worker() {
    if (is_cl_init) {
        for(int i=0; i<N_STREAMS; i++) {
            if (d_stor[i]) clReleaseMemObject(d_stor[i]);
            if (d_tpl[i])  clReleaseMemObject(d_tpl[i]);
            if (queues[i]) clReleaseCommandQueue(queues[i]);
            if (h_tpl[i])  free(h_tpl[i]);
        }
        if (kernel)      clReleaseKernel(kernel);
        if (prog_handle) clReleaseProgram(prog_handle); 
        if (ctx)         clReleaseContext(ctx);
        is_cl_init = 0;
    }
}

extern void run_mining_round(long work_id,
                             long *attempts_out,
                             int *coins_found_out,
                             double *mhs_out, 
                             char coins_out[][COIN_HEX_STRLEN],
                             const char *custom_text)
{
    cl_int err;
    int vec_width = 1;
    int is_gpu = 0;

    // --- INITIALIZATION ---
    if (!is_cl_init) {
        cl_platform_id *platforms = NULL; cl_uint num_platforms;
        cl_device_id device = NULL;
        char *source = NULL; // FIXED: Added missing declaration

        clGetPlatformIDs(0, NULL, &num_platforms);
        platforms = malloc(sizeof(cl_platform_id) * num_platforms);
        clGetPlatformIDs(num_platforms, platforms, NULL);

        // --- DEVICE SELECTION LOGIC ---
#ifdef FORCE_CPU
        printf("[MODE] CPU (Forced)\n");
        for(cl_uint i=0; i<num_platforms; i++) {
            clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, 1, &device, NULL);
            if(device) break;
        }
#else
        // Try GPU First
        for(cl_uint i=0; i<num_platforms; i++) {
            clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 1, &device, NULL);
            if(device) { is_gpu = 1; break; }
        }
        // Fallback to CPU
        if(!device) {
             for(cl_uint i=0; i<num_platforms; i++) {
                clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, 1, &device, NULL);
                if(device) break;
            }
        }
#endif

        if(!device) { fprintf(stderr, "No OpenCL device found\n"); exit(1); }

        clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
        ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
        if(err != CL_SUCCESS) check_cl_error(err, "clCreateContext");
        
        for(int i=0; i<N_STREAMS; i++) {
            queues[i] = clCreateCommandQueue(ctx, device, 0, &err);
            h_tpl[i] = malloc(64); 
        }

        // --- AVX/SCALAR SELECTION ---
        char opts[512] = "-cl-mad-enable"; 
        
        if (is_gpu) {
            strcat(opts, " -DUSE_SCALAR");
            vec_width = 1;
        } else {
            #if defined(__AVX512F__)
                strcat(opts, " -DUSE_AVX512"); vec_width = 16;
            #elif defined(__AVX2__)
                strcat(opts, " -DUSE_AVX2"); vec_width = 8;
            #elif defined(__AVX__)
                strcat(opts, " -DUSE_AVX"); 
                vec_width = 8;
            #else
                // No vectorization fallback
                vec_width = 1; 
            #endif
        }

        source = load_kernel("deti_coin_opencl_kernel.cl");
        prog_handle = clCreateProgramWithSource(ctx, 1, (const char**)&source, NULL, &err);
        
        if(clBuildProgram(prog_handle, 1, &device, opts, NULL, NULL) != CL_SUCCESS) {
            char log[8192];
            clGetProgramBuildInfo(prog_handle, device, CL_PROGRAM_BUILD_LOG, 8192, log, NULL);
            fprintf(stderr, "Build Error:\n%s\n", log); exit(1);
        }
        kernel = clCreateKernel(prog_handle, "miner_kernel", &err);
        free(source); 
        free(platforms); 

        for(int i=0; i<N_STREAMS; i++) {
            d_stor[i] = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, STORAGE_INTS * 4, NULL, &err);
            u32_t *p = clEnqueueMapBuffer(queues[i], d_stor[i], CL_TRUE, CL_MAP_WRITE, 0, 4, 0, NULL, NULL, &err);
            p[0] = 1; 
            clEnqueueUnmapMemObject(queues[i], d_stor[i], p, 0, NULL, NULL);
            d_tpl[i] = clCreateBuffer(ctx, CL_MEM_READ_ONLY, 64, NULL, &err);
        }
        
        u64_t seed = 0, offset = 0; u32_t max_ints = STORAGE_INTS, debug = 0;
        clSetKernelArg(kernel, 1, 8, &seed);
        clSetKernelArg(kernel, 2, 8, &offset);
        clSetKernelArg(kernel, 4, 4, &max_ints);
        clSetKernelArg(kernel, 5, 4, &debug);

        is_cl_init = 1;
        printf("Device: %s (%s)\n", device_name, is_gpu ? "GPU" : "CPU");
        printf("[OPTS] Streams: %d | Vector Width: %d\n", N_STREAMS, vec_width);
    }
    
    // --- START ROUND ---
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    host_lcg_state = ((u64_t)ts.tv_nsec) ^ ((u64_t)getpid() << 32) ^ (u64_t)work_id;

    u64_t total = 0;
    u64_t last_rep = 0;
    int coins_found = 0;
    
    size_t base_batch = is_gpu ? (1024*256) : (1024*64);
    size_t gws = base_batch / vec_width; 

    int custom_len = 0;
    if (custom_text != NULL) {
        size_t len = strlen(custom_text);
        custom_len = (len > MAX_CUSTOM_LEN) ? MAX_CUSTOM_LEN : (int)len;
    }

    struct timespec t0, tcurr;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    
    int fr = 0;
    int keep_looping = 1;

    printf("[RUN] Starting Mining Loop...\n");

    while(keep_looping) {
        if (!keep_running) break;
        
        int s = fr % N_STREAMS;

        if(fr >= N_STREAMS) {
            clFinish(queues[s]);
            u32_t *map = clEnqueueMapBuffer(queues[s], d_stor[s], CL_TRUE, CL_MAP_READ|CL_MAP_WRITE, 0, STORAGE_INTS*4, 0, NULL, NULL, &err);
            
            if(map[0] > 1) {
                for(u32_t k=1; k < map[0]; k+=15) {
                    if (coins_found < MAX_COINS_PER_ROUND) {
                        u08_t *coin_bytes = (u08_t*)&map[k+1];
                        
                        char *out = coins_out[coins_found];
                        int pos = 0;
                        for (int b = 0; b < 55; ++b) pos += snprintf(out+pos, COIN_HEX_STRLEN-pos, "%02x", coin_bytes[b^3]);
                        out[COIN_HEX_STRLEN-1] = 0;
                        
                        printf("\n[Stream %d] FOUND! Nonce: %c\n", s, coin_bytes[53^3]);
                        
                        coins_found++;
                    }
                }
                map[0] = 1;
            }
            clEnqueueUnmapMemObject(queues[s], d_stor[s], map, 0, NULL, NULL);
        }

        generate_host_template(h_tpl[s], custom_text, custom_len);
        u64_t base_counter = get_random_u64();

        clEnqueueWriteBuffer(queues[s], d_tpl[s], CL_FALSE, 0, 64, h_tpl[s], 0, NULL, NULL);
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_stor[s]);
        clSetKernelArg(kernel, 3, 8, &base_counter);
        clSetKernelArg(kernel, 6, sizeof(cl_mem), &d_tpl[s]);
        
        clEnqueueNDRangeKernel(queues[s], kernel, 1, NULL, &gws, NULL, 0, NULL, NULL);
        clFlush(queues[s]);

        total += (gws * vec_width * LOOP_SIZE);
        fr++;

        if(total - last_rep > 20000000) {
            clock_gettime(CLOCK_MONOTONIC, &tcurr);
            double dt = (tcurr.tv_sec - t0.tv_sec) + (tcurr.tv_nsec - t0.tv_nsec)*1e-9;
            printf("\r[%.2f MH/s] [Total: %llu]  ", total/dt/1e6, (unsigned long long)total);
            fflush(stdout);
            last_rep = total;
        }

        clock_gettime(CLOCK_MONOTONIC, &tcurr);
        double elapsed = (tcurr.tv_sec - t0.tv_sec) + (tcurr.tv_nsec - t0.tv_nsec) * 1e-9;
        
        if (elapsed >= MAX_SECONDS || coins_found >= MAX_COINS_PER_ROUND) {
            for(int i=0; i<N_STREAMS; i++) clFinish(queues[i]);
            keep_looping = 0;
        }
    }

    // --- SUMMARY STATS ---
    clock_gettime(CLOCK_MONOTONIC, &tcurr);
    double total_time = (tcurr.tv_sec - t0.tv_sec) + (tcurr.tv_nsec - t0.tv_nsec)*1e-9;
    
    printf("\n\n--- DONE ---\n");
    printf("Attempts: %llu Time: %.2fs\n", (unsigned long long)total, total_time);
    printf("Avg: %.2f MH/s\n", (double)total / total_time / 1e6);
    
    *attempts_out = (long)total;
    *coins_found_out = coins_found;
    *mhs_out = (total_time > 0) ? ((double)total / total_time / 1e6) : 0.0;
}