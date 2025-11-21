// deti_coin_opencl_worker.c
// REVISED: Fixes fread warning and includes cleanup.
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

#define COIN_HEX_STRLEN (55 * 2 + 1) 
#define MAX_CUSTOM_LEN 34
#define STORAGE_INTS  (1 << 16) 
#define MAX_SECONDS 60.0 
#define DISPLAY_INTERVAL_ATTEMPTS 50000000ULL 
#define MAX_COINS_PER_ROUND 1024
#define LOOP_SIZE 95 

extern volatile int keep_running;

static int is_cl_init = 0;
static cl_context ctx;
static cl_command_queue queue;
static cl_kernel kernel;
static cl_mem d_storage;
static cl_mem d_template;
static cl_program prog_handle = NULL;
static u32_t* h_storage = NULL; 
static u64_t host_lcg_state = 0;
static char device_name[128] = {0};

void check_cl_error(cl_int err, const char* op) {
    if (err != CL_SUCCESS) { fprintf(stderr, "OpenCL Error %s: %d\n", op, err); exit(1); }
}

// FIX: Correctly checking fread return value
char* load_kernel(const char* filename) {
    FILE* f = fopen(filename, "rb"); 
    if (!f) { perror("fopen"); exit(1); }
    
    fseek(f, 0, SEEK_END); 
    long sz = ftell(f); 
    rewind(f);
    
    char* buf = malloc(sz + 1);
    if (!buf) { perror("malloc"); fclose(f); exit(1); }

    if (fread(buf, 1, sz, f) != (size_t)sz) {
        fprintf(stderr, "Error: Failed to read complete kernel file %s\n", filename);
        free(buf);
        fclose(f);
        exit(1);
    }
    
    buf[sz] = 0; 
    fclose(f); 
    return buf;
}

static void prepare_template(u32_t *buffer, const char *custom_text, int custom_len) {
    u08_t *bytes = (u08_t *)buffer;
    memset(buffer, 0, 14 * sizeof(u32_t));
    const char header[] = "DETI coin 2 ";
    for(int i = 0; i < 12; i++) bytes[i ^ 3] = (u08_t)header[i];
    int current_idx = 12;
    int text_pos = 0;
    if (custom_text != NULL) {
        while(text_pos < custom_len && current_idx < 54) {
            char c = custom_text[text_pos++];
            if(c < 32 || c > 126) c = ' '; 
            bytes[current_idx ^ 3] = (u08_t)c;
            current_idx++;
        }
    }
}

extern void cleanup_cl_worker() {
    if (is_cl_init) {
        if (h_storage) CL_CALL(clEnqueueUnmapMemObject, (queue, d_storage, h_storage, 0, NULL, NULL));
        if (d_storage) CL_CALL(clReleaseMemObject, (d_storage));
        if (d_template) CL_CALL(clReleaseMemObject, (d_template));
        if (kernel)    CL_CALL(clReleaseKernel, (kernel));
        if (prog_handle) CL_CALL(clReleaseProgram, (prog_handle)); 
        if (queue)     CL_CALL(clReleaseCommandQueue, (queue));
        if (ctx)       CL_CALL(clReleaseContext, (ctx));
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
    if (!is_cl_init) {
        cl_platform_id platform; cl_device_id device; cl_uint num; cl_int err; char* source;
        CL_CALL(clGetPlatformIDs, (1, &platform, NULL));
        if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &num) != CL_SUCCESS)
            CL_CALL(clGetDeviceIDs, (platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL));
        clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);

        ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
        queue = clCreateCommandQueue(ctx, device, 0, &err);
        
        source = load_kernel("deti_coin_opencl_kernel.cl");
        prog_handle = clCreateProgramWithSource(ctx, 1, (const char**)&source, NULL, &err);
        clBuildProgram(prog_handle, 1, &device, "", NULL, NULL);
        
        // Optional: Build log retrieval if compilation fails
        // size_t log_len; clGetProgramBuildInfo(prog_handle, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_len);
        // ...

        kernel = clCreateKernel(prog_handle, "search_deti_coins", &err);
        free(source); 

        d_storage = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, STORAGE_INTS * 4, NULL, &err);
        d_template = clCreateBuffer(ctx, CL_MEM_READ_ONLY, 16 * 4, NULL, &err);
        h_storage = (u32_t*)clEnqueueMapBuffer(queue, d_storage, CL_TRUE, CL_MAP_READ|CL_MAP_WRITE, 0, STORAGE_INTS*4, 0, NULL, NULL, &err);
        h_storage[0] = 1; 

        struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
        host_lcg_state = ((u64_t)ts.tv_nsec) ^ ((u64_t)getpid() << 32);

        u64_t seed = 0, offset = 0; u32_t max_ints = STORAGE_INTS, debug = 0;
        clSetKernelArg(kernel, 1, sizeof(u64_t), &seed);
        clSetKernelArg(kernel, 2, sizeof(u64_t), &offset);
        clSetKernelArg(kernel, 4, sizeof(u32_t), &max_ints);
        clSetKernelArg(kernel, 5, sizeof(u32_t), &debug);
        is_cl_init = 1;
        printf("[OpenCL] Device: %s (Nonce Grinding x%d)\n", device_name, LOOP_SIZE);
    }
    
    u64_t total_attempts = 0;
    int coins_found = 0;
    u64_t last_report = 0;
    u64_t base_counter = (u64_t)work_id * 100000000000ULL;
    
    size_t global = 256 * 1024; 
    size_t local  = 256; 
    
    int custom_len = 0;
    if (custom_text != NULL) {
        size_t len = strlen(custom_text);
        custom_len = (len > MAX_CUSTOM_LEN) ? MAX_CUSTOM_LEN : (int)len;
    }

    u32_t h_template[16];
    prepare_template(h_template, custom_text, custom_len);
    u32_t fixed_len = 12 + custom_len;

    CL_CALL(clEnqueueWriteBuffer, (queue, d_template, CL_TRUE, 0, 64, h_template, 0, NULL, NULL));
    CL_CALL(clSetKernelArg, (kernel, 6, sizeof(cl_mem), &d_template));
    CL_CALL(clSetKernelArg, (kernel, 7, sizeof(u32_t), &fixed_len));

    struct timespec t_start, t_curr;
    clock_gettime(CLOCK_MONOTONIC, &t_start);
    
    int keep_looping = 1;
    while(keep_looping) {
        if (!keep_running) break;
        
        CL_CALL(clEnqueueUnmapMemObject, (queue, d_storage, h_storage, 0, NULL, NULL));
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_storage);
        clSetKernelArg(kernel, 3, sizeof(u64_t), &base_counter);
        
        CL_CALL(clEnqueueNDRangeKernel, (queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL));
        
        u64_t hashes_this_round = (u64_t)global * LOOP_SIZE;
        base_counter += global; 
        total_attempts += hashes_this_round;
        
        cl_int err;
        h_storage = (u32_t*)clEnqueueMapBuffer(queue, d_storage, CL_TRUE, CL_MAP_READ|CL_MAP_WRITE, 0, STORAGE_INTS*4, 0, NULL, NULL, &err);
        
        if (h_storage[0] > 1u) {
             for (u32_t i = 1; i < h_storage[0]; i += 15) {
                 if (coins_found < MAX_COINS_PER_ROUND) { 
                     u32_t val = h_storage[i];
                     u08_t *cb = (u08_t *)&h_storage[i+1];
                     char *out = coins_out[coins_found];
                     int pos = 0;
                     for (int b = 0; b < 55; ++b) pos += snprintf(out+pos, COIN_HEX_STRLEN-pos, "%02x", cb[b^3]);
                     out[COIN_HEX_STRLEN-1] = 0;
                     printf("\n[!] OpenCL FOUND! Val: %u | Nonce: %c\n", val, cb[53^3]);
                     coins_found++;
                 }
             }
             h_storage[0] = 1u; 
        }

        if ((total_attempts - last_report) >= DISPLAY_INTERVAL_ATTEMPTS) {
            clock_gettime(CLOCK_MONOTONIC, &t_curr);
            double elap = (t_curr.tv_sec - t_start.tv_sec) + (t_curr.tv_nsec - t_start.tv_nsec) * 1e-9;
            if(elap > 0) {
                printf("\r[Status] Speed: %.2f MH/s | Hashes: %lu | Coins: %d", 
                       (double)total_attempts/elap/1e6, (unsigned long)total_attempts, coins_found);
                fflush(stdout);
            }
            last_report = total_attempts;
        }
        
        clock_gettime(CLOCK_MONOTONIC, &t_curr);
        double elapsed = (t_curr.tv_sec - t_start.tv_sec) + (t_curr.tv_nsec - t_start.tv_nsec) * 1e-9;
        if (elapsed >= MAX_SECONDS || coins_found >= MAX_COINS_PER_ROUND) keep_looping = 0;
    }

    printf("\n");
    clock_gettime(CLOCK_MONOTONIC, &t_curr);
    double elapsed = (t_curr.tv_sec - t_start.tv_sec) + (t_curr.tv_nsec - t_start.tv_nsec) * 1e-9;
    *attempts_out = (long)total_attempts;
    *coins_found_out = coins_found;
    *mhs_out = (elapsed > 0) ? ((double)total_attempts / elapsed / 1000000.0) : 0.0;
}