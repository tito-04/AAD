// deti_coin_miner_opencl.c
// Exact formatting update

#define CL_TARGET_OPENCL_VERSION 120
#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <signal.h>
#include <unistd.h>
#include "open_cl_util.h"
#include "aad_data_types.h"
#include "aad_sha1_cpu.h"
#include "aad_vault.h"

// Config
#define N_STREAMS 4
#define STORAGE_INTS (1 << 16)
#define LOOP_SIZE 95

static volatile int keep_running = 1;
static u64_t host_lcg_state = 0;
void sig_handler(int s) { (void)s; keep_running = 0; }

static inline u64_t get_random_u64() {
    host_lcg_state = 6364136223846793005ul * host_lcg_state + 1442695040888963407ul;
    return host_lcg_state;
}

void check_cl_error(cl_int err, const char* op) {
    if (err != CL_SUCCESS) { 
        fprintf(stderr, "CL Error %s: %d (%s)\n", op, err, cl_error_string(err)); 
        exit(1); 
    }
}

char* load_kernel_source(const char* f) {
    FILE *fp = fopen(f, "rb");
    if(!fp) return NULL;
    fseek(fp, 0, SEEK_END); long sz = ftell(fp); rewind(fp);
    char *buf = malloc(sz + 1);
    if (!buf) { fclose(fp); return NULL; }
    if (fread(buf, 1, sz, fp) != (size_t)sz) { free(buf); fclose(fp); return NULL; }
    buf[sz] = 0; fclose(fp);
    return buf;
}

static void generate_host_template(u32_t *buffer, const char *custom_text, int custom_len) {
    u08_t *bytes = (u08_t *)buffer;
    memset(buffer, 0, 16 * sizeof(u32_t));
    const char header[] = "DETI coin 2 ";
    for(int i = 0; i < 12; i++) bytes[i ^ 3] = (u08_t)header[i];

    int current_idx = 12;
    const int end_idx = 45; 
    int text_pos = 0;
    
    if (custom_text) {
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
    
    bytes[54 ^ 3] = (u08_t)'\n';
    bytes[55 ^ 3] = 0x80;
}

int main(int argc, char** argv) {
    signal(SIGINT, sig_handler);
    signal(SIGALRM, sig_handler); // <--- ADD THIS LINE
    const char *custom_text = (argc > 1) ? argv[1] : NULL;
    int custom_len = custom_text ? strlen(custom_text) : 0;

    cl_platform_id *platforms = NULL; cl_uint num_platforms;
    clGetPlatformIDs(0, NULL, &num_platforms);
    platforms = malloc(sizeof(cl_platform_id) * num_platforms);
    clGetPlatformIDs(num_platforms, platforms, NULL);
    
    cl_device_id device = NULL;
    cl_int err;
    int is_gpu = 0;
    int vec_width = 1;

#ifdef FORCE_CPU
    printf("[MODE] CPU (Forced)\n");
    for(cl_uint i=0; i<num_platforms; i++) {
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, 1, &device, NULL);
        if(device) break;
    }
#else
    for(cl_uint i=0; i<num_platforms; i++) {
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 1, &device, NULL);
        if(device) { is_gpu = 1; break; }
    }
    if(!device) {
         for(cl_uint i=0; i<num_platforms; i++) {
            clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, 1, &device, NULL);
            if(device) break;
        }
    }
#endif
    if(!device) { fprintf(stderr, "No device found\n"); return 1; }

    char dname[128]; clGetDeviceInfo(device, CL_DEVICE_NAME, 128, dname, NULL);
    printf("Device: %s (%s)\n", dname, is_gpu ? "GPU" : "CPU");

    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if(err != CL_SUCCESS) check_cl_error(err, "clCreateContext");

    cl_command_queue queues[N_STREAMS];
    for(int i=0; i<N_STREAMS; i++) {
        queues[i] = clCreateCommandQueue(ctx, device, 0, &err);
        if(err != CL_SUCCESS) check_cl_error(err, "clCreateCommandQueue");
    }

    char opts[512] = "-cl-mad-enable"; 
    
    if (is_gpu) {
        strcat(opts, " -DUSE_SCALAR");
        vec_width = 1;
        printf("[OPTS] GPU Mode (Scalar)\n");
    } else {
        #if defined(__AVX512F__)
            strcat(opts, " -DUSE_AVX512"); vec_width = 16;
            printf("[OPTS] CPU Mode (AVX-512)\n");
        #elif defined(__AVX2__)
            strcat(opts, " -DUSE_AVX2"); vec_width = 8;
            printf("[OPTS] CPU Mode (AVX2)\n");
        #elif defined(__AVX__)
            strcat(opts, " -DUSE_AVX"); 
            vec_width = 8;
            printf("[OPTS] CPU Mode (AVX Legacy) | Width: 8\n");
        #else
            printf("[OPTS] CPU Mode (No Vectorization)\n");
        #endif
    }

    char *src = load_kernel_source("deti_coin_opencl_kernel.cl");
    if(!src) { fprintf(stderr, "Kernel load failed\n"); return 1; }
    
    cl_program prog = clCreateProgramWithSource(ctx, 1, (const char**)&src, NULL, &err);
    if(clBuildProgram(prog, 1, &device, opts, NULL, NULL) != CL_SUCCESS) {
        char log[8192];
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 8192, log, NULL);
        printf("Build Error:\n%s\n", log); return 1;
    }
    cl_kernel kernel = clCreateKernel(prog, "miner_kernel", &err);
    check_cl_error(err, "clCreateKernel");

    cl_mem d_stor[N_STREAMS], d_tpl[N_STREAMS];
    u32_t h_tpl[N_STREAMS][16];
    
    for(int i=0; i<N_STREAMS; i++) {
        d_stor[i] = clCreateBuffer(ctx, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, STORAGE_INTS*4, NULL, &err);
        u32_t *p = clEnqueueMapBuffer(queues[i], d_stor[i], CL_TRUE, CL_MAP_WRITE, 0, 4, 0, NULL, NULL, &err);
        p[0] = 1; 
        clEnqueueUnmapMemObject(queues[i], d_stor[i], p, 0, NULL, NULL);
        d_tpl[i] = clCreateBuffer(ctx, CL_MEM_READ_ONLY, 64, NULL, &err);
    }
    
    u64_t seed = time(NULL), offset=0; u32_t max_i = STORAGE_INTS, dbg=0;
    clSetKernelArg(kernel, 1, 8, &seed);
    clSetKernelArg(kernel, 2, 8, &offset);
    clSetKernelArg(kernel, 4, 4, &max_i);
    clSetKernelArg(kernel, 5, 4, &dbg);

    size_t base_batch = is_gpu ? (1024*256) : (1024*64);
    size_t gws = base_batch / vec_width; 

    u64_t total=0, last_rep=0;
    struct timespec t0, tcurr; clock_gettime(CLOCK_MONOTONIC, &t0);
    host_lcg_state = t0.tv_nsec;

    printf("[RUN] Starting Mining Loop...\n");

    alarm(60);

    int fr=0;
    while(keep_running) {
        int s = fr % N_STREAMS;
        
        if(fr >= N_STREAMS) {
            clFinish(queues[s]);
            u32_t *map = clEnqueueMapBuffer(queues[s], d_stor[s], CL_TRUE, CL_MAP_READ|CL_MAP_WRITE, 0, STORAGE_INTS*4, 0, NULL, NULL, &err);
            if(map[0] > 1) {
                for(u32_t k=1; k < map[0]; k+=15) {
                    save_coin(&map[k+1]);
                    // FORMATTED PRINT
                    u08_t *coin_bytes = (u08_t*)&map[k+1];
                    printf("\n[Stream %d] FOUND! Nonce: %c\n", s, coin_bytes[53^3]);
                }
                map[0] = 1;
            }
            clEnqueueUnmapMemObject(queues[s], d_stor[s], map, 0, NULL, NULL);
        }

        generate_host_template(h_tpl[s], custom_text, custom_len);
        u64_t base = get_random_u64();
        
        clEnqueueWriteBuffer(queues[s], d_tpl[s], CL_FALSE, 0, 64, h_tpl[s], 0, NULL, NULL);
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_stor[s]);
        clSetKernelArg(kernel, 3, 8, &base);
        clSetKernelArg(kernel, 6, sizeof(cl_mem), &d_tpl[s]);
        
        clEnqueueNDRangeKernel(queues[s], kernel, 1, NULL, &gws, NULL, 0, NULL, NULL);
        clFlush(queues[s]);
        
        total += gws * vec_width * LOOP_SIZE;
        if(total - last_rep > 20000000) {
            clock_gettime(CLOCK_MONOTONIC, &tcurr);
            double dt = (tcurr.tv_sec - t0.tv_sec) + (tcurr.tv_nsec - t0.tv_nsec)*1e-9;
            printf("\r[%.2f MH/s] [Total: %llu]  ", total/dt/1e6, (unsigned long long)total);
            fflush(stdout);
            last_rep = total;
        }
        fr++;
    }
    
    // --- SUMMARY STATS ---
    clock_gettime(CLOCK_MONOTONIC, &tcurr);
    double total_time = (tcurr.tv_sec - t0.tv_sec) + (tcurr.tv_nsec - t0.tv_nsec)*1e-9;
    
    printf("\n\n--- DONE ---\n");
    printf("Attempts: %llu Time: %.2fs\n", (unsigned long long)total, total_time);
    printf("Avg: %.2f MH/s\n", (double)total / total_time / 1e6);

    save_coin(NULL);
    for(int i=0; i<N_STREAMS; i++) {
        clReleaseMemObject(d_stor[i]);
        clReleaseMemObject(d_tpl[i]);
        clReleaseCommandQueue(queues[i]);
    }
    clReleaseKernel(kernel);
    clReleaseProgram(prog);
    clReleaseContext(ctx);
    free(platforms);

    return 0;
}