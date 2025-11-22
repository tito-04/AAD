// deti_coin_opencl.c
// Strategy: Random Prefix (Nonce) + Slow Salt Interval
// Matches behavior of: deti_coin_worker.c, deti_coin_cuda_worker.cu

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

#define STORAGE_INTS  (1 << 16) 
#define LOOP_SIZE 95 

// --- STRATEGY CONFIGURATION ---
#define SALT_UPDATE_INTERVAL 50000ULL 
#define FAST_NONCE_START 46

static volatile int keep_running = 1;
static u64_t host_lcg_state = 0;

void sig_handler(int s) { (void)s; keep_running = 0; printf("\nStopping...\n"); }

// Helper: LCG Random Generator (Matches CPU/CUDA)
static inline u64_t get_random_u64() {
    host_lcg_state = 6364136223846793005ul * host_lcg_state + 1442695040888963407ul;
    return host_lcg_state;
}

void check_cl_error(cl_int err, const char* op) {
    if (err != CL_SUCCESS) {
        fprintf(stderr, "OpenCL Error during %s: %d\n", op, err);
        exit(1);
    }
}

char* load_kernel(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) { perror("fopen"); exit(1); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    rewind(f);
    char* buf = malloc(sz + 1);
    if (fread(buf, 1, sz, f) != (size_t)sz) {
        fprintf(stderr, "Error reading kernel file\n");
        free(buf);
        fclose(f);
        exit(1);
    }
    buf[sz] = '\0';
    fclose(f);
    return buf;
}

// Helper: Generates Template with Header + Text + Random Salt (Matches CPU Logic)
static void generate_host_template(u32_t *buffer, const char *custom_text, int custom_len) {
    u08_t *bytes = (u08_t *)buffer;
    memset(buffer, 0, 14 * sizeof(u32_t));

    // 1. Header
    const char header[] = "DETI coin 2 ";
    for(int i = 0; i < 12; i++) bytes[i ^ 3] = (u08_t)header[i];

    int current_idx = 12;
    const int end_idx = 45; // Salt ends at 45
    int text_pos = 0;

    // 2. Custom Text
    if (custom_text != NULL) {
        while(text_pos < custom_len && current_idx <= end_idx) {
            char c = custom_text[text_pos++];
            if(c < 32 || c > 126) c = ' '; 
            bytes[current_idx ^ 3] = (u08_t)c;
            current_idx++;
        }
    }
    
    // 3. Random Salt (Bytes 12-45)
    while (current_idx <= end_idx) {
        u64_t rnd = get_random_u64();
        u08_t ascii_char = 32 + (u08_t)(( (u08_t)(rnd >> 56) * 95) >> 8);
        bytes[current_idx ^ 3] = ascii_char;
        current_idx++;
    }

    bytes[54 ^ 3] = (u08_t)'\n';
    bytes[55 ^ 3] = 0x80;
}

int main(int argc, char** argv) {
    signal(SIGINT, sig_handler);

    const char *custom_text = (argc > 1) ? argv[1] : NULL;

    // --- OpenCL setup ---
    cl_platform_id platform;
    cl_device_id device;
    CL_CALL(clGetPlatformIDs, (1, &platform, NULL));

    cl_uint num;
    if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &num) != CL_SUCCESS || num == 0) {
        printf("No GPU found, using CPU\n");
        CL_CALL(clGetDeviceIDs, (platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL));
    }
    char name[128];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, NULL);
    printf("=== DETI COIN MINER - OpenCL (Standard Strategy) ===\n");
    printf("Device: %s\n", name);
    printf("Strategy: Random Prefix + Slow Salt (Interval: %llu)\n", (unsigned long long)SALT_UPDATE_INTERVAL);
    if (custom_text) printf("Custom Text: \"%s\"\n", custom_text);
    printf("Mining... (Ctrl+C to stop)\n\n");

    cl_context ctx;
    cl_command_queue queue;
    cl_int err;

    ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    check_cl_error(err, "clCreateContext");

    queue = clCreateCommandQueue(ctx, device, 0, &err);
    check_cl_error(err, "clCreateCommandQueue");

    char* source = load_kernel("deti_coin_opencl_kernel.cl");
    cl_program prog = clCreateProgramWithSource(ctx, 1, (const char**)&source, NULL, &err);
    check_cl_error(err, "clCreateProgramWithSource");
    
    if (clBuildProgram(prog, 1, &device, "", NULL, NULL) != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = malloc(log_size);
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("Build Log:\n%s\n", log);
        free(log);
        exit(1);
    }
    free(source);

    cl_kernel kernel = clCreateKernel(prog, "search_deti_coins", &err);
    check_cl_error(err, "clCreateKernel");

    cl_mem d_storage = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                               STORAGE_INTS * sizeof(u32_t), NULL, &err);
    check_cl_error(err, "clCreateBuffer (Storage)");

    u32_t* storage = (u32_t*)clEnqueueMapBuffer(queue, d_storage, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE,
                                              0, STORAGE_INTS * sizeof(u32_t), 0, NULL, NULL, &err);
    check_cl_error(err, "clEnqueueMapBuffer");
    storage[0] = 1;

    // --- INITIALIZE STRATEGY ---
    int custom_len = (custom_text) ? strlen(custom_text) : 0;
    if (custom_len > 34) custom_len = 34;
    
    u32_t h_template[16];
    // Initial Salt Generation
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    host_lcg_state = ((u64_t)ts.tv_nsec) ^ ((u64_t)getpid() << 32);

    generate_host_template(h_template, custom_text, custom_len);

    // CRITICAL: Set fixed_len to 46 (FAST_NONCE_START)
    // This tells the kernel to Preserve bytes 0-45 (Header + Text + Salt)
    // and only randomize Bytes 46-52 (Nonce)
    u32_t fixed_len = FAST_NONCE_START; 

    cl_mem d_template = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       16 * sizeof(u32_t), h_template, &err);
    check_cl_error(err, "clCreateBuffer (Template)");

    u64_t seed   = time(NULL) ^ ((u64_t)clock() << 16);
    u64_t offset = 0;
    u64_t total_hashes = 0; 
    u32_t max_ints = STORAGE_INTS;
    u32_t debug = 0;

    size_t global = 256 * 1024; 
    size_t local  = 256; 

    struct timespec t0, t_curr;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    u64_t last_report_hashes = 0;

    CL_CALL(clSetKernelArg, (kernel, 1, sizeof(u64_t), &seed));
    CL_CALL(clSetKernelArg, (kernel, 2, sizeof(u64_t), &offset));
    CL_CALL(clSetKernelArg, (kernel, 4, sizeof(u32_t), &max_ints));
    CL_CALL(clSetKernelArg, (kernel, 5, sizeof(u32_t), &debug));
    CL_CALL(clSetKernelArg, (kernel, 6, sizeof(cl_mem), &d_template));
    CL_CALL(clSetKernelArg, (kernel, 7, sizeof(u32_t), &fixed_len));

    u64_t salt_age = 0;

    while (keep_running) {
        
        // 1. Strategy: Update Salt (Bytes 12-45) periodically
        if (salt_age >= SALT_UPDATE_INTERVAL) {
             generate_host_template(h_template, custom_text, custom_len);
             CL_CALL(clEnqueueWriteBuffer, (queue, d_template, CL_TRUE, 0, 64, h_template, 0, NULL, NULL));
             salt_age = 0;
        } else {
             salt_age++;
        }

        // 2. Strategy: Randomize Nonce Base (Bytes 46-52)
        u64_t base_counter = get_random_u64();

        CL_CALL(clEnqueueUnmapMemObject, (queue, d_storage, storage, 0, NULL, NULL));

        CL_CALL(clSetKernelArg, (kernel, 0, sizeof(cl_mem), &d_storage));
        CL_CALL(clSetKernelArg, (kernel, 3, sizeof(u64_t), &base_counter));

        CL_CALL(clEnqueueNDRangeKernel, (queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL));
        
        total_hashes += (u64_t)global * LOOP_SIZE;

        storage = (u32_t*)clEnqueueMapBuffer(queue, d_storage, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE,
                                           0, STORAGE_INTS * sizeof(u32_t), 0, NULL, NULL, &err);
        check_cl_error(err, "clEnqueueMapBuffer (Read)");

        if (storage[0] > 1) {
            for (u32_t i = 1; i < storage[0]; i += 15) {
                u32_t value = storage[i];
                u32_t* coin_data = &storage[i + 1];

                printf("\n========================================\n");
                printf("DETI COIN FOUND! Value = %u\n", value);
                printf("Content: ");
                for (int j = 0; j < 55; j++) {
                    u08_t c = ((u08_t*)coin_data)[j ^ 3];
                    if (c >= 32 && c <= 126) printf("%c", c);
                    else if (c == '\n') printf("\\n");
                    else if (c == '\b') printf("\\b");
                    else printf(".");
                }
                printf("\n========================================\n");

                save_coin(coin_data);
            }
            storage[0] = 1;
        }

        if ((total_hashes - last_report_hashes) > 50000000) {
            clock_gettime(CLOCK_MONOTONIC, &t_curr);
            double elapsed = (t_curr.tv_sec - t0.tv_sec) + (t_curr.tv_nsec - t0.tv_nsec) * 1e-9;
            if (elapsed > 0) {
                printf("\r[%.2f MH/s] %llu hashes", (double)total_hashes / elapsed / 1e6, (unsigned long long)total_hashes);
                fflush(stdout);
            }
            last_report_hashes = total_hashes;
        }
    }

    save_coin(NULL);
    printf("\n\nAll coins saved to deti_coins_v2_vault.txt\n");

    clEnqueueUnmapMemObject(queue, d_storage, storage, 0, NULL, NULL);
    clReleaseMemObject(d_storage);
    clReleaseMemObject(d_template);
    clReleaseKernel(kernel);
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    return 0;
}