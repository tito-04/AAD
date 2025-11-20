#define CL_TARGET_OPENCL_VERSION 120
#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <signal.h>

#include "open_cl_util.h"
#include "aad_data_types.h"
#include "aad_sha1_cpu.h"
#include "aad_vault.h"

#define STORAGE_INTS  (1 << 16) 

static volatile int keep_running = 1;

void sig_handler(int s) { (void)s; keep_running = 0; printf("\nStopping...\n"); }

// Helper to check OpenCL errors explicitly (replacing broken CL_CALL_ALT)
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
    // FIX 1: Check return value of fread
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

int main(int argc, char** argv) {
    signal(SIGINT, sig_handler);

    // --- OpenCL setup ---
    cl_platform_id platform;
    cl_device_id device;
    // CL_CALL works for functions returning cl_int directly
    CL_CALL(clGetPlatformIDs, (1, &platform, NULL));

    cl_uint num;
    if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &num) != CL_SUCCESS || num == 0) {
        printf("No GPU found, using CPU\n");
        CL_CALL(clGetDeviceIDs, (platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL));
    }
    char name[128];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, NULL);
    printf("=== DETI COIN MINER - OpenCL V3 (Optimized) ===\nDevice: %s\nMining... (Ctrl+C to stop)\n\n", name);

    cl_context ctx;
    cl_command_queue queue;
    cl_int err;

    // FIX 2: Replace CL_CALL_ALT with explicit error checking
    ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    check_cl_error(err, "clCreateContext");

    queue = clCreateCommandQueue(ctx, device, 0, &err);
    check_cl_error(err, "clCreateCommandQueue");

    char* source = load_kernel("deti_coin_opencl_kernel.cl");
    cl_program prog;
    
    prog = clCreateProgramWithSource(ctx, 1, (const char**)&source, NULL, &err);
    check_cl_error(err, "clCreateProgramWithSource");
    
    // Build program
    clBuildProgram(prog, 1, &device, "", NULL, NULL);
    
    // Check for build errors
    size_t log_size;
    clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    if (log_size > 1) {
        char* log = malloc(log_size);
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("Build Log:\n%s\n", log);
        free(log);
    }

    cl_kernel kernel;
    kernel = clCreateKernel(prog, "search_deti_coins", &err);
    check_cl_error(err, "clCreateKernel");

    // OPTIMIZATION: Use PINNED memory (CL_MEM_ALLOC_HOST_PTR)
    cl_mem d_storage;
    d_storage = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                               STORAGE_INTS * sizeof(u32_t), NULL, &err);
    check_cl_error(err, "clCreateBuffer");

    // Map buffer to get host pointer
    u32_t* storage = (u32_t*)clEnqueueMapBuffer(queue, d_storage, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE,
                                              0, STORAGE_INTS * sizeof(u32_t), 0, NULL, NULL, &err);
    check_cl_error(err, "clEnqueueMapBuffer");
    
    storage[0] = 1;

    // Setup Randomness
    u64_t seed   = time(NULL) ^ ((u64_t)clock() << 16);
    u64_t offset = ((u64_t)rand() << 32) | rand();
    u64_t total  = 0;
    u32_t max_ints = STORAGE_INTS;
    u32_t debug = (argc > 1 && strcmp(argv[1], "--debug") == 0);

    // Tuning Work Size
    size_t global = 1024 * 1024; 
    size_t local  = 256; // Standard for most GPUs

    struct timespec t0;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    while (keep_running) {
        // Unmap before Kernel execution (release host ownership)
        CL_CALL(clEnqueueUnmapMemObject, (queue, d_storage, storage, 0, NULL, NULL));

        CL_CALL(clSetKernelArg, (kernel, 0, sizeof(cl_mem), &d_storage));
        CL_CALL(clSetKernelArg, (kernel, 1, sizeof(u64_t), &seed));
        CL_CALL(clSetKernelArg, (kernel, 2, sizeof(u64_t), &offset));
        CL_CALL(clSetKernelArg, (kernel, 3, sizeof(u64_t), &total));
        CL_CALL(clSetKernelArg, (kernel, 4, sizeof(u32_t), &max_ints));
        CL_CALL(clSetKernelArg, (kernel, 5, sizeof(u32_t), &debug));

        CL_CALL(clEnqueueNDRangeKernel, (queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL));
        total += global;

        // Re-Map to read results
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
            // Reset counter
            storage[0] = 1;
        }

        if ((total / global) % 20 == 0) {
            struct timespec now;
            clock_gettime(CLOCK_MONOTONIC, &now);
            double elapsed = (now.tv_sec - t0.tv_sec) + (now.tv_nsec - t0.tv_nsec) * 1e-9;
            printf("\r[%.2f MH/s] %llu hashes", total / elapsed / 1e6, (unsigned long long)total);
            fflush(stdout);
        }
    }

    save_coin(NULL);
    printf("\n\nAll coins saved to deti_coins_v2_vault.txt\n");

    // Cleanup
    clEnqueueUnmapMemObject(queue, d_storage, storage, 0, NULL, NULL);
    free(source);
    clReleaseMemObject(d_storage);
    clReleaseKernel(kernel);
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    return 0;
}