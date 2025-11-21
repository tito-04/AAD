#define CL_TARGET_OPENCL_VERSION 120
#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <signal.h>
#include <unistd.h> // Required for sleep/usleep if needed

#include "open_cl_util.h"
#include "aad_data_types.h"
#include "aad_sha1_cpu.h"
#include "aad_vault.h"

#define STORAGE_INTS  (1 << 16) 
#define LOOP_SIZE 95 // Must match the Grinding Loop in Kernel V6

static volatile int keep_running = 1;

void sig_handler(int s) { (void)s; keep_running = 0; printf("\nStopping...\n"); }

// Helper to check OpenCL errors explicitly
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

// Helper to prepare the custom text template (Matches Worker Logic)
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

int main(int argc, char** argv) {
    signal(SIGINT, sig_handler);

    // parse custom text from argv
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
    printf("=== DETI COIN MINER - OpenCL V3 (Optimized) ===\n");
    printf("Device: %s\n", name);
    if (custom_text) printf("Custom Text: \"%s\"\n", custom_text);
    printf("Mining... (Ctrl+C to stop)\n\n");

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

    cl_kernel kernel;
    kernel = clCreateKernel(prog, "search_deti_coins", &err);
    check_cl_error(err, "clCreateKernel");

    // OPTIMIZATION: Use PINNED memory (CL_MEM_ALLOC_HOST_PTR)
    cl_mem d_storage;
    d_storage = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                               STORAGE_INTS * sizeof(u32_t), NULL, &err);
    check_cl_error(err, "clCreateBuffer (Storage)");

    // Map buffer to get host pointer
    u32_t* storage = (u32_t*)clEnqueueMapBuffer(queue, d_storage, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE,
                                              0, STORAGE_INTS * sizeof(u32_t), 0, NULL, NULL, &err);
    check_cl_error(err, "clEnqueueMapBuffer");
    storage[0] = 1;

    // --- PREPARE TEMPLATE FOR KERNEL V6 ---
    int custom_len = (custom_text) ? strlen(custom_text) : 0;
    if (custom_len > 34) custom_len = 34;
    
    u32_t h_template[16];
    prepare_template(h_template, custom_text, custom_len);
    u32_t fixed_len = 12 + custom_len;

    cl_mem d_template = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       16 * sizeof(u32_t), h_template, &err);
    check_cl_error(err, "clCreateBuffer (Template)");

    // Setup Randomness
    u64_t seed   = time(NULL) ^ ((u64_t)clock() << 16);
    u64_t offset = ((u64_t)rand() << 32) | rand();
    u64_t base_counter = 0;
    u64_t total_hashes = 0; // Track total hashes (threads * loop_size)
    u32_t max_ints = STORAGE_INTS;
    u32_t debug = 0;

    // Tuning Work Size
    size_t global = 256 * 1024; // Slightly reduced for grinding kernel
    size_t local  = 256; 

    struct timespec t0, t_curr;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    u64_t last_report_hashes = 0;

    // --- SET KERNEL ARGUMENTS (Args 0-7 for V6 Kernel) ---
    CL_CALL(clSetKernelArg, (kernel, 1, sizeof(u64_t), &seed));
    CL_CALL(clSetKernelArg, (kernel, 2, sizeof(u64_t), &offset));
    CL_CALL(clSetKernelArg, (kernel, 4, sizeof(u32_t), &max_ints));
    CL_CALL(clSetKernelArg, (kernel, 5, sizeof(u32_t), &debug));
    CL_CALL(clSetKernelArg, (kernel, 6, sizeof(cl_mem), &d_template)); // New Arg
    CL_CALL(clSetKernelArg, (kernel, 7, sizeof(u32_t), &fixed_len));   // New Arg

    while (keep_running) {
        // Unmap before Kernel execution (release host ownership)
        CL_CALL(clEnqueueUnmapMemObject, (queue, d_storage, storage, 0, NULL, NULL));

        // Update dynamic args
        CL_CALL(clSetKernelArg, (kernel, 0, sizeof(cl_mem), &d_storage));
        CL_CALL(clSetKernelArg, (kernel, 3, sizeof(u64_t), &base_counter));

        // Launch Kernel
        CL_CALL(clEnqueueNDRangeKernel, (queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL));
        
        // Update counters
        base_counter += global;
        total_hashes += (u64_t)global * LOOP_SIZE; // Account for Nonce Grinding

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

    // Cleanup
    clEnqueueUnmapMemObject(queue, d_storage, storage, 0, NULL, NULL);
    clReleaseMemObject(d_storage);
    clReleaseMemObject(d_template);
    clReleaseKernel(kernel);
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    return 0;
}