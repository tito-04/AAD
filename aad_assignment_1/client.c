#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <signal.h>
#include <errno.h>
#include "tcp_io.h" 

#define DEFAULT_SERVER "127.0.0.1"
#define DEFAULT_PORT 9000

#define MAX_COINS_PER_ROUND 1024
#define COIN_HEX_STRLEN (55 * 2 + 1) 

// --- EXTERNAL WORKER INTERFACE ---
extern void run_mining_round(long work_id,
                             long *attempts_out,
                             int *coins_found_out,
                             double *mhs_out, 
                             char coins_out[][COIN_HEX_STRLEN],
                             const char *custom_text); 

// --- EXTERNAL CLEANUP ---
extern void cleanup_cl_worker(void);
extern void cleanup_cuda_worker(void);

// --- GLOBAL CONTROL ---
volatile int keep_running = 1;

// --- HELPER FUNCTIONS ---

void sig_handler(int s) {
    (void)s;
    keep_running = 0;
    printf("\n[Client] Signal caught! Finishing round and shutting down...\n");
}

static void perform_cleanup() {
    #ifdef USE_CUDA_WORKER
        cleanup_cuda_worker();
    #elif defined(USE_OPENCL_WORKER)
        cleanup_cl_worker();
    #endif
}

// --- MAIN ---

int main(int argc, char *argv[]) {
    const char *server_ip = DEFAULT_SERVER;
    int server_port = DEFAULT_PORT;
    const char *custom_text = NULL; 
    
    // Argument Parsing
    if (argc >= 2) server_ip = argv[1];
    if (argc >= 3) server_port = atoi(argv[2]);
    if (argc >= 4) custom_text = argv[3]; 

    // Register Signal Handler
    signal(SIGINT, sig_handler);

    printf("Client configured for %s:%d\n", server_ip, server_port);
    if (custom_text) printf("Mining with custom text: \"%s\"\n", custom_text);
    printf("Press Ctrl+C to stop safely.\n");

    // --- RECONNECTION LOOP (Outer) ---
    while (keep_running) {
        
        printf("[Client] Connecting to server...\n");

        int sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0) { perror("socket"); sleep(2); continue; }

        struct sockaddr_in srv;
        srv.sin_family = AF_INET;
        srv.sin_port = htons(server_port);
        if (inet_pton(AF_INET, server_ip, &srv.sin_addr) <= 0) {
            fprintf(stderr, "Invalid server IP\n");
            close(sock);
            return 1; // Fatal error, do not retry
        }

        if (connect(sock, (struct sockaddr *)&srv, sizeof(srv)) < 0) {
            perror("connect failed");
            fprintf(stderr, "[Client] Retrying in 5 seconds...\n");
            close(sock);
            sleep(5); 
            continue; // Retry connection
        }

        printf("[Client] Connected! Starting work loop.\n");

        // Initialize Safe Reader
        tcp_reader_t reader;
        tcp_reader_init(&reader, sock);
        char *incoming_line = NULL;

        // --- MINING LOOP (Inner) ---
        while (keep_running) {
            // 1. Send GETWORK
            const char *msg = "GETWORK\n";
            if (send(sock, msg, strlen(msg), 0) < 0) {
                fprintf(stderr, "[Client] Send error (GETWORK). Server might be down.\n");
                break; // Break inner loop to reconnect
            }

            // 2. Receive WORK (Safe Read)
            // This blocks until a full line ending in \n is received
            int status = tcp_read_line(&reader, &incoming_line);

            if (status <= 0) {
                fprintf(stderr, "[Client] Disconnected or read error.\n");
                break; // Break inner loop to reconnect
            }

            // 3. Parse Work ID
            long work_id = -1;
            if (sscanf(incoming_line, "WORK %ld", &work_id) != 1) {
                printf("[Client] Received unexpected message: %s\n", incoming_line);
                sleep(1);
                continue;
            }

            // 4. Execute Mining Round
            long attempts = 0;
            int coins_found = 0;
            double mhs = 0.0;
            char coins[MAX_COINS_PER_ROUND][COIN_HEX_STRLEN];
            
            // Initialize coins memory safety
            for(int i=0; i<MAX_COINS_PER_ROUND; i++) coins[i][0] = '\0';

            // Run the worker (SIMD/CUDA/Scalar)
            run_mining_round(work_id, &attempts, &coins_found, &mhs, coins, custom_text);

            // If user hit Ctrl+C during the round, stop before sending if no coins found
            if (!keep_running && coins_found == 0) break;

            char outmsg[131072];
            int pos = snprintf(outmsg, sizeof(outmsg), "RESULT work=%ld attempts=%ld speed=%.2f coins=%d",
                               work_id, attempts, mhs, coins_found);

            for (int i = 0; i < coins_found; ++i) {
                // Check buffer space
                if (pos >= (int)sizeof(outmsg) - 200) break; 
                pos += snprintf(outmsg + pos, sizeof(outmsg) - pos, " %s", coins[i]);
            }
            pos += snprintf(outmsg + pos, sizeof(outmsg) - pos, "\n");

            // Send Results
            if (send(sock, outmsg, strlen(outmsg), 0) < 0) { 
                fprintf(stderr, "[Client] Failed to send results.\n");
                break; // Break inner loop to reconnect
            }
        }

        // Clean up socket before retrying
        close(sock);
        if (keep_running) {
            // If we broke the inner loop but still want to run, wait a bit
            sleep(2);
        }
    }

    // --- FINAL CLEANUP ---
    printf("[Client] Exiting... Cleaning up resources.\n");
    perform_cleanup();
    
    return 0;
}