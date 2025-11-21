// server.c
// Scalable, Multithreaded Mining Server
// 
// Features:
// 1. Thread Pool: Fixed number of workers handling connections.
// 2. Connection Queue: Thread-safe producer-consumer queue.
// 3. Robust I/O: Uses tcp_io.h for safe line reading.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <pthread.h>

// Include the Safe Reader (assumed to be in the same dir)
#include "tcp_io.h"
#include "aad_data_types.h"
#include "aad_sha1_cpu.h"
#include "aad_vault.h"

#define PORT 9000
#define THREAD_POOL_SIZE 100      // Number of worker threads
#define MAX_CONNECTION_QUEUE 256 // Max pending connections

// --- SHARED STATE ---
static long next_work_id = 0;
static pthread_mutex_t work_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t vault_lock = PTHREAD_MUTEX_INITIALIZER;

// --- CONNECTION QUEUE ---
typedef struct {
    int sockets[MAX_CONNECTION_QUEUE];
    int head;
    int tail;
    int count;
    pthread_mutex_t lock;
    pthread_cond_t cond;
} connection_queue_t;

static connection_queue_t queue = {
    .head = 0, .tail = 0, .count = 0,
    .lock = PTHREAD_MUTEX_INITIALIZER,
    .cond = PTHREAD_COND_INITIALIZER
};

// --- HELPER FUNCTIONS ---

// Check if a hash meets the DETI Coin criteria
// (Moved here to validate before saving)
static int is_valid_coin(u32_t *hash) {
    return (hash[0] == 0xAAD20250u);
}

void enqueue_connection(int client_sock) {
    pthread_mutex_lock(&queue.lock);
    if (queue.count < MAX_CONNECTION_QUEUE) {
        queue.sockets[queue.tail] = client_sock;
        queue.tail = (queue.tail + 1) % MAX_CONNECTION_QUEUE;
        queue.count++;
        pthread_cond_signal(&queue.cond);
    } else {
        fprintf(stderr, "[Server] Queue full! Dropping connection %d\n", client_sock);
        close(client_sock);
    }
    pthread_mutex_unlock(&queue.lock);
}

int dequeue_connection() {
    int sock = -1;
    pthread_mutex_lock(&queue.lock);
    while (queue.count == 0) {
        pthread_cond_wait(&queue.cond, &queue.lock);
    }
    sock = queue.sockets[queue.head];
    queue.head = (queue.head + 1) % MAX_CONNECTION_QUEUE;
    queue.count--;
    pthread_mutex_unlock(&queue.lock);
    return sock;
}

// --- CLIENT HANDLER ---

void handle_client(int client_fd) {
    // Initialize Safe Reader
    tcp_reader_t reader;
    tcp_reader_init(&reader, client_fd);
    char *line = NULL;

    // Loop until disconnect
    while (tcp_read_line(&reader, &line) > 0) {
        // 'line' is a clean, null-terminated string from the buffer.
        // No strtok needed for the command type.

        if (strncmp(line, "GETWORK", 7) == 0) {
            pthread_mutex_lock(&work_lock);
            long my_work = next_work_id++;
            pthread_mutex_unlock(&work_lock);

            char reply[64];
            snprintf(reply, sizeof(reply), "WORK %ld\n", my_work);
            send(client_fd, reply, strlen(reply), 0);
        }
        else if (strncmp(line, "RESULT", 6) == 0) {
            long work_id = -1, attempts = -1;
            int coins_count = 0;
            float speed_mhs = 0.0;
            
            // Safe parsing of the line
            sscanf(line, "RESULT work=%ld attempts=%ld speed=%f coins=%d", 
                   &work_id, &attempts, &speed_mhs, &coins_count);
            
            if (coins_count > 0) {
                printf("\n--- COIN RECEIVED (Client %d) ---\n", client_fd);
                
                // Find the "coins=" part safely
                char *coin_ptr = strstr(line, "coins=");
                if (coin_ptr) {
                    // Move past "coins=N " to get to the first hex string
                    // We look for the first space after coins=
                    coin_ptr = strchr(coin_ptr, ' ');
                    if (coin_ptr) coin_ptr++; // Skip the space
                    
                    // Parse each coin using strtok_r (THREAD SAFE)
                    char *saveptr;
                    char *token = strtok_r(coin_ptr, " \n\r", &saveptr);
                    
                    pthread_mutex_lock(&vault_lock);
                    while (token) {
                        if (strlen(token) == 110) { // 55 bytes * 2 hex chars
                            u32_t coin_data[14];
                            u08_t *bytes = (u08_t *)coin_data;
                            
                            // Parse Hex
                            for (int j = 0; j < 55; j++) {
                                int v1, v2;
                                sscanf(&token[j*2], "%1x%1x", &v1, &v2);
                                bytes[j ^ 3] = (v1 << 4) | v2;
                            }
                            bytes[54 ^ 3] = '\n';
                            bytes[55 ^ 3] = 0x80;

                            // Validate
                            u32_t hash[5];
                            sha1(coin_data, hash);

                            if (is_valid_coin(hash)) {
                                // Display (Matches Vault format)
                                u32_t val = 0;
                                for(val = 0u; val < 128u; val++)
                                    if((hash[1u + val / 32u] >> (31u - val % 32u)) % 2u != 0u) break;
                                if(val > 99u) val = 99u;

                                printf("V%02u:", val);
                                for (int j = 0; j < 55; j++) putchar(bytes[j ^ 3]);
                                
                                save_coin(coin_data);
                            } else {
                                printf("[Security] Invalid coin rejected!\n");
                            }
                        }
                        token = strtok_r(NULL, " \n\r", &saveptr);
                    }
                    save_coin(NULL); // Flush to disk
                    pthread_mutex_unlock(&vault_lock);
                }
                printf("-------------------------------------\n");
            } else {
                 // Just status update
                 // printf("Client %d reported 0 coins\n", client_fd);
            }
        }
    }
    close(client_fd);
}

// --- WORKER THREAD ---

void *worker_thread_func(void *arg) {
    (void)arg;
    while (1) {
        int client_sock = dequeue_connection();
        if (client_sock >= 0) {
            handle_client(client_sock);
            // printf("[Server] Client %d disconnected.\n", client_sock);
        }
    }
    return NULL;
}

// --- MAIN ---

int main(int argc, char *argv[]) {
    (void)argc; (void)argv;
    int server_fd, new_sock;
    struct sockaddr_in address;
    int opt = 1;
    socklen_t addrlen = sizeof(address);

    // 1. Setup Socket
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }
    
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt))) {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }

    if (listen(server_fd, 50) < 0) {
        perror("listen");
        exit(EXIT_FAILURE);
    }

    printf("Server listening on port %d\n", PORT);
    printf("Initializing Thread Pool (%d threads)...\n", THREAD_POOL_SIZE);

    // 2. Start Thread Pool
    pthread_t thread_pool[THREAD_POOL_SIZE];
    for (int i = 0; i < THREAD_POOL_SIZE; i++) {
        if (pthread_create(&thread_pool[i], NULL, worker_thread_func, NULL) != 0) {
            perror("Failed to create worker thread");
        }
    }

    // 3. Accept Loop
    while (1) {
        new_sock = accept(server_fd, (struct sockaddr *)&address, &addrlen);
        if (new_sock < 0) {
            perror("accept");
            continue;
        }
        // Push to queue handled by workers
        enqueue_connection(new_sock);
    }

    return 0;
}