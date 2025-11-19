// server.c
// Simple multithreaded server for distributed DETI mining.
// Accepts GETWORK and RESULT messages from clients and saves coins to vault.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <pthread.h>

#include "aad_data_types.h"
#include "aad_sha1_cpu.h"
#include "aad_vault.h"

#define PORT 9000
#define MAX_BUFFER 131072

static long next_work_id = 0;
static pthread_mutex_t work_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t vault_lock = PTHREAD_MUTEX_INITIALIZER;

void *handle_client(void *arg) {
    int client_fd = *(int*)arg;
    free(arg);

    char buf[MAX_BUFFER];
    ssize_t n;

    while ((n = recv(client_fd, buf, sizeof(buf)-1, 0)) > 0) {
        buf[n] = '\0';
        char *line = strtok(buf, "\n");
        while (line) {
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
                
                sscanf(line, "RESULT work=%ld attempts=%ld speed=%f coins=%d", 
                       &work_id, &attempts, &speed_mhs, &coins_count);
                
                if (coins_count == 0) {
                    printf("[Client %d] Work: %ld | Speed: %.2f MH/s | Coins: 0\n", 
                           client_fd, work_id, speed_mhs);
                } else {
                    printf("\n--- COIN RECEIVED FROM CLIENT %d ---\n", client_fd);
                    
                    char *coin_ptr = strstr(line, "coins=");
                    if (coin_ptr) {
                        coin_ptr = strchr(coin_ptr, ' ');
                        if (coin_ptr) {
                            coin_ptr++; 
                            pthread_mutex_lock(&vault_lock);
                            
                            for (int i = 0; i < coins_count && *coin_ptr; ++i) {
                                char hex_coin[256];
                                int hex_pos = 0;
                                while (*coin_ptr && *coin_ptr != ' ' && hex_pos < 255) {
                                    hex_coin[hex_pos++] = *coin_ptr;
                                    coin_ptr++;
                                }
                                hex_coin[hex_pos] = '\0';
                                
                                if (hex_pos == 110) { 
                                    u32_t coin_data[14];
                                    u08_t *bytes = (u08_t *)coin_data;
                                    
                                    // Parse Hex
                                    for (int j = 0; j < 55; j++) {
                                        u08_t b1 = hex_coin[j*2];
                                        u08_t b2 = hex_coin[j*2 + 1];
                                        u08_t v1 = (b1 >= 'a') ? (b1 - 'a' + 10) : (b1 - '0');
                                        u08_t v2 = (b2 >= 'a') ? (b2 - 'a' + 10) : (b2 - '0');
                                        bytes[j ^ 3] = (v1 << 4) | v2;
                                    }
                                    bytes[54 ^ 3] = '\n';
                                    bytes[55 ^ 3] = 0x80;
                                    
                                    // --- Calculate Value for Display ---
                                    u32_t hash[5];
                                    sha1(coin_data, hash);
                                    
                                    // Count leading zeros (Logic copied from aad_vault.h)
                                    u32_t val = 0;
                                    for(val = 0u; val < 128u; val++)
                                        if((hash[1u + val / 32u] >> (31u - val % 32u)) % 2u != 0u)
                                            break;
                                    if(val > 99u) val = 99u;

                                    // --- Print exactly like Vault ---
                                    // Format: Vxx:CONTENT
                                    printf("V%02u:", val);
                                    for (int j = 0; j < 55; j++) {
                                        putchar(bytes[j ^ 3]);
                                    }
                                    // Note: coin usually ends with \n, so this completes the line
                                    
                                    // --- Save ---
                                    save_coin(coin_data);
                                }
                                if (*coin_ptr == ' ') coin_ptr++;
                            }
                            save_coin(NULL); // Flush
                            pthread_mutex_unlock(&vault_lock);
                        }
                    }
                    printf("-------------------------------------\n");
                }
            }
            line = strtok(NULL, "\n");
        }
    }

    close(client_fd);
    return NULL;
}

int main(int argc, char *argv[]) {
    int server_fd, new_sock;
    struct sockaddr_in address;
    int opt = 1;
    socklen_t addrlen = sizeof(address);

    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("socket");
        exit(EXIT_FAILURE);
    }

    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("bind");
        exit(EXIT_FAILURE);
    }

    if (listen(server_fd, 20) < 0) {
        perror("listen");
        exit(EXIT_FAILURE);
    }

    printf("Server listening on port %d\n", PORT);

    while (1) {
        new_sock = accept(server_fd, (struct sockaddr *)&address, &addrlen);
        if (new_sock < 0) {
            perror("accept");
            continue;
        }
        int *pclient = malloc(sizeof(int));
        *pclient = new_sock;
        pthread_t tid;
        pthread_create(&tid, NULL, handle_client, pclient);
        pthread_detach(tid);
    }

    pthread_mutex_lock(&vault_lock);
    save_coin(NULL);
    pthread_mutex_unlock(&vault_lock);
    
    close(server_fd);
    return 0;
}