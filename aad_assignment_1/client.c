// client.c
// Distributed mining client (Generic).
// Compile with EITHER scalar OR simd worker.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>

#define DEFAULT_SERVER "127.0.0.1"
#define DEFAULT_PORT 9000

#define MAX_COINS_PER_ROUND 1024
#define COIN_HEX_STRLEN (55 * 2 + 1) 

// Generic Mining Interface updated to accept custom_text
extern void run_mining_round(long work_id,
                             long *attempts_out,
                             int *coins_found_out,
                             double *mhs_out, 
                             char coins_out[][COIN_HEX_STRLEN],
                             const char *custom_text); // <--- NOVO ARGUMENTO

static int send_line(int sock, const char *s) {
    size_t L = strlen(s);
    ssize_t w = send(sock, s, L, 0);
    if (w != (ssize_t)L) return -1;
    if (s[L-1] != '\n') send(sock, "\n", 1, 0);
    return 0;
}

int main(int argc, char *argv[]) {
    const char *server_ip = DEFAULT_SERVER;
    int server_port = DEFAULT_PORT;
    const char *custom_text = NULL; // <--- Custom Text Default

    if (argc >= 2) server_ip = argv[1];
    if (argc >= 3) server_port = atoi(argv[2]);
    if (argc >= 4) custom_text = argv[3]; // <--- Ler da command line

    printf("Client connecting to %s:%d\n", server_ip, server_port);
    if (custom_text) printf("Mining with custom text: \"%s\"\n", custom_text);

    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) { perror("socket"); return 1; }

    struct sockaddr_in srv;
    srv.sin_family = AF_INET;
    srv.sin_port = htons(server_port);
    if (inet_pton(AF_INET, server_ip, &srv.sin_addr) <= 0) {
        fprintf(stderr, "Invalid server IP\n");
        return 1;
    }

    if (connect(sock, (struct sockaddr *)&srv, sizeof(srv)) < 0) {
        perror("connect");
        return 1;
    }

    char buf[8192];
    while (1) {
        if (send_line(sock, "GETWORK") != 0) break;

        ssize_t r = recv(sock, buf, sizeof(buf)-1, 0);
        if (r <= 0) break;
        buf[r] = '\0';

        long work_id = -1;
        if (sscanf(buf, "WORK %ld", &work_id) != 1) {
            sleep(1);
            continue;
        }

        long attempts = 0;
        int coins_found = 0;
        double mhs = 0.0;
        char coins[MAX_COINS_PER_ROUND][COIN_HEX_STRLEN];
        for(int i=0; i<MAX_COINS_PER_ROUND; i++) coins[i][0] = '\0';

        // Passamos o custom_text para o worker
        run_mining_round(work_id, &attempts, &coins_found, &mhs, coins, custom_text);

        char outmsg[131072];
        int pos = snprintf(outmsg, sizeof(outmsg), "RESULT work=%ld attempts=%ld speed=%.2f coins=%d",
                           work_id, attempts, mhs, coins_found);

        for (int i = 0; i < coins_found; ++i) {
            pos += snprintf(outmsg + pos, sizeof(outmsg) - pos, " %s", coins[i]);
            if (pos >= (int)sizeof(outmsg) - 200) break;
        }
        pos += snprintf(outmsg + pos, sizeof(outmsg) - pos, "\n");

        if (send(sock, outmsg, strlen(outmsg), 0) < 0) break;
    }

    close(sock);
    return 0;
}