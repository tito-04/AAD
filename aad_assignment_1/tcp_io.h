#ifndef TCP_IO_H
#define TCP_IO_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <errno.h>

#define IO_BUFFER_SIZE 16384

typedef struct {
    int sock_fd;
    char buffer[IO_BUFFER_SIZE];
    int len;        // Bytes currently in buffer
    int processed;  // Bytes already scanned
} tcp_reader_t;

// Initialize the reader
static inline void tcp_reader_init(tcp_reader_t *reader, int sock) {
    reader->sock_fd = sock;
    reader->len = 0;
    reader->processed = 0;
    memset(reader->buffer, 0, IO_BUFFER_SIZE);
}

// Returns: 1 if line read, 0 on disconnect, -1 on error
// 'out_line' will point to a null-terminated string within the reader's buffer.
// Do NOT free 'out_line'. It is valid until the next call.
static inline int tcp_read_line(tcp_reader_t *reader, char **out_line) {
    while (1) {
        // 1. Search for newline in the data we already have
        for (int i = 0; i < reader->len; i++) {
            if (reader->buffer[i] == '\n') {
                reader->buffer[i] = '\0'; // Null-terminate the line
                *out_line = reader->buffer;
                
                // Shift remaining data to the start of the buffer
                // This handles the case where we received "LINE1\nLINE2"
                int leftover_start = i + 1;
                int leftover_len = reader->len - leftover_start;
                
                // If we have leftover data, move it to front. 
                // If not, we just reset len to 0.
                if (leftover_len > 0) {
                    memmove(reader->buffer, reader->buffer + leftover_start, leftover_len);
                }
                reader->len = leftover_len;
                return 1; // Success
            }
        }

        // 2. If buffer is full and no newline, line is too long (Protocol Error)
        if (reader->len >= IO_BUFFER_SIZE - 1) {
            fprintf(stderr, "Error: Line too long, clearing buffer.\n");
            reader->len = 0; // Reset to recover
            return -1;
        }

        // 3. Read more data from network
        ssize_t received = recv(reader->sock_fd, 
                                reader->buffer + reader->len, 
                                IO_BUFFER_SIZE - reader->len - 1, 
                                0);

        if (received > 0) {
            reader->len += received;
            reader->buffer[reader->len] = '\0'; // Safety null
        } else if (received == 0) {
            return 0; // Disconnected
        } else {
            if (errno == EINTR) continue; // Interrupted by signal, retry
            return -1; // Error
        }
    }
}

#endif