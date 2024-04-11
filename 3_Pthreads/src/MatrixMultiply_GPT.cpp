#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>

#define MAX_THREADS 16

// Matrix dimensions
int n;
// Number of threads
int num_threads;

// Matrices
int **A, **B, **C;

// Function to generate random matrix
void generate_matrix(int **matrix) {
    for (int i = 0; i < n; i++) {
        matrix[i] = (int *)malloc(n * sizeof(int));
        for (int j = 0; j < n; j++) {
            matrix[i][j] = rand() % 100; // Random value between 0 and 99
        }
    }
}

// Function for each thread to compute its part of matrix multiplication
void *multiply(void *arg) {
    int thread_id = *(int *)arg;
    int start = thread_id * (n / num_threads);
    int end = (thread_id + 1) * (n / num_threads);

    for (int i = start; i < end; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = 0;
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    pthread_exit(NULL);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s matrix_size num_threads\n", argv[0]);
        return 1;
    }

    n = atoi(argv[1]);
    num_threads = atoi(argv[2]);

    if (n < 128 || n > 2048 || num_threads < 1 || num_threads > MAX_THREADS) {
        printf("Invalid matrix size or number of threads.\n");
        return 1;
    }

    // Allocate memory for matrices
    A = (int **)malloc(n * sizeof(int *));
    B = (int **)malloc(n * sizeof(int *));
    C = (int **)malloc(n * sizeof(int *));

    // Generate random matrices
    generate_matrix(A);
    generate_matrix(B);

    // Allocate memory for result matrix
    for (int i = 0; i < n; i++) {
        C[i] = (int *)malloc(n * sizeof(int));
    }

    // Measure time
    struct timeval start, end;
    gettimeofday(&start, NULL);

    // Create threads
    pthread_t threads[MAX_THREADS];
    int thread_ids[MAX_THREADS];
    for (int i = 0; i < num_threads; i++) {
        thread_ids[i] = i;
        pthread_create(&threads[i], NULL, multiply, &thread_ids[i]);
    }

    // Join threads
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    // Measure time
    gettimeofday(&end, NULL);
    double delta = ((end.tv_sec - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;

    // Print N, num_threads, time
    printf("%d %d %f\n", n, num_threads, delta);

    // Free memory
    for (int i = 0; i < n; i++) {
        free(A[i]);
        free(B[i]);
        free(C[i]);
    }
    free(A);
    free(B);
    free(C);

    return 0;
}
