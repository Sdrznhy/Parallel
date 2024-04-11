#include <pthread.h>
#include <stdlib.h>

#include <ctime>
#include <iostream>

#define MAX_THREADS 16

int N; // 矩阵规模
int nThreads; // 线程数

// double *A, *B, *C; // 矩阵指针
double **A, **B, **C;
clock_t start, end;

void matrixGenerate(double* matrix)
{
    for (int i = 0; i < N * N; i++) {
        matrix[i] = double(rand() / 100.0);
    }
}

void matrixGenerate(double** matrix)
{
    for (int i = 0; i < N; i++) {
        matrix[i] = new double[N];
        for (int j = 0; j < N; j++) {
            matrix[i][j] = double(rand() / 100.0);
        }
    }
}

void* multiply(void* arg)
{
    int threadID = *(int*)arg;
    int numPerThread = N / nThreads;
    int start = threadID * numPerThread;
    int end = (threadID + 1) * numPerThread;

    // for (int i = start; i < end; i++) {
    //     for (int j = 0; j < N; j++) {
    //         C[i * N + j] = 0;
    //         for (int k = 0; k < N; k++) {
    //             C[i * N + j] += A[i * N + k] * B[k * N + j];
    //         }
    //     }
    // }
    for (int i = start; i < end; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    pthread_exit(NULL);
}

int main(int argc, char* argv[])
{
    N = atoi(argv[1]);
    nThreads = atoi(argv[2]);

    // A = new double[N * N];
    // B = new double[N * N];
    // C = new double[N * N];
    A = new double*[N];
    B = new double*[N];
    C = new double*[N];
    matrixGenerate(A);
    matrixGenerate(B);
    for (int i = 0; i < N; i++) {
        C[i] = new double[N];
    }

    start = clock();
    pthread_t threads[MAX_THREADS];
    int threadIDs[MAX_THREADS];
    for (int i = 0; i < nThreads; i++) {
        threadIDs[i] = i;
        pthread_create(&threads[i], NULL, multiply, &threadIDs[i]);
    }

    for (int i = 0; i < nThreads; i++) {
        pthread_join(threads[i], NULL);
    }

    end = clock();

    std::cout << N << " " << nThreads << " " << (double)(end - start) / CLOCKS_PER_SEC << std::endl;

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}