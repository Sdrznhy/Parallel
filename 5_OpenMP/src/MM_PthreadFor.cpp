#include "../include/parallel.h"
#include <iostream>
#include <pthread.h>
#include <stdlib.h>
#include <sys/time.h>

double **A, **B, **C;
int N, num_threads;

// Pthread线程函数，按行实现矩阵相乘
void* matrix_multiply(int i, void* arg)
{
    for (int j = 0; j < N; j++) {
        for (int k = 0; k < N; k++) {
            C[i][j] += A[i][k] * B[k][j];
        }
    }
    return NULL;
}

// 初始化矩阵
void matrix_init()
{
    A = new double*[N];
    B = new double*[N];
    C = new double*[N];
    for (int i = 0; i < N; i++) {
        A[i] = new double[N];
        B[i] = new double[N];
        C[i] = new double[N];
        for (int j = 0; j < N; j++) {
            A[i][j] = rand() / (double)RAND_MAX;
            B[i][j] = rand() / (double)RAND_MAX;
            C[i][j] = 0.0;
        }
    }
}

// 主函数
int main(int argn, char** argv)
{
    // std::cout << "hello?" << std::endl;
    // 读取参数
    N = atoi(argv[1]) ? atoi(argv[1]) : 128;
    num_threads = atoi(argv[2]) ? atoi(argv[2]) : 4;

    // 初始化矩阵
    matrix_init();

    struct timeval start, end;
    gettimeofday(&start, NULL);

    // std::cout << "hello1?" << std::endl;
    // 计算并行矩阵乘法
    parallel_for(0, N, 1, matrix_multiply, NULL, num_threads);

    gettimeofday(&end, NULL);

    // std::cout << "hello2?" << std::endl;
    // 输出结果
    double time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;
    std::cout << N << " " << num_threads << " " << time << std::endl;

    // 释放内存
    for (int i = 0; i < N; i++) {
        delete[] A[i];
        delete[] B[i];
        delete[] C[i];
    }
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}