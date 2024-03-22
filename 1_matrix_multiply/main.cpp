#include <iostream>
#include <cstdlib> // for rand() and srand()
#include <ctime>   // for time()
#include "mkl.h"

#define M 1024
#define N 1024
#define K 1024

// 初始化全0矩阵
double **init_matrix(int m, int n)
{
    double **matrix = new double *[m];
    for (int i = 0; i < m; ++i)
        matrix[i] = new double[n];
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            matrix[i][j] = 0;
        }
    }
    return matrix;
}

// 使用循环方式将两个矩阵相乘，并返回计算时间
double original(double **A, double **B, double **C)
{
    clock_t start, end;
    start = clock();
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < K; j++)
        {
            C[i][j] = 0;
            for (int k = 0; k < N; k++)
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    end = clock();
    return (double)(end - start) / CLOCKS_PER_SEC;
}

// 优化算法1：调整循环顺序
double adjustLoopOrder(double **A, double **B, double **C)
{
    clock_t start, end;
    start = clock();
    for (int i = 0; i < M; i++)
    {
        for (int k = 0; k < N; k++)
        {
            double r = A[i][k];
            for (int j = 0; j < K; j++)
            {
                C[i][j] += r * B[k][j];
            }
        }
    }
    end = clock();
    return (double)(end - start) / CLOCKS_PER_SEC;
}

// 优化算法2：循环展开（2，4，8，16）
double loopUnrolling2(double **A, double **B, double **C)
{
    clock_t start, end;
    start = clock();
    int k;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < K; j++)
        {
            k = 0;
            while (k < N - 1)
            {
                C[i][j] += A[i][k] * B[k][j] + A[i][k + 1] * B[k + 1][j];
                k += 2;
            }
            while (k < N)
            {
                C[i][j] += A[i][k] * B[k][j];
                k++;
            }
        }
    }
    end = clock();
    return (double)(end - start) / CLOCKS_PER_SEC;
}

double loopUnrolling4(double **A, double **B, double **C)
{
    clock_t start, end;
    start = clock();
    int k;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < K; j++)
        {
            k = 0;
            while (k < N - 3)
            {
                C[i][j] += A[i][k] * B[k][j] + A[i][k + 1] * B[k + 1][j] + A[i][k + 2] * B[k + 2][j] + A[i][k + 3] * B[k + 3][j];
                k += 4;
            }
            while (k < N)
            {
                C[i][j] += A[i][k] * B[k][j];
                k++;
            }
        }
    }
    end = clock();
    return (double)(end - start) / CLOCKS_PER_SEC;
}

// 优化算法3：使用Intel MKL库
double mkl(double **A, double **B, double **C)
{
    clock_t start, end;
    start = clock();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, K, N, 1, *A, N, *B, K, 0, *C, K);
    end = clock();
    return (double)(end - start) / CLOCKS_PER_SEC;
}

void test(double **A, double **B, double **C)
{
    std::cout << "original: " << original(A, B, C) << "s" << std::endl;
    C = init_matrix(M, K);
    std::cout << "adjust the order of loop: " << adjustLoopOrder(A, B, C) << "s" << std::endl;
    C = init_matrix(M, K);
    std::cout << "loop unroll(2): " << loopUnrolling2(A, B, C) << "s" << std::endl;
    C = init_matrix(M, K);
    std::cout << "loop unroll(4): " << loopUnrolling4(A, B, C) << "s" << std::endl;
    C = init_matrix(M, K);
    std::cout << "mkl: " << mkl(A, B, C) << "s" << std::endl;
}

void execute()
{
    // 初始化矩阵 A 和 B，使用随机双精度浮点数，与python规格一致
    double **A = new double *[M];
    for (int i = 0; i < M; ++i)
        A[i] = new double[N];

    double **B = new double *[N];
    for (int i = 0; i < N; ++i)
        B[i] = new double[K];

    // 使用数组
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[i][j] = (double)rand() / RAND_MAX;
        }
    }
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < K; j++)
        {
            B[i][j] = (double)rand() / RAND_MAX;
        }
    }

    // 初始化矩阵 C，使用0填充
    double **C = init_matrix(M, K);

    // 通过不同的方式计算矩阵相乘，并输出计算时间
    test(A, B, C);

    // 释放内存
    for (int i = 0; i < M; ++i)
        delete[] A[i];
    delete[] A;

    for (int i = 0; i < N; ++i)
        delete[] B[i];
    delete[] B;
}

// 主函数
int main()
{
    execute();
}
