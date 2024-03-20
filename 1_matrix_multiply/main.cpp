# include <iostream>
# include <cstdlib> // for rand() and srand()
# include <ctime> // for time()
# include "mkl.h"

# define M 777
# define N 787
# define K 747

// 初始化全0矩阵
double** init_matrix(int m, int n) {
    double** matrix = new double*[m];
    for(int i = 0; i < m; ++i)
        matrix[i] = new double[n];
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i][j] = 0;
        }
    }
    return matrix;
}


// 使用循环方式将两个矩阵相乘，并返回计算时间
double original(double **A, double **B, double **C) {
    clock_t start, end;
    start = clock();
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            C[i][j] = 0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    end = clock();
    return (double)(end - start) / CLOCKS_PER_SEC;
}

// 优化算法1：调整循环顺序
double adjustLoopOrder(double **A, double **B, double **C) {
    clock_t start, end;
    start = clock();
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < N; k++) {
            double r = A[i][k];
            for (int j = 0; j < K; j++) {
                C[i][j] += r * B[k][j];
            }
        }
    }
    end = clock();
    return (double)(end - start) / CLOCKS_PER_SEC;
}

// 优化算法2：循环展开（2，4，8，16）
double loopUnrolling2(double **A, double **B, double **C) {
    clock_t start, end;
    start = clock();
    int k;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            k = 0;
            while (k < N-1) {
                C[i][j] += A[i][k] * B[k][j] + A[i][k+1] * B[k+1][j];
                k += 2;
            }
            while (k < N) {
                C[i][j] += A[i][k] * B[k][j];
                k++;
            }
        }
    }
    end = clock();
    return (double)(end - start) / CLOCKS_PER_SEC;
}

double loopUnrolling4(double **A, double **B, double **C) {
    clock_t start, end;
    start = clock();
    int k;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            k = 0;
            while (k < N-3) {
                C[i][j] += A[i][k] * B[k][j] + A[i][k+1] * B[k+1][j] + A[i][k+2] * B[k+2][j] + A[i][k+3] * B[k+3][j];
                k += 4;
            }
            while (k < N) {
                C[i][j] += A[i][k] * B[k][j];
                k++;
            }
        }
    }
    end = clock();
    return (double)(end - start) / CLOCKS_PER_SEC;
}

double loopUnrolling8(double **A, double **B, double **C) {
    clock_t start, end;
    start = clock();
    int k;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            k = 0;
            while (k < N-7) {
                C[i][j] += A[i][k] * B[k][j] + A[i][k+1] * B[k+1][j] + A[i][k+2] * B[k+2][j] + A[i][k+3] * B[k+3][j] + A[i][k+4] * B[k+4][j] + A[i][k+5] * B[k+5][j] + A[i][k+6] * B[k+6][j] + A[i][k+7] * B[k+7][j];
                k += 8;
            }
            while (k < N) {
                C[i][j] += A[i][k] * B[k][j];
                k++;
            }
        }
    }
    end = clock();
    return (double)(end - start) / CLOCKS_PER_SEC;
}

double loopUnrolling16(double **A, double **B, double **C) {
    clock_t start, end;
    start = clock();
    int k;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            k = 0;
            while (k < N-15) {
                C[i][j] += A[i][k] * B[k][j] + A[i][k+1] * B[k+1][j] + A[i][k+2] * B[k+2][j] + A[i][k+3] * B[k+3][j] + A[i][k+4] * B[k+4][j] + A[i][k+5] * B[k+5][j] + A[i][k+6] * B[k+6][j] + A[i][k+7] * B[k+7][j] + A[i][k+8] * B[k+8][j] + A[i][k+9] * B[k+9][j] + A[i][k+10] * B[k+10][j] + A[i][k+11] * B[k+11][j] + A[i][k+12] * B[k+12][j] + A[i][k+13] * B[k+13][j] + A[i][k+14] * B[k+14][j] + A[i][k+15] * B[k+15][j];
                k += 16;
            }
            while (k < N) {
                C[i][j] += A[i][k] * B[k][j];
                k++;
            }
        }
    }
    end = clock();
    return (double)(end - start) / CLOCKS_PER_SEC;
}

// 优化算法3：使用Intel MKL库


// 主函数
int main() {
    // 初始化矩阵 A 和 B，使用随机双精度浮点数，与python规格一致
    double** A = new double*[M];
    for(int i = 0; i < M; ++i)
        A[i] = new double[N];

    double** B = new double*[N];
    for(int i = 0; i < N; ++i)
        B[i] = new double[K];

    // 使用数组
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = (double)rand() / RAND_MAX;
        }
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            B[i][j] = (double)rand() / RAND_MAX;
        }
    }

    // 初始化矩阵 C，使用0填充
    double** C = init_matrix(M, K);

    // 通过不同的方式计算矩阵相乘，并输出计算时间
    std::cout << "original: " << original(A, B, C) << "s" << std::endl;
    C = init_matrix(M, K);
    std::cout << "adjust the order of loop: " << adjustLoopOrder(A, B, C) << "s" << std::endl;
    C = init_matrix(M, K);
    std::cout << "loop unroll(2): " << loopUnrolling2(A, B, C) << "s" << std::endl;
    C = init_matrix(M, K);
    std::cout << "loop unroll(4): " << loopUnrolling4(A, B, C) << "s" << std::endl;
    C = init_matrix(M, K);
    std::cout << "loop unroll(8): " << loopUnrolling8(A, B, C) << "s" << std::endl;
    C = init_matrix(M, K);
    std::cout << "loop unroll(16): " << loopUnrolling16(A, B, C) << "s" << std::endl;

    // 释放内存
    for(int i = 0; i < M; ++i)
        delete [] A[i];
    delete [] A;

    for(int i = 0; i < N; ++i)
        delete [] B[i];
    delete [] B;
}
