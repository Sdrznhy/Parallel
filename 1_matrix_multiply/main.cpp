# include <iostream>
# include <cstdlib> // for rand() and srand()
# include <ctime> // for time()

# define M 777
# define N 787
# define K 747


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
    double** C = new double*[M];
    for(int i = 0; i < M; ++i)
        C[i] = new double[K];

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            C[i][j] = 0;
        }
    }

    // 通过不同的方式计算矩阵相乘，并输出计算时间
    std::cout << "original: " << original(A, B, C) << "s" << std::endl;

    // 释放内存
    for(int i = 0; i < M; ++i)
        delete [] A[i];
    delete [] A;

    for(int i = 0; i < N; ++i)
        delete [] B[i];
    delete [] B;
}
