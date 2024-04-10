# include <pthread.h>
# include <iostream>
# include <stdlib.h>
# include <ctime>

int N;
int tnum;
// double *A, *B, *C;

struct matrixMultiplyArgs
{
    double *A;
    double *B;
    double *C;
    int threadID;
} ;

// 生成随机矩阵
double *matrixGen(int row, int col)
{
    double *matrix = new double[row * col];
    for (int i = 0; i < row * col; i++)
    {
        matrix[i] = double(rand() / 100);
    }
    return matrix;
}

// 生成全0矩阵
double *matrixGenZero(int row, int col)
{
    double *matrix = new double[row * col];
    for (int i = 0; i < row * col; i++)
    {
        matrix[i] = 0;
    }
    return matrix;
}

// 矩阵转置
void matrixTranspose(double *matrix, int row, int col)
{
    double *temp = new double[row * col];
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            temp[j * row + i] = matrix[i * col + j];
        }
    }
    for (int i = 0; i < row * col; i++)
    {
        matrix[i] = temp[i];
    }
    delete[] temp;
}

// 矩阵相乘
void matrixMultiply(double *A, double *B, double *C, int m, int n, int k)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < k; j++)
        {
            for (int l = 0; l < n; l++)
            {
                C[i * k + j] += A[i * n + l] * B[l * k + j];
            }
        }
    }
}

// 矩阵相乘的线程函数
void *matrixMultiplyThread(void *arg)
{
    matrixMultiplyArgs *args = (matrixMultiplyArgs *)arg;
    
    int threadID = args->threadID;
    double *A = args->A;
    double *B = args->B;
    double *C = args->C;
    
    int beginRow, endRow, rowsPerThread;
    rowsPerThread = N / tnum;
    beginRow = threadID * rowsPerThread;
    endRow = (threadID == tnum - 1) ? N - 1 : beginRow + rowsPerThread - 1;

    matrixMultiply(A + beginRow * N, B, C + beginRow * N, endRow - beginRow, N, N);

    pthread_exit(NULL);
}


int main(int argc, char *argv[])
{
    // 声明变量
    clock_t start, end;
    pthread_t *threadID = NULL;
    tnum = strtol(argv[1], NULL, 10);
    N = strtol(argv[2], NULL, 10);

    double *A = matrixGen(N, N);
    double *B = matrixGen(N, N);
    double *C = matrixGenZero(N, N);

    // int beginRow, endRow, rowsPerThread;

    threadID = (pthread_t *)malloc(tnum * sizeof(pthread_t));

    matrixMultiplyArgs args;
    args.A = A;
    args.B = B;
    args.C = C;

    // 计时开始
    start = clock();

    // 使用Pthreads进行矩阵相乘
    for (int i = 0; i < tnum; i++)
    {
        args.threadID = i;
        pthread_create(&threadID[i], NULL, matrixMultiplyThread, (void *)&args);
    }

    // // 计时开始
    // start = clock();

    // 回收线程
    for (int i = 0; i < tnum; i++)
    {
        pthread_join(threadID[i], NULL);
    }

    // 计时结束
    end = clock();

    // 输出时间
    std::cout << N << " " 
                << tnum << " " 
                << (double)(end - start) / CLOCKS_PER_SEC << " "
                << std::endl;

    // 释放内存
    free(A);
    free(B);
    free(C);
    free(threadID);

    return 0;
}