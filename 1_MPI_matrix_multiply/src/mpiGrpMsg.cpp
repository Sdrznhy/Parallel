#include <iostream>
#include <mpi/mpi.h>
#include <stdlib.h>
#include <ctime>

// 用于生成矩阵
double *matrixGenerate(int rows, int cols)
{
    double *matrix = new double[rows * cols];
    for (int i = 0; i < rows * cols; i++)
    {
        matrix[i] = (double)rand() / RAND_MAX;
    }
    return matrix;
}

// 初始化全为0的矩阵
double *matrixGenerateZero(int rows, int cols)
{
    double *matrix = new double[rows * cols];
    for (int i = 0; i < rows * cols; i++)
    {
        matrix[i] = 0;
    }
    return matrix;
}

// 矩阵相乘，使用三层循环嵌套方法
void matrixMultiply(double *matrixA, double *matrixB, double *matrixC, int row, int n)
{
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int k = 0; k < n; k++)
                matrixC[i * n + j] = matrixA[i * n + k] * matrixB[k * n + j];
        }
    }
}

int main(int argc, char **argv)
{
    // std::cout << "Hello, World!" << std::endl;
    // 声明需要用到的参数
    int n = atoi(argv[1]); // 矩阵大小
    // int beginRow, endRow;      // 每个进程的计算范围
    double beginTime, endTime; // 计时

    int rank;       // 进程编号
    int processNum; // 进程数量

    // 初始化 MPI
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processNum);

    double *matrixA = matrixGenerate(n, n);
    double *matrixB = matrixGenerate(n, n);
    double *matrixC = matrixGenerateZero(n, n);

    if (processNum == 1) // 进程数为1，非并行
    {
        // double *matrixA = matrixGenerate(n, n);
        // double *matrixB = matrixGenerate(n, n);
        // double *matrixC = matrixGenerateZero(n, n);

        beginTime = MPI_Wtime();
        matrixMultiply(matrixA, matrixB, matrixC, n, n);
        endTime = MPI_Wtime();

        // Matrix_size | Number_of_Processes | Time
        std::cout << n << " " << processNum << " " << endTime - beginTime << std::endl;

        delete[] matrixA;
        delete[] matrixB;
        delete[] matrixC;
    }

    else // 进程数大于1，使用并行计算
    {
        int rows = n / processNum;

        double *localA = new double[rows * n];
        double *localC = new double[rows * n];

        if (rank == 0)
        {
            // 生成矩阵
            // double *matrixA = matrixGenerate(n, n);
            // double *matrixB = matrixGenerate(n, n);
            // double *matrixC = matrixGenerateZero(n, n);

            // 使用MPI_Bcast和MPI_Scatter发送数据
            beginTime = MPI_Wtime();
            MPI_Scatter(matrixA, rows * n, MPI_DOUBLE, localA, rows * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(matrixB, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
        // 所有进程都需要计算
        matrixMultiply(localA, matrixB, localC, rows, n);

        // 使用MPI_Gather收集数据
        MPI_Gather(localC, rows * n, MPI_DOUBLE, matrixC, rows * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        delete localA;
        delete localC;

        if (rank == 0)
        {
            endTime = MPI_Wtime();
            std::cout << n << " " << processNum << " " << endTime - beginTime << std::endl;
            delete[] matrixA;
            delete[] matrixB;
            delete[] matrixC;
        }
    }

    MPI_Finalize();
}