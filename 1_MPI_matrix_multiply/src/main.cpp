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
    // 为了跑16个进程，需要在mpirun后加上--use-hwthread-cpus
    // mpic++ -o main main.cpp
    // mpirun --use-hwthread-cpus -np 4 ./main

    // 划定计算范围
    // atoi: ASCII to Integer
    int n = atoi(argv[1]);     // 矩阵大小
    int beginRow, endRow;      // 每个进程的计算范围
    double beginTime, endTime; // 计时

    int rank;       // 进程编号
    int processNum; // 进程数量

    // 初始化 MPI
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processNum);

    MPI_Status status;

    if (processNum == 1) // 进程数为1，非并行
    {
        double *matrixA = matrixGenerate(n, n);
        double *matrixB = matrixGenerate(n, n);
        double *matrixC = matrixGenerateZero(n, n);

        beginTime = MPI_Wtime();
        matrixMultiply(matrixA, matrixB, matrixC, n, n);
        endTime = MPI_Wtime();

        // Matrix_size | Number_of_Processes | Time
        std::cout << n << " " << processNum << " " << endTime - beginTime << std::endl;

        delete[] matrixA;
        delete[] matrixB;
        delete[] matrixC;
    }

    else // 并行，进程0负责收发和汇总，其他进程负责计算
    {
        int rowsPerProcess = n / (processNum - 1);

        if (rank == 0) // 进程0，主进程
        {
            double *matrixA = matrixGenerate(n, n);
            double *matrixB = matrixGenerate(n, n);
            double *matrixC = matrixGenerateZero(n, n);

            beginTime = MPI_Wtime();

            // 发送信息：A[beginRow:endRow] 和 B
            // beginRow = rowsPerProcess * (rank - 1)
            // endRow = rowsPerProcess * rank or n - 1
            for (int i = 0; i < processNum - 1; i++)
            {
                beginRow = rowsPerProcess * i;
                endRow = rowsPerProcess * (i + 1);
                if (i == processNum - 2)
                    endRow = n;

                MPI_Send(&beginRow, 1, MPI_INT, i + 1, 4, MPI_COMM_WORLD);
                MPI_Send(&matrixA[beginRow * n + 0], (endRow - beginRow) * n, MPI_DOUBLE, i + 1, 0, MPI_COMM_WORLD);
                MPI_Send(&matrixB[0 * n + 0], n * n, MPI_DOUBLE, i + 1, 1, MPI_COMM_WORLD);
            }

            // 接收信息：C[beginRow:endRow]
            for (int i = 0; i < processNum - 1; i++)
            {
                beginRow = rowsPerProcess * i;
                endRow = rowsPerProcess * (i + 1);
                if (i == processNum - 2)
                    endRow = n;

                MPI_Recv(&matrixC[beginRow * n + 0], (endRow - beginRow) * n, MPI_DOUBLE, i + 1, 2, MPI_COMM_WORLD, &status);
            }

            endTime = MPI_Wtime();
            // Matrix_size | Number_of_Processes | Time
            std::cout << n << " " << processNum << " " << endTime - beginTime << std::endl;
            // std::cout << "Size of matrix: " << n << std::endl;
            // std::cout << "Time: " << endTime - beginTime << std::endl;
            // std::cout << "Using process: " << processNum << std::endl;

            delete[] matrixA;
            delete[] matrixB;
            delete[] matrixC;
        }
        else // 其他进程
        {
            MPI_Recv(&beginRow, 1, MPI_INT, 0, 4, MPI_COMM_WORLD, &status);
            if (rank == processNum - 1)
                endRow = n;
            else
                endRow = rowsPerProcess * rank;

            double *matrixA = new double[(endRow - beginRow) * n];
            double *matrixB = new double[n * n];
            double *matrixC = new double[(endRow - beginRow) * n];

            MPI_Recv(&matrixA[0 * n + 0], (endRow - beginRow) * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
            MPI_Recv(&matrixB[0 * n + 0], n * n, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);

            matrixMultiply(matrixA, matrixB, matrixC, endRow - beginRow, n);

            // 发送信息：C[beginRow:endRow]
            MPI_Send(&matrixC[0 * 0 + 0], rowsPerProcess * n, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);

            delete[] matrixA;
            delete[] matrixB;
            delete[] matrixC;
        }
    }

    MPI_Finalize();
    return 0;
}
