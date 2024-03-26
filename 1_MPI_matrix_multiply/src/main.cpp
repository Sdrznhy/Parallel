#include <iostream>
#include <mpi/mpi.h>
#include <stdlib.h>
#include <ctime>

// 用于生成矩阵
double **matrixGenerate(int rows, int cols)
{
    double **matrix = new double *[rows];
    for (int i = 0; i < rows; i++)
    {
        matrix[i] = new double[cols];
        for (int j = 0; j < cols; j++)
        {
            matrix[i][j] = (double)rand() / RAND_MAX;
        }
    }
    return matrix;
}

// 用于生成整数矩阵
int **matrixGenerateInt(int rows, int cols)
{
    int **matrix = new int *[rows];
    for (int i = 0; i < rows; i++)
    {
        matrix[i] = new int[cols];
        for (int j = 0; j < cols; j++)
        {
            matrix[i][j] = rand() % 10;
        }
    }
    return matrix;
}

int main(int argc, char **argv)
{   
    // only for square matrix

    // to run
    // mpic++ -o main main.cpp
    // mpirun -np 4 ./main
    // 好像还有别的编译指令，之后再加
    
    int rank;       // 进程编号
    int processNum; // 进程数量

    // 初始化 MPI
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processNum);
}
