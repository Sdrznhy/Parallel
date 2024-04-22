// 使用OpenMP实现通用矩阵乘法

#include <iostream>
#include <omp.h>
#include <cstdlib>
#include <sys/time.h>

int numThreads;
int size;

// 随机矩阵初始化
double* matrixGenerate(int row, int col) {
    double* matrix = new double[row * col];
    for (int i = 0; i < row * col; i++) {
        matrix[i] = rand() % 100;
    }
    return matrix;
}

// OpenMP并行矩阵乘法
double* matrixMultiplyOpenMP(double* matrixA, double* matrixB, int rowA, int colA, int colB) {
    double* matrixC = new double[rowA * colB];
    #pragma omp parallel for num_threads(numThreads)
    for (int i = 0; i < rowA; i++) {
        for (int j = 0; j < colB; j++) {
            matrixC[i * colB + j] = 0;
            for (int k = 0; k < colA; k++) {
                matrixC[i * colB + j] += matrixA[i * colA + k] * matrixB[k * colB + j];
            }
        }
    }
    return matrixC;
}

int main(int argc, char* argv[]) {
    size = argv[1] ? atoi(argv[1]) : 1000;
    numThreads = argv[2] ? atoi(argv[2]) : 4;

    struct timeval start_time, end_time;

    // 初始化矩阵
    double* matrixA = matrixGenerate(size, size);
    double* matrixB = matrixGenerate(size, size);
    
    gettimeofday(&start_time, NULL);

    double* matrixC = matrixMultiplyOpenMP(matrixA, matrixB, size, size, size);
    
    gettimeofday(&end_time, NULL);
    double time = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_usec - start_time.tv_usec) / 1000000.0;

    std::cout << size << " " << numThreads << " " << time << std::endl;

    delete[] matrixA;
    delete[] matrixB;
    delete[] matrixC;

    return 0;
}