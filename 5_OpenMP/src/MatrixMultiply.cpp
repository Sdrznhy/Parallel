// 使用OpenMP实现通用矩阵乘法

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <sys/time.h>

#define DYNAMIC 1

int chunkSize = 256;
int numThreads;
int size;

// 随机矩阵初始化
double** matrixGenerate(int row, int col)
{
    double** matrix = new double*[row];
    for (int i = 0; i < row; i++) {
        matrix[i] = new double[col];
        for (int j = 0; j < col; j++) {
            matrix[i][j] = rand() % 100;
        }
    }
    return matrix;
}

// OpenMP并行矩阵乘法
double** matrixMultiplyOpenMP(double** matrixA, double** matrixB, int rowA, int colA, int colB)
{
    double** matrixC = new double*[rowA];
    for (int i = 0; i < rowA; i++) {
        matrixC[i] = new double[colB];
    }

#if DYNAMIC
#pragma omp parallel for num_threads(numThreads) collapse(2) schedule(dynamic, chunkSize)
    for (int i = 0; i < rowA; i++) {
        for (int j = 0; j < colB; j++) {
            matrixC[i][j] = 0;
            for (int k = 0; k < colA; k++) {
                matrixC[i][j] += matrixA[i][k] * matrixB[k][j];
            }
        }
    }
#else

#pragma omp parallel for num_threads(numThreads)
    for (int i = 0; i < rowA; i++) {
        for (int j = 0; j < colB; j++) {
            matrixC[i][j] = 0;
            for (int k = 0; k < colA; k++) {
                matrixC[i][j] += matrixA[i][k] * matrixB[k][j];
            }
        }
    }
#endif

    return matrixC;
}

int main(int argc, char* argv[])
{
    size = argv[1] ? atoi(argv[1]) : 1000;
    numThreads = argv[2] ? atoi(argv[2]) : 4;

    //     if (numThreads == 1) {
    //         // 串行直接计算乘法
    //         double** matrixA = matrixGenerate(size, size);
    //         double** matrixB = matrixGenerate(size, size);

    //         struct timeval start_time, end_time;
    //         gettimeofday(&start_time, NULL);

    //         double** matrixC = new double*[size];
    //         for (int i = 0; i < size; i++) {
    //             matrixC[i] = new double[size];
    //             for (int j = 0; j < size; j++) {
    //                 matrixC[i][j] = 0;
    //                 for (int k = 0; k < size; k++) {
    //                     matrixC[i][j] += matrixA[i][k] * matrixB[k][j];
    //                 }
    //             }
    //         }

    //         gettimeofday(&end_time, NULL);

    //         double time = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_usec - start_time.tv_usec) / 1000000.0;

    // // std::cout << size << " " << numThreads << " " << time << std::endl;
    // #if DYNAMIC
    //         std::ofstream file("output/MM_dynamic.txt", std::ios::app);
    // #else
    //         std::ofstream file("output/MM_static.txt", std::ios::app);
    // #endif
    //         if (!file) {
    //             std::cerr << "File open error!" << std::endl;
    //             return 0;
    //         }
    //         file << size << " " << numThreads << " " << time << std::endl;
    //         file.close();
    //         return 0;
    //     }

    struct timeval start_time, end_time;

    // 初始化矩阵
    double** matrixA = matrixGenerate(size, size);
    double** matrixB = matrixGenerate(size, size);

    gettimeofday(&start_time, NULL);

    double** matrixC = matrixMultiplyOpenMP(matrixA, matrixB, size, size, size);

    gettimeofday(&end_time, NULL);
    double time = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_usec - start_time.tv_usec) / 1000000.0;

// std::cout << size << " " << numThreads << " " << time << std::endl;
// 将结果输出到文件
#if DYNAMIC
    std::ofstream file("output/MM_dynamic_" + std::to_string(chunkSize) + ".txt", std::ios::app);
#else
    std::ofstream file("output/MM_static.txt", std::ios::app);
#endif
    if (!file) {
        std::cerr << "File open error!" << std::endl;
        return 0;
    }
    file << size << " " << numThreads << " " << time << std::endl;
    file.close();

    for (int i = 0; i < size; i++) {
        delete[] matrixA[i];
        delete[] matrixB[i];
        delete[] matrixC[i];
    }
    delete[] matrixA;
    delete[] matrixB;
    delete[] matrixC;

    return 0;
}