#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// CUDA 错误检查
#define CUDA_CHECK_ERROR()                                                \
    {                                                                     \
        cudaError_t err = cudaGetLastError();                             \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    }

// CUDA 内核函数，用于执行矩阵乘法
// 每一个thread的任务是计算C的一个元素
__global__ void matrixMulCUDA(float* A, float* B, float* C, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    if (row < n && col < n) {
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

bool checkResult(float* A, float* B, int N)
{
    for (int i = 0; i < N * N; i++) {
        if (abs(A[i] - B[i]) > 1e-5) {
            return false;
        }
    }
    return true;
}

// 主机代码
int main(int argc, char* argv[])
{
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <matrix_size> <block_size_x> <block_size_y>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    int N = atoi(argv[1]); // 矩阵维度
    int blockSizeX = atoi(argv[2]);
    int blockSizeY = atoi(argv[3]);
    int size = N * N * sizeof(float);

    // 在主机上分配内存
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // 初始化输入矩阵
    srand(time(NULL));
    for (int i = 0; i < N * N; i++) {
        h_A[i] = rand() % 100;
        h_B[i] = rand() % 100;
    }

    // 在设备上分配内存
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // 将输入数据从主机传输到设备
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 定义CUDA网格和块结构
    dim3 threadsPerBlock(blockSizeX, blockSizeY);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // 启动CUDA内核
    clock_t start = clock();
    matrixMulCUDA<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    clock_t end = clock();
    CUDA_CHECK_ERROR();

    // 计算并打印运行时间
    double elapsed_time = (double)(end - start) / CLOCKS_PER_SEC;
    double gflops = 2.0 * N * N * N / (1e9 * elapsed_time);
    printf("Matrix size: %d, Block size: (%d, %d), Time: %f seconds, GFLOPS: %f\n", N, blockSizeX, blockSizeY, elapsed_time, gflops);

    // 将结果从设备传回主机
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 检查结果
    // float* reference = (float*)malloc(size);
    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++) {
    //         reference[i * N + j] = 0.0f;
    //         for (int k = 0; k < N; k++) {
    //             reference[i * N + j] += h_A[i * N + k] * h_B[k * N + j];
    //         }
    //     }
    // }

    // if (checkResult(h_C, reference, N)) {
    //     printf("Result is correct\n");
    // } else {
    //     printf("Result is incorrect\n");
    // }

    // 释放设备和主机内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
