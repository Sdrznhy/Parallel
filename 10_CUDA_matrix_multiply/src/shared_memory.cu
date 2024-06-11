#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BLOCK_SIZE 8
// CUDA 错误检查
#define CUDA_CHECK_ERROR()                                                \
    {                                                                     \
        cudaError_t err = cudaGetLastError();                             \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    }

// // CUDA 内核函数，用于执行矩阵乘法
// __global__ void matrixMulCUDA(float* A, float* B, float* C, int n)
// {
//     extern __shared__ float shared[];
//     float* shared_A = shared;
//     float* shared_B = shared + blockDim.x * blockDim.y;

//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.x * blockDim.x + threadIdx.x;
//     float sum = 0.0f;

//     for (int tileIdx = 0; tileIdx < (n + blockDim.x - 1) / blockDim.x; ++tileIdx) {
//         int tiledRow = row;
//         int tiledCol = tileIdx * blockDim.x + threadIdx.x;

//         if (tiledRow < n && tiledCol < n) {
//             shared_A[threadIdx.y * blockDim.x + threadIdx.x] = A[tiledRow * n + tiledCol];
//         } else {
//             shared_A[threadIdx.y * blockDim.x + threadIdx.x] = 0.0f;
//         }

//         tiledRow = tileIdx * blockDim.y + threadIdx.y;
//         tiledCol = col;

//         if (tiledRow < n && tiledCol < n) {
//             shared_B[threadIdx.y * blockDim.x + threadIdx.x] = B[tiledRow * n + tiledCol];
//         } else {
//             shared_B[threadIdx.y * blockDim.x + threadIdx.x] = 0.0f;
//         }

//         __syncthreads();

//         for (int k = 0; k < blockDim.x; ++k) {
//             sum += shared_A[threadIdx.y * blockDim.x + k] * shared_B[k * blockDim.x + threadIdx.x];
//         }

//         __syncthreads();
//     }

//     if (row < n && col < n) {
//         C[row * n + col] = sum;
//     }
// }

// 核函数的具体实现
__global__ void matmul_ShareMemory(float* M, float* N, float* P, int width)
{
    __shared__ float Mds[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Nds[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Col = bx * BLOCK_SIZE + tx;
    int Row = by * BLOCK_SIZE + ty;

    int Pervalue = 0;
    // 有多少个BLOCK_SIZE，每个循环计算一个块的大小
    for (int i = 0; i < width / BLOCK_SIZE; i++) {
        Mds[ty][tx] = M[Row * width + (i * BLOCK_SIZE + tx)];
        Nds[ty][tx] = N[Col + (i * BLOCK_SIZE + ty) * width];
        __syncthreads();

        // BLOCK_SIZE相乘
        for (int k = 0; k < BLOCK_SIZE; k++)
            Pervalue += Mds[ty][k] * Nds[k][tx];
        __syncthreads();
    }
    P[Row * width + Col] = Pervalue;
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

    // 在主机上分配内存=
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

    size_t sharedMemSize = 2 * blockSizeX * blockSizeY * sizeof(float);

    // 启动CUDA内核
    clock_t start = clock();
    // matrixMulCUDA<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_A, d_B, d_C, N);
    matmul_ShareMemory<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_A, d_B, d_C, N);
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
