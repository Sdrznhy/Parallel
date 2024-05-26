/*
：随机生成n×n的矩阵A，对其进行转置得到A^T。
转置矩阵中第i行j列上的元素为原矩阵中j行i列元素，即A_ij^T=A_ji。
输出：矩阵A及其转置矩阵A^T，及计算所消耗的时间t。
要求：使用CUDA实现并行矩阵转置，分析不同线程块大小，矩阵规模，访存方式，任务/数据划分方式，对程序性能的影响。输入：整数n，其取值范围均为[512, 2048]
问题描述
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Kernel to transpose the matrix
__global__ void transpose(int* A, int* A_T, int n)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < n && y < n) {
        int pos = y * n + x;
        int trans_pos = x * n + y;
        A_T[trans_pos] = A[pos];
    }
}

// Function to print the matrix
void print_matrix(int* matrix, int n)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d ", matrix[i * n + j]);
        }
        printf("\n");
    }
}

int main(int argc, char** argv)
{

    int n = argc > 1 ? atoi(argv[1]) : 512;
    int block_size_x = argc > 2 ? atoi(argv[2]) : 32;
    int block_size_y = argc > 3 ? atoi(argv[3]) : 32;

    size_t size = n * n * sizeof(int);

    // Allocate host memory
    int* h_A = (int*)malloc(size);
    int* h_A_T = (int*)malloc(size);

    // Initialize the matrix with random values
    srand(time(NULL));
    for (int i = 0; i < n * n; i++) {
        h_A[i] = rand() % 100;
    }

    // Allocate device memory
    int *d_A, *d_A_T;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_A_T, size);

    // Copy the matrix to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    // Define block and grid size
    // dim3 block_size(16, 16);
    // dim3 block_size(32, 32);
    dim3 block_size(block_size_x, block_size_y);
    dim3 grid_size((n + block_size.x - 1) / block_size.x, (n + block_size.y - 1) / block_size.y);

    // Record the start time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Launch the kernel
    transpose<<<grid_size, block_size>>>(d_A, d_A_T, n);

    // Record the stop time
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate the elapsed time
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copy the transposed matrix back to host
    cudaMemcpy(h_A_T, d_A_T, size, cudaMemcpyDeviceToHost);

    // Print the matrices and the elapsed time
    // printf("Matrix A:\n");
    // print_matrix(h_A, n);
    // printf("\nMatrix A_T:\n");
    // print_matrix(h_A_T, n);
    // printf("\nMatrix size: %d x %d\n", n, n);
    // printf("\nElapsed time: %f ms\n", elapsed_time);
    printf("%d %d %f\n", n, block_size_x, elapsed_time);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_A_T);

    // Free host memory
    free(h_A);
    free(h_A_T);

    return 0;
}
