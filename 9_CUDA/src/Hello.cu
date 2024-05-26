/*
创建n个线程块，每个线程块的维度为m×k，
每个线程均输出线程块编号、二维块内线程编号及Hello World！
（如，“Hello World from Thread (1, 2) in Block 10!”）。
主线程输出“Hello World from the host!”。
*/

#include <stdio.h>

__global__ void helloFromGPU(int m, int k)
{
    int block_id = blockIdx.x;
    int thread_id_x = threadIdx.x;
    int thread_id_y = threadIdx.y;

    printf("Hello World from Thread (%d, %d) in Block %d!\n", thread_id_x, thread_id_y, block_id);
}

int main()
{
    int n = 3, m = 2, k = 2;
    dim3 threadPerBlock(m, k);
    dim3 numberOfBlocks(n);
    helloFromGPU<<<numberOfBlocks, threadPerBlock>>>(m, k);
    cudaDeviceSynchronize();
    printf("Hello World from the host!\n");
    return 0;
}