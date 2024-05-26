# 9-CUDA矩阵转置

## 实验目的

- 由多个线程并行输出“Hello World！”
- 使用CUDA对矩阵进行并行转置。

## 实验过程和核心代码

### hello world

CUDA核函数

```C
__global__ void helloFromGPU(int m, int k)
{
    int block_id = blockIdx.x;
    int thread_id_x = threadIdx.x;
    int thread_id_y = threadIdx.y;

    printf("Hello World from Thread (%d, %d) in Block %d!\n", thread_id_x, thread_id_y, block_id);
}
```

主函数

```c
int main()
{
    int n = 3, m = 4, k = 5;
    dim3 threadPerBlock(m, k);
    dim3 numberOfBlocks(n);
    helloFromGPU<<<numberOfBlocks, threadPerBlock>>>(m, k);
    cudaDeviceSynchronize();
    printf("Hello World from the host!\n");
    return 0;
}
```

### 矩阵转置

CUDA核函数

```c
// 初始化矩阵
// ...

// 为显卡分配内存
int *d_A, *d_A_T;
cudaMalloc((void**)&d_A, size);
cudaMalloc((void**)&d_A_T, size);

// 拷贝矩阵到显存
cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

// 分配block, grid大小
dim3 block_size(block_size_x, block_size_y);
dim3 grid_size((n + block_size.x - 1) / block_size.x, (n + block_size.y - 1) / block_size.y);

// 启动核函数
transpose<<<grid_size, block_size>>>(d_A, d_A_T, n);

// 将数据拷贝回内存
cudaMemcpy(h_A_T, d_A_T, size, cudaMemcpyDeviceToHost);
```

## 实验结果

### hello world

![image-20240526222306911](./assets/image-20240526222306911.png)

### 矩阵转置

对于不同的矩阵大小和块大小运行结果如下

| 矩阵大小 (n) | 块大小 (block_size) | 运行时间 (ms) |
| ------------ | ------------------- | ------------- |
| 128          | 16                  | 0.753664      |
| 128          | 32                  | 0.635904      |
| 128          | 64                  | 0.004096      |
| 256          | 16                  | 0.859520      |
| 256          | 32                  | 0.735392      |
| 256          | 64                  | 0.003776      |
| 512          | 16                  | 0.846176      |
| 512          | 32                  | 0.294112      |
| 512          | 64                  | 0.003744      |
| 1024         | 16                  | 0.469088      |
| 1024         | 32                  | 0.690304      |
| 1024         | 64                  | 0.003776      |
| 2048         | 16                  | 0.594080      |
| 2048         | 32                  | 1.584096      |
| 2048         | 64                  | 0.003552      |
| 4096         | 16                  | 1.193792      |
| 4096         | 32                  | 2.411104      |
| 4096         | 64                  | 0.003360      |

结果分析：

可以观察到以下几点：

1. 块大小 (16)和(32)
   - 对于较小的矩阵（128x128 到 512x512），运行时间相对较短
   - 随着矩阵大小增加到2048x2048和4096x4096，运行时间显著增加
2. 块大小 (64)
   - 不论矩阵大小如何，运行时间均保持在极低水平，约为0.003到0.004毫秒。

可能的解释：

- 块大小为64的情况运行时间最短，这是因为较大的块大小可以更好地利用GPU的并行计算能力，减少线程之间的同步开销和数据传输的开销。
- 块大小为16和32的情况下，随着矩阵大小的增加，运行时间明显增加。这是因为较小的线程块数量较多，导致线程调度和数据传输的开销更高。
- 对于较大的矩阵，块大小为64能显著降低运行时间，说明在这个实验中，64x64的块大小能够更好地发挥GPU的性能。

当使用32的块大小时，时间甚至较16时有所提升，对此可能的解释如下：

1. 内存访问模式和对齐:
   - 块大小为32时，可能导致内存访问模式不如16和64那样优化。如果线程访问内存的模式未对齐，可能会导致内存访问效率降低，从而影响性能。
2. 线程调度与占用:
   - 块大小为32时，每个块只有一个warp，可能导致线程调度和资源利用不充分。较大的块大小（如64）能够更好地隐藏内存访问延迟，提高硬件资源利用率。
3. 栅栏同步开销:
   - 块大小为32时，可能需要更多的同步操作以确保数据一致性，这些同步操作会增加额外开销。

## 实验总结

这次试验是我熟悉了基本的CUDA编程，学会了如何使用CUDA编写一些简单的并行程序。