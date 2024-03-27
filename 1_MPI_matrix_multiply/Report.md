# 1-MPI矩阵乘法

## 实验目的

使用MPI点对点通信方式实现并行通用矩阵乘法(MPI-v1)，并通过实验分析不同进程数量、矩阵规模时该实现的性能。在实验中，讨论两个优化的方向

1. 在内存有限的情况下，如何进行大规模矩阵乘法计算
2. 如何提高大规模稀疏矩阵乘法性能

对于每一次计算，要求：
> **输入**：$m,\ n,\ k$三个整数，每个整数的取值范围均为$[128, 2048]$
> 
> 随机生成$m\times n$的矩阵$A$及$n\times k$的矩阵$B$，并对这两个矩阵进行矩阵乘法运算，得到矩阵$C$
>
> **输出**：计算矩阵$C$所消耗的时间$t$

## 实验过程和核心代码

矩阵使用一维数组储存，三层嵌套循环计算乘法，这里略去了这些过程的代码

### 使用MPI点对点通信

下面的代码位于`./src/main.cpp`中

1个进程负责收发数据，其余进程负责计算并发回结果，实现如下

```C++
#include <mpi.h>
```

首先声明一些需要用到的参数

```C++
int n = atoi(argv[1]);     // 从主入口获取矩阵大小
int beginRow, endRow;      // 每个进程的计算范围
double beginTime, endTime; // 计时

int rank;       // 进程编号
int processNum; // 进程数量
```

初始化MPI

```C++
MPI_Init(NULL, NULL);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &processNum);

MPI_Status status;
```

对于进程数为1的情况，直接计算即可，下面讨论并行的情况

主进程需要将初始化的矩阵切割好并发送至各个进程

```C++
int rowsPerProcess = n / (processNum - 1); // 切割矩阵每块大小

if (rank == 0) // 进程0，主进程
{
    // 为其余进程发送切割好的数据
    // 若判断为最后一个进程，把剩下的全发过去
    for (int i = 0; i < processNum - 1; i++)
    {
        beginRow = rowsPerProcess * i;
        endRow = rowsPerProcess * (i + 1);
        if (i == processNum - 2)
            endRow = n;

        // 发送起始行，切割的A和全部的B
        MPI_Send(&beginRow, 1, MPI_INT, i + 1, 4, MPI_COMM_WORLD);
        MPI_Send(&matrixA[beginRow * n + 0], (endRow - beginRow) * n, 
                MPI_DOUBLE, i + 1, 0, MPI_COMM_WORLD);
        MPI_Send(&matrixB[0 * n + 0], n * n, 
                MPI_DOUBLE, i + 1, 1, MPI_COMM_WORLD);
    }

    // 接收其余进程计算好的部分的C并合并
    for (int i = 0; i < processNum - 1; i++)
    {
        beginRow = rowsPerProcess * i;
        endRow = rowsPerProcess * (i + 1);
        if (i == processNum - 2)
            endRow = n;

        MPI_Recv(&matrixC[beginRow * n + 0], (endRow - beginRow) * n, 
                MPI_DOUBLE, i + 1, 2, MPI_COMM_WORLD, &status);
    }
}
```

对于其余的进程，计算收到的A的切片与B相乘，再发回C的切片即可

```C++
if (rank != 0) // 其他进程
{   
    // 根据接收的起始行来初始化计算范围
    MPI_Recv(&beginRow, 1, MPI_INT, 0, 4, MPI_COMM_WORLD, &status);
    if (rank == processNum - 1)
        endRow = n;
    else
        endRow = rowsPerProcess * rank;

    // 接收A的切片和B
    MPI_Recv(&matrixA[0 * n + 0], (endRow - beginRow) * n, 
            MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&matrixB[0 * n + 0], n * n, 
            MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);

    // 计算乘法
    matrixMultiply(matrixA, matrixB, matrixC, endRow - beginRow, n);

    // 将完成计算的切片的C发回
    MPI_Send(&matrixC[0 * 0 + 0], rowsPerProcess * n, 
            MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
}
```

这样就实现了使用基础点对点通讯的并行乘法，这种方法显然是存在优化空间的

### 优化1：

## 实验结果

运行平台：`Intel Core i7-11800H`，有8个核心并且支持超线程

## 点对点通信

结果表明超线程并没有进一步提升运行效率，可能的解释如下

> ChatGPT:
>
> 在一些情况下，超线程可以提高并行计算的性能。例如，如果一个应用程序有很多独立的线程，这些线程经常需要等待I/O操作（如磁盘读写或网络通信），那么超线程可以在一个线程等待时，让CPU切换到另一个线程，从而提高CPU的利用率。
>
> 然而，在其他情况下，超线程可能不会提高并行计算的性能，甚至可能降低性能。例如，如果一个应用程序的线程经常需要共享数据，那么超线程可能会增加缓存冲突和内存访问延迟。此外，如果一个应用程序已经充分利用了CPU的所有核心，那么超线程可能会导致线程之间的竞争，降低性能。

## 实验感想