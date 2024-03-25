# 并行程序设计实验-0

| 姓名   | 学号     |
| ------ | -------- |
| 刘思迪 | 21311156 |

## 实验目的

数学上，一个m×n的矩阵是一个由m行n列元素排列成的矩形阵列。矩阵是高等代数中常见的数学工具，也常见于统计分析等应用数学学科中。矩阵运算是数值分析领域中的重要问题。

通用矩阵乘法： $C=A\cdot B$，其中$A$为$m\times n$的矩阵，$B$为$n\times k$的矩阵，则其乘积$C$为$m\times k$的矩阵，$C$中第$i$行第$j$列元素可由矩阵$A$中第$i$行向量与矩阵$B$中第$j$列向量的内积给出，即：

$$C_(i,j)=\sum_{p=1}^n A_{i,p}\ B_{p,j}$$

根据以上定义用C/C++语言实现一个串行矩阵乘法，并通过对比实验分析不同方法的性能。

## 实验过程和核心代码

实验机器：`AMD EPYC 7763 64-Core Processor`其中的一个核

本次实验使用若干中方法测试对于相同规模随机矩阵的计算速度，包括

- python三重嵌套循环
- python调用`numpy`库
- C++三层嵌套循环
- C++循环展开
- C++调整循环顺序
- C++向量化编译
- C++调用MKL库

设定m, n, k的值均为1024，A, B矩阵填充为随机双精度浮点数

其中对于python程序，矩阵的初始化方法如下

```python
import numpy as np

m = 1024
n = 1024
k = 1024

A = np.random.rand(m, n)
B = np.random.rand(n, k)
```

对于C++程序，矩阵的初始化方法如下，以矩阵A为例

```C++
#define M 1024
#define N 1024
#define K 1024

double **A = new double *[M];
for (int i = 0; i < M; ++i)
    A[i] = new double[N];

for (int i = 0; i < M; i++)
{
    for (int j = 0; j < N; j++)
    {
        A[i][j] = (double)rand() / RAND_MAX;
    }
}
```

### Python三层循环

```python
for i in range(m):
    for j in range(k):
        for l in range(n):
            C[i][j] += A[i][l] * B[l][j]
```

### Python调用`numpy`

```python
C = np.dot(A, B)
```

### C++三层循环

```C++
for (int i = 0; i < M; i++)
{
    for (int j = 0; j < K; j++)
    {
        C[i][j] = 0;
        for (int k = 0; k < N; k++)
        {
            C[i][j] += A[i][k] * B[k][j];
        }
    }
}
```

### C++循环展开

```C++
// 将最内层循环展开，每次执行两轮
int k;
for (int i = 0; i < M; i++)
{
    for (int j = 0; j < K; j++)
    {
        k = 0;
        while (k < N - 1)
        {
            C[i][j] += A[i][k] * B[k][j] + A[i][k + 1] * B[k + 1][j];
            k += 2;
        }
        while (k < N)
        {
            C[i][j] += A[i][k] * B[k][j];
            k++;
        }
    }
}
```

### C++调整循环顺序

```C++
// 调整两个内层循环的嵌套顺序，提升访存效率
for (int i = 0; i < M; i++)
{
    for (int j = 0; j < K; j++)
    {
        C[i][j] = 0;
        for (int k = 0; k < N; k++)
        {
            C[i][j] += A[i][k] * B[k][j];
        }
    }
}
```

### C++使用向量化编译

更改编译指令为

```shell
g++ main.cpp -o ./bin/main  -O3 -fomit-frame-pointer  -ffast-math
```

作为对比，原编译指令为

```shell
g++ main.cpp -o ./bin/main
```

### C++调用MKL库

```C++
#include "mkl.h"

cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, K, N, 1, *A, N, *B, K, 0, *C, K);

```

## 实验结果

### Python程序

```text
m=1024, n=1024, k=1024
using numpy:
Time: 76.56ms
using original python:
Time: 768382.38ms
```

### C++程序

不使用向量化编译

```text
original: 4.45308s
adjust the order of loop: 3.38751s
loop unroll(2): 3.52078s
mkl: 0.055746s
```

使用向量化编译

```text
original: 1.1009s
adjust the order of loop: 0.221498s
loop unroll(2): 1.04828s
mkl: 0.054501s
```

### 结果汇总

> GFLOPS计算说明
>
> - 对于`numpy`和MKL，采用理论复杂度最低的Coppersmith–Winograd算法的$O(n^{2.376})$​计算
> - 对于其他方法，均采用$O(n^3)$

|          Function           |    用时     | GFLOPS | 相对加速比 |
| :-------------------------: | :---------: | :----: | :--------: |
|       Python三层循环        | 768382.38ms | 0.002  |  172.550   |
|           `numpy`           |   76.56ms   |  0.37  |   0.017    |
|    C++三层循环**(基准)**    |  4453.08ms  |  0.48  |     1      |
|       C++调整循环顺序       |  3387.51ms  |  0.63  |   0.760    |
|         C++循环展开         |  3520.78ms  |  0.61  |   0.790    |
|        C++调用MKL库         |   55.75ms   |  0.51  |   0.013    |
|   C++三层循环(向量化编译)   |  1100.9ms   |  1.95  |    0.25    |
| C++调整循环顺序(向量化编译) |  221.50ms   |  9.71  |   0.049    |
|   C++循环展开(向量化编译)   |  1048.28ms  |  2.05  |   0.235    |
|  C++调用MKL库(向量化编译)   |   54.50ms   |  0.52  |   0.012    |

### 结果分析

1. 关于python，python作为解释型语言，在拥有易于上手和阅读的优点的同时，解释器需要在程序运行时进行语法分析和错误检查，这会增加额外的开销。因此计算时效率极其低下
2. 关于`numpy`，`numpy`的核心计算模块使用C语言编写，实验中使用的`dot`矩阵点乘函数使用了优化的线性代数库，如 BLAS 和 LAPACK。这些库通常使用一种或多种高效的矩阵乘法算法，如 Strassen 算法或 Coppersmith–Winograd 算法。`numpy` 的 `dot` 函数会根据矩阵规模等因素自动选择最适合的算法。换句话说，`numpy`的成绩基本可以被视为标杆。
3. 关于调整循环顺序，这样做实现了按照主序访问B矩阵，减少了访存开销
4. 关于循环展开，这里选取的是对最内层循环进行展开，一次进行两步操作，这样子减少了矩阵A的加载次数。实验结果发现增加展开的列数对性能提升不大，并且很快达到阈值不在提升，实验结果中选取的是一次展开两列的结果
5. 加入向量化编译选项同样旨在减少访存次数，可以看到对于前面几种仅仅稍微优化的方法提升是十分巨大的
6. 关于MKL库对于矩阵乘法的优化，与`numpy`相同，都会动态使用了Strassen 算法和 Coppersmith-Winograd 算法等进行可能的优化，并且在内存访存上也进行了优化。同时，MKL在Intel处理器上还会利用其SIMD指令来进行向量化运算，由于平台原因，这一部分性能提升可能没有体现出来

## 实验感想

矩阵乘法的计算复杂度是 $O(n^3)$，这意味着当矩阵的大小增加时，所需的计算时间会急剧增加。在单核处理器上，由于不存在并行加速，这个问题更为突出。

在实验中，我观察到了内存访问模式改变对于计算性能的巨大提升，了解了一些先进优化算法的计算方式，如Strassen 算法和 Coppersmith-Winograd 算法，这些知识对于我未来的编程工作十分有用。
