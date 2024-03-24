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

本次实验使用若干中方法测试对于相同规模随机矩阵的计算速度，包括

- python三重嵌套循环
- python调用numpy库
- C++三层嵌套循环
- C++循环展开
- C++调整循环顺序
- C++向量化编译
- C++调用MKL库

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

### Python调用numpy

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

## 实验感想
