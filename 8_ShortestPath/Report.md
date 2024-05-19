# 8-并行多源最短路径搜索

## 实验目的

使用OpenMP/Pthreads/MPI中的一种实现无向图上的多源最短路径搜索，并通过实验分析在不同进程数量、数据下该实现的性能

## 实验过程和核心代码

使用Floyd算法来实现本实验要求的的无向图多源最短路径搜索

Floyd算法大致可以分为以下几个步骤

1. 初始化邻接矩阵，对角线数值为0，其余为无穷大
2. 初始化一个与图矩阵规模相当的`next`矩阵，初始值为-1，`next[i][j]`的值`n`表示从i到j的最短路径上的第一个点为n
3. 读入数据
4. 遍历每一个点，以这个点为中转，若有两点间存在更小的距离，则更新两点间的距离，并将当前的中间点

具体代码如下：

```c++
// 初始化数组
// ...

for (int k = 0; k < SIZE; k++) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            if (graph[i][k] != INF && graph[k][j] != INF && graph[i][k] + graph[k][j] < graph[i][j]) {
                graph[i][j] = graph[i][k] + graph[k][j];
                next[i][j] = k;
            }
        }
    }
}

// 输出结果
// 释放资源
// ...
```

要将程序改为并行，由于外层循环必须串行进行，因此对第二层循环并行即可

```C++
for (int k = 0; k < SIZE; k++) {
#pragma omp parallel for num_threads(threadNum)
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            if (graph[i][k] != INF && graph[k][j] != INF && graph[i][k] + graph[k][j] < graph[i][j]) {
                graph[i][j] = graph[i][k] + graph[k][j];
                next[i][j] = k;
            }
        }
    }
}
```

完成计算后，可以这样获取最短路径和路径长度

```C++
result = to_string(graph[pointA][pointB]);
if (graph[pointA][pointB] == INF) {
    result = "INF";
}
output << pointA << " " << pointB << " " << result << " ";
// 输出路径
if (graph[pointA][pointB] != INF) {
    output << "Path: ";
    output << pointA << "->";
    int k = next[pointA][pointB];
    while (k != -1) {
        output << k << "->";
        k = next[k][pointB];
    }
    output << pointB << endl;
```

## 实验结果

### 程序正确性

并行与串行输出结果一致，可以确认并行程序结果是正确的

![image-20240519203438722](./assets/image-20240519203438722.png)

### 并行效率

分别计算了`flower.csv`的规模930个点的图以及2048和4096个点的图，结果如下

| 矩阵规模 | 1        | 2       | 4        | 8        | 16       |
| -------- | -------- | ------- | -------- | -------- | -------- |
| 930      | 0.803814 | 1.1477  | 0.800355 | 0.584498 | 0.331431 |
| 2048     | 7.10313  | 5.74989 | 3.42627  | 2.81628  | 1.82256  |
| 4096     | 54.431   | 31.4083 | 21.4249  | 16.9208  | 11.1464  |

根据数据绘制加速比如下：

![output/figure1.png](./assets/Figure_111.png)

可以看到，随着数据规模的增加，并行的加速比会逐渐提升，但在目前规模还没有达到理论极值

规模较小时，由于并行的线程开销，并行效率甚至可能不如串行

##  实验总结

这一次的实验相对于前面的实验较为简单，经过前面的学习，已经可以比较熟练将串行程序改写为并行的，实现程序效率的提升
