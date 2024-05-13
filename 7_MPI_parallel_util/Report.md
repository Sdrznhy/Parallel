# 7-MPI并行应用

## 实验目的

使用MPI对快速傅里叶变换进行并行化。

阅读参考文献中的串行傅里叶变换代码(fft_serial.cpp)，并使用MPI对其进行并行化。

1. 并行化：使用MPI多进程对fft_serial.cpp进行并行化。为适应MPI的消息传递机制，可能需要对fft_serial代码进行一定调整。
2. 优化：使用MPI_Pack/MPI-Unpack或MPI_Type_create_struct对数据重组后进行消息传递。
3. 分析：
    1. 改变并行规模（进程数）及问题规模（N），分析程序的并行性能；
    2. 通过实验对比，分析数据打包对于并行程序性能的影响；
    3. 使用Valgrind massif工具集采集并分析并行程序的内存消耗。

## 实验过程和核心代码

原程序实现了一个被称为Cooley-Tukey的蝶形算法，实现了快速傅里叶变换（FFT）

原程序最底层的函数是`step`，考虑到FFT算法的递归和分治特性，将并行化主要集中在`step`函数内部实现可以更好地利用算法本身的结构，因为每个步骤相对独立且可并行化。

 因此，在外层函数中管理高层次的并行任务分配和通信协调，在`step`中实施具体的数据处理并行化，这样既利用了算法的自然结构，又保证了良好的可扩展性和性能，是一种不错的选择

同时，为了方便并行改写，把最外层的循环去掉了，改用命令行来实现不同规模的数组

`step`函数大致的结构如下

```C++
void step(int n, int mj, double a[], double b[], double c[],
    double d[], double w[], double sgn)
{
    // ... 声明和初始化变量

    for (j = 0; j < lj; j++) {
        // ...循环内操作
    }
    return;
}
```

要将该函数并行化处理，一种简单的方式是将for循环展开

在MPI并行中需要注意的点是线程间的通信，也就是数据的分发和回收

在原程序的循环中

```C++
for (j = 0; j < mj; j++) {
        // ...
        for (k = 0; k < mj; k++) {
            c[(jc + k) * 2 + 0] = a[(ja + k) * 2 + 0] + b[(jb + k) * 2 + 0];
            c[(jc + k) * 2 + 1] = a[(ja + k) * 2 + 1] + b[(jb + k) * 2 + 1];
            // ...
        }
    }
```

需要根据数组大小和循环数，为每个线程分配需要计算的数量和数组大小

```C++
int chunk = lj / size;
int remainder = lj % size;

int start = chunk * rank;
int end = start + chunk + (rank == size - 1 ? remainder : 0);
int local_n = end - start;
```

然后使用local数组来储存本地数据，最后使用`MPI_Gatherv`函数来收回数据到主进程

```C++
for (int i = 0; i < local_n * mj; i++) {
    local_c[i] = c[start * mj2 + i];
    local_d[i] = d[start * mj2 + i];
}

// Calculate send counts and displacements for each process
int* sendcounts = new int[size];
int* displs = new int[size];
for (int i = 0; i < size; i++) {
    sendcounts[i] = chunk * mj + (i == size - 1 ? remainder * mj : 0);
    displs[i] = i * chunk * mj;
}

// Use MPI_Gatherv to collect the results from all processes
MPI_Gatherv(local_c, local_n * mj, MPI_DOUBLE, c, sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
MPI_Gatherv(local_d, local_n * mj, MPI_DOUBLE, d, sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
```

## 实验结果

### 实验正确性

如下图，在随机数种子一致的情况下，可以和原程序得到一样的运行结果

<img src="./assets/image-20240513231411104.png" alt="image-20240513231411104" style="zoom:50%;" />

<img src="./assets/image-20240513231441749.png" alt="image-20240513231441749" style="zoom:50%;" />

### 实验数据

各个线程数下运行效率如下

| 线程数 | 1       | 2       | 4       | 8      | 16     |
| ------ | ------- | ------- | ------- | ------ | ------ |
| MFLOPS | 1210.36 | 2061.78 | 2755.32 | 2369.6 | 2743.1 |

- 由于CPU核心性能物理差距，MFLOPS并没有一直增加
- 并行程序并未经过完全的优化
- 数据规模小可能无法让多线程体现出优势

### 内存

原程序：

<img src="./assets/image-20240513233023332.png" alt="image-20240513233023332" style="zoom:50%;" />

并行程序：

<img src="./assets/image-20240513233316156.png" alt="image-20240513233316156" style="zoom:50%;" />

虽然不知道运行时的核心数，不过确实多占用了一些内存，对于性能释放应当有一些帮助

## 实验总结

在这次实验中，我对于快速傅里叶变换的实现有了一些了解，并且对于如何在较为复杂的情况下使用MPI改写程序使其并行化更加熟练了。同时了解了使用Valgrind massif工具来进行堆内存使用情况的可视化观察与分析。