# 3-Pthreads并行矩阵乘法与数组求和

## 实验目的

**并行矩阵乘法：**

使用Pthreads实现并行矩阵乘法，并通过实验分析其性能。

其中，矩阵规模范围为$[128-2048]$，线程数范围为$[1-16]$

**数组求和：**

使用Pthreads实现并行数组求和，并通过实验分析其性能。

其中，数组规模范围为$[1M-128M]$，线程数范围为$[1-16]$

## 实验过程和核心代码

### 矩阵乘法

代码位于`src/MatrixMultiply.cpp`中

使用Pthreads实现矩阵乘法相对简单，只需要为各个线程分配好任务即可，以下是详细的实现方法

Pthreads程序需要使用`pthreads.h`

```c++
#include <pthread.h>
```

声明全局变量

```c++
int N; // 矩阵规模
int nThreads; // 线程数
double **A, **B, **C; // 矩阵指针
```

声明并定义线程中的乘法函数

```c++
// void指针用于在创建线程时传递参数
void* multiply(void* arg)
{	
    // 计算线程计算范围
    int threadID = *(int*)arg; // 进程序号
    int numPerThread = N / nThreads; // 计算行数
    int start = threadID * numPerThread; // 起始行
    int end = (threadID + 1) * numPerThread; // 结束行

    for (int i = start; i < end; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    pthread_exit(NULL);
}
```

在`main`中创建对应数量线程，并在完成计算后回收

```c++
int main(int argc, char* argv[])
{
    // 初始化矩阵
    // ...
	
    // 创建线程并传递线程号
    pthread_t threads[MAX_THREADS];
    int threadIDs[MAX_THREADS];
    for (int i = 0; i < nThreads; i++) {
        threadIDs[i] = i;
        pthread_create(&threads[i], NULL, multiply, &threadIDs[i]);
    }
	
    // 等待并回收线程
    for (int i = 0; i < nThreads; i++) {
        pthread_join(threads[i], NULL);
    }

	// 输出结果和释放资源
    // ...
}
```

### 数组求和

代码位于`src/ArraySum.cpp`中

声明全局变量

```c++
int N, nThreads; // 数组规模（百万），线程数
int* A; // 数组指针
long long totalSum = 0; // 总和
pthread_mutex_t lock; // 互斥锁;
```

声明并定义每个线程中的求和函数

```c++
void* sum(void* arg)
{
    long threadID = (long)arg;
    long long localSum = 0; // 线程被分配的部分和
    int numPerThread = N * 1000000 / nThreads;
    int start = threadID * numPerThread;
    int end = (threadID + 1) * numPerThread;

    for (int i = start; i < end; i++) {
        localSum += A[i];
    }
	
    // 将部分和添加至总和
    // 使用互斥锁保证结果的正确性
    pthread_mutex_lock(&lock);
    totalSum += localSum;
    pthread_mutex_unlock(&lock);

    pthread_exit(NULL);
}
```

在`main`中创建对应数量线程，并在完成计算后回收

```c++
int main(int argc, char* argv[])
{
	// 初始化数组
    // ...

    // 初始化互斥锁
    pthread_mutex_init(&lock, NULL);

    // 创建线程
    pthread_t threads[MAX_THREADS];
    int threadID[MAX_THREADS];
    for (long i = 0; i < nThreads; i++) {
        threadID[i] = i;
        pthread_create(&threads[i], NULL, sum, (void*)i);
    }
	// 等待进程计算完成并回收进程
    for (int i = 0; i < nThreads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    // 输出结果和释放资源
    // ...
}
```

## 实验结果

CPU: Intel Core Ultra 7 155H

### 矩阵乘法

结果输出位于`output/Matrix.txt`

平均加速比结果由20轮结果计算平均用时后计算得到，作图如下：

![Figure_6](D:\Obsidian\Assignments\Parallel_programing\assets\Figure_6.png)

可以从图中看出，当矩阵规模增大时，加速比会逐渐逼近理论上的并行程序加速比极值

关于超出理论值的部分，我认为可能是CPU不同规模的核心性能不同导致的

### 数组求和

结果输出位于`output/Array.txt`

平均加速比结果由50轮结果计算平均用时后计算得到，作图如下：

![Figure_7](D:\Obsidian\Assignments\Parallel_programing\assets\Figure_7.png)

结果比矩阵乘法更能体现并行性能变化

- 当规模较小时，新建线程开销大于计算开销，多线程性能甚至会更差
- 随着数组规模增大，并行优势逐渐凸显，加速比逐渐增大

## 实验感想

pThread的程序实现虽然相较之前的MPI更为简单，但是还是会出现一些奇怪的问题，并且这些问题一般与计算机性能调度策略有关，通常笔记本插电结果就正常了。

最终还是基本了解掌握了使用pThread为程序并行加速的基本方法。
