# 4-Pthreads并行求解及蒙特卡洛

## 实验目的

1. 使用Pthread编写多线程程序，求解一元二次方程组的根，结合数据及任务之间的依赖关系，及实验计时，分析其性能。

    一元二次方程为包含一个未知项且未知项最高次数为2的方程，常写作$ax^2+bx+c=0$，求根公式为：
   $$
   x_{1,2}=\frac{-b\pm \sqrt{b^2-4ac}}{2a},\space b^2-4ac\geq0
   $$

2. 基于Pthreads编写多线程程序，使用蒙特卡洛方法求圆周率$\pi$近似值。

   蒙特卡洛方法是一种基于随机采样的数值计算方法，对于估算圆周率，可以具体表达为

   - 在第一象限做一个半径为1的四分之一扇形和边长为1的正方形

     <img src="assets\Figure_10.png" alt="Figure_10" style="zoom:50%;" />

   - 随机生成正方形内的点，计算落在扇形内的点的数量和总数的比值。根据扇形的面积$S=\frac{1}{4}\pi r^2$和正方形的面积$S=r^2$，得到这个比值应当近似等于$\frac{\pi}{4}$，由此估计$\pi$的值

     <img src="assets\Figure_9.png" alt="Figure_9" style="zoom:50%;" />

## 实验过程和核心代码

 ### 求解一元二次方程

代码位于`src/Fomula.cpp`

根据求根公式，
$$
x_{1,2}=\frac{-b\pm \sqrt{b^2-4ac}}{2a},\space b^2-4ac\geq0
$$
采用并行方法可以使用两个线程分别计算$\sqrt{b^2-4ac}$和$2a$，然后等到这两个线程完成计算后，再使用两个不同的线程分别计算两个根的值。可以使用互斥锁和条件变量来确保前两个线程完成计算后，后两个进程才开始计算

`sqrtDelta`线程

```c++
void* sqrtDelta(void* arg)
{
    sqrt_delta = sqrt(b * b - 4 * a * c);
    // 通知X1和X2线程
    pthread_mutex_lock(&mutex);
    count++;
    pthread_cond_broadcast(&cond1);
    pthread_mutex_unlock(&mutex);
    return NULL;
}
```

`twoTimesA`线程

```c++
void* twoTimesA(void* arg)
{
    two_a = 2 * a;
    // 通知X1和X2线程
    pthread_mutex_lock(&mutex);
    count++;
    pthread_cond_broadcast(&cond1);
    pthread_mutex_unlock(&mutex);
    return NULL;
}
```

`X1(X2)`线程

```c++
void* X1(void* arg)
{
    // 等待two_a和sqrt_delta计算完成
    pthread_mutex_lock(&mutex);
    // 确保两个线程都已经计算完成
    while (count < 2) {
        pthread_cond_wait(&cond1, &mutex);
    }
    x1 = (-b + sqrt_delta) / two_a;
    pthread_mutex_unlock(&mutex);
    return NULL;
}
```

然后在主程序中初始化四个线程并回收即可完成计算

在主程序中，增加了一个变量用于储存串行方法的计算结果，用于验证并行结果的正确性

```c++
// 串行计算过程
// ...

double x11 = x1, x22 = x2;

// 并行计算过程
// ...

bool flag = (x11 == x1) && (x22 == x2);
```

### 蒙特卡洛方法估算圆周率

代码位于`src/MonteCarlo.cpp`

首先生成正方形内部的随机点

```c++
double rand_double()
{
    return rand() / (double)RAND_MAX;
}

void init()
{
    x = new double[N];
    y = new double[N];
    for (int i = 0; i < N; i++) {
        x[i] = rand_double();
        y[i] = rand_double();
    }
    M = 0;
}
```

然后，需要在主程序中将这些点分配给各个线程来计算判断是否位于扇形内部，并在完成计算后将线程的本地计数值加到全局值上，在这一过程中使用互斥锁保证总值不会被错误写，线程函数如下

```c++
void* MonteCarlo(void* arg)
{
    long pNum = (long)arg;
    int start = pNum * num_per_process;
    int end = (pNum + 1) * num_per_process;
    int localM = 0;
    for (int i = start; i < end; i++) {
        if (x[i] * x[i] + y[i] * y[i] <= 1) {
            localM++;
        }
    }
    pthread_mutex_lock(&mutex);
    M += localM;
    pthread_mutex_unlock(&mutex);

    return NULL;
}
```

最后根据计数值计算$\pi$的估计值

```c++
int N; // 生成的点的个数
int M; // 落在圆内的点的个数

// 计算过程
// ...

pi = 4 * M / (double)N;
```

## 实验结果

### 求解方程

程序的输出结果如下

```
Serial: 1e-06
Parallel: 0.011995
Flag: 1
```

显然，求解一元二次方程是一个非常轻量的任务，并行无法为其带来显著的效率提升，甚至反而会降低计算效率。

不过，可以通过计算结果验证并行程序确实得到了正确的结果，说明在类似的场景下，如果计算的规模较大，理论上可以通过并行程序获得理想的性能提升

### 估算圆周率

实验要求中随机点数的上限为65535，然而在我的设备上不足以体现并行带来的性能提升，因此我把上限加了一点，结果显示，随着点数的增多，估测值的误差是有减小的趋势的

<img src="assets\Figure_11.png" alt="Figure_11" style="zoom: 67%;" />

随着线程数的增加，运行估算程序的加速比计算作图如下

<img src="assets\Figure_8.png" alt="Figure_8" style="zoom:50%;" />

可以看出，随著数据规模的增大，更多的线程逐渐体现出了更大的性能优势。

### 结果分析

在并行计算中，当问题的规模增加时，通常可以看到并行效率的提高。这是因为更大的问题规模意味着更多的计算工作，这些工作可以在多个处理器或线程之间分配，从而更好地利用并行性。

然而，当问题规模较小，或者并行开销较大时，串行计算可能反而会更快，这点在方程的计算和蒙特卡洛方法的小规模部分都有体现。此外，当线程数量超过处理器核心数量时，线程切换的开销可能会降低并行效率。

上述现象体现在图片上，就是上图所示的样子，多线程的加速比有一个逐渐反超的过程。

## 实验感想

通过这次实验，我对并行计算和多线程编程有了更深入的理解，了解了如何使用互斥锁和条件变量来同步线程。

在求解一元二次方程的部分，我发现并行计算并不总是能提高效率。当问题规模较小，或者并行开销（如线程创建和同步）较大时，串行计算可能会更快。在使用蒙特卡洛方法估算圆周率的部分，随着问题规模的增加，多线程程序的性能优势逐渐显现出来。这让我更加深刻地理解了并行计算的重要性。

同时，在对实验数据进行分析的过程中，我对`pandas`等数据分析库有了更多的了解，使用起来也更加得心应手了。