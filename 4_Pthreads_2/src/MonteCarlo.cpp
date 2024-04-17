// 使用蒙特卡洛方法估算PI
// 正方形内部有一个相切的圆，它们的面积之比是π/4。
// 现在，在这个正方形内部，随机产生1000000个点（即1000000个坐标对 (x, y)），计算它们与中心点的距离，从而判断是否落在圆的内部。
// 如果这些点均匀分布，那么圆内的点应该占到所有点的 π/4，因此将这个比值乘以4，就是π的值。

#include <cmath>
#include <iostream>
#include <pthread.h>
#include <stdlib.h>
#include <sys/time.h>

#define MAX_THREADS 16

long long N; // 生成的点的个数
long long M; // 落在圆内的点的个数
double pi; // 估算出的π值
int num_threads; // 线程的个数
int num_per_process; // 每个线程需要计算的组数
double *x, *y; // 随机数数组
unsigned int seed = time(NULL);

// 使用一个互斥锁保护count变量
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

// 计时
struct timeval start_time, end_time;

// unsigned int seed = time(NULL);
// unsigned int seed = 62442;
// 生成随机数
double rand_double()
{
    return (double)rand_r(&seed) / RAND_MAX;
}

// 初始化随机数数组
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

// 对照，串行方法计算PI
void serial()
{
    M = 0;
    for (int i = 0; i < N; i++) {
        if (x[i] * x[i] + y[i] * y[i] <= 1) {
            M++;
        }
    }
    pi = 4 * M / (double)N;
}

// 并行线程函数
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

int main(int argc, char** argv)
{
    // 从命令行读入参数
    N = atoi(argv[1]);
    num_threads = atoi(argv[2]);
    num_per_process = N / num_threads;
    // std::cout << N << num_threads << num_per_process << std::endl;

    init();

    // gettimeofday(&start_time, NULL);
    // serial();
    // gettimeofday(&end_time, NULL);
    // double time = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_usec - start_time.tv_usec) / 1000000.0;

    // double loss = fabs(pi - M_PI);
    // std::cout << "serial " << time << " " << loss << std::endl;

    // init();
    // M = 0;
    pthread_t threads[MAX_THREADS];
    // int threadID[MAX_THREADS];
    gettimeofday(&start_time, NULL);

    for (long i = 0; i < num_threads; i++) {
        // threadID[i] = i;
        pthread_create(&threads[i], NULL, MonteCarlo, (void*)i);
    }
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    pi = 4 * M / (double)N;
    gettimeofday(&end_time, NULL);
    double time = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_usec - start_time.tv_usec) / 1000000.0;

    double loss = fabs(pi - M_PI);
    // std::cout << N << " " << num_threads << " " << time << std::endl;
    std::cout << N << " " << loss << std::endl;
}