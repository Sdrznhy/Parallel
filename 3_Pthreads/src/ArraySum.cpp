#include <ctime>
#include <iostream>
#include <pthread.h>
#include <stdlib.h>
#include <sys/time.h>

#define MAX_THREADS 16

int N, nThreads; // 数组规模（百万），线程数
int* A; // 数组指针
long long totalSum = 0; // 总和
pthread_mutex_t lock; // 互斥锁;

// 线程求和函数
void* sum(void* arg)
{
    long threadID = (long)arg;
    // std::cout << threadID << " ";
    long long localSum = 0; // 线程被分配的部分和
    int numPerThread = N * 1000000 / nThreads;
    int start = threadID * numPerThread;
    int end = (threadID + 1) * numPerThread;

    for (int i = start; i < end; i++) {
        localSum += A[i];
    }

    // 使用互斥锁保证和的正确性
    pthread_mutex_lock(&lock);
    totalSum += localSum;
    pthread_mutex_unlock(&lock);

    pthread_exit(NULL);
}

int main(int argc, char* argv[])
{
    N = atoi(argv[1]);
    nThreads = atoi(argv[2]);

    // 初始化随机数组A
    A = new int[N * 1000000];
    for (int i = 0; i < N * 1000000; i++) {
        A[i] = rand() % 100;
    }

    // 初始化互斥锁
    pthread_mutex_init(&lock, NULL);

    // 创建线程
    pthread_t threads[MAX_THREADS];
    int threadID[MAX_THREADS];

    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);

    // time_t start = clock();
    for (long i = 0; i < nThreads; i++) {
        threadID[i] = i;
        pthread_create(&threads[i], NULL, sum, (void*)i);
    }

    for (int i = 0; i < nThreads; i++) {
        pthread_join(threads[i], NULL);
    }
    gettimeofday(&end_time, NULL);
    // time_t end = clock();
    double time = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_usec - start_time.tv_usec) / 1000000.0;

    // std::cout << N << " " << nThreads << " " << (double)(end - start) / CLOCKS_PER_SEC << std::endl;
    std::cout << N << " " << nThreads << " " << time << std::endl;

    pthread_mutex_destroy(&lock);
    delete[] A;
    return 0;
}