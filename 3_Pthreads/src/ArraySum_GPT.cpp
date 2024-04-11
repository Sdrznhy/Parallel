#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define MAX_THREADS 16

// 定义全局变量
int n, np;
int* A;
long long total_sum = 0;
pthread_mutex_t lock;

// 线程函数
void* thread_sum(void* arg)
{
    long thread_id = (long)arg;
    long long partial_sum = 0;
    int elements_per_thread = n * 1000000 / np;
    int start = thread_id * elements_per_thread;
    int end = start + elements_per_thread;

    // 计算本线程负责的部分和
    for (int i = start; i < end; ++i) {
        partial_sum += A[i];
    }

    // 使用互斥锁保护共享的总和变量
    pthread_mutex_lock(&lock);
    total_sum += partial_sum;
    pthread_mutex_unlock(&lock);

    return NULL;
}

int main(int argc, char* argv[])
{
    if (argc != 3) {
        printf("Usage: %s <n> <np>\n", argv[0]);
        return 1;
    }

    n = atoi(argv[1]);
    np = atoi(argv[2]);
    // if (n < 1 || n > 128 || np < 1 || np > MAX_THREADS) {
    //     printf("Invalid input values. n must be in [1, 128] and np must be in [1, 16].\n");
    //     return 1;
    // }

    // 初始化数组A，并随机填充
    A = (int*)malloc(n * 1000000 * sizeof(int));
    if (A == NULL) {
        printf("Memory allocation failed.\n");
        return 1;
    }
    for (int i = 0; i < n * 1000000; ++i) {
        A[i] = rand() % 100; // 随机生成0-99的整数
    }

    // 初始化互斥锁
    if (pthread_mutex_init(&lock, NULL) != 0) {
        printf("Mutex initialization failed.\n");
        return 1;
    }

    // 创建线程
    pthread_t threads[MAX_THREADS];
    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);

    for (long i = 0; i < np; ++i) {
        pthread_create(&threads[i], NULL, thread_sum, (void*)i);
    }

    // 等待线程结束
    for (int i = 0; i < np; ++i) {
        pthread_join(threads[i], NULL);
    }

    gettimeofday(&end_time, NULL);
    double time_taken = (end_time.tv_sec - start_time.tv_sec) * 1e6 + (end_time.tv_usec - start_time.tv_usec);
    time_taken /= 1e6;

    // printf("Total sum: %lld\n", total_sum);
    // printf("Time taken: %.6f seconds\n", time_taken);
    printf("%d %d %.6f\n", n, np, time_taken);

    // 释放资源
    free(A);
    pthread_mutex_destroy(&lock);

    return 0;
}
