#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

int thread_count = 4;
void *Hello(void *rank);

int main()
{
    // 循环变量
    long thread;
    // 进程的标号，数组
    pthread_t *thread_handles;
    // 分配空间
    thread_handles =(pthread_t *)malloc(thread_count * sizeof(pthread_t));

    // 绑定线程，开始分叉
    for (thread = 0; thread < thread_count; ++thread)
    // 将线程hello，和给hello的参数绑定到线程上
    // thread就是循环变量传递给每一个线程中执行的函数
        pthread_create(&thread_handles[thread], NULL,
        Hello, (void *)thread);

    printf("Hello from the main thread\n");
    // 结束线程
    for (thread = 0; thread < thread_count; ++thread)
        pthread_join(thread_handles[thread], NULL);

    free(thread_handles);    
    return 0;
}

void *Hello(void *rank)
{
    long my_rank = (long)rank;
    printf("Hello from thread %ld of %d\n", my_rank, thread_count);
    return NULL;
}