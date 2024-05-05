/*parallel.cpp*/
#include "../include/parallel.h"
#include <iostream>
#include <pthread.h>

struct FunctorArgs {
    int index;
    void* arg;
    int start;
    int end;
    int inc;
    void* (*functor)(int, void*);
};

// Pthread线程函数，用于执行functor
void* pthread_functor(void* arg)
{
    FunctorArgs* args = (FunctorArgs*)arg;
    int start = args->start;
    int end = args->end;
    long index = args->index;
    // std::cout << "Thread " << index << std::endl;

    for (int i = start; i < end; i += args->inc) {
        args->functor(i, &index);
    }

    delete args;
    return NULL;
}

// start, end, inc分别为循环的开始、结束及索引自增量
// functor为函数指针，定义了每次循环所执行的内容
// arg为functor的参数指针，给出了functor执行所需的数据
// num_threads为期望产生的线程数量
void parallel_for(
    int start, int end, int inc,
    void* (*functor)(int, void*),
    void* arg, int num_threads)
{
    // 计算每个线程的工作量
    int work = (end - start) / inc;
    int work_per_thread = work / num_threads;

    // 创建线程
    pthread_t threads[num_threads];
    for (int i = 0; i < num_threads; i++) {
        // 计算每个线程的起始和结束索引
        int thread_start = start + i * work_per_thread * inc;
        int thread_end = thread_start + work_per_thread * inc;
        if (i == num_threads - 1)
            thread_end = end;

        FunctorArgs* args = new FunctorArgs;
        args->index = i;
        args->arg = arg;
        args->functor = functor;
        args->start = thread_start;
        args->end = thread_end;
        args->inc = inc;
        pthread_create(&threads[i], NULL, pthread_functor, args);
    }

    // 等待线程结束
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
}