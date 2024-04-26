/*parallel.h*/
// 构造基于Pthreads的并行for循环分解、分配、执行机制

#ifndef _PARALLEL_H
#define _PARALLEL_H

void parallel_for(
    int start, int end, int inc,
    void* (*functor)(int, void*),
    void* arg, int num_threads);

#endif
