#include <iostream>
#include <stdlib.h>
#include <ctime>
#include <pthread.h>

int N;
int tnum;
int sum = 0;
int numsPerThread;
clock_t start, end;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

struct arraySumArgs
{
    int *A;
    int threadID;
    int localSum;
};

int *arrayGen(int n)
{
    int *array = new int[n];
    for (int i = 0; i < n; i++)
    {
        array[i] = rand() % 100;
    }
    return array;
}

void *arraySum(void *args)
{
    arraySumArgs *arg = (arraySumArgs *)args;
    int *A = arg->A;
    int threadID = arg->threadID;
    arg->localSum = 0;

    int start = threadID * numsPerThread;
    int end = threadID == tnum - 1 ? N : (threadID + 1) * numsPerThread;

    for (int i = start; i < end; i++)
    {
        arg->localSum += A[i];
    }

    pthread_exit(NULL);
}

int main(int argc, char *argv[])
{
    srand(12345);
    N = atoi(argv[1]) * 1000000;
    tnum = atoi(argv[2]);
    numsPerThread = N / tnum;

    int *A = arrayGen(N);

    pthread_t *threads = new pthread_t[tnum];
    arraySumArgs *args = new arraySumArgs[tnum];

    for (int i = 0; i < tnum; i++)
    {
        args[i].A = A;
        args[i].threadID = i;
        pthread_create(&threads[i], NULL, arraySum, (void *)&args[i]);
    }

    start = clock();
    for (int i = 0; i < tnum; i++)
    {
        pthread_join(threads[i], NULL);
        sum += args[i].localSum;
    }
    end = clock();
    std::cout << tnum << " " << N / 1000000 << " " 
                << (double)(end - start) / CLOCKS_PER_SEC * 1000000 << std::endl;

    delete[] A;
    delete[] threads;
    delete[] args;

    return 0;
}
