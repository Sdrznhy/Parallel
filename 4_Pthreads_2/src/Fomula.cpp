// 使用Pthreads进行一元二次方程求解

#include <cmath>
#include <iostream>
#include <pthread.h>
#include <stdlib.h>
#include <sys/time.h>

double a, b, c; // 一元二次方程的系数
double x1, x2; // 一元二次方程的解

// 并行求解一元二次方程的系数
double sqrt_delta; // 一元二次方程的delta的平方根
double two_a; // 一元二次方程的2a

// int con1_met = 0, con2_met = 0;
int count = 0;

// 条件变量
pthread_cond_t cond1 = PTHREAD_COND_INITIALIZER;
pthread_cond_t cond2 = PTHREAD_COND_INITIALIZER;

// 生成一元二次方程的系数
void generate()
{
    a = rand() % 100;
    b = rand() % 100;
    c = rand() % 100;
    if (b * b - 4 * a * c < 0) {
        generate();
    }
}

// 不使用并行的一元二次方程求解
void solve()
{
    double delta = b * b - 4 * a * c;
    double sqrt_delta = sqrt(delta);
    double denominator = 2 * a;
    x1 = (-b + sqrt_delta) / denominator;
    x2 = (-b - sqrt_delta) / denominator;
}

// 使用一个互斥锁保护count变量
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

// 使用并行方法求解的线程函数
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

void* X1(void* arg)
{
    // 等待two_a和sqrt_delta计算完成
    pthread_mutex_lock(&mutex);
    while (count < 2) {
        pthread_cond_wait(&cond1, &mutex);
        // pthread_cond_wait(&cond2, &mutex);
    }
    x1 = (-b + sqrt_delta) / two_a;
    pthread_mutex_unlock(&mutex);
    // std::cout << "x1" << std::endl;
    return NULL;
}

void* X2(void* arg)
{
    // 等待two_a和sqrt_delta计算完成
    pthread_mutex_lock(&mutex);
    while (count < 2) {
        pthread_cond_wait(&cond1, &mutex);
        // pthread_cond_wait(&cond2, &mutex);
    }
    x2 = (-b - sqrt_delta) / two_a;
    pthread_mutex_unlock(&mutex);
    // std::cout << "x2" << std::endl;
    return NULL;
}

int main()
{
    generate();

    // 计时器
    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);

    for (int i = 0; i < 100; i++) {
        solve();
    }
    double x11 = x1, x22 = x2;

    gettimeofday(&end_time, NULL);
    double time1 = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_usec - start_time.tv_usec) / 1000000.0;
    std::cout << "Serial: " << time1 << std::endl;

    pthread_t tid1, tid2, tid3, tid4;

    gettimeofday(&start_time, NULL);
    for (int i = 0; i < 100; i++) {
        pthread_create(&tid1, NULL, sqrtDelta, NULL);
        pthread_create(&tid2, NULL, twoTimesA, NULL);
        pthread_create(&tid3, NULL, X1, NULL);
        pthread_create(&tid4, NULL, X2, NULL);
        pthread_join(tid1, NULL);
        pthread_join(tid2, NULL);
        pthread_join(tid3, NULL);
        pthread_join(tid4, NULL);
        count = 0;
    }

    gettimeofday(&end_time, NULL);
    double time2 = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_usec - start_time.tv_usec) / 1000000.0;
    std::cout << "Parallel: " << time2 << std::endl;
    bool flag = (x11 == x1) && (x22 == x2);
    std::cout << "Flag: " << flag << std::endl;
}