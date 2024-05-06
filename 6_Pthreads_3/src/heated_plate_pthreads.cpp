#include "../include/parallel.h"
#include <fstream>
#include <iostream>
#include <math.h>
#include <omp.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define M 500
#define N 500
#define MAX_THREADS 22

double u[M][N];
double w[M][N];
double mean;
double diff;

// 需要规约的线程私有变量
double my_mean[MAX_THREADS] = { 0 };
double my_diff[MAX_THREADS] = { 0 };

// init, which corresponds with line 137-153 in heated_plate_openmp.c
void* init_i(int i, void* arg)
{
    w[i][0] = 100.0;
    w[i][N - 1] = 100.0;

    return NULL;
}

void* init_j(int j, void* arg)
{
    w[M - 1][j] = 100.0;
    w[0][j] = 0.0;

    return NULL;
}

// Average the boundary values, to come up with a reasonable
// initial value for the interior.
// line158-166 in heated_plate_openmp.c
void* calculate_mean_i(int i, void* arg)
{
    // 获取pthread的线程号
    long thread_num = *((long*)arg);
    my_mean[thread_num] += w[i][0] + w[i][N - 1];

    // 记得手动规约my_mean

    return NULL;
}

void* calculate_mean_j(int j, void* arg)
{
    // 获取pthread的线程号
    long thread_num = *((long*)arg);
    my_mean[thread_num] += w[M - 1][j] + w[0][j];

    // 记得手动规约my_mean

    return NULL;
}

// 初始化内部网格，对应line 179-187 in heated_plate_openmp.c
void* init_interior(int i, void* arg)
{
    for (int j = 1; j < N - 1; j++) {
        w[i][j] = mean;
    }

    return NULL;
}

// 将矩阵w的值复制到矩阵u，对应208-211
void* copy_w_to_u(int i, void* arg)
{
    for (int j = 0; j < N; j++) {
        u[i][j] = w[i][j];
    }

    return NULL;
}

// 生成w矩阵的新值，对应line 218-222
void* generate_w(int i, void* arg)
{
    for (int j = 1; j < N - 1; j++) {
        w[i][j] = (u[i - 1][j] + u[i + 1][j] + u[i][j - 1] + u[i][j + 1]) / 4.0;
        // if (j % 100 == 0) {
        //     printf("w[%d][%d] = %f\n", i, j, w[i][j]);
        // }
    }

    return NULL;
}

// 计算两次迭代之间的最大温度差，对应line 233-249
void* calculate_diff(int i, void* arg)
{
    long thread_num = *((long*)arg);
    // std::cout << thread_num << std::endl;
    for (int j = 1; j < N - 1; j++) {
        if (my_diff[thread_num] < fabs(w[i][j] - u[i][j])) {
            my_diff[thread_num] = fabs(w[i][j] - u[i][j]);
        }
    }
    // 最后需要手动规约my_diff

    return NULL;
}

int main(int argc, char* argv[])
{
    // 存储在两次迭代之间的最大温度差，被发配去当全局变量了
    // double diff;

    // 一个阈值，用于确定何时停止迭代。
    // 当两次迭代之间的最大温度差小于或等于这个阈值时，迭代停止。
    double epsilon = 0.001;

    // 迭代次数
    int iterations;
    int iterations_print;

    // 平均值，已经被发配去当全局变量
    mean = 0;

    // 线程私有变量，存储当前迭代的最大温度差，被发配去当全局变量了
    // double my_diff;

    // 存储当前迭代的温度，被发配去当全局变量了
    // double u[M][N];
    // double w[M][N];

    // int i, j;
    double wtime;

    printf("\n");
    printf("HEATED_PLATE\n");
    printf("  C++/Pthreads version\n");
    printf("  A program to solve for the steady state temperature distribution\n");
    printf("  over a rectangular plate.\n");
    printf("\n");
    printf("  Spatial grid of %d by %d points.\n", M, N);
    printf("  The iteration will be repeated until the change is <= %e\n", epsilon);
    printf("  Number of processors available = %d\n", omp_get_num_procs());
    printf("  Number of threads =              %d\n", omp_get_max_threads());

    mean = 0.0;

    // 初始化互斥锁
    // for (int i = 0; i < MAX_THREADS; i++) {
    //     pthread_mutex_init(&mutex[i], NULL);
    // }

    // init, which corresponds with line 137-166 in heated_plate_openmp.c
    parallel_for(1, M - 1, 1, init_i, NULL, MAX_THREADS);
    parallel_for(0, N, 1, init_j, NULL, MAX_THREADS);

    // Average the boundary values, to come up with a reasonable
    // initial value for the interior.
    parallel_for(1, M - 1, 1, calculate_mean_i, NULL, MAX_THREADS);

    parallel_for(0, N, 1, calculate_mean_j, NULL, MAX_THREADS);

    // 规约并计算平均值
    // 对应line 173-175 in heated_plate_openmp.c
    for (int i = 0; i < MAX_THREADS; i++) {
        mean += my_mean[i];
    }
    mean = mean / (2 * M + 2 * N - 4);
    printf("\n");
    printf("  MEAN = %f\n", mean);

    // 使用平均值初始化内部网格
    // 对应line 179-187 in heated_plate_openmp.c
    parallel_for(1, M - 1, 1, init_interior, NULL, MAX_THREADS);

    // line 188-199 in heated_plate_openmp.c
    // 照搬过来
    /*
      iterate until the  new solution W differs from the old solution U
      by no more than EPSILON.
    */
    iterations = 0;
    iterations_print = 1;
    printf("\n");
    printf(" Iteration  Change\n");
    printf("\n");
    wtime = omp_get_wtime();

    diff = epsilon;

    while (epsilon <= diff) {
        // #pragma omp parallel shared(u, w) private(i, j)

        // line 208-211
        parallel_for(0, M, 1, copy_w_to_u, NULL, MAX_THREADS);
        // std::cout << "hello" << std::endl;
        // line 218-222
        parallel_for(1, M - 1, 1, generate_w, NULL, MAX_THREADS);
        // std::cout << "hello" << std::endl;
        // line 231
        diff = 0.0;

        for (int i = 0; i < MAX_THREADS; i++) {
            my_diff[i] = 0.0;
        }
        // line 233-249
        parallel_for(1, M - 1, 1, calculate_diff, NULL, MAX_THREADS);
        // std::cout << "hello" << std::endl;
        // 得到最大温度差
        for (int i = 0; i < MAX_THREADS; i++) {
            if (diff < my_diff[i]) {
                diff = my_diff[i];
            }
        }

        iterations++;
        if (iterations == iterations_print) {
            double wtime_mid = omp_get_wtime() - wtime;
            printf("  %8d  %f\n", iterations, diff);
            std::ofstream file("output/heated_plate_pthreads.txt", std::ios::app);
            file << iterations << " " << wtime_mid << std::endl;
            iterations_print
                = 2 * iterations_print;
        }
    }

    wtime = omp_get_wtime() - wtime;

    printf("\n");
    printf("  %8d  %f\n", iterations, diff);
    printf("\n");
    printf("  Error tolerance achieved.\n");
    printf("  Wallclock time = %f\n", wtime);
    /*
      Terminate.
    */
    printf("\n");
    printf("HEATED_PLATE_OPENMP:\n");
    printf("  Normal end of execution.\n");

    return 0;
}