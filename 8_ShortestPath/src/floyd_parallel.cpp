/*
并行floyd算法
C++和OpenMP实现
*/
#include <fstream>
#include <iostream>
#include <omp.h>
#include <sstream>
#include <vector>

using namespace std;

#define DEBUG 0
#define INF 2147483647
#define SIZE 4096 // updated_flower.csv
string fileName = "updated_flower.csv";
// string fileName = "updated_mouse.csv";
string testFileName = "test_data.txt";

/*
从csv文件中读取数据
*/
void readData(double** graph)

{
    ifstream csvData(fileName, ios::in);
    if (!csvData.is_open()) {
        cout << "Error opening file" << endl;
        exit(1);
    } else {
        string line;
        istringstream sin;
        vector<string> tokens;
        string token;
        int pointA, pointB;
        double distance;

        // 读取csv标题行
        getline(csvData, line);

        // 读取csv数据
        while (getline(csvData, line)) {
            sin.clear();
            sin.str(line);
            token.clear();
            tokens.clear();
            while (getline(sin, token, ',')) {
                tokens.push_back(token);
            }
            // 转化为int型
            pointA = stod(tokens[0]);
            pointB = stod(tokens[1]);
            distance = stod(tokens[2]);
            // 写入graph邻接矩阵
            graph[pointA][pointB] = distance;
            graph[pointB][pointA] = distance;
        }
    }
}

/*
Floyd算法
*/
void floyd(double** graph, double** next, int threadNum)
{
    // 并行Floyd算法
    // #pragma omp parallel shared(graph) private(i, j)
    //     {
    for (int k = 0; k < SIZE; k++) {
#pragma omp parallel for num_threads(threadNum)
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                if (graph[i][k] != INF && graph[k][j] != INF && graph[i][k] + graph[k][j] < graph[i][j]) {
                    graph[i][j] = graph[i][k] + graph[k][j];
                    next[i][j] = k;
                }
            }
        }
    }
}
// }

/*
读取测试数据，输出其中包含的点对的最短路径到文件
*/
void testAndOutput(double** graph, double** next)
{
    // 读取测试数据
    // 共n行，每行包含两个整型（分别为两个邻接顶点的ID）
    ifstream testData(testFileName, ios::in);
    if (!testData.is_open()) {
        cout << "Error opening file" << endl;
        exit(1);
    } else {
        string line;
        istringstream sin;
        vector<string> tokens;
        string token;
        int pointA, pointB;
        string result;

        // 读取测试数据
        while (getline(testData, line)) {
            sin.clear();
            sin.str(line);
            token.clear();
            tokens.clear();
            while (getline(sin, token, ' ')) {
                tokens.push_back(token);
            }
            // 转化为int型
            pointA = stod(tokens[0]);
            pointB = stod(tokens[1]);
            // 输出最短路径到文件
            fstream output("output/result_parallel.txt", ios::app);
            if (!output.is_open()) {
                cout << "Error opening file" << endl;
                exit(1);
            } else {
                result = to_string(graph[pointA][pointB]);
                if (graph[pointA][pointB] == INF) {
                    result = "INF";
                }
                output << pointA << " " << pointB << " " << result << " ";
                // 输出路径
                if (graph[pointA][pointB] != INF) {
                    output << "Path: ";
                    output << pointA << "->";
                    int k = next[pointA][pointB];
                    while (k != -1) {
                        output << k << "->";
                        k = next[k][pointB];
                    }
                    output << pointB << endl;
                }
            }
        }
    }
}

int main()
{
    double** graph = new double*[SIZE];
    for (int i = 0; i < SIZE; i++) {
        graph[i] = new double[SIZE];
    }
    // 初始化邻接矩阵
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            if (i == j) {
                graph[i][j] = 0;
            } else {
                graph[i][j] = INF;
            }
        }
    }
    // 初始化next邻接矩阵
    double** next = new double*[SIZE];
    for (int i = 0; i < SIZE; i++) {
        next[i] = new double[SIZE];
        for (int j = 0; j < SIZE; j++) {
            next[i][j] = -1;
        }
    }
    // 读取数据
    readData(graph);
#if DEBUG
    floyd(graph, next, 1);
    // 测试数据
    testAndOutput(graph, next);
#else
    // Floyd算法
    double start, end;
    for (int i = 1; i <= 16; i *= 2) {
        start = omp_get_wtime();
        floyd(graph, next, i);
        end = omp_get_wtime();
        cout << "Thread number: " << i << " Time: " << end - start << endl;
    }
    cout << "Finish" << endl;
#endif
    // 释放内存
    for (int i = 0; i < SIZE; i++) {
        delete[] graph[i];
        delete[] next[i];
    }
    return 0;
}