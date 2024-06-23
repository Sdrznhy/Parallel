#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

using namespace std;
using namespace chrono;

// im2col 转换的 CUDA kernel
__global__ void im2col_kernel(const float* data_im, float* data_col, int channels,
    int height, int width, int ksize, int stride, int height_col, int width_col)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < channels * height_col * width_col) {
        int w_out = index % width_col;
        int h_out = (index / width_col) % height_col;
        int c = index / (width_col * height_col);

        int w_in = w_out * stride;
        int h_in = h_out * stride;
        int offset = (c * height + h_in) * width + w_in;

        for (int i = 0; i < ksize; ++i) {
            for (int j = 0; j < ksize; ++j) {
                data_col[index * ksize * ksize + i * ksize + j] = data_im[offset + i * width + j];
            }
        }
    }
}

// GPU 矩阵乘法的 CUDA kernel
__global__ void matrixMulCUDA(float* C, float* A, float* B, int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k) {
        float value = 0;
        for (int e = 0; e < n; ++e) {
            value += A[row * n + e] * B[e * k + col];
        }
        C[row * k + col] = value;
    }
}

void cpu_conv2d(const vector<float>& input, const vector<float>& kernel, vector<float>& output, int width, int height, int depth, int ksize, int stride, int out_width, int out_height)
{
    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < out_height; ++y) {
            for (int x = 0; x < out_width; ++x) {
                float sum = 0.0f;
                for (int i = 0; i < ksize; ++i) {
                    for (int j = 0; j < ksize; ++j) {
                        int x_in = x * stride + i;
                        int y_in = y * stride + j;
                        if (x_in < width && y_in < height) {
                            sum += input[(z * height + y_in) * width + x_in] * kernel[(z * ksize + j) * ksize + i];
                        }
                    }
                }
                output[(z * out_height + y) * out_width + x] = sum;
            }
        }
    }
}

int main(int argc, char* argv[])
{
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <image_width> <image_height>" << endl;
        return 1;
    }

    int width = stoi(argv[1]);
    int height = stoi(argv[2]);
    int depth = 3; // 通道数量
    int ksize = 3; // 卷积核大小
    int strides[] = { 1, 2, 3 };
    int num_kernels = 3; // 卷积核个数

    // 初始化随机种子
    mt19937 gen(42);
    uniform_real_distribution<> dis(0.0, 1.0);

    // 初始化输入图像和卷积核
    vector<float> input(width * height * depth);
    vector<float> kernel(ksize * ksize * depth * num_kernels);
    vector<vector<float>> output_cpu(num_kernels, vector<float>());
    vector<vector<float>> output_gpu(num_kernels, vector<float>());

    for (auto& val : input) {
        val = dis(gen);
    }
    for (auto& val : kernel) {
        val = dis(gen);
    }

    // 对每个步幅进行卷积操作
    for (int s = 0; s < 3; ++s) {
        int stride = strides[s];
        int out_width = (width - ksize) / stride + 1;
        int out_height = (height - ksize) / stride + 1;

        for (int k = 0; k < num_kernels; ++k) {
            output_cpu[k].resize(out_width * out_height * depth);
            output_gpu[k].resize(out_width * out_height * depth);
        }

        // CPU 端 im2col 卷积
        // auto start_cpu = high_resolution_clock::now();
        // for (int k = 0; k < num_kernels; ++k) {
        //     cpu_conv2d(input, kernel, output_cpu[k], width, height, depth, ksize, stride, out_width, out_height);
        // }
        // auto end_cpu = high_resolution_clock::now();
        // auto duration_cpu = duration_cast<milliseconds>(end_cpu - start_cpu).count();

        // 设备端内存分配
        float *d_input, *d_kernel, *d_output, *d_col;
        cudaMalloc(&d_input, input.size() * sizeof(float));
        cudaMalloc(&d_kernel, kernel.size() * sizeof(float));
        cudaMalloc(&d_output, out_width * out_height * depth * sizeof(float));
        cudaMalloc(&d_col, depth * ksize * ksize * out_width * out_height * sizeof(float));

        // 数据拷贝到设备
        cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_kernel, kernel.data(), kernel.size() * sizeof(float), cudaMemcpyHostToDevice);

        // 定义 block 和 grid 大小
        int height_col = (height - ksize) / stride + 1;
        int width_col = (width - ksize) / stride + 1;
        int channels_col = depth * ksize * ksize;

        dim3 blockSize(16, 16);
        dim3 gridSize((channels_col * height_col * width_col + blockSize.x - 1) / blockSize.x);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        // GPU im2col 转换
        im2col_kernel<<<gridSize, blockSize>>>(d_input, d_col, depth, height, width, ksize, stride, height_col, width_col);
        cudaDeviceSynchronize();

        // 定义矩阵乘法的 block 和 grid 大小
        dim3 blockSizeMul(16, 16);
        dim3 gridSizeMul((out_width + blockSizeMul.x - 1) / blockSizeMul.x, (out_height + blockSizeMul.y - 1) / blockSizeMul.y);

        // GPU 矩阵乘法
        for (int k = 0; k < num_kernels; ++k) {
            matrixMulCUDA<<<gridSizeMul, blockSizeMul>>>(d_output, d_col, d_kernel, out_height * out_width, channels_col, 1);
            cudaMemcpy(output_gpu[k].data(), d_output, out_width * out_height * depth * sizeof(float), cudaMemcpyDeviceToHost);
        }
        cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float duration_gpu;
        cudaEventElapsedTime(&duration_gpu, start, stop);

        // 释放设备内存
        cudaFree(d_input);
        cudaFree(d_kernel);
        cudaFree(d_output);
        cudaFree(d_col);

        // 打印结果
        // cout << "Stride: " << stride << endl;
        // cout << "Output size: " << out_width << " x " << out_height << endl;
        // cout << "CPU Time: " << duration_cpu << " ms" << endl;
        // cout << "GPU Time: " << duration_gpu << " ms" << endl;
        cout << width << " " << out_width << " " << stride << " " << duration_gpu << endl;

        // 验证结果
        // for (int k = 0; k < num_kernels; ++k) {
        //     bool valid = true;
        //     for (int i = 0; i < output_cpu[k].size(); ++i) {
        //         if (abs(output_cpu[k][i] - output_gpu[k][i]) > 1e-5) {
        //             valid = false;
        //             break;
        //         }
        //     }
        //     if (valid) {
        //         cout << "Kernel " << k << ": Results are valid." << endl;
        //     } else {
        //         cout << "Kernel " << k << ": Results are invalid!" << endl;
        //     }
        // }
    }

    return 0;
}
