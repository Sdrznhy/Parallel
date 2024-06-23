#include <chrono>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <random>
#include <vector>

using namespace std;
using namespace chrono;

// 检查 CUDA 错误
#define CUDA_CHECK(status)                                            \
    if (status != 0) {                                                \
        cerr << "CUDA Error: " << cudaGetErrorString(status) << endl; \
        exit(1);                                                      \
    }

// 检查 cuDNN 错误
#define CUDNN_CHECK(status)                                             \
    if (status != CUDNN_STATUS_SUCCESS) {                               \
        cerr << "cuDNN Error: " << cudnnGetErrorString(status) << endl; \
        exit(1);                                                        \
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

    cudaEvent_t start, stop;
    float duration_gpu = 0.0f;

    // 初始化随机种子
    mt19937 gen(42);
    uniform_real_distribution<> dis(0.0, 1.0);

    // 初始化输入图像和卷积核
    vector<float> input(width * height * depth);
    vector<float> kernel(ksize * ksize * depth * num_kernels);
    vector<vector<float>> output(num_kernels, vector<float>());

    for (auto& val : input) {
        val = dis(gen);
    }
    for (auto& val : kernel) {
        val = dis(gen);
    }

    // 初始化 cuDNN
    cudnnHandle_t cudnn;
    CUDNN_CHECK(cudnnCreate(&cudnn));

    // 设备端内存分配
    float *d_input, *d_kernel, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, input.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_kernel, kernel.size() * sizeof(float)));

    // 拷贝数据到设备
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, kernel.data(), kernel.size() * sizeof(float), cudaMemcpyHostToDevice));

    // 对每个步幅进行卷积操作
    for (int s = 0; s < 3; ++s) {
        int stride = strides[s];
        int out_width = (width - ksize) / stride + 1;
        int out_height = (height - ksize) / stride + 1;

        for (int k = 0; k < num_kernels; ++k) {
            output[k].resize(out_width * out_height * depth);
        }

        CUDA_CHECK(cudaMalloc(&d_output, out_width * out_height * depth * sizeof(float) * num_kernels));

        // 创建和设置卷积描述符
        cudnnTensorDescriptor_t input_descriptor;
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_descriptor));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_descriptor,
            CUDNN_TENSOR_NHWC,
            CUDNN_DATA_FLOAT,
            1, depth, height, width));

        cudnnTensorDescriptor_t output_descriptor;
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_descriptor));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_descriptor,
            CUDNN_TENSOR_NHWC,
            CUDNN_DATA_FLOAT,
            1, num_kernels, out_height, out_width));

        cudnnFilterDescriptor_t kernel_descriptor;
        CUDNN_CHECK(cudnnCreateFilterDescriptor(&kernel_descriptor));
        CUDNN_CHECK(cudnnSetFilter4dDescriptor(kernel_descriptor,
            CUDNN_DATA_FLOAT,
            CUDNN_TENSOR_NCHW,
            num_kernels, depth, ksize, ksize));

        cudnnConvolutionDescriptor_t convolution_descriptor;
        CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(convolution_descriptor,
            0, 0, stride, stride, 1, 1,
            CUDNN_CROSS_CORRELATION,
            CUDNN_DATA_FLOAT));

        // 获取前向卷积算法
        cudnnConvolutionFwdAlgoPerf_t convolution_algorithm;
        int returnedAlgoCount;
        CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(cudnn,
            input_descriptor,
            kernel_descriptor,
            convolution_descriptor,
            output_descriptor,
            1, // 请求返回的算法数量
            &returnedAlgoCount,
            &convolution_algorithm));

        // 获取前向卷积所需的工作空间大小
        size_t workspace_bytes = 0;
        CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
            input_descriptor,
            kernel_descriptor,
            convolution_descriptor,
            output_descriptor,
            convolution_algorithm.algo,
            &workspace_bytes));

        // 分配工作空间
        void* d_workspace = nullptr;
        CUDA_CHECK(cudaMalloc(&d_workspace, workspace_bytes));

        // 执行前向卷积
        float alpha = 1.0f, beta = 0.0f;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        // auto start_gpu = high_resolution_clock::now();
        CUDNN_CHECK(cudnnConvolutionForward(cudnn,
            &alpha,
            input_descriptor,
            d_input,
            kernel_descriptor,
            d_kernel,
            convolution_descriptor,
            convolution_algorithm.algo,
            d_workspace,
            workspace_bytes,
            &beta,
            output_descriptor,
            d_output));
        // auto end_gpu = high_resolution_clock::now();
        // auto duration_gpu = duration_cast<milliseconds>(end_gpu - start_gpu).count();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&duration_gpu, start, stop);

        // 拷贝结果回主机
        for (int k = 0; k < num_kernels; ++k) {
            CUDA_CHECK(cudaMemcpy(output[k].data(), d_output + k * out_width * out_height * depth,
                out_width * out_height * depth * sizeof(float), cudaMemcpyDeviceToHost));
        }

        // 打印结果
        // cout << "Stride: " << stride << endl;
        // cout << "Output size: " << out_width << " x " << out_height << endl;
        // cout << "GPU Time: " << duration_gpu << " ms" << endl;

        cout << width << " " << out_width << " " << stride << " " << duration_gpu << endl;

        // 清理工作空间
        CUDA_CHECK(cudaFree(d_workspace));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_descriptor));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_descriptor));
        CUDNN_CHECK(cudnnDestroyFilterDescriptor(kernel_descriptor));
        CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(convolution_descriptor));
    }

    // 清理设备内存
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_kernel));
    CUDA_CHECK(cudaFree(d_output));

    // 清理 cuDNN
    CUDNN_CHECK(cudnnDestroy(cudnn));

    return 0;
}
