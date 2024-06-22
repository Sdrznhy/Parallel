#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

using namespace std;
using namespace chrono;

// GPU kernel for 2D convolution
__global__ void conv2d(float* input, float* kernel, float* output, int width, int height, int depth, int ksize, int stride, int out_width, int out_height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < out_width && y < out_height && z < depth) {
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
    int depth = 3; // Number of channels
    int ksize = 3; // Kernel size
    int strides[] = { 1, 2, 3 };
    int num_kernels = 3; // Number of kernels

    // Initialize random seed
    mt19937 gen(42);
    uniform_real_distribution<> dis(0.0, 1.0);

    // Initialize input image and kernels
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

    // For each stride
    for (int s = 0; s < 3; ++s) {
        int stride = strides[s];
        int out_width = (width - ksize) / stride + 1;
        int out_height = (height - ksize) / stride + 1;

        for (int k = 0; k < num_kernels; ++k) {
            output_cpu[k].resize(out_width * out_height * depth);
            output_gpu[k].resize(out_width * out_height * depth);
        }

        // CPU Convolution
        auto start_cpu = high_resolution_clock::now();
        for (int k = 0; k < num_kernels; ++k) {
            cpu_conv2d(input, kernel, output_cpu[k], width, height, depth, ksize, stride, out_width, out_height);
        }
        auto end_cpu = high_resolution_clock::now();
        auto duration_cpu = duration_cast<milliseconds>(end_cpu - start_cpu).count();

        // Allocate device memory
        float *d_input, *d_kernel, *d_output;
        cudaMalloc(&d_input, input.size() * sizeof(float));
        cudaMalloc(&d_kernel, kernel.size() * sizeof(float));
        cudaMalloc(&d_output, out_width * out_height * depth * sizeof(float));

        // Copy data to device
        cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_kernel, kernel.data(), kernel.size() * sizeof(float), cudaMemcpyHostToDevice);

        // Define block and grid sizes
        dim3 blockSize(16, 16, 1);
        dim3 gridSize((out_width + blockSize.x - 1) / blockSize.x, (out_height + blockSize.y - 1) / blockSize.y, depth);

        // GPU Convolution
        auto start_gpu = high_resolution_clock::now();
        for (int k = 0; k < num_kernels; ++k) {
            conv2d<<<gridSize, blockSize>>>(d_input, d_kernel, d_output, width, height, depth, ksize, stride, out_width, out_height);
            cudaMemcpy(output_gpu[k].data(), d_output, out_width * out_height * depth * sizeof(float), cudaMemcpyDeviceToHost);
        }
        cudaDeviceSynchronize();
        auto end_gpu = high_resolution_clock::now();
        auto duration_gpu = duration_cast<milliseconds>(end_gpu - start_gpu).count();

        // Free device memory
        cudaFree(d_input);
        cudaFree(d_kernel);
        cudaFree(d_output);

        // Print results
        cout << "Stride: " << stride << endl;
        cout << "Output size: " << out_width << " x " << out_height << endl;
        cout << "CPU Time: " << duration_cpu << " ms" << endl;
        cout << "GPU Time: " << duration_gpu << " ms" << endl;

        // Validate results
        for (int k = 0; k < num_kernels; ++k) {
            bool valid = true;
            for (int i = 0; i < output_cpu[k].size(); ++i) {
                if (abs(output_cpu[k][i] - output_gpu[k][i]) > 1e-5) {
                    valid = false;
                    break;
                }
            }
            if (valid) {
                cout << "Kernel " << k << ": Results are valid." << endl;
            } else {
                cout << "Kernel " << k << ": Results are invalid!" << endl;
            }
        }
    }

    return 0;
}
