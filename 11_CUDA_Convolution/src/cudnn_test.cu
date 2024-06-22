#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>

#define CUDNN_CHECK(status)                                                       \
    if (status != CUDNN_STATUS_SUCCESS) {                                         \
        std::cerr << "cuDNN Error: " << cudnnGetErrorString(status) << std::endl; \
        exit(1);                                                                  \
    }

int main()
{
    cudaError_t cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }

    cudnnHandle_t cudnn;
    CUDNN_CHECK(cudnnCreate(&cudnn));
    std::cout << "cuDNN successfully initialized." << std::endl;
    CUDNN_CHECK(cudnnDestroy(cudnn));
    return 0;
}
