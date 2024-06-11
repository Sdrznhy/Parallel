#include <cuda_runtime.h>
#include <iostream>

int main()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); // 获取第一个设备的属性

    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max block dimensions: "
              << prop.maxThreadsDim[0] << " x "
              << prop.maxThreadsDim[1] << " x "
              << prop.maxThreadsDim[2] << std::endl;
    std::cout << "Max grid dimensions: "
              << prop.maxGridSize[0] << " x "
              << prop.maxGridSize[1] << " x "
              << prop.maxGridSize[2] << std::endl;

    return 0;
}