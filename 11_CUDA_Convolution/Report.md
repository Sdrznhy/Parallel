# 11-CUDA卷积

# 实验内容

## 1. 滑窗法实现CUDA并行卷积

使用CUDA实现二维图像的直接卷积（滑窗法）。在信号处理、图像处理和其他工程/科学领域，卷积是一种使用广泛的技术。在深度学习领域，卷积神经网络(CNN)这种模型架构就得名于这种技术。在本实验中，我们将在GPU上实现卷积操作，注意这里的卷积是指神经网络中的卷积操作，与信号处理领域中的卷积操作不同，它不需要对Filter进行翻转，不考虑bias。

下图展示了滑窗法实现的CUDA卷积，其中蓝色网格表示输入图像，红色网格表示输出图像，橙色网格展示了一个­$3\times 3$卷积核，卷积核中每个元素为对应位置像素的权重，该卷积核的输出值为像素值的加权和，输出位置位于橙色网格中心，即红色网格中的绿色元素。滑窗法移动该卷积核的中心，从而产生红色网格中的所有元素。

![image-20240611103022798](./assets/image-20240611103022798.png)

> **输入：**一张二维图像（$height \times width$）与一个卷积核（$3 \times 3$)。
>
> **问题描述：**用直接卷积的方式对输入二维图像进行卷积，通道数量（channel, depth）设置为3，卷积核个数为3，步幅（stride）分别设置为1/2/3，可能需要通过填充（padding）配合步幅（stride）完成卷积操作。注：实验的卷积操作不需要考虑bias (b)，bias设置为0。
>
> **输出**：卷积结果图像（$height -2 \times width -2$）及计算时间。
>
> **要求：**使用CUDA实现并行图像卷积，分析不同图像大小、访存方式、任务/数据划分方式、线程块大小等因素对程序性能的影响。

## 2. 使用im2col方法实现CUDA并行卷积

滑窗法使用$3 \times 3$的卷积核对$3 \times 3$窗口内的图像像素求加权和，此过程可以写做矩阵乘法形式$w^T \cdot x$，其中$w^T$为$1\times 9$的权重矩阵，$x$为$9\times 1$的像素值矩阵。将图像中每个需要进行卷积的窗口平铺为的$9 \times 1$矩阵（列向量)并进行拼接，可将卷积计算变为矩阵乘法，从而利用此前实现的并行矩阵乘法模块实现并行卷积。具体拼接方式见下图：

![image-20240611104534783](./assets/image-20240611104534783.png)

> 问题描述：用 im2col 方法对输入二维图像进行卷积。其他设置与任务 1（滑窗法并行卷积）相同。

## 3. 使用cuDNN方法实现CUDA并行卷积

NVIDIA cuDNN 是用于深度神经网络的 GPU 加速库。它强调性能、易用性和低内存开销。

> 要求：使用 cuDNN 提供的卷积方法进行卷积操作，记录其相应 Input的卷积时间，与自己实现的卷积操作进行比较。如果性能不如 cuDNN，用文字描述可能的改进方法。

# 实验过程和核心代码

实现卷积计算的过程大体一致，主要区别在于卷积核函数的不同，下面将其余部分与核函数分别展示

## 主要过程

### 变量初始化

```C++
// 声明变量
// ...

// 初始化随机数种
mt19937 gen(42);
uniform_real_distribution<> dis(0.0, 1.0);

// 使用随机数初始化矩阵和卷积核
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
```

### 卷积

由于需要进行三个步长的运算，这里使用`for`循环实现，下面是循环内部的内容

计算步长对应的输出大小

```C++
int stride = strides[s];
int out_width = (width - ksize) / stride + 1;
int out_height = (height - ksize) / stride + 1;

for (int k = 0; k < num_kernels; ++k) {
    output_cpu[k].resize(out_width * out_height * depth);
    output_gpu[k].resize(out_width * out_height * depth);
}
```

分配CUDA内存，字母`d`表示`device`设备内存，`h`表示`host`主机内存

```C++
float *d_input, *d_kernel, *d_output;
cudaMalloc(&d_input, input.size() * sizeof(float));
cudaMalloc(&d_kernel, kernel.size() * sizeof(float));
cudaMalloc(&d_output, out_width * out_height * depth * sizeof(float));
```

将数据拷贝到CUDA设备

```C++
cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_kernel, kernel.data(), kernel.size() * sizeof(float), cudaMemcpyHostToDevice);
```

定义`block`和`grid`大小

```C++
dim3 blockSize(16, 16, 1);
dim3 gridSize((out_width + blockSize.x - 1) / blockSize.x, (out_height + blockSize.y - 1) / blockSize.y, depth);
```

使用核函数进行卷积运算并计时，这里以滑窗法实现的`con2d`函数为例

```C++ 
cudaEvent_t start, stop;
cudaEventRecord(start);
for (int k = 0; k < num_kernels; ++k) {
    conv2d<<<gridSize, blockSize>>>(d_input, d_kernel, d_output, width, height, depth, ksize, stride, out_width, out_height);
    cudaMemcpy(output_gpu[k].data(), d_output, out_width * out_height * depth * sizeof(float), cudaMemcpyDeviceToHost);
}
cudaDeviceSynchronize();
cudaEventRecord(stop);
```

最后释放内存并输出用时

为了验证CUDA卷积程序的正确性，可以用CPU进行同样的运算诸位比较来验证，CPU实现卷积的过程与核函数类似，这里就不展开了

```C++
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
```

下面是卷积实现的核函数

## 滑窗法卷积

```C++
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
```

这个核函数的形参分别为输入矩阵、卷积核、输出矩阵的指针，以及输入矩阵的宽度、高度、深度（通道），卷积核大小，步长，输出矩阵的高度和宽度

核函数首先计算该线程应当处理的元素在输出矩阵中的位置，然后

- 检查当前线程计算的位置是否在输出矩阵的范围内。
- 如果在范围内，线程计算该位置的输出值。通过遍历卷积核的每个元素，并将其与输入矩阵中相应位置的元素相乘，然后将这些乘积加起来来实现，具体包括
  - 计算当前卷积核元素对应的输入矩阵中的位置。
  - 检查这个位置是否在输入矩阵的范围内。
  - 如果在范围内，进行乘法操作并累加到`sum`变量中。
- 最后，将计算得到的`sum`值赋给输出矩阵的相应位置。

## im2col方法卷积

im2col卷积包括了im2col转换和矩阵乘法两步，由于在之前的实验中已经实现过矩阵乘法，这里只展示了实现im2col的核函数

```C++
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
```

这个核函数的形参包括了输入图像，输出列向量，输入的宽、高、通道数，卷积核大小，步长，转换后的列向量高度和宽度，实现了

- 对于每个输出位置，计算对应的输入图像中的起始位置。
- 对于卷积核覆盖的每个像素，将其值复制到输出矩阵的相应位置。这一步实际上是**将卷积核覆盖的区域“拉直”成一个列向量**。

经过转化后，卷积操作可以通过转化后的列向量和卷积核进行矩阵乘法来得到，简化了卷积的实现，并可以利用已有矩阵乘法库来加速计算

## CUDNN卷积

CUDNN的卷积有别于前面两种，属于调库实现，主要步骤包括

1. 初始化cuDNN库：通过`cudnnCreate`函数创建一个cuDNN句柄，用于后续的所有cuDNN操作。
2. 创建和设置张量描述符：使用`cudnnCreateTensorDescriptor`和`cudnnSetTensor4dDescriptor`函数创建并设置输入和输出张量的描述符。这些描述符定义了张量的格式、数据类型以及维度信息。
3. 创建和设置过滤器（卷积核）描述符：通过`cudnnCreateFilterDescriptor`和`cudnnSetFilter4dDescriptor`函数创建并设置卷积核的描述符，包括卷积核的数量、维度以及数据类型。
4. 创建和设置卷积描述符：使用`cudnnCreateConvolutionDescriptor`和`cudnnSetConvolution2dDescriptor`函数创建并设置卷积操作的描述符，包括卷积的填充、步长、膨胀等参数。
5. 选择最优的卷积算法：`cudnnGetConvolutionForwardAlgorithm_v7`函数用于选择最适合当前配置（包括输入、输出、卷积核和卷积参数）的前向卷积算法。
6. 查询工作空间大小：`cudnnGetConvolutionForwardWorkspaceSize`函数查询给定卷积算法所需的工作空间大小。
7. 执行卷积操作：`cudnnConvolutionForward`函数执行实际的卷积操作，需要提供输入和输出张量、卷积核、卷积描述符、选择的卷积算法以及工作空间。
8. 资源清理：最后，代码清理了所有分配的资源，包括设备内存、张量描述符、过滤器描述符、卷积描述符以及cuDNN句柄。

在看起来比较繁琐的初始化过程中，比较重要的是这一步，使用`cudnnGetConvolutionForwardAlgorithm_v7`函数选择最适合当前配置（包括输入、输出、卷积核和卷积参数）的前向卷积算法。

```C++
CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
    cudnn,
    input_descriptor,
    kernel_descriptor,
    convolution_descriptor,
    output_descriptor,
    1, // 请求返回的算法数量
    &returnedAlgoCount,
    &convolution_algorithm));
```

# 实验结果

首先可以通过CPU计算来验证程序正确性，经过验证，都可以得到类似这样的输出

![image-20240623155418723](./assets/image-20240623155418723.png)

然后通过命令行给出不同的输入矩阵大小，比较计算结果，最终结果如下

## 滑窗法

| 矩阵规模 | 步长 | 时间     |
| -------- | ---- | -------- |
| 512      | 1    | 4.90966  |
| 512      | 2    | 0.808576 |
| 512      | 3    | 0.468224 |
| 1024     | 1    | 10.7297  |
| 1024     | 2    | 1.97709  |
| 1024     | 3    | 2.41162  |
| 2048     | 1    | 29.5756  |
| 2048     | 2    | 11.2278  |
| 2048     | 3    | 5.55238  |

## im2col

| 矩阵规模 | 步长 | 时间    |
| -------- | ---- | ------- |
| 512      | 1    | 6.16531 |
| 512      | 2    | 1.12051 |
| 512      | 3    | 1.1489  |
| 1024     | 1    | 13.9953 |
| 1024     | 2    | 4.13581 |
| 1024     | 3    | 10.8266 |
| 2048     | 1    | 46.0935 |
| 2048     | 2    | 11.3484 |
| 2048     | 3    | 5.4335  |

## CUDNN

| 矩阵规模 | 步长 | 时间     |
| -------- | ---- | -------- |
| 512      | 1    | 12.4805  |
| 512      | 2    | 0.062592 |
| 512      | 3    | 0.074176 |
| 1024     | 1    | 14.6422  |
| 1024     | 2    | 0.221088 |
| 1024     | 3    | 0.172896 |
| 2048     | 1    | 14.5213  |
| 2048     | 2    | 1.156    |
| 2048     | 3    | 0.943232 |

## 结果分析

观察实验结果可以发现，总体来说运行速度由快至慢的排名是CUDNN，滑窗，im2col

步长的增加通常能减少计算时间，但具体效果因方法和矩阵规模而异

理论上应该稍快的**im2col总体慢于滑窗法**，可能的原因如下

- im2col方法需要将输入图像数据展开成矩阵，这个展开操作涉及大量的内存复制和重新排列。内存操作的开销在大矩阵或高步长时可能非常显著，从而导致总体计算时间增加。
- 在大规模数据下，矩阵乘法的开销（包括内存读取和写入、缓存命中率等）可能会超过预期。尤其是在卷积核较小时，矩阵乘法的优势不明显
- 并没有使用高效的矩阵乘法库
- 矩阵转换和矩阵乘法的核函数是分离的，增加了额外的调度和访存开销

理论应当碾压的**CUDNN并没有在所有任务都做到最快**，可能的原因如下：

- CUDNN的设计和优化是为了在大规模数据和高计算需求的场景下发挥最佳性能。对于小规模数据或某些特定步长，CUDNN的初始化和调度开销可能导致总体性能不如预期
- CUDNN的启动和任务调度会引入一定的开销。这些开销在处理大规模数据时可以被计算时间所掩盖，但在处理小规模数据时可能显著影响性能
- CUDNN在不同卷积操作（如不同步长、不同卷积核大小）上的优化程度可能有所不同。某些特定卷积操作的实现细节和优化策略可能未能完全适应当前任务的需求，从而导致性能不如其他方法

要想使我们的手搓算法在大规模计算时接近CUDNN的效率，以下是几种可能的方法

- 优化内存管理和数据布局
- 增加算法选择，选择对于特定任务最优的算法
- 在矩阵乘法时使用一些库进行操作

# 实验总结

通过这次实验，我了解掌握了矩阵卷据的方法，以及在不同平台进行计算的方法

通过对于CUDNN的使用，以及分析实验结果时查阅网络资料，了解了更多的关于大规模计算时的优化方法

