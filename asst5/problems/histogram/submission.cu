#include <cuda_runtime.h>
#include <torch/extension.h>
#include <stdint.h>

#define THREADS_NUM 256
#define CHANNELS_PER_BLOCK 32

//
// craete your function: __global__ void kernel(...) here
// Note: input data is of type uint8_t
//


// histogram_kernel_naive -- (CUDA Device Code)
//
// Block -> channel
__global__ void histogram_kernel_naive(
    const uint8_t* __restrict__ data, 
    int* __restrict__ histogram,
    int length,
    int num_channels,
    int num_bins
) {
    // 一个 Block 处理一个 channel
    int c = blockIdx.x; 
    if(c >= num_channels) return;

    // Bin Bucket
    extern __shared__ int local_hist[];
    
    for (int b = threadIdx.x; b < num_bins; b += blockDim.x) {
        local_hist[b] = 0;
    }
    __syncthreads();    // 同步避免脏读

    // Offset Calculation
    for (int i = threadIdx.x; i < length; i += blockDim.x) {
        uint8_t val = data[i * num_channels + c];
        atomicAdd(&local_hist[val], 1);
    }
    __syncthreads();

    // Global Write Back
    for (int b = threadIdx.x; b < num_bins; b += blockDim.x) {
        atomicAdd(&histogram[c * num_bins + b], local_hist[b]);
    }
    
}

// histogram_kernel_coalesced -- (CUDA Device Code)
//
// Coalesced Access
__global__ void histogram_kernel_coalesced (
    const uint8_t* __restrict__ data, // 使用 u32 进行向量化读取
    int* __restrict__ histogram,
    int length,
    int num_channels,
    int num_bins
) {
    int c_base = blockIdx.x * CHANNELS_PER_BLOCK;
    if (c_base >= num_channels) return;

    // 每一个 block 负责处理一组 32 个 channels
    extern __shared__ int smem_hist[]; 

    for (int i = threadIdx.x; i < CHANNELS_PER_BLOCK * num_bins; i += blockDim.x) {
        smem_hist[i] = 0;
    }
    __syncthreads();


    // 合并读取与局部累计
    // threadIdx.x % 32 负责通道内偏移，threadIdx.x / 32 处理行步长
    int t_c = threadIdx.x % CHANNELS_PER_BLOCK;         // 线程负责的通道内索引
    int t_r_step = blockDim.x / CHANNELS_PER_BLOCK;     // 每一轮处理多少行
    int t_r_idx = threadIdx.x / CHANNELS_PER_BLOCK;     // 线程负责的行起始偏移
    if (c_base + t_c < num_channels) {
        for (int r = t_r_idx; r < length; r += t_r_step) {
            // 合并访问
            uint8_t val = data[r * num_channels + (c_base + t_c)];
            atomicAdd(&smem_hist[val * CHANNELS_PER_BLOCK + t_c], 1);
        }
    }
    __syncthreads();

    // 写回全局显存
    for (int i = threadIdx.x; i < CHANNELS_PER_BLOCK * num_bins; i += blockDim.x) {
        int bin = i / CHANNELS_PER_BLOCK;
        int c_offset = i % CHANNELS_PER_BLOCK;
        int c_idx = c_base + c_offset;
        
        if (c_idx < num_channels) {
            int count = smem_hist[i];
            if (count > 0) {
                atomicAdd(&histogram[c_idx * num_bins + bin], count);
            }
        }
    }
}

// histogram_kernel_vectorized -- (CUDA Device Code)
//
// Coalesced Access && Vectorized Read / Write
__global__ void histogram_kernel_vectorized(
    const uint32_t* __restrict__ data_u32, // [length * (num_channels / 4)]
    int* __restrict__ histogram,           // [num_channels * num_bins]
    int length,
    int num_channels,
    int num_bins
) {
    int c_base = blockIdx.x * CHANNELS_PER_BLOCK;
    if (c_base >= num_channels) return;
    
    extern __shared__ int smem[]; 
    
    for (int i = threadIdx.x; i < num_bins * CHANNELS_PER_BLOCK; i += blockDim.x) {
        smem[i] = 0;
    }
    __syncthreads();
    // 向量化读取与统计
    int num_channels_v = num_channels / 4;
    int c_v_base = c_base / 4;              // 向量化后的起始通道索引
    
    // 计算该线程负责的向量列索引
    int v_offset = threadIdx.x % (CHANNELS_PER_BLOCK / 4);
    int t_r_step = blockDim.x / (CHANNELS_PER_BLOCK / 4);
    int t_r_start = threadIdx.x / (CHANNELS_PER_BLOCK / 4);

    for (int r = t_r_start; r < length; r += t_r_step) {
        // Warp 内线程读取连续的 uint32
        uint32_t packed_val = data_u32[r * num_channels_v + (c_v_base + v_offset)];
        
        // 解包 4 个 uint8
        uint8_t b1 = (uint8_t)(packed_val & 0xFF);
        uint8_t b2 = (uint8_t)((packed_val >> 8) & 0xFF);
        uint8_t b3 = (uint8_t)((packed_val >> 16) & 0xFF);
        uint8_t b4 = (uint8_t)((packed_val >> 24) & 0xFF);

        int local_c = v_offset * 4;
        
        // 不同通道的写入会落到不同 Bank
        atomicAdd(&smem[b1 * CHANNELS_PER_BLOCK + local_c], 1);
        atomicAdd(&smem[b2 * CHANNELS_PER_BLOCK + local_c + 1], 1);
        atomicAdd(&smem[b3 * CHANNELS_PER_BLOCK + local_c + 2], 1);
        atomicAdd(&smem[b4 * CHANNELS_PER_BLOCK + local_c + 3], 1);
    }
    __syncthreads();

    // 写回
    for (int i = threadIdx.x; i < num_bins * CHANNELS_PER_BLOCK; i += blockDim.x) {
        int bin = i / CHANNELS_PER_BLOCK;
        int local_c = i % CHANNELS_PER_BLOCK;
        int global_c = c_base + local_c;
        
        if (global_c < num_channels) {
            int count = smem[i];
            if (count > 0) {
                atomicAdd(&histogram[global_c * num_bins + bin], count);
            }
        }
    }
}


// Host function to launch kernel
torch::Tensor histogram_kernel(
    torch::Tensor data,  // [length, num_channels], dtype=uint8
    int num_bins
) {
    TORCH_CHECK(data.device().is_cuda(), "Tensor data must be a CUDA tensor");

    const int length = data.size(0);
    const int num_channels = data.size(1);
    
    // Allocate output tensor
    auto options = torch::TensorOptions()
        .dtype(torch::kInt32)
        .device(data.device());
    torch::Tensor histogram = torch::zeros({num_channels, num_bins}, options);
    
    /* Version 1 - Naive */
    // int blocks = num_channels;
    // size_t shared_mem_size = num_bins * sizeof(int)

    /* Version 2 - Coalesced Global Memory Access */
    // dim3 blocks((num_channels + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK);
    // dim3 threads(256);
    // size_t shared_mem_size = CHANNELS_PER_BLOCK * num_bins * sizeof(int);

    /* Version 3 - Coalesced Global Memory Access && Vectorized */
    dim3 threads(256); 
    dim3 blocks((num_channels + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK);
    
    size_t shared_mem_size = CHANNELS_PER_BLOCK * num_bins * sizeof(int);
    // 检查 Shared Memory 限制
    cudaFuncSetAttribute(histogram_kernel_vectorized, 
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 
                         shared_mem_size);
    ////
    // Launch your kernel here
    ////

    /* Version 1 - Naive */
    // histogram_kernel_naive<<<blocks, threads, shared_mem_size>>>(
    //     data.data_ptr<uint8_t>(),
    //     histogram.data_ptr<int>(),
    //     length,
    //     num_channels,
    //     num_bins
    // );

    /* Version 2 - Coalesced Global Memory Access */
    // histogram_kernel_coalesced<<<blocks, threads, shared_mem_size>>>(
    //     data.data_ptr<uint8_t>(),
    //     histogram.data_ptr<int>(),
    //     length,
    //     num_channels,
    //     num_bins
    // );

    /* Version 3 - Coalesced Global Memory Access && Vectorized */
    histogram_kernel_vectorized<<<blocks, threads, shared_mem_size>>>(
        reinterpret_cast<const uint32_t*>(data.data_ptr<uint8_t>()),
        histogram.data_ptr<int>(),
        length,
        num_channels,
        num_bins
    );

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    return histogram;
}