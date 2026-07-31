// template.cu
// Template for SwiGLU CUDA Kernel Submission
//
// This file shows the expected signature for your CUDA implementation.
// You must implement the kernel_body and custom_kernel functions below.

/**
 * SwiGLU(x, W, V, b, c, beta) = Swish(xW + b) ⊙ (xV + c)
 * 其中 Swish(x) = x * sigmoid(beta * x)
 */

#include <torch/extension.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256
#define TILE_SIZE 16

// TODO: Implement your CUDA kernel here
// Example kernel signature (you can modify as needed):
template <typename scalar_t>
__global__ void kernel_body(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ W,
    const scalar_t* __restrict__ V,
    const scalar_t* __restrict__ b,
    const scalar_t* __restrict__ c,
    float beta,
    scalar_t* __restrict__ output
) {
    // TODO: Your CUDA kernel implementation
}

// kernel_body_1D_reflection -- (CUDA Device Code)
//
// 最朴素的 loop 求和版本
// 慢的爆炸
template <typename scalar_t> 
__global__ void kernel_body_1D_reflection(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ W,
    const scalar_t* __restrict__ V,
    const scalar_t* __restrict__ b,
    const scalar_t* __restrict__ c,
    float beta,
    scalar_t* __restrict__ output,
    // New Param
    const int M, const int N, const int K
) {

    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index >= M * N) return;

    int m = index / N;
    int n = index % N;

    float sum_xw = 0.f;
    float sum_xv = 0.f;

    // Sum - (m, k) * (k, n)
    for(int k = 0; k < K; ++k) {
        sum_xw += (float)x[m * K + k] * (float)W[k * N + n];
        sum_xv += (float)x[m * K + k] * (float)V[k * N + n];
    }

    // Bias
    sum_xw += b[n];
    sum_xv += c[n];

    float swish = sum_xw / (1.0f + expf(-beta * sum_xw));

    output[index] = (scalar_t)(swish * sum_xv);
}

// kernel_body_tiled -- (CUDA Device Code)
//
// 分块搬运数据并求和版本
template <typename scalar_t>
__global__ void kernel_body_tiled(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ W,
    const scalar_t* __restrict__ V,
    const scalar_t* __restrict__ b,
    const scalar_t* __restrict__ c,
    float beta,
    scalar_t* __restrict__ output,
    int M, int N, int K
) {
    // 申请共享内存
    __shared__ float s_x[TILE_SIZE][TILE_SIZE];
    __shared__ float s_W[TILE_SIZE][TILE_SIZE];
    __shared__ float s_V[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float acc_xw = 0.0f;
    float acc_xv = 0.0f;

    // 在 K 维度上分块迭代
    for (int ph = 0; ph < (K + TILE_SIZE - 1) / TILE_SIZE; ++ph) {
        // 协同加载数据到 Shared Memory
        // 加载 x 的块
        if (row < M && (ph * TILE_SIZE + tx) < K)
            s_x[ty][tx] = (float)x[row * K + ph * TILE_SIZE + tx];
        else
            s_x[ty][tx] = 0.0f;

        // 加载 W 和 V 的块
        if (col < N && (ph * TILE_SIZE + ty) < K) {
            s_W[ty][tx] = (float)W[(ph * TILE_SIZE + ty) * N + col];
            s_V[ty][tx] = (float)V[(ph * TILE_SIZE + ty) * N + col];
        } else {
            s_W[ty][tx] = 0.0f;
            s_V[ty][tx] = 0.0f;
        }

        // 同步确保完成加载
        __syncthreads();

        // 在 Shared Memory 中计算局部点积
        for (int k = 0; k < TILE_SIZE; ++k) {
            acc_xw += s_x[ty][k] * s_W[k][tx];
            acc_xv += s_x[ty][k] * s_V[k][tx];
        }

        // 同步
        __syncthreads();
    }

    if (row < M && col < N) {
        acc_xw += (float)b[col];
        acc_xv += (float)c[col];
        float swish = acc_xw / (1.0f + expf(-beta * acc_xw));
        output[row * N + col] = (scalar_t)(swish * acc_xv);
    }
}

// Required: Main function that will be called from Python
// Signature must match: torch::Tensor custom_kernel(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, float)
torch::Tensor custom_kernel(
    torch::Tensor x,
    torch::Tensor W,
    torch::Tensor V,
    torch::Tensor b,
    torch::Tensor c,
    float beta
) {
    // TODO: Configure your kernel launch parameters
    const int M = x.size(0) * x.size(1);    // batch * seq_len
    const int N = W.size(1);                // hidden_size
    const int K = x.size(2);

    // /* Version 1 - 1D reflection */
    // int total_nums = M * N;
    // int num_blocks = (total_nums + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    // auto output = torch::empty({x.size(0), x.size(1), N}, x.options());
    // AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "kernel_body", ([&] {
    //     kernel_body_1D_reflection<scalar_t><<<num_blocks, THREADS_PER_BLOCK>>>(
    //         x.data_ptr<scalar_t>(),
    //         W.data_ptr<scalar_t>(),
    //         V.data_ptr<scalar_t>(),
    //         b.data_ptr<scalar_t>(),
    //         c.data_ptr<scalar_t>(),
    //         beta,
    //         output.data_ptr<scalar_t>(),
    //         M, N, K
    //     );
    // }));

    /* Version 2 - Tile with Shared Memory */
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    auto output = torch::empty({x.size(0), x.size(1), N}, x.options());
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "kernel_body", ([&] {
        kernel_body_tiled<scalar_t><<<gridDim, blockDim>>>(
            x.data_ptr<scalar_t>(),
            W.data_ptr<scalar_t>(),
            V.data_ptr<scalar_t>(),
            b.data_ptr<scalar_t>(),
            c.data_ptr<scalar_t>(),
            beta,
            output.data_ptr<scalar_t>(),
            M, N, K
        );
    }));


    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    // Synchronize to ensure kernel completion
    cudaDeviceSynchronize();

    return output;

}