// flash_attention.cu
// Template for FlashAttention CUDA Kernel Submission

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cmath>
#include <vector>

// Constants
#define TILE_SIZE 16
#define KV_TILE_SIZE 32
#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define ROWS_PER_BLOCK WARPS_PER_BLOCK
#define THREADS_PER_BLOCK (WARPS_PER_BLOCK * WARP_SIZE)
#define MAX_HEAD_DIM 128
#define VALUES_PER_LANE ((MAX_HEAD_DIM + WARP_SIZE - 1) / WARP_SIZE)
// ------------------------------------------------------------------------
// CUDA Kernel Implementation
// ------------------------------------------------------------------------

__device__ __forceinline__ float warp_sum(float value) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    // 允许 Warp 内的线程直接交换寄存器数据
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return __shfl_sync(0xffffffff, value, 0);
}

template <typename scalar_t> __global__ void cooperative_flash_attention_kernel (
    const scalar_t* __restrict__ Q, const scalar_t* __restrict__ K,
    const scalar_t* __restrict__ V, scalar_t* __restrict__ O,
    const int batch_size, const int num_heads, const int seq_len,
    const int head_dim, const float scale
) {
  extern __shared__ float sKV[];
  float* sK = sKV;
  float* sV = sKV + KV_TILE_SIZE * head_dim;

  const int tid = threadIdx.x;
  const int lane = tid & (WARP_SIZE - 1);
  const int warp_id = tid >> 5;
  const int q_row = blockIdx.x * ROWS_PER_BLOCK + warp_id;
  const int batch_head_idx = blockIdx.y;
  const int batch_idx = batch_head_idx / num_heads;
  const int head_idx = batch_head_idx % num_heads;
  const int base_offset = (batch_idx * num_heads * seq_len * head_dim) +
                          (head_idx * seq_len * head_dim);
  const bool valid_q = q_row < seq_len;

  float q_frag[VALUES_PER_LANE];
  float o_frag[VALUES_PER_LANE];

  #pragma unroll
  for (int i = 0; i < VALUES_PER_LANE; ++i) {
    const int dim = lane + i * WARP_SIZE;
    q_frag[i] = (valid_q && dim < head_dim)
                    ? static_cast<float>(Q[base_offset + q_row * head_dim + dim])
                    : 0.0f;
    o_frag[i] = 0.0f;
  }

  float m_i = -INFINITY;
  float l_i = 0.0f;

  for (int j_block = 0; j_block < seq_len; j_block += KV_TILE_SIZE) {
    const int valid_tile = min(KV_TILE_SIZE, seq_len - j_block);
    const int tile_elems = valid_tile * head_dim;

    for (int i = tid; i < tile_elems; i += THREADS_PER_BLOCK) {
      const int row = i / head_dim;
      const int dim = i - row * head_dim;
      const int offset = base_offset + (j_block + row) * head_dim + dim;
      sK[i] = static_cast<float>(K[offset]);
      sV[i] = static_cast<float>(V[offset]);
    }
    __syncthreads();

    for (int j = 0; j < valid_tile; ++j) {
      float partial = 0.0f;

      // 展开循环
      #pragma unroll
      for (int i = 0; i < VALUES_PER_LANE; ++i) {
        const int dim = lane + i * WARP_SIZE;
        if (dim < head_dim) {
          partial += q_frag[i] * sK[j * head_dim + dim];
        }
      }

      const float score = warp_sum(partial) * scale;

      if (valid_q) {
        const float m_prev = m_i;
        m_i = fmaxf(m_i, score);
        const float alpha = expf(m_prev - m_i);
        const float beta = expf(score - m_i);
        l_i = l_i * alpha + beta;

        #pragma unroll
        for (int i = 0; i < VALUES_PER_LANE; ++i) {
          const int dim = lane + i * WARP_SIZE;
          if (dim < head_dim) {
            o_frag[i] = o_frag[i] * alpha + sV[j * head_dim + dim] * beta;
          }
        }
      }
    }
    __syncthreads();
  }

  if (valid_q) {
    const float inv_l = 1.0f / l_i;
    #pragma unroll
    for (int i = 0; i < VALUES_PER_LANE; ++i) {
      const int dim = lane + i * WARP_SIZE;
      if (dim < head_dim) {
        O[base_offset + q_row * head_dim + dim] =
            static_cast<scalar_t>(o_frag[i] * inv_l);
      }
    }
  }

}

template <typename scalar_t>
__global__ void naive_flash_attention_kernel(
    const scalar_t* __restrict__ Q, const scalar_t* __restrict__ K,
    const scalar_t* __restrict__ V, scalar_t* __restrict__ O,
    const int batch_size, const int num_heads, const int seq_len,
    const int head_dim, const float scale) {
  // 1. Calculate Thread Indices
  // We map one block to one row of the Query (one token's attention)
  int bx = blockIdx.x;

  // Map linear block index to (Batch, Head, Token)
  int q_idx = bx;
  int token_idx = q_idx % seq_len;
  int head_idx = (q_idx / seq_len) % num_heads;
  int batch_idx = q_idx / (seq_len * num_heads);

  // 2. Calculate Global Memory Offsets
  // Standard layout: (Batch, Head, Seq, Dim)
  int base_offset = (batch_idx * num_heads * seq_len * head_dim) +
                    (head_idx * seq_len * head_dim);

  // Pointers to the specific row/vector for this thread
  const scalar_t* q_vec = Q + base_offset + (token_idx * head_dim);
  scalar_t* o_vec = O + base_offset + (token_idx * head_dim);

  // 3. Online Softmax State (Keep in float32 for numerical stability)
  float m_i = -INFINITY;  // Running max
  float l_i = 0.0f;       // Running sum of exponentials

  // Temporary Output Accumulator (in float32)
  // Since we can't dynamically allocate registers, we reuse the global output
  // buffer for storage, but we must be careful to read/write it as we go.
  // For this implementation, we simply assume we can overwrite O_vec.
  // Initialize Output to 0
  for (int d = 0; d < head_dim; ++d) {
    o_vec[d] = static_cast<scalar_t>(0.0f);
  }

  // 4. Iterate over Key/Value Blocks (Tiling)
  for (int j_block = 0; j_block < seq_len; j_block += TILE_SIZE) {
    // Handle edge case for last tile
    int valid_tile_size = min(TILE_SIZE, seq_len - j_block);

    // Process Tile
    for (int j = 0; j < valid_tile_size; ++j) {
      int k_idx = j_block + j;
      const scalar_t* k_vec = K + base_offset + (k_idx * head_dim);
      const scalar_t* v_vec = V + base_offset + (k_idx * head_dim);

      // A. Compute Dot Product: Q_i . K_j
      float score = 0.0f;
      for (int d = 0; d < head_dim; ++d) {
        // Cast to float for accumulation precision
        score += static_cast<float>(q_vec[d]) * static_cast<float>(k_vec[d]);
      }
      score *= scale;

      // B. Online Softmax Updates
      float m_prev = m_i;
      m_i = fmaxf(m_i, score);

      float alpha = expf(m_prev - m_i);
      float beta = expf(score - m_i);

      l_i = (l_i * alpha) + beta;

      // C. Update Output Accumulator
      // O_new = (O_old * alpha) + (V_j * beta)
      for (int d = 0; d < head_dim; ++d) {
        float o_val = static_cast<float>(o_vec[d]);
        float v_val = static_cast<float>(v_vec[d]);

        o_val = o_val * alpha + v_val * beta;

        o_vec[d] = static_cast<scalar_t>(o_val);
      }
    }
  }

  // 5. Final Normalization
  // O_final = O_acc / l_i
  for (int d = 0; d < head_dim; ++d) {
    float o_val = static_cast<float>(o_vec[d]);
    o_vec[d] = static_cast<scalar_t>(o_val / l_i);
  }
}

// ------------------------------------------------------------------------
// C++ / Python Interface
// ------------------------------------------------------------------------

// Required: Main function that will be called from Python
torch::Tensor flash_attention_forward(torch::Tensor Q, torch::Tensor K,
                                      torch::Tensor V) {
  // 1. Setup Output Tensor
  auto O = torch::empty_like(Q);

  // 2. Extract Dimensions
  const int batch_size = Q.size(0);
  const int num_heads = Q.size(1);
  const int seq_len = Q.size(2);
  const int head_dim = Q.size(3);
  const float scale = 1.0f / sqrtf(head_dim);

  if (head_dim > MAX_HEAD_DIM) {
    throw std::runtime_error("flash_attention_forward only supports head_dim <= 128");
  }

  // 3. Configure Kernel Launch Parameters

  // /* Version 1 - Naive FlashAttn */
  // // Grid: One block per query token (Total threads = B * H * L)
  // // Block: 1 thread (This is a simplified naive kernel)
  // int total_threads = batch_size * num_heads * seq_len;
  // dim3 blocks(total_threads);
  // dim3 threads(1);

  /* Version 2 - Shared Memory */
  int M = (seq_len + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
  int B_H = batch_size * num_heads;
  dim3 blocks(M, B_H);
  dim3 threads(THREADS_PER_BLOCK);
  int shmem_size = 2 * (KV_TILE_SIZE * head_dim) * sizeof(float);

  // 4. Dispatch and Launch

  // /* Version 1 - Naive FlashAttn */
  // AT_DISPATCH_FLOATING_TYPES_AND_HALF(
  //     Q.scalar_type(), "naive_flash_attention_kernel", ([&] {
  //       naive_flash_attention_kernel<scalar_t><<<blocks, threads>>>(
  //           Q.data_ptr<scalar_t>(), K.data_ptr<scalar_t>(),
  //           V.data_ptr<scalar_t>(), O.data_ptr<scalar_t>(), batch_size,
  //           num_heads, seq_len, head_dim, scale);
  //     }));


  /* Version 2 - Shared Memory */
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      Q.scalar_type(), "cooperative_flash_attention_kernel", ([&] {
        cooperative_flash_attention_kernel<scalar_t><<<blocks, threads, shmem_size>>>(
            Q.data_ptr<scalar_t>(), K.data_ptr<scalar_t>(),
            V.data_ptr<scalar_t>(), O.data_ptr<scalar_t>(), batch_size,
            num_heads, seq_len, head_dim, scale);
      }));

  // 5. Check for kernel launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(err));
  }

  return O;
}
