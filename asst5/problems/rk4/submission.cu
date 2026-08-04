#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <stdexcept>
#include <utility>

#define THREADS_PER_BLOCK 256
#define TILE_DIM 8    // 分块大小
#define RADIUS 4    // 光晕宽度
#define SHMEM_D (TILE_DIM + 2 * RADIUS)

#ifndef RK4_USE_SLIDING_WINDOW
#define RK4_USE_SLIDING_WINDOW 1
#endif

const float C0 = -205.0f / 72.0f;
const float C1 = 8.0f / 5.0f;
const float C2 = -1.0f / 5.0f;
const float C3 = 8.0f / 315.0f;
const float C4 = -1.0f / 560.0f;
const float c = 0.05;

// 
// Create your kernel functions here
// Example: __global__ void kernel(...) { ... }
// 

// 将 3D 局部索引映射到 Shared Memory 的 1D 偏移
__device__ int offset_3d_2_1d(int x, int y, int z, int dim) {
    return z * (dim * dim) + y * dim + x;
}

__device__ __forceinline__ float load_or_zero(
    const float* __restrict__ u,
    int x, int y, int z,
    int Nx, int Ny, int Nz
) {
    if (x >= 0 && x < Nx && y >= 0 && y < Ny && z >= 0 && z < Nz) {
        return u[(size_t)z * Ny * Nx + (size_t)y * Nx + x];
    }
    return 0.0f;
}

// rk4_kernel_tiling -- (CUDA Device Code)
//
// 将 3D 网格切成小立方体加载进 Shared Memory
__global__ void rk4_kernel_tiling(
    const float* __restrict__ u_src,    // 当前读取的场 (用于算导数)
    const float* __restrict__ u_base,   // 步起点 u_n
    float* u_dest,                      // 写入的中间状态或结果
    float* k_out,                       // 存储当前计算出的 k (alpha * Lap)
    int Nx, int Ny, int Nz,
    float alpha, float dt, float weight,
    float inv_hx2, float inv_hy2, float inv_hz2,
    bool is_final_step // 是否是最后一步合并
) {

    extern __shared__ float s_u[];

    int tx = threadIdx.x; 
    int ty = threadIdx.y; 
    int tz = threadIdx.z;
    
    int gx = blockIdx.x * TILE_DIM + tx;
    int gy = blockIdx.y * TILE_DIM + ty;
    int gz = blockIdx.z * TILE_DIM + tz;
    
    // 协同加载数据
    int threads_in_block = blockDim.x * blockDim.y * blockDim.z;
    int tid = tz * (TILE_DIM * TILE_DIM) + ty * TILE_DIM + tx;
    
    for (int i = tid; i < SHMEM_D * SHMEM_D * SHMEM_D; i += threads_in_block) {
        int sz = i / (SHMEM_D * SHMEM_D);
        int sy = (i / SHMEM_D) % SHMEM_D;
        int sx = i % SHMEM_D;
    
        int cur_gx = blockIdx.x * TILE_DIM + sx - RADIUS;
        int cur_gy = blockIdx.y * TILE_DIM + sy - RADIUS;
        int cur_gz = blockIdx.z * TILE_DIM + sz - RADIUS;
    
        if (
            cur_gx >= 0 && cur_gx < Nx && 
            cur_gy >= 0 && cur_gy < Ny && 
            cur_gz >= 0 && cur_gz < Nz
        ) {    
            s_u[i] = u_src[cur_gz * Ny * Nx + cur_gy * Nx + cur_gx];
        } else {
            s_u[i] = 0.0f;
        } 
    }
    
    __syncthreads();
    
    // 计算内部点的 Laplacian
    if (gx < Nx && gy < Ny && gz < Nz) {
        size_t idx = (size_t)gz * Ny * Nx + (size_t)gy * Nx + gx;
        // 如果在边界 4 层内，直接保持原值
        if (
            gx < RADIUS || gx >= Nx - RADIUS || 
            gy < RADIUS || gy >= Ny - RADIUS || 
            gz < RADIUS || gz >= Nz - RADIUS
        ) {
            u_dest[idx] = u_base[idx];
            return;
        }

        // 计算中心和偏移索引
        int sid = (tz + RADIUS) * (SHMEM_D * SHMEM_D) + (ty + RADIUS) * SHMEM_D + (tx + RADIUS);
        int sy_step = SHMEM_D;
        int sz_step = SHMEM_D * SHMEM_D;

        float u_xx = (C0 * s_u[sid] +
                     C1 * (s_u[sid + 1] + s_u[sid - 1]) +
                     C2 * (s_u[sid + 2] + s_u[sid - 2]) +
                     C3 * (s_u[sid + 3] + s_u[sid - 3]) +
                     C4 * (s_u[sid + 4] + s_u[sid - 4])
        ) * inv_hx2;
        
        float u_yy = (C0 * s_u[sid] +
                C1 * (s_u[sid + sy_step] + s_u[sid - sy_step]) +
                C2 * (s_u[sid + 2 * sy_step] + s_u[sid - 2 * sy_step]) +
                C3 * (s_u[sid + 3 * sy_step] + s_u[sid - 3 * sy_step]) +
                C4 * (s_u[sid + 4 * sy_step] + s_u[sid - 4 * sy_step])
        ) * inv_hy2;

        float u_zz = (C0 * s_u[sid] +
                C1 * (s_u[sid + sz_step] + s_u[sid - sz_step]) +
                C2 * (s_u[sid + 2 * sz_step] + s_u[sid - 2 * sz_step]) +
                C3 * (s_u[sid + 3 * sz_step] + s_u[sid - 3 * sz_step]) +
                C4 * (s_u[sid + 4 * sz_step] + s_u[sid - 4 * sz_step])
        ) * inv_hz2;
        
        float lap = u_xx + u_yy + u_zz;
        float k = alpha * lap;

        if (k_out) k_out[idx] = k; // 存储当前阶段的 k
        if (!is_final_step) {
            u_dest[idx] = u_base[idx] + weight * dt * k;
        }
    }
}

// rk4_kernel_tiling_accumulate -- (CUDA Device Code)
//
// 在原 3D Tiling 结构上增加 RK4 加权累加：
//   k_accum <- k1 + 2*k2 + 2*k3 + k4
// 这样每个时间步只需要一个 accumulator，而不是同时保存 k1/k2/k3/k4。
__global__ void rk4_kernel_tiling_accumulate(
    const float* __restrict__ u_src,
    const float* __restrict__ u_base,
    float* u_dest,
    float* k_accum,
    int Nx, int Ny, int Nz,
    float alpha, float dt, float stage_weight, float accum_weight,
    float inv_hx2, float inv_hy2, float inv_hz2,
    bool reset_accum,
    bool write_stage
) {
    extern __shared__ float s_u[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    int gx = blockIdx.x * TILE_DIM + tx;
    int gy = blockIdx.y * TILE_DIM + ty;
    int gz = blockIdx.z * TILE_DIM + tz;

    int threads_in_block = blockDim.x * blockDim.y * blockDim.z;
    int tid = tz * (TILE_DIM * TILE_DIM) + ty * TILE_DIM + tx;

    for (int i = tid; i < SHMEM_D * SHMEM_D * SHMEM_D; i += threads_in_block) {
        int sz = i / (SHMEM_D * SHMEM_D);
        int sy = (i / SHMEM_D) % SHMEM_D;
        int sx = i % SHMEM_D;

        int cur_gx = blockIdx.x * TILE_DIM + sx - RADIUS;
        int cur_gy = blockIdx.y * TILE_DIM + sy - RADIUS;
        int cur_gz = blockIdx.z * TILE_DIM + sz - RADIUS;

        s_u[i] = load_or_zero(u_src, cur_gx, cur_gy, cur_gz, Nx, Ny, Nz);
    }

    __syncthreads();

    if (gx < Nx && gy < Ny && gz < Nz) {
        size_t idx = (size_t)gz * Ny * Nx + (size_t)gy * Nx + gx;
        bool is_boundary =
            gx < RADIUS || gx >= Nx - RADIUS ||
            gy < RADIUS || gy >= Ny - RADIUS ||
            gz < RADIUS || gz >= Nz - RADIUS;

        if (is_boundary) {
            if (reset_accum) {
                k_accum[idx] = 0.0f;
            }
            if (write_stage && u_dest) {
                u_dest[idx] = u_base[idx];
            }
            return;
        }

        int sid = (tz + RADIUS) * (SHMEM_D * SHMEM_D) + (ty + RADIUS) * SHMEM_D + (tx + RADIUS);
        int sy_step = SHMEM_D;
        int sz_step = SHMEM_D * SHMEM_D;

        float center = s_u[sid];
        float u_xx = (C0 * center +
                C1 * (s_u[sid + 1] + s_u[sid - 1]) +
                C2 * (s_u[sid + 2] + s_u[sid - 2]) +
                C3 * (s_u[sid + 3] + s_u[sid - 3]) +
                C4 * (s_u[sid + 4] + s_u[sid - 4])
        ) * inv_hx2;

        float u_yy = (C0 * center +
                C1 * (s_u[sid + sy_step] + s_u[sid - sy_step]) +
                C2 * (s_u[sid + 2 * sy_step] + s_u[sid - 2 * sy_step]) +
                C3 * (s_u[sid + 3 * sy_step] + s_u[sid - 3 * sy_step]) +
                C4 * (s_u[sid + 4 * sy_step] + s_u[sid - 4 * sy_step])
        ) * inv_hy2;

        float u_zz = (C0 * center +
                C1 * (s_u[sid + sz_step] + s_u[sid - sz_step]) +
                C2 * (s_u[sid + 2 * sz_step] + s_u[sid - 2 * sz_step]) +
                C3 * (s_u[sid + 3 * sz_step] + s_u[sid - 3 * sz_step]) +
                C4 * (s_u[sid + 4 * sz_step] + s_u[sid - 4 * sz_step])
        ) * inv_hz2;

        float k = alpha * (u_xx + u_yy + u_zz);

        if (reset_accum) {
            k_accum[idx] = accum_weight * k;
        } else {
            k_accum[idx] += accum_weight * k;
        }

        if (write_stage && u_dest) {
            u_dest[idx] = u_base[idx] + stage_weight * dt * k;
        }
    }
}

// rk4_apply_accumulated_kernel -- (CUDA Device Code)
//
// u_{n+1} = u_n + dt/6 * k_accum
__global__ void rk4_apply_accumulated_kernel(
    const float* __restrict__ u_base,
    const float* __restrict__ k_accum,
    float* u_next,
    int Nx, int Ny, int Nz,
    float dt
) {
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    int gz = blockIdx.z * blockDim.z + threadIdx.z;

    if (gx < Nx && gy < Ny && gz < Nz) {
        size_t idx = (size_t)gz * Ny * Nx + (size_t)gy * Nx + gx;
        if (
            gx < RADIUS || gx >= Nx - RADIUS ||
            gy < RADIUS || gy >= Ny - RADIUS ||
            gz < RADIUS || gz >= Nz - RADIUS
        ) {
            u_next[idx] = u_base[idx];
            return;
        }
        u_next[idx] = u_base[idx] + (dt / 6.0f) * k_accum[idx];
    }
}

// rk4_final_combine_kernel -- (CUDA Device Code)
//
// RK4 的最后加权合并
__global__ void rk4_final_combine_kernel(
    const float* u_n, const float* k1, const float* k2, const float* k3, const float* k4,
    float* u_next, int Nx, int Ny, int Nz, float dt
) {
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    int gz = blockIdx.z * blockDim.z + threadIdx.z;
    if (gx < Nx && gy < Ny && gz < Nz) {
        size_t idx = (size_t)gz * Ny * Nx + (size_t)gy * Nx + gx;
        if (gx < RADIUS || gx >= Nx - RADIUS || gy < RADIUS || gy >= Ny - RADIUS || gz < RADIUS || gz >= Nz - RADIUS) {
            u_next[idx] = u_n[idx];
            return;
        }
        // RK4: u_next = u_n + (dt/6)*(k1 + 2k2 + 2k3 + k4)
        u_next[idx] = u_n[idx] + (dt / 6.0f) * (k1[idx] + 2.0f * k2[idx] + 2.0f * k3[idx] + k4[idx]);
    }
}


// rk4_kernel_sliding_window -- (CUDA Device Code)
//
// 缓存 X-Y 平面，在 Z 方向使用寄存器轮转
__global__ void rk4_kernel_sliding_window(
    const float* __restrict__ u_src,
    const float* __restrict__ u_base,
    float* u_dest,
    float* k_accum,
    int Nx, int Ny, int Nz,
    float alpha, float dt, float stage_weight, float accum_weight,
    float inv_hx2, float inv_hy2, float inv_hz2,
    bool reset_accum,
    bool write_stage
) {
    extern __shared__ float s_xy[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int gx = blockIdx.x * TILE_DIM + tx;
    int gy = blockIdx.y * TILE_DIM + ty;
    int gz_start = blockIdx.z * TILE_DIM;

    int tid = ty * blockDim.x + tx;
    int threads_in_block = blockDim.x * blockDim.y;
    int plane_size = SHMEM_D * SHMEM_D;

    float z_m4 = load_or_zero(u_src, gx, gy, gz_start - 4, Nx, Ny, Nz);
    float z_m3 = load_or_zero(u_src, gx, gy, gz_start - 3, Nx, Ny, Nz);
    float z_m2 = load_or_zero(u_src, gx, gy, gz_start - 2, Nx, Ny, Nz);
    float z_m1 = load_or_zero(u_src, gx, gy, gz_start - 1, Nx, Ny, Nz);
    float z_0  = load_or_zero(u_src, gx, gy, gz_start,     Nx, Ny, Nz);
    float z_p1 = load_or_zero(u_src, gx, gy, gz_start + 1, Nx, Ny, Nz);
    float z_p2 = load_or_zero(u_src, gx, gy, gz_start + 2, Nx, Ny, Nz);
    float z_p3 = load_or_zero(u_src, gx, gy, gz_start + 3, Nx, Ny, Nz);
    float z_p4 = load_or_zero(u_src, gx, gy, gz_start + 4, Nx, Ny, Nz);

    for (int local_z = 0; local_z < TILE_DIM; ++local_z) {
        int gz = gz_start + local_z;
        if (gz >= Nz) {
            break;
        }

        for (int i = tid; i < plane_size; i += threads_in_block) {
            int sy = i / SHMEM_D;
            int sx = i % SHMEM_D;
            int cur_gx = blockIdx.x * TILE_DIM + sx - RADIUS;
            int cur_gy = blockIdx.y * TILE_DIM + sy - RADIUS;
            s_xy[i] = load_or_zero(u_src, cur_gx, cur_gy, gz, Nx, Ny, Nz);
        }

        __syncthreads();

        if (gx < Nx && gy < Ny) {
            size_t idx = (size_t)gz * Ny * Nx + (size_t)gy * Nx + gx;
            bool is_boundary =
                gx < RADIUS || gx >= Nx - RADIUS ||
                gy < RADIUS || gy >= Ny - RADIUS ||
                gz < RADIUS || gz >= Nz - RADIUS;

            if (is_boundary) {
                if (reset_accum) {
                    k_accum[idx] = 0.0f;
                }
                if (write_stage && u_dest) {
                    u_dest[idx] = u_base[idx];
                }
            } else {
                int sid = (ty + RADIUS) * SHMEM_D + (tx + RADIUS);

                float center = s_xy[sid];
                float u_xx = (C0 * center +
                        C1 * (s_xy[sid + 1] + s_xy[sid - 1]) +
                        C2 * (s_xy[sid + 2] + s_xy[sid - 2]) +
                        C3 * (s_xy[sid + 3] + s_xy[sid - 3]) +
                        C4 * (s_xy[sid + 4] + s_xy[sid - 4])
                ) * inv_hx2;

                float u_yy = (C0 * center +
                        C1 * (s_xy[sid + SHMEM_D] + s_xy[sid - SHMEM_D]) +
                        C2 * (s_xy[sid + 2 * SHMEM_D] + s_xy[sid - 2 * SHMEM_D]) +
                        C3 * (s_xy[sid + 3 * SHMEM_D] + s_xy[sid - 3 * SHMEM_D]) +
                        C4 * (s_xy[sid + 4 * SHMEM_D] + s_xy[sid - 4 * SHMEM_D])
                ) * inv_hy2;

                float u_zz = (C0 * z_0 +
                        C1 * (z_p1 + z_m1) +
                        C2 * (z_p2 + z_m2) +
                        C3 * (z_p3 + z_m3) +
                        C4 * (z_p4 + z_m4)
                ) * inv_hz2;

                float k = alpha * (u_xx + u_yy + u_zz);

                if (reset_accum) {
                    k_accum[idx] = accum_weight * k;
                } else {
                    k_accum[idx] += accum_weight * k;
                }

                if (write_stage && u_dest) {
                    u_dest[idx] = u_base[idx] + stage_weight * dt * k;
                }
            }
        }

        __syncthreads();

        z_m4 = z_m3;
        z_m3 = z_m2;
        z_m2 = z_m1;
        z_m1 = z_0;
        z_0 = z_p1;
        z_p1 = z_p2;
        z_p2 = z_p3;
        z_p3 = z_p4;
        z_p4 = load_or_zero(u_src, gx, gy, gz + 5, Nx, Ny, Nz);
    }

}

// Host function to launch kernel
torch::Tensor custom_kernel(
    torch::Tensor u0,
    float alpha,
    float hx,
    float hy,
    float hz,
    int n_steps
) {
    TORCH_CHECK(u0.device().is_cuda(), "Tensor u0 must be a CUDA tensor");
    TORCH_CHECK(u0.scalar_type() == torch::kFloat32, "u0 must be float32");
    TORCH_CHECK(u0.dim() == 3, "u0 must have shape (Nz, Ny, Nx)");

    const int Nz = u0.size(0);
    const int Ny = u0.size(1);
    const int Nx = u0.size(2);
    
    if (Nx < 9 || Ny < 9 || Nz < 9) {
        throw std::runtime_error("All dimensions must be >= 9 for radius-4 stencil.");
    }

    torch::Tensor result = u0.clone();
    torch::Tensor u = result;
    torch::Tensor u_next = torch::empty_like(u0);
    torch::Tensor u_stage_a = torch::empty_like(u0);
    torch::Tensor u_stage_b = torch::empty_like(u0);
    torch::Tensor k_accum = torch::empty_like(u0);
    

    ////
    // Launch your kernel here
    ////
    
    float inv_hx2 = 1.f / (hx * hx);
    float inv_hy2 = 1.f / (hy * hy);
    float inv_hz2 = 1.f / (hz * hz);

    float S = inv_hx2 + inv_hy2 + inv_hz2;
    float dt = c / (alpha * S);

    /* Version 1 - Shared Memory */
    int tileX = (Nx + TILE_DIM - 1) / TILE_DIM;
    int tileY = (Ny + TILE_DIM - 1) / TILE_DIM;
    int tileZ = (Nz + TILE_DIM - 1) / TILE_DIM;

    size_t shmem_size = SHMEM_D * SHMEM_D * SHMEM_D * sizeof(float);

    dim3 grids(tileX, tileY, tileZ);
    dim3 blocks(TILE_DIM, TILE_DIM, TILE_DIM); 

#if RK4_USE_SLIDING_WINDOW
    size_t shmem_size_xy = SHMEM_D * SHMEM_D * sizeof(float);
    dim3 grids_xy(tileX, tileY, tileZ);
    dim3 blocks_xy(TILE_DIM, TILE_DIM);
#endif

    for (int i = 0; i < n_steps; ++i) {
        // Stage 1: k_accum = k1, u_stage_a = u + 0.5*dt*k1
#if RK4_USE_SLIDING_WINDOW
        rk4_kernel_sliding_window<<<grids_xy, blocks_xy, shmem_size_xy>>>(
            u.data_ptr<float>(), u.data_ptr<float>(),
            u_stage_a.data_ptr<float>(), k_accum.data_ptr<float>(),
            Nx, Ny, Nz,
            alpha, dt, 0.5f, 1.0f,
            inv_hx2, inv_hy2, inv_hz2,
            true, true
        );
#else
        rk4_kernel_tiling_accumulate<<<grids, blocks, shmem_size>>>(
            u.data_ptr<float>(), u.data_ptr<float>(),
            u_stage_a.data_ptr<float>(), k_accum.data_ptr<float>(),
            Nx, Ny, Nz, 
            alpha, dt, 0.5f, 1.0f,
            inv_hx2, inv_hy2, inv_hz2,
            true, true
        );
#endif

        // Stage 2: k_accum += 2*k2, u_stage_b = u + 0.5*dt*k2
#if RK4_USE_SLIDING_WINDOW
        rk4_kernel_sliding_window<<<grids_xy, blocks_xy, shmem_size_xy>>>(
            u_stage_a.data_ptr<float>(), u.data_ptr<float>(),
            u_stage_b.data_ptr<float>(), k_accum.data_ptr<float>(),
            Nx, Ny, Nz,
            alpha, dt, 0.5f, 2.0f,
            inv_hx2, inv_hy2, inv_hz2,
            false, true
        );
#else
        rk4_kernel_tiling_accumulate<<<grids, blocks, shmem_size>>>(
            u_stage_a.data_ptr<float>(), u.data_ptr<float>(),
            u_stage_b.data_ptr<float>(), k_accum.data_ptr<float>(),
            Nx, Ny, Nz, 
            alpha, dt, 0.5f, 2.0f,
            inv_hx2, inv_hy2, inv_hz2,
            false, true
        );
#endif
        
        // Stage 3: k_accum += 2*k3, u_stage_a = u + dt*k3
#if RK4_USE_SLIDING_WINDOW
        rk4_kernel_sliding_window<<<grids_xy, blocks_xy, shmem_size_xy>>>(
            u_stage_b.data_ptr<float>(), u.data_ptr<float>(),
            u_stage_a.data_ptr<float>(), k_accum.data_ptr<float>(),
            Nx, Ny, Nz,
            alpha, dt, 1.0f, 2.0f,
            inv_hx2, inv_hy2, inv_hz2,
            false, true
        );
#else
        rk4_kernel_tiling_accumulate<<<grids, blocks, shmem_size>>>(
            u_stage_b.data_ptr<float>(), u.data_ptr<float>(),
            u_stage_a.data_ptr<float>(), k_accum.data_ptr<float>(),
            Nx, Ny, Nz, 
            alpha, dt, 1.0f, 2.0f,
            inv_hx2, inv_hy2, inv_hz2,
            false, true
        );
#endif
        
        // Stage 4: k_accum += k4
#if RK4_USE_SLIDING_WINDOW
        rk4_kernel_sliding_window<<<grids_xy, blocks_xy, shmem_size_xy>>>(
            u_stage_a.data_ptr<float>(), u.data_ptr<float>(),
            static_cast<float*>(nullptr), k_accum.data_ptr<float>(),
            Nx, Ny, Nz,
            alpha, dt, 0.0f, 1.0f,
            inv_hx2, inv_hy2, inv_hz2,
            false, false
        );
#else
        rk4_kernel_tiling_accumulate<<<grids, blocks, shmem_size>>>(
            u_stage_a.data_ptr<float>(), u.data_ptr<float>(),
            static_cast<float*>(nullptr), k_accum.data_ptr<float>(),
            Nx, Ny, Nz, 
            alpha, dt, 0.0f, 1.0f,
            inv_hx2, inv_hy2, inv_hz2,
            false, false
        );
#endif
        
        // Final Combine: u_next = u + (dt/6)*k_accum
        rk4_apply_accumulated_kernel<<<grids, blocks>>>(
            u.data_ptr<float>(),
            k_accum.data_ptr<float>(),
            u_next.data_ptr<float>(),
            Nx, Ny, Nz, dt
        );
        
        // 指针交换
        std::swap(u, u_next);
    }

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    // Synchronize to ensure kernel completion
    cudaDeviceSynchronize();
    
    return u;
}
