from task import input_t, output_t
import torch
import triton
import triton.language as tl

# Stencil 系数
C0 = -205.0 / 72.0
C1 = 8.0 / 5.0
C2 = -1.0 / 5.0
C3 = 8.0 / 315.0
C4 = -1.0 / 560.0

@triton.jit

def rk4_step_kernel(
    u_src_ptr, u_base_ptr, u_dest_ptr, k_accum_ptr,
    Nx, Ny, Nz,
    alpha, dt, stage_weight, accum_weight,
    inv_hx2, inv_hy2, inv_hz2,
    reset_accum: tl.constexpr, write_stage: tl.constexpr,
    BLOCK_SIZE_X: tl.constexpr, BLOCK_SIZE_Y: tl.constexpr,
    RADIUS: tl.constexpr = 4
):
    # 获取 2D 程序的 ID
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    # 计算线程块处理的范围
    off_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    off_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    
    # 基础偏移逻辑 
    def get_offset(x, y, z):
        return z * Ny * Nx + y * Nx + x

    # Z 方向循环
    for z in range(0, Nz):
        # 加载中心平面和 X/Y 方向的邻居
        mask = (off_x >= 0) & (off_x < Nx) & (off_y >= 0) & (off_y < Ny)
        
        # 辅助函数：加载带偏移的平面数据
        def load_u(ox, oy, oz):
            # 处理边界条件（简单起见，这里假设越界为0，与你CUDA load_or_zero一致）
            curr_mask = mask & (off_x + ox >= 0) & (off_x + ox < Nx) & \
                        (off_y + oy >= 0) & (off_y + oy < Ny) & \
                        (z + oz >= 0) & (z + oz < Nz)
            ptr = u_src_ptr + get_offset(off_x + ox, off_y + oy, z + oz)
            return tl.load(ptr, mask=curr_mask, other=0.0)
        # 加载 X, Y 方向 Stencil 点
        u_mid = load_u(0, 0, 0)
        
        # Laplacian X
        lap_x = C0 * u_mid + \
                C1 * (load_u(1,0,0) + load_u(-1,0,0)) + \
                C2 * (load_u(2,0,0) + load_u(-2,0,0)) + \
                C3 * (load_u(3,0,0) + load_u(-3,0,0)) + \
                C4 * (load_u(4,0,0) + load_u(-4,0,0))
        
        # Laplacian Y
        lap_y = C0 * u_mid + \
                C1 * (load_u(0,1,0) + load_u(0,-1,0)) + \
                C2 * (load_u(0,2,0) + load_u(0,-2,0)) + \
                C3 * (load_u(0,3,0) + load_u(0,-3,0)) + \
                C4 * (load_u(0,4,0) + load_u(0,-4,0))

        # Laplacian Z
        lap_z = C0 * u_mid + \
                C1 * (load_u(0,0,1) + load_u(0,0,-1)) + \
                C2 * (load_u(0,0,2) + load_u(0,0,-2)) + \
                C3 * (load_u(0,0,3) + load_u(0,0,-3)) + \
                C4 * (load_u(0,0,4) + load_u(0,0,-4))

        # 边界处理
        is_inner = (off_x >= RADIUS) & (off_x < Nx - RADIUS) & \
                   (off_y >= RADIUS) & (off_y < Ny - RADIUS) & \
                   (z >= RADIUS) & (z < Nz - RADIUS)
        
        # 计算 k
        k = alpha * (lap_x * inv_hx2 + lap_y * inv_hy2 + lap_z * inv_hz2)
        
        # 索引计算
        curr_idx = get_offset(off_x, off_y, z)
        
        # 更新 k_accum
        if reset_accum:
            new_k_acc = k * accum_weight
        else:
            old_k_acc = tl.load(k_accum_ptr + curr_idx, mask=mask)
            new_k_acc = old_k_acc + k * accum_weight
        
        tl.store(k_accum_ptr + curr_idx, new_k_acc, mask=mask)
        # 更新 u_dest
        if write_stage:
            u_base = tl.load(u_base_ptr + curr_idx, mask=mask)
            u_dest = u_base + stage_weight * dt * k
            tl.store(u_dest_ptr + curr_idx, u_dest, mask=mask)

@triton.jit
def apply_final_kernel(
    u_base_ptr, k_accum_ptr, u_next_ptr,
    Nx, Ny, Nz, dt,
    BLOCK_SIZE_X: tl.constexpr, BLOCK_SIZE_Y: tl.constexpr,
    RADIUS: tl.constexpr = 4
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    off_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    off_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    for z in range(0, Nz):
        idx = z * Ny * Nx + off_y * Nx + off_x
        mask = (off_x >= 0) & (off_x < Nx) & (off_y >= 0) & (off_y < Ny)
        
        is_inner = (off_x >= RADIUS) & (off_x < Nx - RADIUS) & \
                   (off_y >= RADIUS) & (off_y < Ny - RADIUS) & \
                   (z >= RADIUS) & (z < Nz - RADIUS)
        
        u_base = tl.load(u_base_ptr + idx, mask=mask)
        k_acc = tl.load(k_accum_ptr + idx, mask=mask)
        
        # u_{n+1} = u_n + dt/6 * (k1 + 2k2 + 2k3 + k4)
        res = tl.where(is_inner, u_base + (dt / 6.0) * k_acc, u_base)
        tl.store(u_next_ptr + idx, res, mask=mask)

def custom_kernel(data: input_t) -> output_t:
    """
    Args:
        data:
            Tuple of (u0, alpha, hx, hy, hz, n_steps) where:
                u0:      Tensor of shape (Nz, Ny, Nx), containing initial field values (float32)
                alpha:   Diffusion coefficient (float)
                hx:      Grid spacing in x direction (float)
                hy:      Grid spacing in y direction (float)
                hz:      Grid spacing in z direction (float)
                n_steps: Number of RK4 time steps (int)
    Returns:
        u:
            Tensor of shape (Nz, Ny, Nx), containing the field after n_steps RK4 updates
            of the 3D heat equation: u_t = alpha * Laplacian(u)
    """
    u0, alpha, hx, hy, hz, n_steps = data
    Nz, Ny, Nx = u0.shape
    u0 = u0.cuda()

    # 常量
    inv_hx2 = 1.0 / (hx * hx)
    inv_hy2 = 1.0 / (hy * hy)
    inv_hz2 = 1.0 / (hz * hz)
    dt = 0.05 / (alpha * (inv_hx2 + inv_hy2 + inv_hz2))

    # 分配内存
    u = u0.clone()
    u_stage = torch.empty_like(u)
    k_accum = torch.empty_like(u)
    u_next = torch.empty_like(u)

    # 配置 Triton Grid
    BLOCK_X, BLOCK_Y = 16, 16
    grid = (triton.cdiv(Nx, BLOCK_X), triton.cdiv(Ny, BLOCK_Y))
    for _ in range(n_steps):
        
        # Stage 1: k_acc = 1*k1, u_stage = u + 0.5*dt*k1
        rk4_step_kernel[grid](
            u, u, u_stage, k_accum, Nx, Ny, Nz,
            alpha, dt, 0.5, 1.0, inv_hx2, inv_hy2, inv_hz2,
            reset_accum=True, write_stage=True, BLOCK_SIZE_X=BLOCK_X, BLOCK_SIZE_Y=BLOCK_Y
        )
        
        # Stage 2: k_acc += 2*k2, u_stage = u + 0.5*dt*k2
        rk4_step_kernel[grid](
            u_stage, u, u_stage, k_accum, Nx, Ny, Nz,
            alpha, dt, 0.5, 2.0, inv_hx2, inv_hy2, inv_hz2,
            reset_accum=False, write_stage=True, BLOCK_SIZE_X=BLOCK_X, BLOCK_SIZE_Y=BLOCK_Y
        )
        # Stage 3: k_acc += 2*k3, u_stage = u + 1.0*dt*k3
        rk4_step_kernel[grid](
            u_stage, u, u_stage, k_accum, Nx, Ny, Nz,
            alpha, dt, 1.0, 2.0, inv_hx2, inv_hy2, inv_hz2,
            reset_accum=False, write_stage=True, BLOCK_SIZE_X=BLOCK_X, BLOCK_SIZE_Y=BLOCK_Y
        )
        # Stage 4: k_acc += 1*k4
        rk4_step_kernel[grid](
            u_stage, u, u_stage, k_accum, Nx, Ny, Nz,
            alpha, dt, 0.0, 1.0, inv_hx2, inv_hy2, inv_hz2,
            reset_accum=False, write_stage=False, BLOCK_SIZE_X=BLOCK_X, BLOCK_SIZE_Y=BLOCK_Y
        )
        # Final Combine
        apply_final_kernel[grid](
            u, k_accum, u_next, Nx, Ny, Nz, dt,
            BLOCK_SIZE_X=BLOCK_X, BLOCK_SIZE_Y=BLOCK_Y
        )
        
        u, u_next = u_next, u
    return u