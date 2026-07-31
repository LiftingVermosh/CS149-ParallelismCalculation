from task import input_t, output_t
import torch
import triton
import triton.language as tl

@triton.jit
def swiglu_kernel_triton(
    x_ptr, W_ptr, V_ptr, b_ptr, c_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_vk, stride_vn,
    stride_om, stride_on,
    beta,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 获取当前块的索引
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # 计算当前块对应的行和列范围
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    # 初始化累加器
    acc_xw = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_xv = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K 维度循环进行矩阵乘法 
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offsets = k * BLOCK_K + rk
        
        # 加载 x 的分块 (M, K)
        mask_x = (rm[:, None] < M) & (k_offsets[None, :] < K)   # if (index >= M * N)

        x_tile = tl.load(x_ptr + rm[:, None] * stride_xm + k_offsets[None, :] * stride_xk, mask=mask_x, other=0.0)
        
        # 加载 W 和 V 的分块 (K, N)
        mask_wv = (k_offsets[:, None] < K) & (rn[None, :] < N)
        w_tile = tl.load(W_ptr + k_offsets[:, None] * stride_wk + rn[None, :] * stride_wn, mask=mask_wv, other=0.0)
        v_tile = tl.load(V_ptr + k_offsets[:, None] * stride_vk + rn[None, :] * stride_vn, mask=mask_wv, other=0.0)
        # 矩阵乘法点积
        acc_xw += tl.dot(x_tile, w_tile)
        acc_xv += tl.dot(x_tile, v_tile)

    # 加载 Bias 并进行 SwiGLU 激活计算
    # 加载 b 和 c (N,)
    mask_n = rn < N
    b_tile = tl.load(b_ptr + rn, mask=mask_n, other=0.0)
    c_tile = tl.load(c_ptr + rn, mask=mask_n, other=0.0)
    
    acc_xw += b_tile[None, :]
    acc_xv += c_tile[None, :]
    
    # Swish: x * sigmoid(beta * x)
    swish_res = acc_xw * tl.sigmoid(beta * acc_xw)
    
    # SwiGLU: Swish(xW+b) * (xV+c)
    result = swish_res * acc_xv

    # 写回 Global Memory
    mask_out = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(out_ptr + rm[:, None] * stride_om + rn[None, :] * stride_on, result, mask=mask_out)


def custom_kernel(data: input_t) -> output_t:
    """
    Reference implementation of SwiGLU activation function.
    SwiGLU(x, W, V, b, c, beta) = Swish(xW + b) ⊙ (xV + c)
    where Swish(x) = x * sigmoid(beta * x)
    
    Args:
        data: tuple of (x, W, V, b, c, beta, seq) where:
            x: input tensor of shape (batch_size, seq_len, in_features)
            W: weight matrix of shape (in_features, hidden_size)
            V: weight matrix of shape (in_features, hidden_size)
            b: bias vector of shape (hidden_size,)
            c: bias vector of shape (hidden_size,)
            beta: scalar value for Swish activation
    Returns:
        Output tensor of shape (batch_size, seq_len, hidden_size)
    """
    x, W, V, b, c, beta = data
    # M = batch * seq_len
    M = x.shape[0] * x.shape[1]
    N = W.shape[1]
    K = x.shape[2]
    
    output = torch.empty((x.shape[0], x.shape[1], N), device=x.device, dtype=x.dtype)
    
    # 定义 Block Size 
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N'])
    )

    swiglu_kernel_triton[grid](
        x, W, V, b, c, output,
        M, N, K,
        x.stride(1), x.stride(2), 
        W.stride(0), W.stride(1),
        V.stride(0), V.stride(1),
        output.stride(1), output.stride(2),
        beta,
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
    )

    return output
