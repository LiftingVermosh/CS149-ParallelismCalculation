import torch
import triton
import triton.language as tl

from task import input_t, output_t

@triton.jit
def histogram_kernel_triton(
        input_ptr,
        output_ptr,
        length,
        num_channels,
        num_bins,
        stride_al,
        stride_ac,
        stride_oc,
        stride_ob,
        BLOCK_SIZE_L: tl.constexpr,
        BLOCK_SIZE_C: tl.constexpr
):
    pid_l = tl.program_id(0)
    pid_c = tl.program_id(1)

    # Offset
    offs_l = pid_l * BLOCK_SIZE_L + tl.arrange(0, BLOCK_SIZE_L)
    offs_c = pid_c * BLOCK_SIZE_C + tl.arrange(0, BLOCK_SIZE_C)

    # Mask
    mask_l = offs_l < length
    mask_c = offs_c < num_channels

    # input_data = input_ptr + row * stride_l + col * stride_c
    input_ptrs = input_ptr + (offs_l[:, None] * stride_al + offs_c[:, None] * stride_ac)
    vals = tl.load(input_ptrs, mask=mask_l[:, None] & mask_c[None, :], other=0) # uint8

    # output_ptr + channel_idx * stride_oc + val * stride_ob
    out_channel_idx = tl.broadcast_to(offs_c[None, :], [BLOCK_SIZE_L, BLOCK_SIZE_C])
    
    # 计算目标地址偏移
    dest_ptrs = output_ptr + (out_channel_idx * stride_oc + vals.to(tl.int32) * stride_ob)

    tl.atomic_add(dest_ptrs, 1, mask=mask_l[:, None] & mask_c[None, :])

    
def custom_kernel(data: input_t) -> output_t:
    """
    Args:
        data:
            Tuple of (array, num_bins) where:
                array:    Tensor of shape [length, num_channels], dtype=uint8, containing
                          integer values in the range [0, num_bins - 1]
                num_bins: Number of histogram bins (defines allowed value range)

    Returns:
        histogram:
            Tensor of shape [num_channels, num_bins], where histogram[c][b]
            contains the count of how many times value b appears in channel c.
    """
    array, num_bins = data
    length, num_channels = array.shape
    
    # 准备输出
    histogram = torch.zeros((num_channels, num_bins), dtype=torch.int32, device=array.device)
    BLOCK_SIZE_L = 1024 
    BLOCK_SIZE_C = 32   

    grid = (
        triton.cdiv(length, BLOCK_SIZE_L),
        triton.cdiv(num_channels, BLOCK_SIZE_C)
    )

    histogram_kernel_triton[grid](
        array, histogram,
        length, num_channels, num_bins,
        array.stride(0), array.stride(1),
        histogram.stride(0), histogram.stride(1),
        BLOCK_SIZE_L=BLOCK_SIZE_L,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
    )

    return histogram
