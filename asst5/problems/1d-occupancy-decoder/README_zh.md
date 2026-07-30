# 1D 占用解码器交叉注意力

模型定义在 `model.py` 中。

### 1. 为何这个问题很重要
- **生产相关背景**：此解码器模块的尺寸和输入大小与 [Roblox 开源的 Cube3D 三维网格生成模型](https://github.com/Roblox/cube) 中使用的模块一致。在该模型中，密集的三维查询网格从紧凑的潜码中重建占用场。
- **前向传播结构**：流水线先对原始查询坐标进行嵌入，然后核对潜特征进行注意力计算、归一化，最后输出占用 logits。
- **序列长度不平衡**：查询（`250,000 × 768`）的数量远多于潜键/值（`1,024 × 768`）。相比之下，普通的 FlashAttention 基准测试通常使用大致平衡或中等程度偏斜的长度（例如 2k × 2k 或 4k × 1k）。

### 2. 形状统计与调用图
- **查询**：`[B, 250k, 3]` → `MLPEmbedder` → `[B, 250k, 768]`
- **潜变量**：`[B, 1024, 768]`
- **交叉注意力**：`queries × latents → [B, 250k, 768]`
- **LayerNorm + 输出投影**：`[B, 250k, 768] → [B, 250k, 1]`

### 2.1. 接口规范

你的实现必须匹配以下接口：

**输入类型（`input_t`）：**
```python
input_t = Tuple[torch.Tensor, torch.Tensor, ModelWeights]
```

输入是一个元组，包含：
1. **`queries`**：`torch.Tensor`，形状为 `[batch_size, num_queries, q_in_dim]`
   - 输入查询坐标（通常 `q_in_dim=3`，用于 3D 坐标）
 
2. **`latents`**：`torch.Tensor`，形状为 `[batch_size, num_latents, width]`
   - 查询将对其进行注意力计算的潜特征向量
 
3. **`weights`**：`ModelWeights` 字典，包含所有模型权重和偏置：
   - **MLPEmbedder（query_in）**：
     - `query_in_in_layer_weight`：`[width, q_in_dim]` — 第一个线性层的权重
     - `query_in_in_layer_bias`：`[width]` — 第一个线性层的偏置
     - `query_in_out_layer_weight`：`[width, width]` — 第二个线性层的权重
     - `query_in_out_layer_bias`：`[width]` — 第二个线性层的偏置
   - **CrossAttention（attn）**：
     - `attn_c_q_weight`：`[width, width]` — 查询投影权重
     - `attn_c_q_bias`：`[width]` — 查询投影偏置
     - `attn_c_k_weight`：`[width, width]` — 键投影权重
     - `attn_c_k_bias`：`[width]` — 键投影偏置
     - `attn_c_v_weight`：`[width, width]` — 值投影权重
     - `attn_c_v_bias`：`[width]` — 值投影偏置
     - `attn_c_proj_weight`：`[width, width]` — 输出投影权重
     - `attn_c_proj_bias`：`[width]` — 输出投影偏置
   - **输出投影（Output Projection）**：
     - `out_proj_weight`：`[out_features, width]` — 最终线性层的权重（通常为 `[1, width]`）
     - `out_proj_bias`：`[out_features]` — 最终线性层的偏置（通常为 `[1]`）

**输出类型（`output_t`）：**
```python
output_t = torch.Tensor  # 形状：[batch_size, num_queries, 1]
```

**函数签名：**

对于 Python/Triton 提交：
```python
def custom_kernel(data: input_t) -> output_t:
    """
    OneDOccupancyDecoder 前向传播的实现。
  
    Args:
        data: (queries, latents, weights) 的元组
      
    Returns:
        形状为 [batch_size, num_queries, 1] 的输出张量
    """
    queries, latents, weights = data
    # 在此处实现
    return output
```

对于 CUDA 提交（`.cu` 文件）：
如果你提交 CUDA 代码，`wrap_submission.py` 会自动解包权重字典，并将所有 14 个权重/偏置张量作为独立参数传递。你的 `custom_kernel` 函数签名必须为：
```cpp
torch::Tensor custom_kernel(
    torch::Tensor queries,
    torch::Tensor latents,
    torch::Tensor query_in_in_layer_weight,
    torch::Tensor query_in_in_layer_bias,
    torch::Tensor query_in_out_layer_weight,
    torch::Tensor query_in_out_layer_bias,
    torch::Tensor attn_c_q_weight,
    torch::Tensor attn_c_q_bias,
    torch::Tensor attn_c_k_weight,
    torch::Tensor attn_c_k_bias,
    torch::Tensor attn_c_v_weight,
    torch::Tensor attn_c_v_bias,
    torch::Tensor attn_c_proj_weight,
    torch::Tensor attn_c_proj_bias,
    torch::Tensor out_proj_weight,
    torch::Tensor out_proj_bias
);
```

**重要注意事项：**

**精度要求（关键）：**
- **所有张量（输入、权重、偏置及中间计算）均使用 `float16`（半精度）**
- **例外：注意力中的 Softmax 操作必须在 `float32` 下计算以保证数值稳定性**
  - 注意力分数以 float16 计算，但在 softmax 之前转换为 float32
  - 在 float32 下执行 softmax，然后将结果转换回 float16
  - 这与 PyTorch 的 `scaled_dot_product_attention` 的行为一致，后者内部对 softmax 使用 float32
- 所有张量均位于 CUDA 设备上

**其他要求：**
- 所有权重和偏置张量已在输入中提供 — 你不应初始化或创建新的模型权重
- LayerNorm 层的 `elementwise_affine=False`，因此没有可学习参数（仅进行归一化）
- 你的实现必须产生与参考实现相同的输出，以保证正确性

### 3. 参考资料
- FlashAttention 论文：https://arxiv.org/abs/2205.14135
- 博客文章 + 实现概述：https://tridao.me/blog/2024/flash3/
- Triton：https://triton-lang.org/main/getting-started/tutorials/index.html

### 4. 硬件背景（H100 参考）
- NVIDIA H100 Tensor Core 概述：https://resources.nvidia.com/en-us-tensor-core/nvidia-hopper-architecture
- 尽管 Triton 抽象掉了许多细节，但理解 tensor core 和 TMA 仍有助于你推理理想的数据布局、MMA 形状和流水线机会。

### 5. CA 的基线与 Triton 尝试
所有测量均在 H100 上使用 batch size 1、`250k` 查询、`1024` 潜变量、`12` 头、宽度 `768` 进行。

| 变体 | 时间（毫秒） | 相对于 PyTorch 的加速比 |
| --- | --- | --- |
| PyTorch MLPEmbedder + PyTorch CrossAttention | `7.399 ± 0.003` | `1.00×` |
| Triton MLPEmbedder + Triton CrossAttention（无融合） | `9.158 ± 0.005` | `0.81×`（慢 19%） |
| Triton MLPEmbedder + PyTorch CrossAttention | `7.277 ± 0.080` | `1.02×`（快 2%） |

交叉注意力的 Triton 移植耗时约 5–6 小时。