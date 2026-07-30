# 实现FlashAttention

## 1. 为什么这个问题很重要
标准自注意力在序列长度上扩展性差，成为现代大型语言模型（LLM）的一个主要瓶颈。
* **复杂度问题：** 标准注意力的计算和内存复杂度为 $O(N^2)$。
* **内存墙问题：** 现代GPU（如H100）拥有巨大的算力，但内存带宽相对有限。标准注意力是**内存受限**的；它在高带宽存储器（HBM）和片上SRAM之间移动数据所花的时间可能比实际计算的时间还多。

FlashAttention通过感知I/O的方式解决了这些挑战。它通过将数据保留在快速的片上SRAM中，最小化HBM访问次数，提高了数据局部性，让计算单元保持忙碌，从而实现了显著更高的性能。

## 2. FlashAttention的高级思路
FlashAttention避免实例化完整的注意力矩阵，而是在GPU有限的片上SRAM内逐块计算输出。

### 关键概念
1.  **分块（Tiling）：** 将 $Q$、$K$ 和 $V$ 的小块从HBM加载到SRAM中，并在本地计算部分注意力结果。
2.  **融合（Fusion）：** 不是将完整的 $N \times N$ 注意力矩阵写入HBM再读回，而是将多个操作直接融合在一个单一的内核中。
3.  **在线Softmax（Online Softmax）：** 利用运行统计量（行内最大值和累积和）跨块增量地归一化softmax分数，无需一次性将整行数据保存在内存中。

**推荐阅读：**
> **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** ——*Tri Dao 等人*
> [在arXiv上阅读论文](https://arxiv.org/abs/2205.14135)

> [从Online Softmax到FlashAttention](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf) （华盛顿大学课程笔记，由[FlashInfer](https://github.com/flashinfer-ai/flashinfer)作者编写）

## 3. 输入/输出与测试策略
你需要实现一个CUDA内核，或使用你选择的DSL（通过PyTorch扩展暴露），接口如下。

### 参数
你的函数应接受以下张量（假设为FP16格式）：

* **`Q`（查询）：** 形状 $(B, N, S, D)$
* **`K`（键）：** 形状 $(B, N, S, D)$
* **`V`（值）：** 形状 $(B, N, S, D)$

**其中：**
* $B$：批大小
* $N$：头数
* $S$：序列长度
* $D$：头维度

**输出：**
* **`O`（输出）：** 形状 $(B, N, S, D)$
    * 应为 $\text{softmax}\left(\frac{Q K^T}{\sqrt{D}}\right)V$ 的结果。

### 测试协议
我们在 `test_cases/test.txt` 中提供了三种输入形状。建议使用较小的案例进行正确性测试，而远程服务器将只使用最大的案例进行基准测试和分析（在本地使用最大配置时可能会遇到内存不足的错误）。

## 4. 性能参考
以下是供你参考的性能数据。你还可以查看排行榜，了解同学们的最佳成果。

| 配置 $(B, N, H, D)$ | 实现方式 | 延迟（毫秒） | 相对于基线的加速比 |
| :--- | :--- | :--- | :--- |
| **小** $(1, 64, 1024, 128)$（在RTX 5090上） | PyTorch（参考） | 0.61 | 1.0x |
| | **PyTorch（FlashAttention）** | **0.23** | **2.6x** |
| **中** $(2, 64, 4096, 128)$（在RTX 5090上） | PyTorch（参考） | 24.7 | 1.0x |
| | **PyTorch（FlashAttention）** | **6.3** | **3.9x** |
| **大** $(4, 64, 8192, 128)$（在H100上） | PyTorch（参考） | 127 | 1.0x |
| | **PyTorch（FlashAttention）** | **28** | **4.5x** |