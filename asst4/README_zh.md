# 作业4：编程机器学习加速器

**截止日期：11月13日周四晚11:59**

**总分：100分**

## 概述 ##

在本作业中，你将学习如何在 [AWS Trainium2](https://aws.amazon.com/ai/machine-learning/trainium/) 架构上实现和优化内核。该架构具有多个面向张量的加速处理引擎以及软件管理的片上存储，为这些引擎提供高带宽的数据访问。

作业分为两个部分。在第1部分中，你将通过学习一些简单的向量加法内核并编写你自己的矩阵转置内核，熟悉 Trainium 架构和数据移动模式。在第2部分中，你将在 Trainium2 上实现一个融合的卷积+最大池化层。

总体而言，本作业将：

1) 让你获得处理张量低级细节和管理加速器片上 SRAM 的经验。

2) 展示关键的保持局部性的优化技术（如循环分块和循环融合）的价值。

## 环境设置 ##

你将在配备 Trainium 加速器的 AWS 虚拟机上编程和测试代码。请按照 [cloud_readme.md](cloud_readme.md) 中的说明设置运行作业的机器。

登录到 AWS 机器后，使用以下命令从课程 GitHub 下载作业起始代码：

`git clone https://github.com/stanford-cs149/asst4-trainium2`

下载作业4仓库后，进入 `asst4-trainium2` 目录并**运行我们提供的安装脚本**：
```
cd asst4-trainium2
source install.sh
```
安装脚本将激活一个 Python [虚拟环境](https://builtin.com/data-science/python-virtual-environment)，其中包含作业所需的所有依赖项。它还会修改你的 `~/.bashrc` 文件，以便在将来登录机器时自动激活该虚拟环境。最后，脚本会设置你的 InfluxDB 凭证，以便使用 `neuron-profile`。

## 第0部分：熟悉 Trainium 和 Neuron Core 架构

### Trainium 架构概述

首先，让我们带你了解 Trainium。

本作业中使用的 `Trn2.3xlarge` 实例配备单个 Trainium 设备，该设备包含八个 NeuronCore。每个核心都配有专用 HBM（高带宽内存），如下图所示。每个 NeuronCore 可视为一个独立的处理单元，拥有自己的片上存储以及一组专门的运算引擎，用于执行 128x128 矩阵运算（Tensor Engine）、128 宽向量运算（Vector Engine）等。虽然每个 Trainium 设备有八个 NeuronCore，但在本作业中，我们将编写在单个 NeuronCore 上执行的内核。

<p align="center">
  <img src="handout/trainium_chip.png" width=45% height=45%>
  <img src="handout/neuroncore_v3.png" width=30% height=30%>
</p>

有关 NeuronCore 中四个不同计算引擎的更多详细信息，请参见[此处](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/neuron-core-v3.html)。

### Trainium 内存层次

在作业3中，关键概念之一是学习 CUDA 呈现的 GPU 内存层次：主机主内存、GPU 设备全局内存、每个线程块的共享内存以及每个 CUDA 线程的私有内存。在 Trainium 上，内存层次由四个级别组成：**主机内存 (DRAM)**、**设备内存 (HBM)** 以及两种快速的片上内存类型；**SBUF（状态缓冲区）** 和 **PSUM（部分和缓冲区）**。在本作业中，我们将编写仅针对设备/片上内存的内核，因此可以忽略 DRAM（位于 Trainium 设备外部）而专注于 HBM、SBUF 和 PSUM。

<p align="center">
  <img src="/handout/memory_hierarchy.png" width=80% height=80%>
</p>

* __HBM__ 是位于 Trainium 设备上的高带宽内存。HBM 作为设备的主内存，提供大容量存储（96 GiB）。大多数在内核外部创建的数据类型（例如 NumPy 数组）默认分配在 HBM 中。
* __SBUF__ 是 NeuronCore 上的片上存储。相比之下，SBUF 比 HBM 小得多（28 MiB），但提供高得多的带宽（约 HBM 的 20 倍）。程序员必须显式地将数据移入和移出 SBUF，才能使用 NeuronCore 执行计算。
* __PSUM__ 是一个小型专用存储库（2 MiB），专门用于存放张量引擎产生的矩阵乘法结果。

<p align="center">
  <img src="/handout/neuron_core.png" width=40% height=40%>
</p>

回想一下，在具有传统数据缓存的系统中，关于哪些来自片外内存的数据被复制并存储在片上存储中的决定是由缓存（基于缓存组织和驱逐策略）做出的。软件在给定的内存地址加载数据，硬件负责从内存中获取该数据，并管理缓存中存储哪些数据以便高效地未来访问。换句话说，从软件正确性的角度来看，缓存并不存在——它是一个硬件实现细节。

相比之下，NeuronCore 可用的内存是*软件管理的*。这意味着软件必须使用数据移动命令显式地将数据移入和移出这些内存。要么程序员必须在其程序中显式描述数据移动，要么 NKI 编译器必须分析应用程序并生成适当的数据移动操作。高效使用 NeuronCore 架构的最大挑战之一涉及高效地编排数据在机器中的移动。

## 第1部分：通过向量加法和矩阵转置学习 Neuron 内核接口（30 分）

在本节中，我们通过提供几个不同的向量元素相加应用程序的实现来介绍 Trainium 编程模型的基础知识。然后我们将编写一个简单内核来转置二维矩阵。

相应代码位于 `/part1` 目录中。具体来说，这里讨论的向量加法内核可以在 `kernels.py` 中找到。此外，我们提供了一个脚本 `run_benchmark.py`，它提供了一个便捷的命令行界面来使用不同向量大小执行这些内核。该脚本还包含一个用于收集性能分析指标的可选标志。

```
usage: run_benchmark.py [-h] --kernel {naive,tiled,stream,transpose} -n N [-m M] [--profile_name PROFILE_NAME]

options:
  -h, --help            show this help message and exit
  --kernel {naive,tiled,stream,transpose}
  -n N
  -m M
  --profile_name PROFILE_NAME
                        Name used to save .NEFF and .NTFF files
```

### NKI 编程模型：

Neuron 内核接口 (NKI) 是一种用于开发在 Trainium 设备上运行的内核的语言和编译器。NKI 内核使用 Python 编写，并利用三种类型的 NKI 操作：
1. **加载数据**：从 HBM 加载到片上 SBUF。
2. **计算**：在 NeuronCore 计算引擎上执行。
3. **存储输出**：从 SBUF 存回 HBM。

例如，以下内核定义了如何使用 NKI 执行向量加法。注意 `@nki.jit` 是一个 Python 装饰器，用于指示一个函数应被编译以在 NeuronDevices 上运行，类似于 CUDA C++ 中的 `__global__` 函数装饰器指定函数作为设备端函数在 GPU 上运行。

类似于 CUDA 内核的参数是 CUDA 设备全局内存中的数组，用 `@nki.jit` 装饰的 Python 函数的参数是位于 NeuronCore 可访问的 HBM 中的张量。`@nki.compiler.skip_middle_end_transformations` 装饰器禁用了一些可能会以意外方式转换内核的编译器优化，这将简化调试。

在以下代码中，假设 `a_vec` 和 `b_vec` 是 HBM 中长度为 128 的向量（对于大于 128 的向量，代码将无法正常工作。我们稍后会解释原因）。
```
@nki.compiler.skip_middle_end_transformations
@nki.jit
def vector_add_naive(a_vec, b_vec):
  
    # 在 HBM 中为输出向量分配空间
    out = nl.ndarray(shape=a_vec.shape, dtype=a_vec.dtype, buffer=nl.hbm)

    # 在 SBUF 中为输入向量分配空间并从 HBM 复制
    a_sbuf = nl.ndarray(shape=(a_vec.shape[0], 1), dtype=a_vec.dtype, buffer=nl.sbuf)
    b_sbuf = nl.ndarray(shape=(b_vec.shape[0], 1), dtype=b_vec.dtype, buffer=nl.sbuf)
  
    nisa.dma_copy(src=a_vec, dst=a_sbuf)
    nisa.dma_copy(src=b_vec, dst=b_sbuf)

    # 将输入向量相加
    res = nisa.tensor_scalar(a_sbuf, nl.add, b_sbuf)

    # 将结果存入 HBM
    nisa.dma_copy(src=res, dst=out)

    return out
```

在上面的代码中…

- `a_vec` 和 `b_vec` 是在内核外部创建的 NumPy 数组，位于 HBM 中。
- `a_sbuf` 和 `b_sbuf` 是显式分配在 SBUF 中的数组，形状和数据类型与 `a_vec` 和 `b_vec` 相同。
- `nisa.tensor_scalar(..., nl.add, ...)` 使用向量引擎执行向量加法。签名 `tensor_scalar` 意味着第二个操作数应是一个向量（即形状为 (N, 1)）或一个常量标量，这使其比通用的 `tensor_tensor` 操作更快。
- `nisa.dma_copy` 在 HBM 和 SBUF 之间移动相关数据（概念上类似于 NVIDIA GPU 上的 `cudaMemcpyAsync`）。

<p align="center">
  <img src="/handout/sbuf_layout.png" width=60% height=60%>
</p>

**查看上面的代码时，请注意 NKI 操作作用于张量，而非标量值。** 具体来说，片上内存 SBUF 和 PSUM 存储的数据排列为二维内存数组。二维数组的第一维称为“分区维度” `P`。第二维称为“自由维度” `F`。NeuronCore 能够并行加载和处理沿分区维度的数据，*但架构还限制了分区维度的大小必须为 128 或更小。* 换句话说，当从 HBM 向 SBUF 加载张量时，张量的分区维度最多可以为 128。我们稍后将讨论自由维度的限制。

因此，在上面的代码中，由于 `a_vec` 和 `b_vec` 是一维向量，它们的唯一维度就是分区维度，因此它们的大小被限制为 128 个元素。换句话说，该代码仅适用于大小为 128 或更小的向量。

### 步骤 1：将向量分块以在 128 条计算通道上并行化（6 分）

为了使代码能够处理大于 128 的向量，我们需要以块（原始张量的子集）的形式加载向量。

```
@nki.compiler.skip_middle_end_transformations
@nki.jit
def vector_add_tiled(a_vec, b_vec):
  
    # 在 HBM 中为输出向量分配空间
    out = nl.ndarray(shape=a_vec.shape, dtype=a_vec.dtype, buffer=nl.hbm)

    # 获取向量总行数
    M = a_vec.shape[0]
  
    # TODO：对于步骤 1，你应该修改此变量
    ROW_CHUNK = 1

    # 循环遍历总块数，我们可以使用 affine_range
    # 因为没有循环携带的依赖
    for m in nl.affine_range(M // ROW_CHUNK):

        # 为输入向量分配行块大小的分块
        a_tile = nl.ndarray((ROW_CHUNK, 1), dtype=a_vec.dtype, buffer=nl.sbuf)
        b_tile = nl.ndarray((ROW_CHUNK, 1), dtype=b_vec.dtype, buffer=nl.sbuf)
      
        # 加载一行块
        nisa.dma_copy(src=a_vec[m * ROW_CHUNK : (m + 1) * ROW_CHUNK], dst=a_tile)
        nisa.dma_copy(src=b_vec[m * ROW_CHUNK : (m + 1) * ROW_CHUNK], dst=b_tile)

        # 将行块相加
        res = nisa.tensor_scalar(a_tile, nl.add, b_tile)

        # 将结果块存入 HBM
        nisa.dma_copy(src=res, dst=out[m * ROW_CHUNK : (m + 1) * ROW_CHUNK])
  
    return out
```

上面的示例将向量行分解为单个元素块（块大小为 1 个向量元素——是的，这效率很低，我们稍后会讨论）。这是通过使用标准的 Python 切片语法 `Tensor[Index:Index:...]` 索引向量实现的。有关 NKI 中张量索引的更多详细信息，请参见[此处](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/programming_model.html#nki-tensor-indexing)。

在上述代码中，[affine_range](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/generated/nki.language.affine_range.html) 用于生成循环迭代器的数字序列，类似于 Python 的 `range` 函数，但它要求迭代之间没有循环携带的依赖关系。对于存在循环携带依赖的情况，NKI 还提供了 [sequential_range](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/generated/nki.language.sequential_range.html)。

通常，`affine_range` 让 NKI 编译器能够更积极地进行优化以增加跨计算引擎的流水线。但由于我们为了透明性/可重复性而禁用了编译器优化，因此这两种构造实际上效果相同。

**你需要做的：**
1. 运行上述 `vector_add_tiled` 实现，其中 *row_chunk = 1*，向量大小为 25600（*这可能需要几分钟*）。你可以使用以下命令：

   ```
   python run_benchmark.py --kernel tiled -n 25600
   ```

   执行时间是多少微秒（μs）？

2. 请记住，NeuronDevice 一次能加载的最大分区大小（行数）为 128。在 `kernels.py` 中，修改 `vector_add_tiled` 使其使用 *row_chunk = 128*。记录 `vector_add_tiled` 使用 *row_chunk = 128* 处理向量大小为 25600 时的执行时间（微秒）。当 *row_chunk = 128* 时，`vector_add_tiled` 在向量大小为 25600 上的速度比 *row_chunk = 1* 快多少？你认为为什么更快？（*提示：* 你应该将执行视为并行从 HBM 加载 `ROW_CHUNK` 个元素，然后对 SBUF 中的向量执行 `ROW_CHUNK` 宽的向量加法。）

3. 尝试在 *row_chunk = 256* 的情况下运行 `vector_add_tiled`，向量大小为 25600。你应该会看到错误。用一句话解释为什么尝试运行 *row_chunk = 256* 时会出现错误。

### 步骤 2a：改进的数据流（4 分）

到目前为止，我们利用了向量引擎可以并行使用所有 128 条向量通道进行计算，每条通道向/从单个 SBUF/PSUM 分区流式传输单个元素。

然而，我们可以通过沿自由维度流式传输更多元素来进一步提高性能。为此，让我们进一步思考直接内存访问 (DMA) 传输。你应该将一次 DMA 传输（即对 `nisa.dma_copy` 的调用）视为一个从 HBM 到 SBUF 或反之亦然的异步操作，它会移动一块数据。

每个 NeuronCore 有 16 个 DMA 引擎，它们都可以并行处理不同的数据传输操作。但需要注意的是，设置一次 DMA 传输并为 DMA 引擎分配任务存在开销成本。为了减少这种设置开销，高效实现应旨在每次传输中移动大量数据，以摊销 DMA 传输开销。

尽管 SBUF 张量的第一维（分区维度）不能超过 128，但单个 SBUF 向量指令的第二维可以最大为 64K 个元素。这意味着可以使用单条指令从 HBM 向 SBUF 加载 128 * 64k = 8192k 个元素。此外，我们可以在一条 `nisa.tensor_tensor` 指令中对两个大小为 8192k 元素的 SBUF 分块执行向量加法。因此，我们不应为向量的每个 128 元素块执行一次 `nisa.dma_copy`，而应在每次 DMA 传输请求中移动多个 128 行块。这种流式方法允许我们摊销数据传输的设置时间。

为了改善 DMA 传输开销，我们需要将向量重塑为二维分块，而不是线性化的数组。在作业3中，我们处理了跨整个图像分区的 CUDA 线程块，为了将 CUDA 线程映射到图像像素，我们通过计算线程的全局线性索引来平铺网格。你可以将 NeuronCore 的重塑过程视为逆过程：目标是将一维向量转换为密集的二维矩阵。NumPy 带有内置的 [reshape 函数](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html)，允许你将数组重塑为任意形状。

<p align="center">
  <img src="/handout/non_reshaped_DMA.png" width=48% height=48%>
  <img src="/handout/reshaped_DMA.png" width=48% height=48%>
</p>

看一下 `vector_add_stream`，它扩展了 `vector_add_tiled`，减少了 DMA 传输次数：
```
@nki.compiler.skip_middle_end_transformations
@nki.jit
def vector_add_stream(a_vec, b_vec):

    # 获取向量总行数
    M = a_vec.shape[0]

    # TODO：对于步骤 2a，你应该修改此变量
    FREE_DIM = 2

    # 分区维度的最大大小
    PARTITION_DIM = 128

    a_vec_re = a_vec.reshape((PARTITION_DIM, M // PARTITION_DIM))
    b_vec_re = b_vec.reshape((PARTITION_DIM, M // PARTITION_DIM))
    out = nl.ndarray(shape=a_vec_re.shape, dtype=a_vec_re.dtype, buffer=nl.hbm)

    # 循环遍历总块数
    for m in nl.affine_range(M // (PARTITION_DIM * FREE_DIM)):

        # 为重塑后的块分配空间
        a_tile = nl.ndarray((PARTITION_DIM, FREE_DIM), dtype=a_vec.dtype, buffer=nl.sbuf)
        b_tile = nl.ndarray((PARTITION_DIM, FREE_DIM), dtype=b_vec.dtype, buffer=nl.sbuf)

        # 加载输入块
        nisa.dma_copy(src=a_vec_re[:, m * FREE_DIM : (m + 1) * FREE_DIM], dst=a_tile)
        nisa.dma_copy(src=b_vec_re[:, m * FREE_DIM : (m + 1) * FREE_DIM], dst=b_tile)

        # 将块相加。注意，我们必须改用 tensor_tensor 而不是 tensor_scalar
        res = nisa.tensor_tensor(a_tile, b_tile, op=nl.add)

        # 将结果块存入 HBM
        nisa.dma_copy(src=res, dst=out[:, m * FREE_DIM : (m + 1) * FREE_DIM])

    # 将输出向量重塑为原始形状
    out = out.reshape(a_vec.shape)

    return out
```

**你需要做的：**
1. 运行上述 `vector_add_stream` 实现，其中 *FREE_DIM = 2*。对于向量大小为 25600，它花费了多少微秒（μs）？与步骤 1 中 *row_chunk = 128* 的 `vector_add_tiled` 相比，快了多少？
2. 当前的 `vector_add_stream` 实现略微减少了 DMA 传输次数，但 DMA 传输次数还可以进一步减少。在 `kernels.py` 中，修改 `vector_add_stream` 的 *FREE_DIM* 值，以尽可能减少在向量大小为 25600 时的 DMA 传输次数。

   你选择了什么 *FREE_DIM* 值？对于该 *FREE_DIM* 值，向量大小为 25600 时的执行时间是多少微秒（μs）？

   你选择的 *FREE_DIM* 值下的 `vector_add_stream` 比 *FREE_DIM = 2* 时快多少？你选择的 *FREE_DIM* 值下的 `vector_add_stream` 比 *row_chunk = 128* 时的 `vector_add_tiled` 快多少？

### 步骤 2b：学习使用 Neuron-Profile（5 分）

在选择块的自由维度大小时存在权衡：
1. 块大小太小会暴露显著的指令开销，导致引擎执行效率低下。
2. 块大小太大通常会导致引擎间的流水线效率低下，并在数据复用情况下产生高 SBUF 内存压力（“内存压力”意味着 SBUF 可能被填满）。

目前，我们已经探索了将块大小增加到最大以摊销指令开销和 DMA 传输设置/拆除的好处。现在，我们将探讨为什么使自由维度尽可能大并不总是最佳解决方案。

对于此任务，你将需要使用 NeuronDevices 的性能分析工具：`neuron-profile`，它可以提供对 NeuronCore 上运行的应用程序性能的详细分析。为了运行性能分析工具，你必须确保已按照 [环境设置](#环境设置) 中的说明运行了安装脚本，并且在 SSH 到你的机器时转发了端口 3001 和 8086。重申后一点，你应该运行的命令是：

 `ssh -i path/to/key_name.pem ubuntu@<public_dns_name> -L 3001:localhost:3001 -L 8086:localhost:8086`
 
 有关为何需要这样做的更多详细信息，请参见 [cloud_readme.md](/cloud_readme.md)。

**你需要做的：**
1.  这次，我们将向量大小增加 10 倍，这样我们不是添加 25600 个元素，而是添加 256000 个元素。这将使我们能够看到处理过大块大小所带来的权衡。

    首先，在 `vector_add_stream` 中设置 *FREE_DIM = 2000*。现在，就像之前的步骤一样，我们将执行我们的内核，但这次我们将把编译后的内核保存到 **.neff** 文件中，并将内核执行轨迹保存到 **.ntff** 轨迹文件中。让我们在向量大小为 256000 时运行 `vector_add_stream`，并将编译后的内核和轨迹保存到前缀为 `stream_256k_fd2k` 的文件中，使用以下命令：

    ```
    python run_benchmark.py --kernel stream -n 256000 --profile_name stream_256k_fd2k
    ```

    你应该已经生成了两个文件：***stream_256k_fd2k.neff*** 和 ***stream_256k_fd2k.ntff***。（你可能会在标准输出中看到一条错误消息说“hw profiler overview not found” —— 这可以安全地忽略，不用担心。）
  
    现在，使用类似的工作流，在向量大小为 256000 时，以 *FREE_DIM = 1000* 运行 `vector_add_stream`，并将编译后的内核和轨迹保存到前缀为 `stream_256k_fd1k` 的文件中。
2.  这些生成的文件将允许我们使用 `neuron-profile` 工具收集内核执行指标。这些性能指标对于分析内核性能非常有用。让我们首先查看 `vector_add_stream` 在 *FREE_DIM = 2000* 时的执行指标简要摘要，运行以下命令：

    ```
    neuron-profile view --output-format summary-text -n stream_256k_fd2k.neff -s stream_256k_fd2k.ntff
    ```

    你将看到一个按字母顺序排列的各种执行指标的摘要输出。让我们关注两个特定指标：
  
     * **dma_transfer_count**：DMA 传输次数
     * **total_time**：内核执行时间（秒）

    当 *FREE_DIM = 2000* 时，内核执行时间是多少秒？进行了多少次 DMA 传输？
  
    使用与之前相同的工作流程，查看 *FREE_DIM = 1000* 时的执行指标摘要。
  
    当 *FREE_DIM = 1000* 时，内核执行时间是多少秒？进行了多少次 DMA 传输？

3. 尽管 *FREE_DIM = 1000* 的内核有更多的 DMA 传输，但它更快！让我们分析原因。

   我们可以使用 `neuron-profile` 的 GUI 功能更深入地了解内核执行指标。让我们启动 `vector_add_stream` 在 *FREE_DIM = 2000* 时的 GUI，运行以下命令：

   ```
   neuron-profile view -n stream_256k_fd2k.neff -s stream_256k_fd2k.ntff
   ```

   运行命令后，你将看到类似如下的输出：

   `View profile at http://localhost:3001/profile/...`

   将此 *http* 链接粘贴到你选择的浏览器中，以查看更深入的分析器分析。（可以忽略页面顶部出现的任何警告。）
 
> [!NOTE]
> 只有在你正确转发了端口 3001 和 8086 并 SSH 到机器后，才能看到此界面。

   你应该会看到分析器生成的一个图表，显示随时间分配给不同引擎的指令。

   为了使我们更容易查看，请转到底部的 `View Settings` 并执行以下操作：
   * 将 `Instructions color group` 改为 `Instruction Type`
   * 在 `Timeline display options` 下关闭 `Show individual NeuronCore layout`
   * 在 `DMA display options` 下关闭 `Show expanded DMA`
   * 单击底部的 `Save`。

   完成这些步骤后，分析器图表应如下所示：

   ![分析器 GUI 示例](/handout/profiler_gui.png)
 
   你也可以将鼠标悬停在图表中的各种事件上以查看更多信息。尝试悬停在以下类别中的事件上：
 
   * **DMA-E79**：显示将输入和输出数据移动到/移动出适当缓冲区的 DMA 引擎（计算指令数量——是否与对 `nisa.dma_copy` 的预期调用次数匹配？）
   * **VectorE**：显示通过 `nisa.tensor_tensor` 添加两个输入向量的向量引擎（应突出显示为绿色）
   * **Pending DMA Count**：显示随时间推移的待处理 DMA 传输数
   * **DMA Throughput**：显示随时间推移的设备带宽利用率

   现在，在终端中按 `ctrl-c` 退出当前的 `neuron-profile view`。请注意，你仍然可以在浏览器中查看 `vector_add_stream` 在 *FREE_DIM = 2000* 情况下的分析器分析，因为它们已临时存储在数据库中。按照相同的工作流程，启动 `vector_add_stream` 在 *FREE_DIM = 1000* 情况下的分析器分析。

4. 在分析了 `vector_add_stream` 在 *FREE_DIM = 2000* 和 *FREE_DIM = 1000* 两种情况下的 GUI 分析图表后，简要解释为什么 FREE_DIM = 1000 的执行时间比 FREE_DIM = 2000 更快，尽管它需要更多的 DMA 传输（*提示：* 流水线）。

   你也可以随意尝试 `neuron-profile` GUI 的各种功能。你可能还想查看底部工具栏中的 `Summary` 选项卡。此选项卡显示与问题 2 中运行
   `neuron-profile view --output-format summary-text ...` 时看到的相同的执行指标简要摘要。欢迎从[用户指南](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/neuron-profile-user-guide.html) 中了解更多关于 `neuron-profile` 功能的信息，并从 [NKI 性能指南](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/nki_perf_guide.html) 中了解有趣的 NKI 内核性能指标。

### 步骤 3：矩阵转置（15 分 = 10 编程 + 5 书面报告 (+ 1 额外加分)）
### NeuronCore 上的矩阵操作
在开始之前，我们将演示如何在 NeuronCore 上执行矩阵操作。如前所述，NeuronCore 配备了各种计算引擎，每个引擎都针对特定类型的算术运算进行了优化。Trainium 上的张量引擎专门设计用于加速这些矩阵操作，例如矩阵乘法和矩阵转置。

<p align="center">
  <img src="/handout/tensor_engine.png" width=60% height=60%>
</p>

上图描绘了张量引擎的架构。张量引擎围绕一个 128x128 [脉动处理阵列](https://gfxcourses.stanford.edu/cs149/fall25/lecture/proghardware/slide_10) 构建，该阵列从 SBUF（片上存储）流式传输矩阵数据输入，并将输出写入 PSUM（也是片上存储）。与 SBUF 一样，PSUM 是快速的片上内存，但它比 SBUF 小得多（2MiB vs 28 MiB），并且专用于存储张量引擎计算出的矩阵乘法结果。张量引擎能够对 PSUM 中的每个地址进行读-加-写操作。因此，当以分块方式执行大型矩阵乘法时，PSUM 非常有用，其中每次矩阵乘法的结果都会累加到同一个输出块中。

### 编写内核
在这里，你将尝试编写自己的小型内核，使用张量引擎转置矩阵，然后再进入第 2 部分中涉及实际矩阵乘法的更复杂内核。查看 `kernels.py` 中的起始代码。你的内核应接受一个形状为 (M, N) 的单个二维张量作为输入，并返回一个形状为 (N, M) 的二维张量。M 和 N 的唯一限制是它们都能被 128 整除，最大分区维度为 128。

```
@nki.compiler.skip_middle_end_transformations
@nki.jit
def matrix_transpose(a_tensor):
    M, N = a_tensor.shape
    out = nl.ndarray((N, M), dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    tile_dim = 128

    assert M % tile_dim == N % tile_dim == 0, "矩阵维度不能被块维度整除！"

    # TODO：你的实现在此处。你唯一应使用的计算指令是 `nisa.nc_transpose`。

    return out
```

要实际执行转置，你必须调用 [nisa.nc_transpose](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.nc_transpose.html#nki.isa.nc_transpose)，这是一个内置指令，使用张量引擎转置大小最大为 128x128 的块，并将结果存储在 PSUM 中。你**不**允许使用其他计算指令，包括 `nisa.dma_tranpose` 或 `nl.transpose`。（当然允许内存指令，包括 `nisa.dma_copy` 和 `nl.ndarray`。）

由于你将转置比 128x128 大得多的矩阵，你的内核应管理数据块在 HBM/SBUF 之间的移动。回顾前面的向量加法内核可能会有所帮助，看看它们如何分配和移动数据。

> [!TIP]
> `nisa.dma_copy` 仅适用于 SBUF/HBM 中的张量。由于 `nisa.nc_transpose` 的输出是一个 PSUM 块，你需要先将其复制到 SBUF。你可能会发现 [`nisa.tensor_copy`](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.tensor_copy.html#nki.isa.tensor_copy) 对此很有用。

**你需要做的：**
1.  填写你的内核实现。然后通过运行以下命令在 1024x1024 矩阵上测试它：
    ```
    python run_benchmark.py --kernel transpose -n 1024
    ```
    并记录执行时间（微秒 μs）。
2. 在不使用分析器的情况下，你认为你的内核是内存受限还是计算受限？解释你的答案。然后，通过使用与 `vector_add_stream` 相同的方式对你的代码进行分析来确认这一点。（你可以包含截图，但请提供书面描述来验证你的答案。）
3.  **(额外加分，1 分)** 优化你的实现以最小化延迟。要获得加分，你应该能够在 4096x4096 转置上达到 <700 μs。确保在*不*传递 `--profile_name` 的情况下测量延迟（分析器会改变执行时间）。

    对于此部分，欢迎尝试除 `nisa.nc_transpose` 之外的其他 API。请同时提交一份简短的书面报告，解释你如何识别性能瓶颈以及如何解决它们。

## 第2部分：实现融合的卷积 - 最大池化层（70 分）

既然你已经学会了如何在 NeuronCore 上高效地移动数据，是时候自己编写一个真实的 Trainium 内核了。在本节中，你的任务是实现一个同时执行卷积和称为“最大池化”操作的内核。正如我们在课堂上讨论的那样，这两个操作是现代卷积神经网络 (CNN) 的基本组成部分，广泛应用于计算机视觉任务。一个重要的细节是，你的这两个操作的实现将是“融合”的，意味着你将在 Trainium 上实现计算，而无需将中间结果转储到片外 HBM。

### NKI 矩阵乘法内核

回想一下，向量引擎能够对大小为 (128, 64k) 的 SBUF 块进行操作。然而，张量引擎具有与向量引擎不同的独特 SBUF 块大小约束。假设我们希望张量引擎执行矩阵乘法 C = A x B，其中 A 和 B 位于 SBUF 中，结果 C 存储在 PSUM 中。Trainium 施加了以下约束：
  - 矩阵 A（左侧块）不能大于 (128, 128)
  - 矩阵 B（右侧块）不能大于 (128, 512)。
  - PSUM 中的输出块 C 的大小限制为 (128, 512)。

鉴于张量引擎的约束，在 Trainium 上为任意矩阵维度实现矩阵乘法需要将计算分块，使其作为一系列固定大小块上的矩阵乘法执行。（这类似于第 1 部分中的向量加法被分块以适用于大的输入向量大小）。下面的示例（我们从 [NKI 教程](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/tutorials) 修改而来）演示了如何使用分块方法实现矩阵乘法，其中块的大小被设置为满足 Trainium 张量引擎的块大小约束。注意：代码说明在代码列表之后提供。

```
@nki.compiler.skip_middle_end_transformations
@nki.jit
def nki_matmul_tiled_(lhsT, rhs, result):
  """NKI 内核，以分块方式计算矩阵乘法操作"""

  K, M = lhsT.shape
  K_, N = rhs.shape
  assert K == K_, "lhsT 和 rhs 必须具有相同的收缩维度"

  # 张量引擎上通用矩阵乘法的固定操作数的最大自由维度
  TILE_M = nl.tile_size.gemm_stationary_fmax  # 128

  # 块的最大分区维度
  TILE_K = nl.tile_size.pmax  # 128

  # 张量引擎上通用矩阵乘法的移动操作数的最大自由维度
  TILE_N = nl.tile_size.gemm_moving_fmax  # 512

  # 使用 affine_range 循环遍历块
  for m in nl.affine_range(M // TILE_M):
    for n in nl.affine_range(N // TILE_N):
      # 在 PSUM 中分配一个张量
      res_psum = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)

      for k in nl.affine_range(K // TILE_K):
        # 在 SBUF 上声明块
        lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
        rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)

        # 从 lhsT 和 rhs 加载块
        nisa.dma_copy(dst=lhsT_tile, src=lhsT[k * TILE_K:(k + 1) * TILE_K, m * TILE_M:(m + 1) * TILE_M])
        nisa.dma_copy(dst=rhs_tile, src=rhs[k * TILE_K:(k + 1) * TILE_K, n * TILE_N:(n + 1) * TILE_N])

        # 将部分和累加到 PSUM 中
        res_psum += nisa.nc_matmul(lhsT_tile[...], rhs_tile[...])

      # 将结果从 PSUM 复制回 SBUF，并转换为预期的输出数据类型
      res_sb = nl.copy(res_psum, dtype=result.dtype)
      nisa.dma_copy(dst=result[m * TILE_M:(m + 1) * TILE_M, n * TILE_N:(n + 1) * TILE_N], src=res_sb)
```

让我们分解计算矩阵乘法 `result = lhsT x rhs` 的内核的组成部分。

  - 输入张量：
      - `lhsT` 是左侧矩阵。但该矩阵以__转置格式__提供，形状为 `[K,M]`，其中 `K` 和 `M` 都是 128 的倍数。
      - `rhs` 是右侧矩阵，形状为 `[K,N]`，其中 `K` 是 128 的倍数，`N` 是 512 的倍数。
      - `result` 是输出矩阵，形状为 `[M,N]`
      - 在矩阵乘法中，**收缩维度** 指的是左侧矩阵的列维度和右侧矩阵的行维度。例如，假设我们有矩阵乘法：`A x B = C`。矩阵 `A` 的形状为 `[M, N]`，矩阵 `B` 的形状为 `[N, M]`。那么 `C` 的形状为 `[M, M]`。因此，被消除的维度是 `A` 的列维度和 `B` 的行维度。
      - 请注意，在上面的 `nki_matmul_tiled_` 示例中，矩阵是转置形式，其中 `lhsT=A^T`。`nisa.nc_matmul` 接受 `lhsT=A^T` 和 `rhs=B` 作为参数，并返回 `A x B`。
  - 块维度：
      - 块大小根据张量引擎矩阵乘法操作的约束设置，如[此处](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/generated/nki.isa.nc_matmul.html)所述。
        - `TILE_M`：128 — `M` 维度的块大小。
        - `TILE_K`：128 — `K` 维度的块大小。
        - `TILE_N`：512 — `N` 维度的块大小。
  - 循环遍历块：
      - 内核使用 `affine_range` 循环在 `result` 矩阵的 `M` 和 `N` 维度上迭代块。
      - 对于每个形状为 `(TILE_M, TILE_N)` 的输出块，它在 PSUM 内存中分配一个临时部分和张量 `res_psum`。
  - 加载块：
      - 对于每个输出块，将 `lhsT` 和 `rhs` 的块加载到片上 SBUF 内存中以便高效访问。
      - `lhsT_tile` 加载形状为 `[TILE_K, TILE_M]` 的切片，`rhs_tile` 加载形状为 `[TILE_K, TILE_N]` 的切片。
  - 矩阵乘法：
      - 使用加载的块执行部分矩阵乘法，并将部分结果累加到 `res_psum` 中。
  - 存储结果：
      - 一旦给定结果块的块完全计算完毕，`res_psum` 中的部分和将被复制到 SBUF 并转换为所需的数据类型。
      - 最终结果被存储回 `result` 张量中的相应位置。

> 请注意，我们已将在线教程中的 `nl.matmul()` 和 `nl.load()/nl.store()` 替换为 `nisa.nc_matmul()` 和 `nisa.dma_copy()`。这会将 nki.lang API 降低到 nki.nisa。我们建议对任何计算指令使用 nki.isa API。这在其降低方式上具有更确定的行为，并且可以减少可能导致错误编译错误的不期望行为。

总之，这种分块实现通过将大矩阵维度分解为硬件兼容的块大小来处理它们。它利用专用内存缓冲区（即 PSUM）来最小化内存延迟并优化矩阵乘法性能。你可以在此处阅读有关 NKI 矩阵乘法的更多信息 [此处](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/tutorials/matrix_multiplication.html)。

### 卷积层概述

现在让我们将注意力转向卷积层。回顾课堂上讨论过的[卷积运算](https://gfxcourses.stanford.edu/cs149/fall25/lecture/dnninference/slide_26)。它涉及在__输入特征图__上滑动一个滤波器，在每个位置上，滤波器与重叠的输入区域相互作用。在每个重叠区域中，滤波器权重与输入区域值之间执行逐元素乘法。这些逐元素乘法的结果然后相加，生成输出特征图中对应位置的单个值。这个过程捕获了相邻特征之间的局部空间模式和关系。

<p align="center">
  <img src="/handout/convolution.png" width=55% height=55%>
</p>

输入特征图通常由多个通道组成。例如，图像通常包含三个 RGB 通道（红、绿、蓝）。在这种情况下，卷积不只是在二维空间区域上计算加权和，而是计算二维空间区域和通道深度上的加权和。下图描绘了对 32x32 输入图像（具有三个 RGB 通道）执行卷积层的示例。在图中，一个 5x5x3 的滤波器应用于 32x32x3 的图像，产生 28x28x1 的输出特征图。

<p align="center">
  <img src="/handout/cs231n_convolution.png" width=55% height=55%>
  <br>
  <em>来源：CS231N https://cs231n.stanford.edu/slides/2025/lecture_5.pdf </em>
</p>

__从图像中可以看出，每个滤波器产生单个输出通道。__ 要生成多个输出通道，需要将多个滤波器应用于输入特征图。此外，每个卷积滤波器还包含一个标量偏置值，该值将添加到每个加权和中。

卷积运算的输入和输出可以总结如下（暂时忽略偏置）：

<p align="left">
  <img src="/handout/conv2d_summary.png" width=50% height=50%>
</p>

此外，[卷积层](https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html)除了输入特征图、滤波器权重和标量偏置外，还可以接受额外的超参数，如填充和步长。然而，我们*简化了卷积的约束*，以便于你实现。你**只需要支持步长为 1**，并且**不必担心填充**，因为我们会在输入特征图传入内核之前为你填充它。

### 将卷积映射到矩阵乘法

现在，我们的目标是将卷积算子映射到 Trainium 张量引擎支持的高性能矩阵运算上。为此，我们可以比较卷积与矩阵乘法的数学公式。

**Conv2D：**

<p align="center">
  <img src="/handout/conv2d_formula.png" width=65% height=65%>
</p>

**矩阵乘法：**

<p align="center">
  <img src="/handout/matmul_formula.png" width=25% height=25%>
</p>

在课堂上，我们讨论了一种将具有多个滤波器的卷积转换为单个大矩阵乘法的方法。我们在这里也会做同样的事情，但采用一种不同的方法，该方法在 Trainium 上产生高效实现。在这种方法中，卷积运算被表述为一系列独立的矩阵乘法。下图直观地说明了这种公式。

> [!NOTE]
> **这与课堂讲座中描述的 conv -> matmul 归约不同，课堂讲座中为每个空间块创建单独的行。**

<p align="center">
  <img src="/handout/conv2d_matmul_diagram.png" width=100% height=100%>
</p>

在这种方法中，输入特征图的高度和宽度维度被展平为单个维度，将输入重塑为 `(Height × Width) × Input Channels`。然后，这个重塑后的输入乘以滤波器的每个位置，其中 `i` 和 `j` 的范围分别从 `0` 到 `Filter Height - 1` 和从 `0` 到 `Filter Width - 1`。每个滤波器切片的形状为 `Input Channels × Output Channels`，生成的矩阵乘法沿着 `Input Channels` 维度收缩。为了使输入与每个滤波器切片对齐，输入必须根据滤波器的当前位置 `(i, j)` 偏移一个对应的偏移量。这些矩阵乘法的结果被累积起来以产生输出张量。

以下是所述算法的伪代码：
```
- 输入图像形状为 (Input Channels, Image Height * Image Width)
- 滤波器权重形状为 (Filter Height, Filter Weight, Input Channels, Output Channels)
- 将输出初始化为适当的形状 (Output Channels, Output Height * Output Width)

# 遍历滤波器高度
for i in range(Filter_Height):
    # 遍历滤波器宽度
    for j in range(Filter_Width):

        # 将输入张量按 (i, j) 偏移，以与滤波器的当前位置对齐
        input_shifted = shift(input, (i, j))

        # 在输入和滤波器切片之间执行矩阵乘法
        # 请注意，这是一个完整的矩阵乘法，对输入大小没有限制
        output += matmul(transpose(weight[i,j,:,:]), input_shifted)
```

> [!NOTE]
> **这只是一个算法描述，本作业的目的是让你弄清楚如何将该算法描述映射到该硬件上的高效实现！**

### 最大池化层概述

最大池化层通常用于 CNN 中连续的卷积层之间，以减小特征图的大小。这不仅防止了可能对计算资源造成问题的过大特征图，还减少了 CNN 中的参数数量，从而有效减少模型过拟合。

最大池化层的操作类似于卷积层，它在输入特征图上空间滑动一个滤波器。然而，最大池化层不是为每个重叠区域计算加权和，而是从每个区域中选择最大值并将其存储在输出特征图中。此操作独立应用于特征图的每个通道，因此通道数保持不变。例如，考虑一个 4x4 输入图像，具有三个 RGB 通道，通过一个具有 2x2 滤波器的最大池化层。生成的输出是一个 2x2 图像，具有三个 RGB 通道，表明空间维度减少了 2 倍，而通道数保持不变。

<p align="center">
  <img src="/handout/maxpool.png" width=37% height=37%>
</p>

如上所示，[最大池化层](https://pytorch.org/docs/stable/generated/torch.nn.functional.max_pool2d.html#torch.nn.functional.max_pool2d)通常具有单独的步长和滤波器大小超参数。与卷积层类似，我们简化了你需要实现的最大池化层的约束。无需同时定义这两个参数，你的内核将使用单个参数 `pool_size`，它同时对应于滤波器大小和步长。`pool_size` 只能设置为 1 或 2。当 `pool_size` 为 2 时，最大池化操作如上图所示。当 `pool_size` 为 1 时，最大池化层作为无操作，产生与输入相同的输出。虽然 `pool_size` 为 1 可能看起来毫无意义，但它实际上为你的融合层提供了额外的灵活性，你很快就会发现这一点。

### 融合卷积和最大池化

你将实现一个 NKI 内核，将卷积层和最大池化层结合为单个融合操作。下面，我们将概述你的融合层的详细规范和需求。

<p align="center">
  <img src="/handout/fused_kernel.png" width=95% height=95%>
</p>

上图说明了你的融合内核在 6x6 输入（具有单个输入通道）上执行的计算。融合内核执行标准的卷积，使用一个滤波器和步长 1。然后，融合内核在卷积结果上使用 2x2 池化滤波器执行最大池化。

你的融合内核接受以下参数：
  - `X` - 一批输入图像。`X` 的形状为 `(Batch Size, Input Channels, Input Height, Input Width)`。保证 `Input Channels` 是 128 的倍数。
  - `W` - 卷积滤波器权重。`W` 的形状为 `(Output Channels, Input Channels, Filter Height, Filter Width)`。保证 `Filter Height == Filter Width`。还保证 `Output Channels` 是 128 的倍数。此外，你可以假设权重的大小始终可以完全放入 SBUF 中。
  - `bias` - 卷积滤波器偏置。`bias` 的形状为 `(Output Channels)`。
  - `pool_size` - 最大池化滤波器和池化步长的大小。保证输入的大小、滤波器的大小和 `pool_size` 使得一切都能被整除。更具体地说，`(Input Height - Filter Height + 1) % Pool Size == 0`。注意，如果 `pool_size` 的值为 `1`，则融合内核作为普通卷积内核运行。这给了我们选择是否想要最大池化的灵活性。

你可以参考[课程幻灯片](https://gfxcourses.stanford.edu/cs149/fall25/lecture/dnninference/slide_57)中关于卷积层实现的讨论作为起点。如果你参考课程幻灯片，在我們的命名方案中，`INPUT_DEPTH` 与 `Input Channels` 同义，`LAYER_NUM_FILTERS` 与 `Output Channels` 同义。请注意，你的融合内核的输入参数与卷积课程幻灯片中描绘的形状不同。你可以自由地使用 [NumPy reshape 方法](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html) 将输入重塑为你想要的任何形状，就像第 1 部分中的 `vector_add_stream` 内核所做的那样。我们还在 `part2/conv2d_numpy.py` 中提供了一个卷积层和最大池化层的 NumPy 实现。NumPy 实现应该为你提供每个层编程逻辑的总体轮廓。思考如何将 NumPy 实现融合到单个层中可能是一个很好的练习，这正是在内核中要做的事情。欢迎查看 [NKI 教程](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/tutorials.html) 了解有关其他优化或其他 API 函数的更多信息。你还可以查看 [NKI API 参考手册](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/index.html) 以查看所有可用的 API 函数及其用法。你可能会发现其中一些很有用。*提示：* [nisa.tensor_reduce(nl.max, ...)](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.tensor_reduce.html) 对于最大池化应该很有帮助。[nisa.tensor_tensor](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.tensor_tensor.html) 对于添加偏置应该很有帮助。

### 你需要做的事情

对于这部分作业，请仅关注文件 `part2/conv2d.py`。我们提供了基本的起始代码；你的任务是完成函数 `fused_conv2d_maxpool` 中（融合的）Conv2D 内核的实现。
#### 通用提示
* **优先考虑正确性。** 我们建议从最简单的情况开始：小图像，无偏置，无最大池化。一旦你的内核适用于小图像，扩展其功能以处理太大而无法完全放入 SBUF 缓冲区的图像。然后，加入偏置加法，然后将最大池化操作融合到内核中。一旦你有了完全正确的解决方案，就开始优化性能/额外加分。
  * 测试框架将按从易到难的顺序对你的内核进行测试。此外，你可以选择在省略最大池化测试用例的情况下运行测试框架，以便你选择在具有高性能实现之后再将融合的最大池化与 conv2d 内核一起工作。
* **彻底理解你的算法。** 在考虑任何分块策略之前，请确保你充分理解上述算法所需的矩阵操作（乘法、移位、加法）。然后，绘制出矩阵及其维度，并思考如何将它们映射到硬件上，特别是关于内存层次结构。
  * 你可能还需要对输入数组进行预处理（例如，重塑或转置）以实现更高效的访问。提示：如果你想知道为什么可能需要转置，请考虑 NKI 的矩阵乘法接口的独特之处——第一个输入矩阵是转置的。
* **跟踪块维度。** 由于你无法一次计算出整个输出，你必须考虑将哪个输出维度分解为块。回想 SBUF 块的约束——分区维度最多为 128，并且必须是张量的第一维。一旦你决定了输出形状，那对你的输入形状意味着什么？换句话说，你需要 X 和 W 的哪个子集来计算单个输出块？
* **在循环排序时记住数据局部性。** 你需要的 `for` 循环来自多个来源：算法定义的滤波器高度和宽度、分块矩阵乘法以及批处理。
  * 在识别出这些循环后，一个推荐的目标是对它们进行排序，以便中间结果保留在 `PSUM` 中，直到每个块的计算完全完成。这确保 SBUF 中结果数组的每个部分只被写入一次，从而提高输出数据局部性——尽管其他方法也可能达到类似的性能。
  * 一旦就位，再对剩余的循环进行排序以优化输入数据局部性。如果你不确定，尝试不同的数据访问模式，找出效果最好的，并思考原因！
* **使用分析器指导性能调整。** 一旦你有了一个工作内核，你很可能需要进一步调整性能以获得满分/额外加分。分析器是你的朋友：寻找张量引擎空闲且利用率低的大间隙/阶段，并尝试重构代码以最小化在这些部分花费的时间。
  * 回顾第 1 部分中我们优化简单向量加法内核（以及如果你尝试了额外加分，还有转置内核）的经验也可能有所帮助。

#### 测试
使用提供的测试框架脚本来验证你的实现。要运行测试，请进入 `part2/` 目录并执行：
```
python3 test_harness.py
```

要检查融合了最大池化的 Conv2D 内核的正确性和性能，请使用 `--test_maxpool` 标志调用测试框架。

测试框架将首先运行正确性测试，然后运行性能检查。满分解决方案必须在保持正确性的同时，性能达到参考内核的 120% 以内。它将使用数据类型为 float32 和 float16 的输入张量调用你的内核，其中 float16 的性能要求更严格。请确保编写内核时记住这一点！

请注意，你的内核将在*没有* `--profile` 的情况下进行性能测试（这会略微改变执行时间），以便与性能阈值的设置方式保持一致。

#### 书面报告和分析
学生需要提交一份简短的书面报告，描述他们的实现。还要描述你是如何优化实现的。确保对你的实现进行分析，并报告使用 `float16` 和 `float32` 数据类型时达到的 MFU（模型 FLOPs 利用率）。你可以通过使用 `--profile <profile_name>` 标志运行测试框架来捕获轨迹，然后运行
```
neuron-profile view -n [profile_name].neff -s [profile_name].ntff
```

> [!TIP]
> 当你打开分析器时，可能会看到一些关于缺少基准测试参数的警告。你在此处需要提交的唯一参数是 MFU 值，它仍然可以通过将鼠标悬停在 GUI 的“Estimated MFU”部分中的“Cumulative Utilization”线上来获得，如下所示。（确保获取最末端的 MFU。）

<p align="center">
  <img src="handout/mfu.png" alt="分析器警告" width="90%">
</p>

### 使用 NKI 的技巧
* 在以下场景中优先使用 nki.isa API：
    * 对于所有计算操作
      * 使用 nisa.nc_matmul 代替 nl.matmul
      * 使用 nisa.tensor_scalar(op=nl.add, <>) 代替 nl.add
    * 优先使用 nisa.dma_copy() 代替 nl.load()/nl.store()。
    * 在调用 nisa 计算操作时，确保只传递 op=nl.* 代码作为这些操作的参数。例如，不要传递 op=math.sin。
* 避免使用嵌套函数。所有函数应在模块级别定义。
* 要调试你的实现，你可以使用 `--simulate` 标志运行测试框架。这会用对 `nki.simulate_kernel()` 的调用来包装你的实现：你可以在此处阅读更多关于它的信息 [此处](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.simulate_kernel.html#nki.simulate_kernel)。在模拟模式下运行时，你可以插入 `nl.device_print(str, tensor)` 在内核中打印设备张量的中间值。然而，CPU 模拟与设备执行之间__可能存在__一些差异。如果你对结果不确定，建议通过返回中间张量进行调试。
* 在可变张量赋值时要小心。一些 nisa API 将 dst 张量作为参数，例如 nisa.dma_copy(src=<>, dst=<>)。其他 API 通过函数本身产生 dst 张量，并且可能需要用于修改现有张量。在未来的 NKI 版本中，所有 ISA API 都将把 dst 作为参数。例如：
  * x_sbuf = nl.zeros(shape=hbm_tensor.shape, buffer=nl.sbuf)  (创建数组)
  * nisa.dma_copy(src=hbm_tensor, dst=x_sbuf) (复制到数组中)
  * 具体来说，如果你选择使用 `nl.load(...)`，`x = nl.load(...)`（创建一个新数组）与 `x[...] = nl.load(...)`（修改现有数组）是不同的。
* 避免使用块维度，它是一个纯软件构造，不影响硬件。（如果你不知道它是什么，就不用担心。）要么将其放入自由维度，要么使用张量列表。请参阅公开[文档](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.26.0/general/nki/nki_block_dimension_migration_guide.html#nki-block-dimension-migration-guide)。
* 对于张量索引，首选使用整数切片。当需要更高级的索引时，请使用 [`nl.mgrid`](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.language.mgrid.html)。不要使用嵌套切片/mgrid（例如 t[0:128, 128:256][0:64, 0:64]）。不要使用 nl.arrange()。

## 额外加分

再次对小图像运行 `neuron-profile`。小图像和大图像之间的 MFU 是否存在差异？如果是，你将如何优化你的融合卷积层以适用于小图像？（了解 `nisa.nc_matmul` 可以接受 >2D 张量作为 `moving` 参数，只要 PSUM 的硬件约束得到满足，可能会有所帮助。）

最多 5 分额外加分将奖励给达到小图像性能目标（更严格的目标）的解决方案。你的书面报告必须清楚解释你的方法以及你为优化解决方案所采取的步骤。

## 评分指南

对于正确性测试，我们使用两种类型的图像。第一种类型是小图像，尺寸为 32×16。第二种类型是大图像，尺寸为 224×224，超过了 SBUF 的容量，无法一次完全放入其中。你的代码必须通过所有正确性测试才能获得性能分数。

对于性能测试，我们评估你的内核与参考内核在不同配置下的性能：带和不带最大池化，使用 float16 和 float32 精度。

作为中间目标，我们包含了来自未优化版本的参考内核的宽松延迟。如果你的 p99 延迟在宽松延迟的 120% 以内，你将获得 95% 的性能分数。如果在优化参考延迟的 120% 以内，你将获得全部性能分数。

EC 部分只有一个性能阈值，即参考延迟的 120%。

**书面报告：30 分**
  - 第 1 部分问题：20 分
  - 第 2 部分问题：10 分

**矩阵转置内核正确性：10 分 (+1 分性能额外加分)**

**融合卷积 - 最大池化内核正确性：10 分**
  - 小图像：2.5 分
  - 大图像：2.5 分
  - 带偏置加法：2.5 分
  - 带最大池化：2.5 分

**融合卷积 - 最大池化内核性能：50 分 (+5 分额外加分)**
  - 不带最大池化 (float16)：17.5 分
  - 不带最大池化 (float32)：17.5 分
  - 带最大池化 (float16)：7.5 分
  - 带最大池化 (float32)：7.5 分
  - 小图像上不带最大池化 (float16)：1.25 分额外加分
  - 小图像上不带最大池化 (float32)：1.25 分额外加分
  - 小图像上带最大池化 (float16)：1.25 分额外加分
  - 小图像上带最大池化 (float32)：1.25 分额外加分

## 提交说明

请使用 Gradescope 提交你的作业。如果你与伙伴合作，请记得在 Gradescope 上标记你的伙伴。

1. **请将你的书面报告作为文件 `writeup.pdf` 提交。**
2. **请运行 `sh create_submission.sh` 生成 `asst4.tar.gz` 提交到 Gradescope。** 如果脚本报错 'Permission denied'，你应该运行 `chmod +x create_submission.sh` 然后重新运行脚本。请同时仔细检查生成的 `tar.gz` 包含：
  * 来自第 1 部分的包含你的转置内核的文件 `kernels.py`。
  * 来自第 2 部分的包含你的融合 Conv2D 内核的文件 `conv2d.py`。