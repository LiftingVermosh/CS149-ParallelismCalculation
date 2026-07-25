# 作业 3：一个简单的 CUDA 渲染器

**截止日期：10月30日星期四，太平洋标准时间晚上11:59**

**总分：100 分**

![My Image](handout/teaser.jpg?raw=true)

## 概述

在本作业中，你将用 CUDA 编写一个并行渲染器，用于绘制彩色圆圈。虽然这个渲染器非常简单，但将其并行化需要你设计并实现能够在并行环境下高效构建和操作的数据结构。这是一项具有挑战性的作业，因此建议你尽早开始。**真的，强烈建议你尽早开始。** 祝你好运！

## 环境搭建

1.  你将需要在亚马逊云服务（AWS）的 GPU 虚拟机上收集本作业的结果（即运行性能测试）。请按照 [cloud_readme.md](cloud_readme.md) 中的说明设置一台机器来运行作业。

2.  从课程 Github 上使用以下命令下载作业模板代码：

`git clone https://github.com/stanford-cs149/asst3`

CUDA C 编程指南 [PDF版](http://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf) 或 [网页版](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) 是学习 CUDA 编程的绝佳参考资料。网上（只需谷歌一下！）和 [NVIDIA 开发者网站](http://docs.nvidia.com/cuda/) 上有大量的 CUDA 教程和 SDK 示例。特别推荐免费的 Udacity 课程 [CUDA 并行编程入门](https://www.udacity.com/blog/2014/01/update-on-udacity-cs344-intro-to.html)。

[CUDA C 编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capabilities) 中的表 21 是一个方便的参考，列出了你将在本作业中使用的 NVIDIA T4 GPU 的每线程块最大 CUDA 线程数、线程块大小、共享内存等参数。NVIDIA T4 GPU 支持 CUDA 计算能力 7.5。

对于 C++ 问题（例如 *virtual* 关键字是什么意思），[C++ Super-FAQ](https://isocpp.org/faq) 是一个很好的资源，它详细且易于理解地解释了问题（不像很多 C++ 资源那样晦涩），由 C++ 之父 Bjarne Stroustrup 合著。

## 第 1 部分：CUDA 热身 1：SAXPY（5 分）

为了练习编写 CUDA 程序，你的热身任务是使用 CUDA 重新实现作业 1 中的 SAXPY 函数。本作业部分的模板代码位于作业仓库的 `/saxpy` 目录中。你可以在 `/saxpy` 目录中通过调用 `make` 和 `./cudaSaxpy` 来构建和运行 saxpy CUDA 程序。

请完成 `saxpy.cu` 中 `saxpyCuda` 函数的 SAXPY 实现。你需要在计算之前分配设备全局内存数组，并将主机输入数组 `X`、`Y` 和 `result` 的内容复制到 CUDA 设备内存中。CUDA 计算完成后，需要将结果复制回主机内存。请参考编程指南（网页版）第 3.2.2 节中 `cudaMemcpy` 函数的定义，或者查看作业模板代码中指向的有用教程。

作为实现的一部分，请在 `saxpyCuda` 中的 CUDA 内核调用周围添加计时器。添加完成后，你的程序应该对两种执行过程进行计时：

- 提供的模板代码包含计时器，用于测量**整个流程**：将数据复制到 GPU、运行内核以及将数据复制回 CPU。

- 你还应该插入计时器，用来测量*仅运行内核所花费的时间*。（这些计时不应包括 CPU 到 GPU 的数据传输时间，也不包括将结果从 GPU 传回 CPU 的时间。）

**在添加后一种情况的计时代码时，你需要小心：** 默认情况下，CUDA 内核在 GPU 上的执行与在 CPU 上运行的主应用程序线程是*异步*的。例如，如果你编写如下代码：

```
double startTime = CycleTimer::currentSeconds();
saxpy_kernel<<<blocks, threadsPerBlock>>>(N, alpha, device_x, device_y, device_result);
double endTime = CycleTimer::currentSeconds();
```

你将会测量到一个快得惊人的内核执行时间！（因为你只计了 API 调用本身的开销，而不是实际在 GPU 上执行计算结果的时间。）

因此，你需要在内核调用之后加入对 `cudaDeviceSynchronize()` 的调用，以等待 GPU 上的所有 CUDA 工作完成。这个 `cudaDeviceSynchronize()` 调用会在 GPU 上所有先前的 CUDA 工作完成后返回。注意，在 `cudaMemcpy()` 之后不需要使用 `cudaDeviceSynchronize()` 来确保内存传输到 GPU 完成，因为在我们使用的情况下 `cudaMemcpy()` 是同步的。（想了解更多细节，请参阅 [此文档](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html#api-sync-behavior__memcpy-sync)。）

```
double startTime = CycleTimer::currentSeconds();
saxpy_kernel<<<blocks, threadsPerBlock>>>(N, alpha, device_x, device_y, device_result);
cudaDeviceSynchronize();
double endTime = CycleTimer::currentSeconds();
```

请注意，在你的测量中，如果包含了将数据传输到 CPU 和从 CPU 返回的时间，那么在最后一个计时器之后（在调用将数据返回 CPU 的 `cudaMemcpy()` 之后）**不需要**调用 `cudaDeviceSynchronize()`，因为 `cudaMemcpy()` 只有在复制完成后才会返回调用线程。

**问题 1.** 与基于 CPU 的 SAXPY 顺序实现（回忆你在作业 1 的程序 5 中得到的 saxpy 结果）相比，你观察到了什么性能差异？

**问题 2.** 比较并解释两组计时器提供的结果之间的差异（仅内核执行计时 vs. 包括数据移到 GPU 并返回的完整过程的内核执行计时）。观察到的带宽值是否与机器不同组件可用的报告带宽大致一致？（你应该使用网络来查找 NVIDIA T4 GPU 的内存带宽。提示：<https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf>。AWS 的内存总线预期带宽为 5.3 GB/s，这与 16 通道 [PCIe 3.0](https://en.wikipedia.org/wiki/PCI_Express) 的带宽不符。有多个因素会阻止达到峰值带宽，包括 CPU 主板芯片组性能以及主机 CPU 用作传输源的内存是否“固定”——后者允许 GPU 直接访问内存而无需经过虚拟内存地址转换。如果你感兴趣，可以在这里找到更多信息：<https://kth.instructure.com/courses/12406/pages/optimizing-host-device-data-communication-i-pinned-host-memory>）

## 第 2 部分：CUDA 热身 2：并行前缀和（10 分）

现在你已经熟悉了 CUDA 程序的基本结构和布局，第二个练习是要求你提出 `find_repeats` 函数的并行实现，该函数给定一个整数列表 `A`，返回所有满足 `A[i] == A[i+1]` 的索引 `i` 的列表。

例如，给定数组 `{1,2,2,1,1,1,3,5,3,3}`，你的程序应输出数组 `{1,3,4,8}`。

#### 独占前缀和

我们希望你通过首先实现并行独占前缀和操作来实现 `find_repeats`。

独占前缀和接收数组 `A` 并生成一个新数组 `output`，在索引 `i` 处，其值为所有在 `A[i]` 之前的元素之和（不包括 `A[i]`）。例如，给定数组 `A={1,4,6,8,2}`，独占前缀和的输出为 `output={0,1,5,11,19}`。

以下“类 C”代码是 scan 的迭代版本。在下边的伪代码中，我们使用 `parallel_for` 来表示潜在的并行循环。这就是我们在课堂上讨论的算法：<https://gfxcourses.stanford.edu/cs149/fall25/lecture/dataparallel/slide_17>

```
void exclusive_scan_iterative(int* start, int* end, int* output) {

    int N = end - start;
    memmove(output, start, N*sizeof(int));

    // 上扫阶段
    for (int two_d = 1; two_d <= N/2; two_d*=2) {
        int two_dplus1 = 2*two_d;
        parallel_for (int i = 0; i < N; i += two_dplus1) {
            output[i+two_dplus1-1] += output[i+two_d-1];
        }
    }

    output[N-1] = 0;

    // 下扫阶段
    for (int two_d = N/2; two_d >= 1; two_d /= 2) {
        int two_dplus1 = 2*two_d;
        parallel_for (int i = 0; i < N; i += two_dplus1) {
            int t = output[i+two_d-1];
            output[i+two_d-1] = output[i+two_dplus1-1];
            output[i+two_dplus1-1] += t;
        }
    }
}
```

我们希望你使用此算法在 CUDA 中实现一个并行前缀和版本。你必须在 `scan/scan.cu` 中实现 `exclusive_scan` 函数。你的实现将包含主机代码和设备代码。该实现需要多次 CUDA 内核启动（对应上面伪代码中的每个 `parallel_for` 循环）。

**注意：** 在模板代码中，上述参考扫描实现假设输入数组的长度（`N`）是 2 的幂。在 `cudaScan` 函数中，我们通过将输入数组长度向上取整到下一个 2 的幂来解决这个问题，以便在 GPU 上分配相应的缓冲区。但是，代码只将 `N` 个元素从 GPU 缓冲区复制回 CPU 缓冲区。这一事实应该会简化你的 CUDA 实现。

编译后会生成二进制文件 `cudaScan`。命令行用法如下：

```
Usage: ./cudaScan [options]

Program Options:
  -m  --test <TYPE>      Run specified function on input.  Valid tests are: scan, find_repeats (default: scan)
  -i  --input <NAME>     Run test on given input type. Valid inputs are: ones, random (default: random)
  -n  --arraysize <INT>  Number of elements in arrays
  -t  --thrust           Use Thrust library implementation
  -?  --help             This message
```

#### 使用前缀和实现“寻找重复元素”

一旦你完成了 `exclusive_scan`，请在 `scan/scan.cu` 中实现 `find_repeats` 函数。这将涉及编写更多的设备代码，以及对 `exclusive_scan()` 的一次或多次调用。你的代码应将重复元素的列表写入提供的输出指针（在设备内存中），然后返回输出列表的大小。

在调用你的 `exclusive_scan` 实现时，请记住 `start` 数组的内容会被复制到 `output` 数组中。此外，传递给 `exclusive_scan` 的数组假定位于设备内存中。

**评分：** 我们将在随机输入数组上测试你的代码的正确性和性能。

作为参考，下面给出了一个扫描分数表，显示了在 K80 GPU 上简单 CUDA 实现的性能。要检查你的 `scan` 和 `find_repeats` 实现的正确性和性能分数，请分别运行 **`./checker.py scan`** 和 **`./checker.py find_repeats`**。这样做会产生一个如下所示的参考表；你的分数完全基于代码的性能。为了获得满分，你的代码性能必须在所提供的参考解决方案的 20% 以内。

```
-------------------------
Scan Score Table:
-------------------------
-------------------------------------------------------------------------
| Element Count   | Ref Time        | Student Time    | Score           |
-------------------------------------------------------------------------
| 1000000         | 0.766           | 0.143 (F)       | 0               |
| 10000000        | 8.876           | 0.165 (F)       | 0               |
| 20000000        | 17.537          | 0.157 (F)       | 0               |
| 40000000        | 34.754          | 0.139 (F)       | 0               |
-------------------------------------------------------------------------
|                                   | Total score:    | 0/5             |
-------------------------------------------------------------------------
```

这部分作业主要是为了让你更多地练习编写 CUDA 并进行数据并行思维，而不是为了优化代码性能。要在这部分获得满分，并不需要过多的性能调优（实际上根本不需要），只需将算法的伪代码直接移植到 CUDA 即可。但是，有一个技巧：scan 的一个朴素实现可能会为伪代码中每个并行循环的每次迭代启动 N 个 CUDA 线程，并使用内核中的条件执行来确定哪些线程实际需要工作。这种解决方案性能不佳！（考虑上扫阶段的最外层循环的最后一次迭代，只有两个线程会工作！）。一个满分的解决方案只会为最内层并行循环的每次迭代启动一个 CUDA 线程。

**测试工具：** 默认情况下，测试工具会在一个每次程序运行时都相同的伪随机生成数组上运行，以帮助调试。你可以传递参数 `-i random` 来运行随机数组——我们在评分时会这样做。我们鼓励你为程序提供其他输入来评估它。你也可以使用 `-n <size>` 选项来更改输入数组的长度。

参数 `--thrust` 将使用 [Thrust 库](http://thrust.github.io/) 的 [exclusive scan](https://docs.nvidia.com/cuda/archive/12.2.2/thrust/index.html?highlight=group%20prefix%20sums#prefix-sums) 实现。**对于能够创建与 Thrust 竞争的性能实现的任何人，可获得最多两分的额外加分。**

## 第 3 部分：一个简单的圆形渲染器（85 分）

现在到了真正的重头戏！

作业模板代码的 `/render` 目录包含一个渲染器的实现，该渲染器绘制彩色圆圈。构建代码，并使用以下命令行运行渲染器：`./render -r cpuref rgb`。程序将输出一个包含三个圆圈的图像 `output_0000.ppm`。现在使用命令行 `./render -r cpuref snow` 运行渲染器。输出的图像将会是飘雪。PPM 图像可以在 OSX 上通过预览直接查看。对于 Windows，你可能需要下载一个查看器。

注意：你也可以使用 `-i` 选项将渲染器输出发送到显示器而不是文件。（对于雪景的情况，你会看到飘雪的动画。）但是，要使用交互模式，你需要能够设置 X-windows 转发到你的本地机器。（[此参考](http://atechyblog.blogspot.com/2014/12/google-cloud-compute-x11-forwarding.html) 或 [此参考](https://stackoverflow.com/questions/25521486/x11-forwarding-from-debian-on-google-compute-engine) 可能会有所帮助。）

作业模板代码包含两个渲染器版本：一个顺序的、单线程的 C++ 参考实现，实现在 `refRenderer.cpp` 中；以及一个*不正确的*并行 CUDA 实现，实现在 `cudaRenderer.cu` 中。

### 渲染器概述

我们鼓励你通过检查 `refRenderer.cpp` 中的参考实现来熟悉渲染器代码库的结构。`setup` 方法在渲染第一帧之前被调用。在你的 CUDA 加速渲染器中，这个方法可能包含你所有的渲染器初始化代码（分配缓冲区等）。`render` 在每一帧被调用，负责将所有圆圈绘制到输出图像中。渲染器的另一个主要函数 `advanceAnimation` 也每帧调用一次。它更新圆圈的位置和速度。在本作业中，你不需要修改 `advanceAnimation`。

渲染器接收一个圆圈数组（3D 位置、速度、半径、颜色）作为输入。渲染每帧的基本顺序算法是：

    清除图像
    对于每个圆圈
        更新位置和速度
    对于每个圆圈
        计算屏幕包围盒
        对于包围盒内的所有像素
            计算像素中心点
            如果中心点在圆圈内
                计算该点圆圈的颜◉
                将圆圈对该像素的贡献混合到图像中

下图说明了使用点在圆内测试计算圆圈-像素覆盖的基本算法。请注意，只有当像素中心位于圆圈内时，圆圈才会为该输出像素贡献颜色。

![点圆测试](handout/point_in_circle.jpg?raw=true "一个计算圆圈对输出图像贡献的简单算法：所有在圆圈包围盒内的像素都进行覆盖测试。对于包围盒中的每个像素，如果其中心点（黑点）包含在圆圈内，则该像素被视为被圆圈覆盖。圆圈内的像素中心被涂成红色。圆圈对图像的贡献将仅针对被覆盖的像素计算。")

渲染器的一个重要细节是它渲染的是**半透明**圆圈。因此，任何像素的颜色都不是单个圆圈的颜色，而是重叠在该像素上的所有半透明圆圈贡献的混合结果（注意上面伪代码中的“混合贡献”部分）。渲染器通过 RGBA 四元组（红、绿、蓝、不透明度 alpha 值）来表示圆圈的颜色。Alpha = 1 对应于完全不透明的圆圈。Alpha = 0 对应于完全透明的圆圈。要在颜色为 `(P_r, P_g, P_b)` 的像素上绘制颜色为 `(C_r, C_g, C_b, C_alpha)` 的半透明圆圈，渲染器使用以下数学公式：

<pre>
   result_r = C_alpha * C_r + (1.0 - C_alpha) * P_r
   result_g = C_alpha * C_g + (1.0 - C_alpha) * P_g
   result_b = C_alpha * C_b + (1.0 - C_alpha) * P_b
</pre>

注意，合成是不可交换的（对象 X 在 Y 之上与对象 Y 在 X 之上看起来不一样），因此渲染器必须按照应用程序提供的顺序绘制圆圈。（你可以假设应用程序按深度顺序提供圆圈。）例如，考虑下面两幅图像，其中蓝色圆圈绘制在绿色圆圈之上，而绿色圆圈又绘制在红色圆圈之上。在左图中，圆圈以正确的顺序绘制到输出图像中。在右图中，圆圈以不同的顺序绘制，输出图像看起来不正确。

![顺序](handout/order.jpg?raw=true "渲染器必须小心地生成与按应用程序提供的顺序顺序绘制所有圆圈时生成的输出相同的输出。")

### CUDA 渲染器

在熟悉了参考代码中实现的圆圈渲染算法之后，现在学习一下 `cudaRenderer.cu` 中提供的 CUDA 渲染器实现。你可以使用 `--renderer cuda (或 -r cuda)` cuda 程序选项来运行 CUDA 渲染器。

提供的 CUDA 实现将计算并行化到所有输入圆圈上，为每个 CUDA 线程分配一个圆圈。虽然这个 CUDA 实现完成了圆圈渲染器数学的完整实现，但它包含几个你将在本作业中修复的主要错误。具体来说：当前实现不能确保图像更新是原子操作，并且它没有保持所需的图像更新顺序（顺序要求将在下面描述）。

### 渲染器要求

你的并行 CUDA 渲染器实现必须维持两个不变量，这些不变量在顺序实现中自然得到保证。

1.  **原子性：** 所有图像更新操作必须是原子的。临界区包括读取四个 32 位浮点值（像素的 RGBA 颜色），将当前圆圈的贡献与当前图像值混合，然后将像素的颜色写回内存。
2.  **顺序：** 你的渲染器必须按照*圆圈输入顺序*对图像像素执行更新。也就是说，如果圆圈 1 和圆圈 2 都对像素 P 有贡献，那么圆圈 1 对 P 的任何图像更新必须在圆圈 2 对 P 的更新之前应用到图像。如前所述，保持顺序要求可以实现透明圆圈的正确渲染。（对图形系统还有许多其他好处。如果你好奇，可以找 Kayvon 聊聊。）**一个关键观察结果是，顺序的定义仅指定了对同一像素的更新顺序。** 因此，如下所示，对于不对同一像素贡献的圆圈，没有顺序要求。这些圆圈可以独立处理。

![依赖关系](handout/dependencies.jpg?raw=true "圆圈 1、2 和 3 的贡献必须按照提供给渲染器的顺序应用于重叠的像素。")

由于提供的 CUDA 实现不满足这两个要求中的任何一个，不正确地尊重顺序或原子性的结果可以通过在 RGB 和圆圈场景上运行 CUDA 渲染器实现看到。你将在生成的图像中看到水平条纹，如下所示。这些条纹每帧都会变化。

![顺序错误](handout/bug_example.jpg?raw=true "由于帧缓冲区更新缺乏原子性导致的输出错误（注意图像底部的条纹）。")

### 你需要做什么

**你的工作是编写一个最快且正确的 CUDA 渲染器实现。** 你可以采用任何你认为合适的方法，但你的渲染器必须遵守上述原子性和顺序要求。不满足这两个要求的解决方案在第 3 部分中将最多获得 12 分。我们已经给了你这样一个解决方案！

一个好的起点是通读 `cudaRenderer.cu`，并确信它*不*满足正确性要求。特别地，查看 `CudaRenderer::render` 如何启动 CUDA 内核 `kernelRenderCircles`。（`kernelRenderCircles` 是所有工作发生的地方。）要直观地看到违反上述两个要求的效果，使用 `make` 编译程序。然后运行 `./render -r cuda rand10k`，它应该显示一个包含 10K 个圆圈的图像，显示在上图底部一行。将这个（不正确的）图像与通过运行 `./render -r cpuref rand10k` 由顺序代码生成的图像进行比较。

我们建议你：

1.  首先重写 CUDA 模板代码的实现，使其在并行运行时逻辑正确（我们建议一种不需要锁或同步的方法）。
2.  然后确定你的解决方案存在什么性能问题。
3.  此时，作业的真正思考开始了……（提示：`circleBoxTest.cu_inl` 中提供的圆与包围盒相交测试是你的朋友。我们鼓励你使用这些子程序。）

以下是 `./render` 的命令行选项：

```
Usage: ./render [options] scenename
Valid scenenames are: rgb, rgby, rand10k, rand100k, rand1M, biglittle, littlebig, pattern, micro2M,
                      bouncingballs, fireworks, hypnosis, snow, snowsingle
Program Options:
  -r  --renderer <cpuref/cuda>  Select renderer: ref or cuda (default=cuda)
  -s  --size  <INT>             Rendered image size: <INT>x<INT> pixels (default=1024)
  -b  --bench <START:END>       Run for frames [START,END) (default=[0,1))
  -c  --check                   Check correctness of CUDA output against CPU reference
  -i  --interactive             Render output to interactive display
  -f  --file  <FILENAME>        Output file name (FILENAME_xxxx.ppm) (default=output)
  -?  --help                    This message
```

**检查器代码：** 为了检测程序的正确性，`render` 有一个方便的 `--check` 选项。此选项会同时运行顺序版本的参考 CPU 渲染器和你的 CUDA 渲染器，然后比较生成的图像以确保正确性。还会打印你的 CUDA 渲染器实现所花费的时间。

我们总共提供了八个你将接受评分的圆圈数据集。然而，为了获得满分，你的代码必须通过我们所有的正确性测试。要检查代码的正确性和性能评分，请在 `/render` 目录中运行 **`./checker.py`**（注意 .py 扩展名）。如果你在模板代码上运行它，程序将打印一个如下所示的表格，以及我们整个测试集的结果：

```
Score table:
------------
--------------------------------------------------------------------------
| Scene Name      | Ref Time (T_ref) | Your Time (T)   | Score           |
--------------------------------------------------------------------------
| rgb             | 0.2622           | (F)             | 0               |
| rand10k         | 3.0658           | (F)             | 0               |
| rand100k        | 29.6144          | (F)             | 0               |
| pattern         | 0.4043           | (F)             | 0               |
| snowsingle      | 19.7155          | (F)             | 0               |
| biglittle       | 15.2422          | (F)             | 0               |
| rand1M          | 230.478          | (F)             | 0               |
| micro2M         | 439.9369         | (F)             | 0               |
--------------------------------------------------------------------------
|                                    | Total score:    | 0/72            |
--------------------------------------------------------------------------
```

注意：在某些运行中，*你可能会*在某些场景中获得分数，因为提供的渲染器的运行时是非确定性的，有时可能是正确的。但这并不能改变当前 CUDA 渲染器通常不正确的事实。

“参考时间”是参考解决方案在当前机器上的性能（在提供的 `render_ref` 可执行文件中）。“你的时间”是你当前 CUDA 渲染器解决方案的性能，其中 `(F)` 表示不正确的解决方案。你的成绩将取决于你的实现与这些参考实现相比的性能（见评分指南）。

除了你的代码外，我们希望你提交一份清晰的、高层次的实现工作原理描述，以及简要描述你是如何得到这个解决方案的。请具体说明你尝试过的方法，以及你是如何确定优化代码的方向的（例如，你进行了哪些测量来指导你的优化工作？）。

你应该在报告中提及的你的工作方面包括：

1.  在报告顶部包含两位合作伙伴的姓名和 SUNet ID。
2.  复制为你解决方案生成的分数表，并说明你在哪台机器上运行了代码。
3.  描述你是如何分解问题的，以及你如何将工作分配给 CUDA 线程块和线程（可能还有 warp）。
4.  描述你的解决方案中同步发生的位置。
5.  你采取了哪些步骤（如果有）来减少通信需求（例如，同步或主存带宽需求）？
6.  简要描述你是如何得出最终解决方案的。你尝试过哪些其他方法？它们有什么问题？

### 评分指南

-   作业的书面报告价值 18 分。
-   你的并行前缀实现价值 10 分。
-   你的渲染器实现价值 72 分。这些分数按以下方式平均分配到每个场景 9 分：
    -   每个场景 2 个正确性分数。我们只会测试你的程序在尺寸为 256 的倍数的图像上。
    -   每个场景 7 个性能分数（仅当解决方案正确时才能获得）。你的性能将相对于提供的基准参考渲染器 T<sub>ref</sub> 的性能进行评分：
        -   对于时间（T）是 T<sub>ref</sub> 的 10 倍的解决方案，不会获得任何性能分数。
        -   对于在优化解决方案的 20% 以内的解决方案（T <= 1.20 \* T<sub>ref</sub>），将获得全部性能分数。
        -   对于其他 T 值（对于 1.20 T<sub>ref</sub> < T < 10 _ T<sub>ref</sub>），你的性能分数在 1 到 7 的范围内将计算为：`7 _ T_ref / T`。

-   对于性能显著高于要求的解决方案，最多可获得 5 分额外加分（由教师自行决定）。你的报告必须清楚全面地解释你的方法。
-   对于高质量、仅使用 CPU 的并行渲染器实现，能够充分利用所有核心和核心的 SIMD 向量单元，最多可获得 5 分额外加分（由教师自行决定）。你可以自由使用任何可用工具（例如，SIMD 内联函数、ISPC、pthreads）。要获得加分，你应该分析基于 GPU 和 CPU 的解决方案的性能，并讨论实现选择差异的原因。

因此，该项目总分如下：

-   第 1 部分（5 分）
-   第 2 部分（10 分）
-   第 3 部分报告（13 分）
-   第 3 部分实现（72 分）
-   潜在的**额外**加分（最多 10 分）

## 作业技巧和提示

以下是前几年整理的一些技巧和提示。请注意，有多种方法可以实现你的渲染器，并非所有提示都适用于你的方法。

-   本作业有两个潜在的并行维度。一个是*像素级并行*，另一个是*圆圈级并行*（前提是对重叠的圆圈要尊重顺序要求）。解决方案需要利用这两种类型的并行性，可能是在计算的不同部分。
-   `circleBoxTest.cu_inl` 中提供的圆-包围盒相交测试是你的朋友。我们鼓励你使用这些子程序。
-   `exclusiveScan.cu_inl` 中提供的共享内存前缀和操作可能对本作业很有价值（并非所有解决方案都会选择使用它）。请参见[这里](https://docs.nvidia.com/cuda/archive/12.2.1/thrust/index.html#prefix-sums)关于前缀和的简单描述。我们在共享内存中提供了一个在**数组大小为 2 的幂**的独占前缀和实现。**提供的代码不适用于非 2 的幂输入，而且它还要求线程块中的线程数等于数组大小。请阅读代码中的注释。**
-   查看正在被调用的 `shadePixel` 方法。注意它为了更新像素的颜色执行了许多全局内存操作。在你的 `kernelRenderCircles` 方法中使用局部累加器可能更明智。然后你可以在一个寄存器中累加像素值，一旦最终像素值累加完成，你就只需要执行一次到全局内存的写入。
-   如果你愿意，可以在实现中使用 [Thrust 库](http://thrust.github.io/)。为了实现优化的 CUDA 参考实现的性能，并不需要使用 Thrust。有一种流行的解决问题的方法使用了我们提供给你的共享内存前缀和实现。另一种流行的方法使用了 Thrust 库中的前缀和例程。两种都是有效的解决策略。
-   渲染器中是否存在数据重用？可以做些什么来利用这种重用？
-   你将如何确保图像更新的原子性，因为没有 CUDA 语言原语可以原子地执行图像更新操作的逻辑？使用全局内存原子操作构造锁是一种解决方案，但请记住，即使你的图像更新是原子的，也必须按要求的顺序执行更新。**我们建议你首先在并行解决方案中考虑如何确保顺序，然后再考虑原子性问题（如果它仍然存在于你的解决方案中）。**
-   对于包含大量圆圈的测试——`rand1M` 和 `micro2M`——你应该小心在全局内存中分配临时结构。如果分配了太多全局内存，你将耗尽设备上的所有内存。如果你没有检查 `cudaMalloc` 调用返回的 `cudaError_t` 值，那么程序仍会执行，但你不会知道设备内存已经用完。相反，你将无法通过正确性检查，因为你无法创建你的临时结构。这就是为什么我们建议你使用下面的 CUDA API 调用包装器，这样你可以包装你的 `cudaMalloc` 调用，并在设备内存耗尽时产生错误。
-   如果你有空闲时间，可以尝试创建自己的场景，享受乐趣！

### 捕获 CUDA 错误

默认情况下，如果你访问数组越界、分配过多内存或其他错误，CUDA 通常不会通知你；相反，它会静默失败并返回一个错误代码。你可以使用以下宏（随意修改）来包装 CUDA 调用：

```
#define DEBUG

#ifdef DEBUG
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr, "CUDA Error: %s at %s:%d\n",
        cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#else
#define cudaCheckError(ans) ans
#endif
```

请注意，一旦你的代码正确，你可以取消定义 DEBUG 来禁用错误检查以提高性能。

然后你可以像这样包装 CUDA API 调用来处理它们返回的错误：

```
cudaCheckError( cudaMalloc(&a, size*sizeof(int)) );
```

请注意，你不能直接包装内核启动。相反，它们的错误将在你包装的下一个 CUDA 调用中被捕获：

```
kernel<<<1,1>>>(a); // 假设内核导致错误！
cudaCheckError( cudaDeviceSynchronize() ); // 错误将在这行打印
```

所有 CUDA API 函数，如 `cudaDeviceSynchronize`、`cudaMemcpy`、`cudaMemset` 等都可以被包装。

**重要：** 如果之前某个 CUDA 函数出错但未被捕获，该错误将出现在下一次错误检查中，即使那次检查包装的是不同的函数。例如：

```
...
line 742: cudaMalloc(&a, -1); // 执行后继续
line 743: cudaCheckError(cudaMemcpy(a,b)); // 打印 "CUDA Error: out of memory at cudaRenderer.cu:743"
...
```

因此，在调试期间，建议你包装**所有** CUDA API 调用（至少在你编写的代码中）。

（来源：改编自 [这个 Stack Overflow 帖子](https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api)）

## 3.4 提交说明

请使用 Gradescope 提交你的作业。如果你与合作伙伴一起工作，请记得在 Gradescope 上标记你的合作伙伴。

1.  **请将你的报告提交为文件 `writeup.pdf`。**
2.  **请运行 `sh create_submission.sh` 生成一个 zip 文件提交到 Gradescope。** 请注意，这将运行 make clean 清理你的代码目录，因此你需要再次运行 make 才能运行你的代码。如果脚本报错“Permission denied”，你应该运行 `chmod +x create_submission.sh`，然后重新运行脚本。

我们的评分脚本将重新运行检查器代码，使我们能够验证你的分数与你提交的 `writeup.pdf` 中的分数是否一致。我们也可能会尝试在其他数据集上运行你的代码，以进一步检查其正确性。