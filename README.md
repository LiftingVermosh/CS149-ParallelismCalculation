# CS149 — Parallel Computing

> Stanford CS149 (Fall 2024): Parallel Computing — 个人实现仓库

本仓库包含 Stanford CS149 课程全部 5 个编程作业的个人实现代码。课程涵盖从多核 CPU 并行到 GPU 编程的完整并行计算技术栈。

---

## 作业概览

| 作业 | 主题 | 关键技术 |
| ------ | ------ | ---------- |
| **asst1** | 多核并行与 SIMD 向量化 | pthreads, ISPC, SIMD intrinsics, 性能分析 |
| **asst2** | 任务并行调度系统 | C++ 线程池, 任务图, work stealing |
| **asst3** | CUDA 大规模并行渲染 | CUDA C++, GPU 架构, 内存层次优化 |
| **asst4** | 云端 AI 加速器训练 | AWS Trainium, Neuron SDK, 卷积优化 |
| **asst5** | GPU Kernel 编程进阶 | CUDA kernel, FlashAttention, Triton, 算子实现 |

---

## 目录结构

```txt
assignment/
├── asst1/                  # 多核并行 & SIMD
│   ├── prog1_mandelbrot_threads/   # pthread 多线程 Mandelbrot
│   ├── prog2_vecintrin/            # SIMD 向量化指令
│   ├── prog3_mandelbrot_ispc/      # ISPC Mandelbrot
│   ├── prog4_sqrt/                 # ISPC 向量化 sqrt
│   ├── prog5_saxpy/                # ISPC SAXPY 带宽测试
│   └── prog6_kmeans/               # K-Means 聚类并行化
│
├── asst2/                  # 任务并行调度
│   ├── part_a/                     # 基础任务调度器
│   ├── part_b/                     # 高级任务调度器
│   ├── tests/                      # 测试框架
│   └── tutorial/                   # 入门教程
│
├── asst3/                  # CUDA 渲染
│   ├── render/                     # CUDA 渲染器核心
│   └── handout/                    # 作业文档 & 图示
│
├── asst4/                  # AWS Trainium / Neuron
│   ├── part1/                      # 基础 kernel 算子
│   └── part2/                      # Conv2D 卷积实现
│
└── asst5/                  # GPU Kernel 编程
    ├── problems/                   # 各算子实现题目
    │   ├── 1d-occupancy-decoder/
    │   ├── flashattention/
    │   ├── histogram/
    │   ├── rk4/
    │   └── swiglu/
    ├── src/                        # Rust CLI 工具源码
    └── binary/                     # 预编译 CLI 工具
```

---

## 环境要求

- **C/C++**: GCC / Clang（asst1, asst2）
- **ISPC**: [Intel ISPC Compiler](https://ispc.github.io/)（asst1 prog3–prog5）
- **CUDA**: NVIDIA GPU + CUDA Toolkit（asst3, asst5）
- **Python 3.8+**: asst4（Neuron SDK）, asst5 测试框架
- **Rust**: asst5 CLI 工具（可选，可使用预编译 binary）

---

## 构建与运行

### asst1 — 多核并行与 SIMD

```bash
# 各子程序独立构建
cd asst1/prog1_mandelbrot_threads && make
cd asst1/prog2_vecintrin && make
cd asst1/prog3_mandelbrot_ispc && make   # 需要 ISPC 编译器
cd asst1/prog4_sqrt && make              # 需要 ISPC 编译器
cd asst1/prog5_saxpy && make             # 需要 ISPC 编译器
cd asst1/prog6_kmeans && make
```

### asst2 — 任务调度器

```bash
cd asst2/part_a && make    # 基础版本
cd asst2/part_b && make    # 高级版本
```

### asst3 — CUDA 渲染器

```bash
cd asst3/render
# 需要 NVIDIA GPU + CUDA Toolkit
nvcc -O3 -o cudaRenderer cudaRenderer.cu *.cpp
```

### asst4 — AWS Trainium

```bash
# 在 AWS Trainium 实例上运行（参考 asst4/cloud_readme.md）
cd asst4/part1 && python run_benchmark.py
cd asst4/part2 && python test_harness.py
```

### asst5 — GPU Kernel 编程

```bash
# 安装 CLI 工具
cd asst5 && bash install.sh    # Linux/macOS
# 或 Windows: .\install.ps1

# 运行各题目
python problems/eval.py --problem flashattention
python problems/eval.py --problem histogram
python problems/eval.py --problem rk4
python problems/eval.py --problem swiglu
python problems/eval.py --problem 1d-occupancy-decoder
```

---

## 笔记

- 每道题的详细思路和性能分析见各子目录下的 `README.md`
- 原始课程资料：[CS149 课程官网](https://cs149.stanford.edu/)
- 本仓库仅包含个人实现部分，不含课程提供的骨架代码之外的原始分发文件

---

## 许可

本仓库为个人学习用途，课程原始材料版权归 Stanford University 所有。
