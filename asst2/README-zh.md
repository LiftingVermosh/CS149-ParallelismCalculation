# 作业2：从零构建一个任务执行库

**截止时间：10月16日（周四）晚上11:59**

**总分：100分**

## 概述 ##

每个人都喜欢快速完成任务，在这项作业中，我们要求你做到这一点！你将实现一个C++库，该库能够尽可能高效地在多核CPU上执行应用程序提供的任务。

在作业的第一部分，你将实现一个支持批量（数据并行）启动多个相同任务实例的任务执行库。此功能类似于你在作业1中用于跨核并行化代码的[ISPC任务启动行为](http://ispc.github.io/ispc.html#task-parallelism-launch-and-sync-statements)。

在作业的第二部分，你将扩展你的任务运行时系统，以执行更复杂的**任务图**，其中任务的执行可能依赖于其他任务产生的结果。这些依赖关系限制了你的任务调度系统能够安全并行运行的任务。在并行机器上调度数据并行任务图的执行是众多流行并行运行时系统的特性，范围从流行的[Thread Building Blocks](https://github.com/intel/tbb)库，到[Apache Spark](https://spark.apache.org/)，再到现代深度学习框架如[PyTorch](https://pytorch.org/)和[TensorFlow](https://www.tensorflow.org/)。

本作业将要求你：

* 使用线程池管理任务执行
* 使用同步原语（如互斥锁和条件变量）协调工作线程的执行
* 实现一个反映任务图定义依赖关系的任务调度器
* 理解工作负载特征，以做出高效的任务调度决策

我们建议你查看我们的[C++同步教程](tutorial/README.md)，以获取关于C++标准库中同步原语的更多信息。此外，查阅[测试用例描述](tests/)可能有助于理解你的库将支持的工作负载类型。

### 等等，我以前做过这个？ ###

你可能已经在CS107或CS111等课程中创建过线程池和任务执行库。然而，本作业提供了一个独特的机会来更好地理解这些系统。你将实现多个任务执行库，有的没有线程池，有的使用不同类型的线程池。通过实现多种任务调度策略并在不同工作负载上比较它们的性能，你将更好地理解创建并行系统时关键设计选择的影响。

## 环境设置 ##

**我们将使用Amazon AWS `c7g.4xlarge` 实例对本次作业进行评分 - 我们在此提供了虚拟机设置说明：[链接](https://github.com/stanford-cs149/asst2/blob/master/cloud_readme.md)。请确保你的代码在此虚拟机上能正常工作，因为我们将用它进行性能测试和评分。**

作业启动代码可在[Github](https://github.com/stanford-cs149/asst2)上找到。请从以下地址下载作业2的启动代码：

    https://github.com/stanford-cs149/asst2/archive/refs/heads/master.zip

**重要提示：** 不要修改提供的 `Makefile`。否则可能会破坏我们的评分脚本。

## 部分A：同步批量任务启动

在作业1中，你使用了ISPC的任务启动原语来启动N个ISPC任务实例（`launch[N] myISPCFunction()`）。在本作业的第一部分，你将实现在任务执行库中的类似功能。

首先，请熟悉 `itasksys.h` 中 `ITaskSystem` 的定义。这个[抽象类](https://www.tutorialspoint.com/cplusplus/cpp_interfaces.htm)定义了你的任务执行系统的接口。该接口包含一个 `run()` 方法，其签名如下：

    virtual void run(IRunnable* runnable, int num_total_tasks) = 0;

`run()` 执行指定任务的 `num_total_tasks` 个实例。由于这个单一函数调用会导致许多任务的执行，我们将每次对 `run()` 的调用称为一次**批量任务启动**。

`tasksys.cpp` 中的启动代码包含一个正确的、但串行的 `TaskSystemSerial::run()` 实现，作为任务系统如何使用 `IRunnable` 接口执行批量任务启动的示例。（`IRunnable` 的定义在 `itasksys.h` 中）。注意，在每次调用 `IRunnable::runTask()` 时，任务系统会向任务提供当前任务标识符（一个介于0和 `num_total_tasks` 之间的整数）以及批量任务启动中的任务总数。任务的实现将使用这些参数来确定任务应该做什么工作。

关于 `run()` 的一个重要细节是：它必须相对于调用线程**同步**执行任务。换句话说，当 `run()` 返回时，应用程序可以保证任务系统已经完成了批量任务启动中**所有任务**的执行。启动代码中提供的串行实现会在调用线程上执行所有任务，因此满足这一要求。

### 运行测试 ###

启动代码包含一组使用你的任务系统的测试应用程序。有关测试框架的描述，请参阅 `tests/README.md`；有关测试定义本身，请参阅 `tests/tests.h`。要运行测试，请使用 `runtasks` 脚本。例如，要运行名为 `mandelbrot_chunked` 的测试（该测试通过批量启动处理图像连续块的任务来计算曼德博分形图像），请输入：

```bash
./runtasks -n 16 mandelbrot_chunked
```

不同的测试具有不同的性能特征——有些每个任务的工作量很少，有些则需要进行大量的处理。有些测试每次启动会创建大量任务，有些则很少。有时启动中的所有任务具有相似的计算成本，而在另一些情况下，单个批量启动中任务的计算成本是可变的。我们在 `tests/README.md` 中描述了大部分测试，但我们建议你检查 `tests/tests.h` 中的代码，以更详细地理解所有测试的行为。

> [!TIP]
> 一个有助于在实现解决方案时调试正确性的测试是 `simple_test_sync`，这是一个非常小的测试，不应用来衡量性能，但足够小，可以通过打印语句或调试器进行调试。参见 `tests/tests.h` 中的 `simpleTest` 函数。

我们鼓励你创建自己的测试。请查看 `tests/tests.h` 中已有的测试以获得灵感。我们还提供了一个由 `class YourTask` 和函数 `yourTest()` 组成的骨架测试，你可以根据需要在此基础上进行构建。对于你创建的测试，请确保将它们添加到 `tests/main.cpp` 中的测试列表和测试名称中，并相应地调整变量 `n_tests`。请注意，虽然你可以使用自己的解决方案运行自己的测试，但无法编译参考解决方案来运行你的测试。

`-n` 命令行选项指定任务系统实现可以使用的最大线程数。在上面的示例中，我们选择了 `-n 16`，因为AWS实例中的CPU具有16个执行上下文。可运行测试的完整列表可通过命令行帮助（`-h` 命令行选项）获得。

`-i` 命令行选项指定在性能测量期间运行测试的次数。为了获得准确的性能测量，`./runtasks` 会多次运行测试，并记录多次运行中的**最小**运行时间；通常默认值就足够了——更大的值可能会产生更精确的测量，但会增加测试运行时间。

此外，我们还提供了用于评分性能的测试框架：

```bash
>>> python3 ../tests/run_test_harness.py
```

该框架具有以下命令行参数：

```bash
>>> python3 run_test_harness.py -h
usage: run_test_harness.py [-h] [-n NUM_THREADS]
                           [-t TEST_NAMES [TEST_NAMES ...]] [-a]

运行任务系统性能测试

optional arguments:
  -h, --help            # 显示此帮助信息并退出
  -n NUM_THREADS, --num_threads NUM_THREADS
                        # 任务系统可以使用的最大线程数（默认为16）
  -t TEST_NAMES [TEST_NAMES ...], --test_names TEST_NAMES [TEST_NAMES ...]
                        # 要运行的测试列表
  -a, --run_async       # 运行异步测试
```

它会产生一份详细的性能报告，如下所示：

```bash
>>> python3 ../tests/run_test_harness.py -t super_light super_super_light
python3 ../tests/run_test_harness.py -t super_light super_super_light
================================================================================
运行任务系统评分框架...（共2个测试）
  - 检测到CPU具有16个执行上下文
  - 任务系统配置为最多使用16个线程
================================================================================
================================================================================
执行测试: super_super_light...
参考二进制文件: ./runtasks_ref_linux
结果: super_super_light
                                        学生    参考    PERF?
[Serial]                                9.053     9.022       1.00  (OK)
[Parallel + Always Spawn]               8.982     33.953      0.26  (OK)
[Parallel + Thread Pool + Spin]         8.942     12.095      0.74  (OK)
[Parallel + Thread Pool + Sleep]        8.97      8.849       1.01  (OK)
================================================================================
执行测试: super_light...
参考二进制文件: ./runtasks_ref_linux
结果: super_light
                                        学生    参考    PERF?
[Serial]                                68.525    68.03       1.01  (OK)
[Parallel + Always Spawn]               68.178    40.677      1.68  (NOT OK)
[Parallel + Thread Pool + Spin]         67.676    25.244      2.68  (NOT OK)
[Parallel + Thread Pool + Sleep]        68.464    20.588      3.33  (NOT OK)
================================================================================
总体性能结果
[Serial]                                : 所有性能测试通过
[Parallel + Always Spawn]               : 性能测试未全部通过
[Parallel + Thread Pool + Spin]         : 性能测试未全部通过
[Parallel + Thread Pool + Sleep]        : 性能测试未全部通过
```

在上面的输出中，`PERF` 是你的实现运行时间与参考解决方案运行时间的比率。因此，小于1的值表示你的任务系统实现比参考实现更快。

> [!TIP]
> Mac用户：虽然我们提供了部分A和部分B的参考解决方案二进制文件，但我们将使用Linux二进制文件来测试你的代码。因此，我们建议你在提交之前在AWS实例中检查你的实现。如果你使用的是配备M1芯片的新型Mac，本地测试时请使用 `runtasks_ref_osx_arm` 二进制文件，否则使用 `runtasks_ref_osx_x86` 二进制文件。

> [!IMPORTANT]
我们将使用AWS上的 `runtasks_ref_linux_arm` 版本的参考解决方案对你的解决方案进行评分。请确保你的解决方案在AWS ARM实例上能正确工作。

### 你需要做的事情 ###

你的工作是实现一个能够高效利用多核CPU的任务执行引擎。你的评分将基于实现的正确性（必须正确运行所有任务）以及性能。这将是一个有趣的编程挑战，但也是一项不简单的工作。为了帮助你保持在正确的轨道上，完成部分A，我们将让你多次实现任务系统，逐步增加实现的复杂性和性能。你的三个实现将在 `tasksys.cpp/.h` 中定义的类中完成：

* `TaskSystemParallelSpawn`
* `TaskSystemParallelThreadPoolSpinning`
* `TaskSystemParallelThreadPoolSleeping`

__请将部分A的实现放在 `part_a/` 子目录中，以便与正确的参考实现（`part_a/runtasks_ref_*`）进行比较。__

_专业提示：请注意下面的说明采用的思路是“先尝试最简单的改进”。每一步都增加了任务执行系统实现的复杂性，但沿着每一步你应该拥有一个工作（完全正确）的任务运行时系统。_

我们还期望你至少创建一个测试，可以测试正确性或性能。有关更多信息，请参阅上面的“运行测试”部分。

#### 步骤1：转向并行任务系统 ####

__在此步骤中，请实现类 `TaskSystemParallelSpawn`。__

启动代码为你提供了一个在 `TaskSystemSerial` 中可用的串行任务系统实现。在此步骤中，你将扩展启动代码以并行执行批量任务启动。

* 你将需要创建额外的控制线程来执行批量任务启动的工作。注意，`TaskSystem` 的构造函数提供了一个参数 `num_threads`，这是你的实现可以用于运行任务的**工作线程最大数量**。

* 本着“先做最简单的事情”的精神，我们建议你在 `run()` 开始时生成工作线程，并在 `run()` 返回之前从主线程 join 这些线程。这将是一个正确的实现，但会因频繁创建线程而产生显著的额外开销。

* 如何将任务分配给工作线程？你应该考虑静态还是动态分配任务给线程？

* 是否存在需要保护以免多个线程同时访问的共享变量（任务执行系统的内部状态）？

#### 步骤2：使用线程池避免频繁创建线程 ####

__在此步骤中，请实现类 `TaskSystemParallelThreadPoolSpinning`。__

你在步骤1中的实现会因为在每次调用 `run()` 时创建线程而产生额外开销。这种开销在任务计算成本较低时尤为明显。此时，我们建议你转向“线程池”实现，让任务执行系统预先创建所有工作线程（例如，在 `TaskSystem` 构造期间，或在第一次调用 `run()` 时）。

* 作为初始实现，我们建议你让工作线程持续循环，始终检查是否有更多工作可以执行。（线程进入 while 循环直到条件为真通常被称为“自旋”。）工作线程如何确定有工作可做？

* 现在确保 `run()` 实现所需的同步行为并非易事。你需要如何修改 `run()` 的实现，以确定批量任务启动中的所有任务都已完成？

#### 步骤3：在没有任务时将线程置于休眠 ####

__在此步骤中，请实现类 `TaskSystemParallelThreadPoolSleeping`。__

步骤2实现的一个缺点是线程在“自旋”等待任务时占用了CPU核心的执行资源。例如，工作线程可能会循环等待新任务到达。再比如，主线程可能会循环等待工作线程完成所有任务，以便从 `run()` 调用返回。这会损害性能，因为CPU资源被用于运行这些线程，即使线程没有做有用工作。

在此部分作业中，我们希望你将线程置于休眠状态，直到它们等待的条件得到满足，从而提高任务系统的效率。

* 你的实现可以选择使用条件变量来实现此行为。条件变量是一种同步原语，允许线程在等待某个条件成立时休眠（并且不占用CPU处理资源）。其他线程“唤醒”等待的线程，让它们检查所等待的条件是否已满足。例如，如果没有工作可做，你可以让工作线程休眠（这样它们就不会从试图做有用工作的线程那里夺走CPU资源）。再比如，调用 `run()` 的主应用程序线程可能希望在等待工作线程完成批量任务启动中的所有任务时休眠（否则，自旋的主线程会夺走工作线程的CPU资源！）

* 你在本部分作业中的实现可能涉及棘手的竞态条件。你需要考虑线程行为的许多可能交错。

* 你可能需要考虑编写额外的测试用例来练习你的系统。__作业启动代码包含评分脚本将用于评分代码性能的工作负载，但我们也会使用启动代码中未提供的更广泛的工作负载来测试你实现的正确性！__

## 部分B：支持任务图的执行

在作业的第二部分，你将扩展部分A的任务系统实现，以支持异步启动可能依赖于先前任务的任务。这些任务间的依赖关系创建了你的任务执行库必须遵守的调度约束。

`ITaskSystem` 接口有一个额外的方法：

    virtual TaskID runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                    const std::vector<TaskID>& deps) = 0;

`runAsyncWithDeps()` 与 `run()` 类似，也用于执行 `num_total_tasks` 任务的批量启动。然而，它在多个方面与 `run()` 不同...

#### 异步任务启动 ####

首先，使用 `runAsyncWithDeps()` 创建的任务由任务系统**异步**（相对于调用线程）执行。这意味着 `runAsyncWithDeps()` 应立即返回给调用者，即使任务尚未完成执行。该方法返回与此批量任务启动关联的唯一标识符。

调用线程可以通过调用 `sync()` 来确定批量任务启动实际已完成的时间。

    virtual void sync() = 0;

`sync()` **仅当所有之前批量任务启动关联的任务都已完成时才**返回给调用者。例如，考虑以下代码：

    // 假设 taskA 和 taskB 是 IRunnable 的有效实例...

    std::vector<TaskID> noDeps;  // 空向量

    ITaskSystem *t = new TaskSystem(num_threads);

    // 批量启动4个任务
    TaskID launchA = t->runAsyncWithDeps(taskA, 4, noDeps);

    // 批量启动8个任务
    TaskID launchB = t->runAsyncWithDeps(taskB, 8, noDeps);

    // 此时，与 launchA 和 launchB 关联的任务可能仍在运行

    t->sync();

    // 此时，与 launchA 和 launchB 关联的所有12个任务都保证已终止

如上述注释所述，调用线程不能保证之前调用 `runAsyncWithDeps()` 的任务已完成，直到线程调用 `sync()`。准确地说，`runAsyncWithDeps()` 告诉你的任务系统执行一个新的批量任务启动，但你的实现可以在下一次调用 `sync()` 之前的任何时间灵活地执行这些任务。注意，这个规范意味着不能保证你的实现会在开始执行launchB的任务之前先执行launchA的任务！

#### 支持显式依赖关系 ####

`runAsyncWithDeps()` 的第二个关键细节是其第三个参数：一个 `TaskID` 标识符向量，这些标识符必须引用先前使用 `runAsyncWithDeps()` 进行的批量任务启动。这个向量指定了当前批量任务启动中的任务依赖于哪些先前任务。__因此，只有当依赖向量中所有启动的任务都完成时，你的任务运行时才能开始执行当前批量任务启动中的任何任务！__ 例如，考虑以下示例：

```cpp
    std::vector<TaskID> noDeps;  // 空向量
    std::vector<TaskID> depOnA;
    std::vector<TaskID> depOnBC;

    ITaskSystem *t = new TaskSystem(num_threads);

    TaskID launchA = t->runAsyncWithDeps(taskA, 128, noDeps);
    depOnA.push_back(launchA);

    TaskID launchB = t->runAsyncWithDeps(taskB, 2, depOnA);
    TaskID launchC = t->runAsyncWithDeps(taskC, 6, depOnA);
    depOnBC.push_back(launchB);
    depOnBC.push_back(launchC);

    TaskID launchD = t->runAsyncWithDeps(taskD, 32, depOnBC);
    t->sync();
```

上面的代码中包含四次批量任务启动（taskA：128个任务，taskB：2个任务，taskC：6个任务，taskD：32个任务）。注意，taskB和taskC的启动依赖于taskA。taskD的批量启动（`launchD`）依赖于 `launchB` 和 `launchC` 的结果。因此，虽然你的任务运行时可以按任意顺序（包括并行）处理与 `launchB` 和 `launchC` 关联的任务，但这些启动中的所有任务必须在 `launchA` 的任务完成后才能开始执行，并且它们必须完成，然后你的运行时才能开始执行 `launchD` 中的任何任务。

我们可以将这些依赖关系可视化为一个**任务图**。任务图是一个有向无环图（DAG），图中的节点对应批量任务启动，从节点X到节点Y的边表示Y依赖于X的输出。上述代码的任务图为：

<p align="center">
    <img src="figs/task_graph.png" width=400>
</p>

注意，如果你在具有八个执行上下文的Myth机器上运行上述示例，能够并行调度来自 `launchB` 和 `launchC` 的任务可能非常有用，因为单独每个批量任务启动都不足以充分利用机器的所有执行资源。

### 测试 ###
所有后缀为 `Async` 的测试应用于测试部分B。评分框架中包含的测试子集在 `tests/README.md` 中描述，所有测试可以在 `tests/tests.h` 中找到，并在 `tests/main.cpp` 中列出。为了调试正确性，我们提供了一个小型测试 `simple_test_async`。查看 `tests/tests.h` 中的 `simpleTest` 函数。`simple_test_async` 应该足够小，可以使用 `simpleTest` 中的打印语句或断点进行调试。

我们鼓励你创建自己的测试。请查看 `tests/tests.h` 中已有的测试以获得灵感。我们还提供了一个由 `class YourTask` 和函数 `yourTest()` 组成的骨架测试，你可以根据需要在此基础上进行构建。对于你创建的测试，请确保将它们添加到 `tests/main.cpp` 中的测试列表和测试名称中，并相应地调整变量 `n_tests`。请注意，虽然你可以使用自己的解决方案运行自己的测试，但无法编译参考解决方案来运行你的测试。

### 你需要做的事情 ###

你必须扩展在部分A中实现的、使用线程池（并休眠）的任务系统，以正确实现 `TaskSystemParallelThreadPoolSleeping::runAsyncWithDeps()` 和 `TaskSystemParallelThreadPoolSleeping::sync()`。我们还期望你至少创建一个测试，可以测试正确性或性能。有关更多信息，请参阅上面的“测试”部分。需要澄清的是，你*需要*在报告中描述你自己的测试，但我们的自动评分器*不会*测试你的测试。

**你不需要在部分B中实现其他 `TaskSystem` 类。**

与部分A一样，我们为你提供以下入门提示：

* 将 `runAsyncWithDeps()` 的行为视为将对应批量任务启动的记录（或可能对应批量任务启动中每个任务的记录）推送到“工作队列”上可能会有所帮助。一旦工作记录进入队列，`runAsyncWithDeps()` 就可以返回给调用者。

* 本部分作业的诀窍在于执行适当的簿记来跟踪依赖关系。当批量任务启动中的所有任务完成时，必须执行什么操作？（这是新任务可能变得可运行的时间点。）

* 在你的实现中使用两个数据结构可能会有所帮助：(1) 一个结构，表示已通过调用 `runAsyncWithDeps()` 添加到系统中但尚未准备好执行的任务（因为它们依赖于仍在运行的任务——这些任务“等待”其他任务完成）；(2) 一个“就绪队列”，其中包含不等待任何先前任务完成的任务，并且一旦工作线程可用就可以安全地运行。

* 在生成唯一的任务启动ID时，你无需担心整数回绕。我们不会向你的任务系统发出超过2^31次批量任务启动。

* 你可以假定所有程序要么只调用 `run()`，要么只调用 `runAsyncWithDeps()`；也就是说，你不需要处理 `run()` 调用需要等待所有之前的 `runAsyncWithDeps()` 调用完成的情况。注意，这个假设意味着你可以通过适当调用 `runAsyncWithDeps()` 和 `sync()` 来实现 `run()`。

* 你可以假定唯一的并发行为是由你的实现创建/使用的多个线程。也就是说，我们不会从那些线程外部再生成额外的线程并调用你的实现。

__请将部分B的实现放在 `part_b/` 子目录中，以便与正确的参考实现（`part_b/runtasks_ref_*`）进行比较。__

## 评分 ##

本次作业的分数分配如下：

**部分A（50分）**
- 5分用于 `TaskSystemParallelSpawn::run()` 的正确性 + 5分用于其性能。（共10分）
- 每个 `TaskSystemParallelThreadPoolSpinning::run()` 和 `TaskSystemParallelThreadPoolSleeping::run()` 的正确性各10分 + 这些方法的性能各10分。（共40分）

**部分B（40分）**
- 30分用于 `TaskSystemParallelThreadPoolSleeping::runAsyncWithDeps()`、`TaskSystemParallelThreadPoolSleeping::run()` 和 `TaskSystemParallelThreadPoolSleeping::sync()` 的正确性
- 10分用于 `TaskSystemParallelThreadPoolSleeping::runAsyncWithDeps()`、`TaskSystemParallelThreadPoolSleeping::run()` 和 `TaskSystemParallelThreadPoolSleeping::sync()` 的性能。对于部分B，你可以忽略 `Parallel + Always Spawn` 和 `Parallel + Thread Pool + Spin` 的结果。也就是说，你只需要在每个测试用例中通过 `Parallel + Thread Pool + Sleep`。

**报告（10分）**
- 请参阅“提交”部分获取更多详情。

对于每个测试，如果实现运行时间在提供的参考实现的20%（部分A）和50%（部分B）以内，将获得全部性能分数。性能分数仅授予返回正确结果的实现。如前所述，我们还可能使用启动代码中未提供的更广泛的工作负载来测试你实现的**正确性**。

## 提交 ##

请使用[Gradescope](https://www.gradescope.com/)提交你的作业。你的提交应包含你的任务系统代码以及一份描述你实现的报告。我们期望提交中包含以下五个文件：

 * part_a/tasksys.cpp
 * part_a/tasksys.h
 * part_b/tasksys.cpp
 * part_b/tasksys.h
 * 你的报告PDF（提交到gradescope的报告作业）

#### 代码提交 ####

我们要求你将源文件 `part_a/tasksys.cpp|.h` 和 `part_b/tasksys.cpp|.h` 放入一个压缩文件中。你可以创建一个目录（例如命名为 `asst2_submission`），其中包含子目录 `part_a` 和 `part_b`，将相关文件放入，然后通过运行 `tar -czvf asst2.tar.gz asst2_submission` 压缩该目录，并上传。请将**压缩文件** `asst2.tar.gz` 提交到Gradescope上的 *Assignment 2 (Code)* 作业。

在提交源文件之前，请确保所有代码都可编译和运行！我们应该能够将这些文件放入一个干净的启动代码树中，输入 `make`，然后无需手动干预即可执行你的程序。

我们的评分脚本将运行启动代码中提供的检查器代码来确定性能分数。_我们还将在启动代码中未提供的其他应用程序上运行你的代码，以进一步测试其正确性！_ 评分脚本将在作业截止后运行。

#### 报告提交 ####

请向Gradescope上的 *Assignment 2 (Write-up)* 作业提交一份简短的报告，涉及以下内容：

 1. 描述你的任务系统实现（一页即可）。除了总体工作原理的描述外，请确保回答以下问题：
  * 你如何决定管理线程？（例如，你是否实现了线程池？）
  * 你的系统如何将任务分配给工作线程？你使用了静态还是动态分配？
  * 在部分B中，你如何跟踪依赖关系以确保任务图的正确执行？

 2. 在部分A中，你可能已经注意到，更简单的任务系统实现（例如，完全串行实现，或每次启动都生成线程的实现）在某些情况下性能与更高级的实现一样好，甚至更好。请解释为什么会这样，并引用某些测试作为例子。例如，在什么情况下串行任务系统实现表现最好？为什么？在什么情况下每次启动都生成线程的实现与使用线程池的更高级并行实现性能相当？什么时候不行？

 3. 描述你为本作业实现的一个测试。该测试做什么？它旨在检查什么？你如何验证你对作业的解决方案在你的测试中表现良好？你添加的测试结果是否导致你更改了作业实现？