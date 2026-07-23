#include "tasksys.h"

IRunnable::~IRunnable() {}

ITaskSystem::ITaskSystem(int num_threads) {}
ITaskSystem::~ITaskSystem() {}

/*
 * ================================================================
 * Serial task system implementation
 * ================================================================
 */

const char* TaskSystemSerial::name() {
    return "Serial";
}

TaskSystemSerial::TaskSystemSerial(int num_threads): ITaskSystem(num_threads) {
}

TaskSystemSerial::~TaskSystemSerial() {}

void TaskSystemSerial::run(IRunnable* runnable, int num_total_tasks) {
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }
}

TaskID TaskSystemSerial::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                          const std::vector<TaskID>& deps) {
    // You do not need to implement this method.
    return 0;
}

void TaskSystemSerial::sync() {
    // You do not need to implement this method.
    return;
}

/*
 * ================================================================
 * Parallel Task System Implementation
 * ================================================================
 */

const char* TaskSystemParallelSpawn::name() {
    return "Parallel + Always Spawn";
}

TaskSystemParallelSpawn::TaskSystemParallelSpawn(int num_threads): ITaskSystem(num_threads) {
    //
    // TODO: CS149 student implementations may decide to perform setup
    // operations (such as thread pool construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //
    this->num_threads = num_threads;
}

TaskSystemParallelSpawn::~TaskSystemParallelSpawn() {}

void TaskSystemParallelSpawn::run(IRunnable* runnable, int num_total_tasks) {


    //
    // TODO: CS149 students will modify the implementation of this
    // method in Part A.  The implementation provided below runs all
    // tasks sequentially on the calling thread.
    //

    // Init
    int call_max_thread_nums = std::min(this->num_threads, num_total_tasks);
    std::vector<std::thread> workers;

    // Allocate
    for(int i = 0; i < call_max_thread_nums; ++i) {
        
        // /* 分块 */ 
        // workers.emplace_back([&, i, call_max_thread_nums]() {
        //     int tasks_per_thread = num_total_tasks / call_max_thread_nums;  // Block Size
        //     int start = i * tasks_per_thread;                               // Start idx
        //     int end = (tasks_per_thread * (i + 1)) >= num_total_tasks ? 
        //                 num_total_tasks : tasks_per_thread * (i + 1);       // End idx 
        //     for(int j = start; j < end; ++j) {
        //         runnable->runTask(j, num_total_tasks);
        //     }
        // });

        /* 跨步 */
        workers.emplace_back([&, i, call_max_thread_nums]() {
            for(int j = i; j < num_total_tasks; j += call_max_thread_nums) {
                runnable->runTask(j, num_total_tasks);
            }
        });
    }

    for (auto& t : workers) {   // std::thread& t : workers
        t.join();
    }
}

TaskID TaskSystemParallelSpawn::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                 const std::vector<TaskID>& deps) {
    // You do not need to implement this method.
    return 0;
}

void TaskSystemParallelSpawn::sync() {
    // You do not need to implement this method.
    return;
}

/*
 * ================================================================
 * Parallel Thread Pool Spinning Task System Implementation
 * ================================================================
 */

const char* TaskSystemParallelThreadPoolSpinning::name() {
    return "Parallel + Thread Pool + Spin";
}

TaskSystemParallelThreadPoolSpinning::TaskSystemParallelThreadPoolSpinning(int num_threads): ITaskSystem(num_threads) {
    //
    // TODO: CS149 student implementations may decide to perform setup
    // operations (such as thread pool construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //


    this->killed = false;
    this->state.runnable = nullptr;
    this->state.next_task_id = 0;
    this->state.completed_tasks = 0;

    for (int i = 0; i < num_threads; i++) {
        workers.emplace_back([this]() {
            while (!this->killed) {
                IRunnable* r = this->state.runnable;
                // 当 runnable 不为空且还有任务时才抢
                if (r != nullptr) {
                    int task_id = this->state.next_task_id.fetch_add(1);
                    if (task_id < this->state.num_total_tasks) {
                        // 执行
                        r->runTask(task_id, this->state.num_total_tasks);
                        // 标记完成
                        this->state.completed_tasks.fetch_add(1);
                    }
                    // 没领到活：循环直到下一次 run() 
                }
            }
        });
    }
}

TaskSystemParallelThreadPoolSpinning::~TaskSystemParallelThreadPoolSpinning() {
    this->killed.store(true);
    for(auto& t : this->workers) {
        t.join();
    }
}

void TaskSystemParallelThreadPoolSpinning::run(IRunnable* runnable, int num_total_tasks) {


    //
    // TODO: CS149 students will modify the implementation of this
    // method in Part A.  The implementation provided below runs all
    // tasks sequentially on the calling thread.
    //

    this->state.runnable = runnable;
    this->state.num_total_tasks = num_total_tasks;
    this->state.completed_tasks = 0;
    
    // 重置 next_task_id
    this->state.next_task_id = 0;
    // 主线程自旋等待
    // while (this->state.completed_tasks.load() < num_total_tasks) {
    //     // 方案一 - 啥也不干
    //     // (.  , .)

    //     // 方案二 - 帮着干
    //     int task_id = this->state.next_task_id.fetch_add(1);
    //     if (task_id < this->state.num_total_tasks) {
    //         // 执行
    //         this->state.runnable->runTask(task_id, this->state.num_total_tasks);
    //         // 标记完成
    //         this->state.completed_tasks.fetch_add(1);
    //     }
    // }

    // 优化：避免原子变量竞争
    while (true) {
        int task_id = this->state.next_task_id.fetch_add(1);
        if (task_id >= num_total_tasks) break; // 没活了，立刻退出这个抢活循环
        runnable->runTask(task_id, num_total_tasks);
        this->state.completed_tasks.fetch_add(1);
    }
    // 没活，咬打火机
    while (this->state.completed_tasks.load() < num_total_tasks) {
        // (. ; .)
    }

    // // 再优化：先读后抢
    // while (this->state.completed_tasks.load() < num_total_tasks) {
    //     // 先做一次读取检查
    //     if (this->state.next_task_id.load() < num_total_tasks) {
    //         // 还有活就尝试执行原子抢夺
    //         int task_id = this->state.next_task_id.fetch_add(1);
    //         if (task_id < num_total_tasks) {
    //             runnable->runTask(task_id, num_total_tasks);
    //             this->state.completed_tasks.fetch_add(1);
    //         }
    //     }
    //     // 如果任务领完了就 load 自旋
    // }

    this->state.runnable = nullptr; 
}

TaskID TaskSystemParallelThreadPoolSpinning::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                              const std::vector<TaskID>& deps) {
    // You do not need to implement this method.
    return 0;
}

void TaskSystemParallelThreadPoolSpinning::sync() {
    // You do not need to implement this method.
    return;
}

/*
 * ================================================================
 * Parallel Thread Pool Sleeping Task System Implementation
 * ================================================================
 */

const char* TaskSystemParallelThreadPoolSleeping::name() {
    return "Parallel + Thread Pool + Sleep";
}

TaskSystemParallelThreadPoolSleeping::TaskSystemParallelThreadPoolSleeping(int num_threads): ITaskSystem(num_threads) {
    //
    // TODO: CS149 student implementations may decide to perform setup
    // operations (such as thread pool construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //
}

TaskSystemParallelThreadPoolSleeping::~TaskSystemParallelThreadPoolSleeping() {
    //
    // TODO: CS149 student implementations may decide to perform cleanup
    // operations (such as thread pool shutdown construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //
}

void TaskSystemParallelThreadPoolSleeping::run(IRunnable* runnable, int num_total_tasks) {


    //
    // TODO: CS149 students will modify the implementation of this
    // method in Parts A and B.  The implementation provided below runs all
    // tasks sequentially on the calling thread.
    //

    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }
}

TaskID TaskSystemParallelThreadPoolSleeping::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                    const std::vector<TaskID>& deps) {


    //
    // TODO: CS149 students will implement this method in Part B.
    //

    return 0;
}

void TaskSystemParallelThreadPoolSleeping::sync() {

    //
    // TODO: CS149 students will modify the implementation of this method in Part B.
    //

    return;
}
