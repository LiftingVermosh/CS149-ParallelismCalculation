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


    this->killed.store(false);
    this->active_workers.store(0);
    this->state.runnable.store(nullptr);
    this->state.next_task_id.store(0);
    this->state.completed_tasks.store(0);

    for (int i = 0; i < num_threads; i++) {
        workers.emplace_back([this]() {
            while (!this->killed.load()) {
                IRunnable* r = this->state.runnable.load(std::memory_order_acquire);
                // 当 runnable 不为空且还有任务时才抢
                if (r != nullptr) {
                    this->active_workers.fetch_add(1, std::memory_order_acq_rel);
                    if (this->state.runnable.load(std::memory_order_acquire) != r) {
                        this->active_workers.fetch_sub(1, std::memory_order_acq_rel);
                        continue;
                    }

                    int task_id = this->state.next_task_id.fetch_add(1);
                    if (task_id < this->state.num_total_tasks) {
                        // 执行
                        r->runTask(task_id, this->state.num_total_tasks);
                        // 标记完成
                        this->state.completed_tasks.fetch_add(1);
                    }
                    this->active_workers.fetch_sub(1, std::memory_order_acq_rel);
                    // 没领到活：循环直到下一次 run() 
                }
            }
        });
    }
}

TaskSystemParallelThreadPoolSpinning::~TaskSystemParallelThreadPoolSpinning() {
    this->killed.store(true);
    this->state.runnable.store(nullptr, std::memory_order_release);
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

    if (num_total_tasks <= 0) {
        return;
    }

    this->state.num_total_tasks = num_total_tasks;
    this->state.completed_tasks.store(0);
    
    // 重置 next_task_id
    this->state.next_task_id.store(0);
    this->state.runnable.store(runnable, std::memory_order_release);
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

    this->state.runnable.store(nullptr, std::memory_order_release);
    while (this->active_workers.load(std::memory_order_acquire) > 0) {
        // 等待已经读到旧 runnable 的 worker 离开当前任务状态
    }
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

    this->num_threads = num_threads;
    this->_killed = false;
    this->state.runnable.store(nullptr);
    this->state.num_total_tasks = 0;
    this->state.next_task_id.store(0);
    this->state.completed_tasks.store(0);

    for (int i = 0; i < num_threads; i++) {
        workers.emplace_back([this]() {
            while(true) {
                // 尝试领任务
                std::unique_lock<std::mutex> lock(this->_mutex);

                // 如果没活干且还没被杀掉，睡觉去
                this->_cv_worker.wait(lock, [this]() {
                    return (this->state.next_task_id.load() < this->state.num_total_tasks) || this->_killed;
                });
                if (this->_killed) break; 
                
                // 抢一个 ID
                // NOTE: 貌似抢一个会导致性能开销过不去
                // int task_id = this->state.next_task_id++;
                // IRunnable* r = this->state.runnable.load();
                // int total = this->state.num_total_tasks;

                // 我要打十个
                // int total = this->state.num_total_tasks;
                // IRunnable* r = this->state.runnable.load();

                // int start_id = this->state.next_task_id;
                // int end_id = std::min(total, start_id + 10);
                // this->state.next_task_id = end_id;

                // 启发式调度
                int total = this->state.num_total_tasks;
                int remaining = total - this->state.next_task_id;
                IRunnable* r = this->state.runnable.load();
                
                // 计算动态块大小：剩余任务 / (线程数的 2 倍)
                int chunk_size = std::max(1, remaining / (2 * this->num_threads));
                
                int start_id = this->state.next_task_id;
                int end_id = std::min(total, start_id + chunk_size);
                this->state.next_task_id = end_id;

                // 干活前解锁
                lock.unlock();

                for(int j = start_id; j < end_id; ++j) r->runTask(j, total);
                
                // 汇报进度
                lock.lock();
                this->state.completed_tasks += (end_id - start_id);
                
                // 如果我是最后一个干完的，通知老板
                if (this->state.completed_tasks == total) {
                    this->_cv_main.notify_all();
                }
                lock.unlock();
            }
        });
    }
}

TaskSystemParallelThreadPoolSleeping::~TaskSystemParallelThreadPoolSleeping() {
    //
    // TODO: CS149 student implementations may decide to perform cleanup
    // operations (such as thread pool shutdown construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //
    {
        std::lock_guard<std::mutex> lock(this->_mutex);
        this->_killed = true;
    }
    this->_cv_worker.notify_all(); // 唤醒所有线程
    for (auto& t : workers) {
        t.join();
    }
}

void TaskSystemParallelThreadPoolSleeping::run(IRunnable* runnable, int num_total_tasks) {


    //
    // TODO: CS149 students will modify the implementation of this
    // method in Parts A and B.  The implementation provided below runs all
    // tasks sequentially on the calling thread.
    //

    if (num_total_tasks <= 0) {
        return;
    }

    std::unique_lock<std::mutex> lock(this->_mutex);
    
    // 准备新任务的状态
    this->state.runnable.store(runnable);
    this->state.num_total_tasks = num_total_tasks;
    this->state.next_task_id.store(0);
    this->state.completed_tasks.store(0);
    // 唤醒所有睡着的工人
    this->_cv_worker.notify_all();
    // // 方案一 - 干等
    // // (. , .)
    // this->_cv_main.wait(lock, [this, num_total_tasks]() {
    //     return this->state.completed_tasks.load() == num_total_tasks;
    // });

    // 方案二 - 主线程帮忙
    // while (this->state.next_task_id < num_total_tasks) {
    //     int start_id = this->state.next_task_id;
    //     int end_id = std::min(num_total_tasks, start_id + 10);
    //     this->state.next_task_id = end_id;
        
    //     // 领到任务后解锁
    //     lock.unlock(); 
    //     for (int i = start_id; i < end_id; ++i) runnable->runTask(i, num_total_tasks);
    //     lock.lock(); 
    //     // 执行完后更新完成数
    //     this->state.completed_tasks += (end_id - start_id);
    // }
    // // 注意主线程这里不需要 notify_all，因为工人都在抢活
    // // 或者如果主线程是最后一个干完的，逻辑上也不需要唤醒自己

    // this->_cv_main.wait(lock, [this, num_total_tasks]() {
    //     return this->state.completed_tasks == num_total_tasks;
    // });

    // 方案三 - 主线程帮忙 & 启发式调度
    int n_threads = this->num_threads + 1; // 包含主线程自己
    while (this->state.next_task_id < num_total_tasks) {
        int remaining = num_total_tasks - this->state.next_task_id;
        int chunk_size = std::max(1, remaining / (2 * n_threads));
        int start_id = this->state.next_task_id;
        int end_id = std::min(num_total_tasks, start_id + chunk_size);
        this->state.next_task_id = end_id;
        
        // 领到任务后解锁
        lock.unlock(); 
        for (int i = start_id; i < end_id; ++i) runnable->runTask(i, num_total_tasks);
        lock.lock(); 
        // 执行完后更新完成数
        this->state.completed_tasks += (end_id - start_id);
    }

    this->_cv_main.wait(lock, [this, num_total_tasks]() {
        return this->state.completed_tasks == num_total_tasks;
    });


    this->state.runnable.store(nullptr);
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
