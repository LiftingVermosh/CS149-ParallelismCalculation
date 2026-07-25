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
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }

    return 0;
}

void TaskSystemSerial::sync() {
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
    this->num_threads = num_threads;
}

TaskSystemParallelSpawn::~TaskSystemParallelSpawn() {}

void TaskSystemParallelSpawn::run(IRunnable* runnable, int num_total_tasks) {
    if (num_total_tasks <= 0) {
        return;
    }

    int call_max_thread_nums = std::min(this->num_threads, num_total_tasks);
    std::vector<std::thread> workers;

    for (int i = 0; i < call_max_thread_nums; ++i) {
        workers.emplace_back([&, i, call_max_thread_nums]() {
            for (int j = i; j < num_total_tasks; j += call_max_thread_nums) {
                runnable->runTask(j, num_total_tasks);
            }
        });
    }

    for (auto& t : workers) {
        t.join();
    }
}

TaskID TaskSystemParallelSpawn::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                 const std::vector<TaskID>& deps) {
    run(runnable, num_total_tasks);
    return 0;
}

void TaskSystemParallelSpawn::sync() {
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
    this->num_threads = num_threads;
    this->killed.store(false);
    this->active_workers.store(0);
    this->state.runnable.store(nullptr);
    this->state.num_total_tasks = 0;
    this->state.next_task_id.store(0);
    this->state.completed_tasks.store(0);
    this->graph_has_work.store(false);
    this->graph_next_task_id = 0;
    this->graph_task_id_base = 0;
    this->graph_batches_submitted = 0;
    this->graph_batches_completed = 0;
    this->graph_tasks.reserve(4096);

    for (int i = 0; i < num_threads; i++) {
        workers.emplace_back([this]() {
            while (!this->killed.load()) {
                if (this->graph_has_work.load(std::memory_order_acquire)) {
                    TaskBatch* batch = nullptr;
                    IRunnable* r = nullptr;
                    int total = 0;
                    int start_id = 0;
                    int end_id = 0;

                    std::unique_lock<std::mutex> graph_lock(this->graph_mutex);
                    if (!this->graph_ready_queue.empty()) {
                        batch = this->graph_ready_queue.front();
                        start_id = batch->tasks_started;
                        int chunk_size = std::max(1, (batch->num_total_tasks + this->num_threads - 1) / this->num_threads);
                        end_id = std::min(batch->num_total_tasks, start_id + chunk_size);
                        batch->tasks_started = end_id;
                        if (batch->tasks_started == batch->num_total_tasks) {
                            this->graph_ready_queue.pop();
                        }
                        this->graph_has_work.store(!this->graph_ready_queue.empty(), std::memory_order_release);
                        r = batch->runnable;
                        total = batch->num_total_tasks;
                    } else {
                        this->graph_has_work.store(false, std::memory_order_release);
                    }
                    graph_lock.unlock();

                    if (batch != nullptr) {
                        for (int i = start_id; i < end_id; i++) {
                            r->runTask(i, total);
                        }

                        graph_lock.lock();
                        batch->tasks_completed += (end_id - start_id);
                        if (batch->tasks_completed == total) {
                            batch->completed = true;
                            for (TaskBatch* next_batch : batch->successors) {
                                next_batch->deps_remaining--;
                                if (next_batch->deps_remaining == 0) {
                                    this->graph_ready_queue.push(next_batch);
                                    this->graph_has_work.store(true, std::memory_order_release);
                                }
                            }
                            this->graph_batches_completed++;
                        }
                        graph_lock.unlock();
                        continue;
                    }
                }

                IRunnable* r = this->state.runnable.load(std::memory_order_acquire);
                if (r != nullptr) {
                    this->active_workers.fetch_add(1, std::memory_order_acq_rel);
                    if (this->state.runnable.load(std::memory_order_acquire) != r) {
                        this->active_workers.fetch_sub(1, std::memory_order_acq_rel);
                        continue;
                    }

                    int task_id = this->state.next_task_id.fetch_add(1);
                    if (task_id < this->state.num_total_tasks) {
                        r->runTask(task_id, this->state.num_total_tasks);
                        this->state.completed_tasks.fetch_add(1);
                    }

                    this->active_workers.fetch_sub(1, std::memory_order_acq_rel);
                } else {
                    std::this_thread::yield();
                }
            }
        });
    }
}

TaskSystemParallelThreadPoolSpinning::~TaskSystemParallelThreadPoolSpinning() {
    this->killed.store(true);
    this->state.runnable.store(nullptr, std::memory_order_release);
    for (auto& t : this->workers) {
        t.join();
    }

    for (TaskBatch* batch : this->graph_tasks) {
        delete batch;
    }
    this->graph_tasks.clear();
}

void TaskSystemParallelThreadPoolSpinning::run(IRunnable* runnable, int num_total_tasks) {
    if (num_total_tasks <= 0) {
        return;
    }

    this->state.num_total_tasks = num_total_tasks;
    this->state.completed_tasks.store(0);
    this->state.next_task_id.store(0);
    this->state.runnable.store(runnable, std::memory_order_release);

    while (true) {
        int task_id = this->state.next_task_id.fetch_add(1);
        if (task_id >= num_total_tasks) {
            break;
        }
        runnable->runTask(task_id, num_total_tasks);
        this->state.completed_tasks.fetch_add(1);
    }

    while (this->state.completed_tasks.load() < num_total_tasks) {
    }

    this->state.runnable.store(nullptr, std::memory_order_release);
    while (this->active_workers.load(std::memory_order_acquire) > 0) {
    }
}

TaskID TaskSystemParallelThreadPoolSpinning::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                              const std::vector<TaskID>& deps) {
    std::lock_guard<std::mutex> graph_lock(this->graph_mutex);

    TaskID cur_id = this->graph_next_task_id++;
    TaskBatch* batch = new TaskBatch();
    batch->id = cur_id;
    batch->runnable = runnable;
    batch->num_total_tasks = num_total_tasks;
    batch->tasks_started = 0;
    batch->tasks_completed = 0;
    batch->deps_remaining = 0;
    batch->completed = false;

    for (TaskID dep_id : deps) {
        size_t dep_index = static_cast<size_t>(dep_id - this->graph_task_id_base);
        if (dep_id >= this->graph_task_id_base && dep_index < this->graph_tasks.size() && !this->graph_tasks[dep_index]->completed) {
            batch->deps_remaining++;
            this->graph_tasks[dep_index]->successors.push_back(batch);
        }
    }

    this->graph_tasks.push_back(batch);
    this->graph_batches_submitted++;

    if (batch->deps_remaining == 0) {
        this->graph_ready_queue.push(batch);
        this->graph_has_work.store(true, std::memory_order_release);
    }

    return cur_id;
}

void TaskSystemParallelThreadPoolSpinning::sync() {
    while (true) {
        TaskBatch* batch = nullptr;
        IRunnable* r = nullptr;
        int total = 0;
        int start_id = 0;
        int end_id = 0;

        std::unique_lock<std::mutex> graph_lock(this->graph_mutex);
        if (this->graph_batches_completed == this->graph_batches_submitted) {
            break;
        }

        if (!this->graph_ready_queue.empty()) {
            batch = this->graph_ready_queue.front();
            start_id = batch->tasks_started;
            int chunk_size = std::max(1, (batch->num_total_tasks + this->num_threads) / (this->num_threads + 1));
            end_id = std::min(batch->num_total_tasks, start_id + chunk_size);
            batch->tasks_started = end_id;
            if (batch->tasks_started == batch->num_total_tasks) {
                this->graph_ready_queue.pop();
            }
            this->graph_has_work.store(!this->graph_ready_queue.empty(), std::memory_order_release);
            r = batch->runnable;
            total = batch->num_total_tasks;
        }
        graph_lock.unlock();

        if (batch == nullptr) {
            std::this_thread::yield();
            continue;
        }

        for (int i = start_id; i < end_id; i++) {
            r->runTask(i, total);
        }

        graph_lock.lock();
        batch->tasks_completed += (end_id - start_id);
        if (batch->tasks_completed == total) {
            batch->completed = true;
            for (TaskBatch* next_batch : batch->successors) {
                next_batch->deps_remaining--;
                if (next_batch->deps_remaining == 0) {
                    this->graph_ready_queue.push(next_batch);
                    this->graph_has_work.store(true, std::memory_order_release);
                }
            }
            this->graph_batches_completed++;
        }
    }

    for (TaskBatch* batch : this->graph_tasks) {
        delete batch;
    }
    this->graph_tasks.clear();
    while (!this->graph_ready_queue.empty()) {
        this->graph_ready_queue.pop();
    }
    this->graph_task_id_base = this->graph_next_task_id;
    this->graph_batches_submitted = 0;
    this->graph_batches_completed = 0;
    this->graph_has_work.store(false, std::memory_order_release);
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
    this->next_task_id = 0;
    this->task_id_base = 0;
    this->killed = false;
    this->total_batches_submitted = 0;
    this->total_batches_completed = 0;
    this->all_tasks.reserve(4096);

    for (int i = 0; i < num_threads; i++) {
        workers.emplace_back([this]() {
            while(true) {
                std::unique_lock<std::mutex> lock(this->mutex);
                
                // 等待有活干或者被 kill
                cv_worker.wait(lock, [this]() {
                    return !ready_queue.empty() || killed;
                });
                if (killed) break;

                // 获取就绪的任务批次
                TaskBatch* batch = ready_queue.front();
                
                // 领走一批子任务
                int start_id = batch->tasks_started;

                // 原启发式调度：按剩余任务数动态缩小 chunk。
                // int chunk_size = std::max(1, (batch->num_total_tasks - start_id) / (2 * this->num_threads));
                
                int chunk_size = std::max(1, (batch->num_total_tasks + this->num_threads - 1) / this->num_threads);
                if (batch->num_total_tasks > 1) {
                    chunk_size = std::max(chunk_size, 2);
                }
                int end_id = std::min(batch->num_total_tasks, start_id + chunk_size);
                batch->tasks_started = end_id;
                
                // 如果这个 Batch 的子任务全被领走了，移出就绪队列
                if (batch->tasks_started == batch->num_total_tasks) {
                    ready_queue.pop();
                }

                // 缓存执行所需变量，准备解锁干活
                IRunnable* r = batch->runnable;
                int total = batch->num_total_tasks;
                
                lock.unlock(); // 干活前释放锁
                
                for (int i = start_id; i < end_id; i++) {
                    r->runTask(i, total);
                }
                // 汇报进度
                lock.lock();
                batch->tasks_completed += (end_id - start_id);
                
                // 如果这个 Batch 彻底干完了
                if (batch->tasks_completed == total) {
                    batch->completed = true;
                    for (TaskBatch* next_batch : batch->successors) {
                        next_batch->deps_remaining--;
                        if (next_batch->deps_remaining == 0) {
                            ready_queue.push(next_batch);
                            int chunk_size = std::max(1, (next_batch->num_total_tasks + this->num_threads - 1) / this->num_threads);
                            if (next_batch->num_total_tasks > 1) {
                                chunk_size = std::max(chunk_size, 2);
                            }
                            int chunks = std::min(this->num_threads, (next_batch->num_total_tasks + chunk_size - 1) / chunk_size);
                            for (int i = 0; i < chunks; i++) {
                                cv_worker.notify_one();
                            }
                            cv_sync.notify_all();
                        }
                    }

                    total_batches_completed++;
                    if (total_batches_completed == total_batches_submitted) {
                        cv_sync.notify_all(); // 唤醒 sync()
                    }
                }
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
        std::lock_guard<std::mutex> lock(this->mutex);
        killed = true;
    }
    cv_worker.notify_all();

    for (auto& t : workers) {
        t.join();
    }

    for (TaskBatch* batch : all_tasks) {
        delete batch;
    }
    all_tasks.clear();
}

void TaskSystemParallelThreadPoolSleeping::run(IRunnable* runnable, int num_total_tasks) {


    //
    // TODO: CS149 students will modify the implementation of this
    // method in Parts A and B.  The implementation provided below runs all
    // tasks sequentially on the calling thread.
    //
    runAsyncWithDeps(runnable, num_total_tasks, {});
    sync();
}

TaskID TaskSystemParallelThreadPoolSleeping::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                    const std::vector<TaskID>& deps) {


    //
    // TODO: CS149 students will implement this method in Part B.
    //
    std::lock_guard<std::mutex> lock(this->mutex);

    // 创建并初始化 TaskBatch
    TaskID cur_id = next_task_id++;
    TaskBatch* batch = new TaskBatch();
    batch->id = cur_id;
    batch->runnable = runnable;
    batch->num_total_tasks = num_total_tasks;
    batch->tasks_started = 0;
    batch->tasks_completed = 0;
    batch->deps_remaining = 0;
    batch->completed = false;

    // 检查依赖关系
    for (TaskID dep_id : deps) {
        // 如果依赖的任务还没彻底完成，就建立关联
        size_t dep_index = static_cast<size_t>(dep_id - this->task_id_base);
        if (dep_id >= this->task_id_base && dep_index < this->all_tasks.size() && !this->all_tasks[dep_index]->completed) {
            batch->deps_remaining++;
            this->all_tasks[dep_index]->successors.push_back(batch);
        }
    }

    // 注册到全局表
    this->all_tasks.push_back(batch);
    this->total_batches_submitted++;

    // 如果没有依赖，直接进入就绪队列
    if (batch->deps_remaining == 0) {
        this->ready_queue.push(batch);
        int chunk_size = std::max(1, (batch->num_total_tasks + this->num_threads - 1) / this->num_threads);
        if (batch->num_total_tasks > 1) {
            chunk_size = std::max(chunk_size, 2);
        }
        int chunks = std::min(this->num_threads, (batch->num_total_tasks + chunk_size - 1) / chunk_size);
        for (int i = 0; i < chunks; i++) {
            this->cv_worker.notify_one();
        }
        this->cv_sync.notify_all();
    }

    return cur_id;
}

void TaskSystemParallelThreadPoolSleeping::sync() {

    //
    // TODO: CS149 students will modify the implementation of this method in Part B.
    //
    std::unique_lock<std::mutex> lock(this->mutex);
    while (total_batches_completed < total_batches_submitted) {
        if (ready_queue.empty()) {
            cv_sync.wait(lock, [this]() {
                return total_batches_completed == total_batches_submitted || !ready_queue.empty();
            });
            continue;
        }

        TaskBatch* batch = ready_queue.front();
        int start_id = batch->tasks_started;
        // 原启发式调度：按剩余任务数动态缩小 chunk。
        // int chunk_size = std::max(1, (batch->num_total_tasks - start_id) / (2 * (this->num_threads + 1)));

        int chunk_size = std::max(1, (batch->num_total_tasks + this->num_threads) / (this->num_threads + 1));
        if (batch->num_total_tasks > 1) {
            chunk_size = std::max(chunk_size, 2);
        }
        int end_id = std::min(batch->num_total_tasks, start_id + chunk_size);
        batch->tasks_started = end_id;

        if (batch->tasks_started == batch->num_total_tasks) {
            ready_queue.pop();
        }

        IRunnable* r = batch->runnable;
        int total = batch->num_total_tasks;

        lock.unlock();
        for (int i = start_id; i < end_id; i++) {
            r->runTask(i, total);
        }
        lock.lock();

        batch->tasks_completed += (end_id - start_id);
        if (batch->tasks_completed == total) {
            batch->completed = true;
            for (TaskBatch* next_batch : batch->successors) {
                next_batch->deps_remaining--;
                if (next_batch->deps_remaining == 0) {
                    ready_queue.push(next_batch);
                    int worker_chunk_size = std::max(1, (next_batch->num_total_tasks + this->num_threads - 1) / this->num_threads);
                    if (next_batch->num_total_tasks > 1) {
                        worker_chunk_size = std::max(worker_chunk_size, 2);
                    }
                    int chunks = std::min(this->num_threads, (next_batch->num_total_tasks + worker_chunk_size - 1) / worker_chunk_size);
                    for (int i = 0; i < chunks; i++) {
                        cv_worker.notify_one();
                    }
                }
            }

            total_batches_completed++;
            if (total_batches_completed == total_batches_submitted) {
                cv_sync.notify_all();
            }
        }
    }

    for (TaskBatch* batch : all_tasks) {
        delete batch;
    }
    all_tasks.clear();
    task_id_base = next_task_id;
    while (!ready_queue.empty()) {
        ready_queue.pop();
    }
    total_batches_submitted = 0;
    total_batches_completed = 0;
}
