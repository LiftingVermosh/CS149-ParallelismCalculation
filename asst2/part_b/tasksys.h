#ifndef _TASKSYS_H
#define _TASKSYS_H

#include "itasksys.h"
#include <algorithm>
#include <mutex>
#include <thread>
#include <vector>
#include <atomic>
#include <queue>
#include <condition_variable>

struct TaskState {
    std::atomic<IRunnable*> runnable;
    int num_total_tasks;
    std::atomic<int> next_task_id;
    std::atomic<int> completed_tasks;
};

struct TaskBatch {
    // 任务基本信息
    TaskID id;
    IRunnable* runnable;
    int num_total_tasks;

    // 执行进度
    int tasks_started;     // 已领取的子任务数 (0 ~ num_total_tasks)
    int tasks_completed;   // 已完成的子任务数 (0 ~ num_total_tasks)

    // 依赖管理
    int deps_remaining;
    bool completed;

    std::vector<TaskBatch*> successors;
};

/*
 * TaskSystemSerial: This class is the student's implementation of a
 * serial task execution engine.  See definition of ITaskSystem in
 * itasksys.h for documentation of the ITaskSystem interface.
 */
class TaskSystemSerial: public ITaskSystem {
    public:
        TaskSystemSerial(int num_threads);
        ~TaskSystemSerial();
        const char* name();
        void run(IRunnable* runnable, int num_total_tasks);
        TaskID runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                const std::vector<TaskID>& deps);
        void sync();
};

/*
 * TaskSystemParallelSpawn: This class is the student's implementation of a
 * parallel task execution engine that spawns threads in every run()
 * call.  See definition of ITaskSystem in itasksys.h for documentation
 * of the ITaskSystem interface.
 */
class TaskSystemParallelSpawn: public ITaskSystem {
    public:
        TaskSystemParallelSpawn(int num_threads);
        ~TaskSystemParallelSpawn();
        const char* name();
        void run(IRunnable* runnable, int num_total_tasks);
        TaskID runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                const std::vector<TaskID>& deps);
        void sync();

        int num_threads;
};

/*
 * TaskSystemParallelThreadPoolSpinning: This class is the student's
 * implementation of a parallel task execution engine that uses a
 * thread pool. See definition of ITaskSystem in itasksys.h for
 * documentation of the ITaskSystem interface.
 */
class TaskSystemParallelThreadPoolSpinning: public ITaskSystem {
    public:
        TaskSystemParallelThreadPoolSpinning(int num_threads);
        ~TaskSystemParallelThreadPoolSpinning();
        const char* name();
        void run(IRunnable* runnable, int num_total_tasks);
        TaskID runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                const std::vector<TaskID>& deps);
        void sync();

        std::vector<std::thread> workers;
        std::atomic<bool> killed;
        std::atomic<int> active_workers;
        TaskState state;
        int num_threads;

        std::mutex graph_mutex;
        std::atomic<bool> graph_has_work;
        TaskID graph_next_task_id;
        TaskID graph_task_id_base;
        std::vector<TaskBatch*> graph_tasks;
        std::queue<TaskBatch*> graph_ready_queue;
        int graph_batches_submitted;
        int graph_batches_completed;
};

/*
 * TaskSystemParallelThreadPoolSleeping: This class is the student's
 * optimized implementation of a parallel task execution engine that uses
 * a thread pool. See definition of ITaskSystem in
 * itasksys.h for documentation of the ITaskSystem interface.
 */
class TaskSystemParallelThreadPoolSleeping: public ITaskSystem {
    public:
        TaskSystemParallelThreadPoolSleeping(int num_threads);
        ~TaskSystemParallelThreadPoolSleeping();
        const char* name();
        void run(IRunnable* runnable, int num_total_tasks);
        TaskID runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                const std::vector<TaskID>& deps);
        void sync();

        std::vector<std::thread> workers;
        std::mutex mutex;
        std::condition_variable cv_worker;           // 唤醒工人
        std::condition_variable cv_sync;             // 唤醒 sync()
        bool killed;
        int num_threads;
        
        TaskID next_task_id;                         // 用于分配唯一的 ID
        TaskID task_id_base;                         // all_tasks[0] 对应的 TaskID
        std::vector<TaskBatch*> all_tasks;           // 记录所有任务，方便查找
        std::queue<TaskBatch*> ready_queue;          // 就绪队列：只放 deps_remaining == 0 的任务
        
        int total_batches_submitted;                 // 已提交的 Batch 总数
        int total_batches_completed;                 // 已彻底完成的 Batch 总数
};

#endif
