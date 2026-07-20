#!/bin/bash

# 配置路径
LOG_DIR="/home/vermosh/projects/CS149-ParallelismCalculation/assignment/asst1/prog1_mandelbrot_threads/logs"
mkdir -p $LOG_DIR

# 待测试的线程数和模式
THREADS=(64)
MODES=("serial" "interleaved")
RUNS=5

echo "Starting Mandelbrot Benchmark..."
echo "Results will be saved to $LOG_DIR"

# 编译
make clean && make

for mode in "${MODES[@]}"; do
    echo "------------------------------------------------"
    echo "Testing Mode: $mode"
    echo "------------------------------------------------"
    
    for t in "${THREADS[@]}"; do
        LOG_FILE="$LOG_DIR/mode_${mode}_t${t}.log"
        echo "Running $t threads, output to $LOG_FILE..."
        
        # Warmup 
        ./mandelbrot -t $t -m $mode > /dev/null
        
        # 正式运行多次
        echo "Configuration: Mode=$mode, Threads=$t" > $LOG_FILE
        echo "Timestamp: $(date)" >> $LOG_FILE
        echo "------------------------------------" >> $LOG_FILE
        
        for ((i=1; i<=$RUNS; i++)); do
            echo "Run #$i..."
            ./mandelbrot -t $t -m $mode >> $LOG_FILE
            echo "------------------------------------" >> $LOG_FILE
        done
        
        # 从 Log 中提取加速比并显示在终端
        avg_speedup=$(grep "speedup" $LOG_FILE | awk '{print $1}' | sed 's/(//' | sed 's/x//' | awk '{sum+=$1} END {print sum/NR}')
        echo "  Done. Average Speedup: ${avg_speedup}x"
    done
done

echo "Benchmark Complete!"