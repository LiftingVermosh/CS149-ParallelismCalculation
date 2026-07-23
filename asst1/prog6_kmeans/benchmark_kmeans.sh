#!/bin/bash

make clean && make

LOG_FILE="./logs/kmeans.log"
mkdir -p ./logs
rm -f $LOG_FILE

# 测试不同的线程数
THREAD_COUNTS=(1 2 4 8 12 16 24 32)
MODES=("assignments" "full")

echo "Starting Benchmark on $(hostname)" | tee -a $LOG_FILE

for mode in "${MODES[@]}"
do
    for t in "${THREAD_COUNTS[@]}"
    do
        echo "------------------------------------------" | tee -a $LOG_FILE
        echo "Running with $t threads, mode=$mode..." | tee -a $LOG_FILE
        ./kmeans -t $t -m $mode >> $LOG_FILE 2>&1
    done
done

echo "Benchmark Complete. Results saved to $LOG_FILE"
