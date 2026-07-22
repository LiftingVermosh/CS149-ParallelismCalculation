#!/bin/bash

LOG_DIR="./logs"
mkdir -p $LOG_DIR
ISPC_SRC="sqrt.ispc"
ISPC_BAK="sqrt.ispc.bak"

echo "Creating backup of $ISPC_SRC..."
cp $ISPC_SRC $ISPC_BAK

cleanup() {
    echo -e "\nRestoring $ISPC_SRC from backup..."
    mv $ISPC_BAK $ISPC_SRC
    make clean > /dev/null
    exit
}
trap cleanup EXIT INT TERM

# 测评配置
TASK_COUNTS=(1 8 16 32 64 128 256 512 1024)
MODES=("RANDOM" "BEST_CASE" "WORST_CASE")
ITERATIONS=3

echo "Mode,TaskCount,Serial,ISPC,TaskISPC" > $LOG_DIR/sqrt_data.csv

for mode in "${MODES[@]}"; do
    echo "------------------------------------------------"
    echo "Testing Data Mode: $mode"
    
    # 根据模式设置编译参数
    if [ "$mode" == "RANDOM" ]; then FLAG="";
    else FLAG="-DBUILD_$mode"; fi

    for n in "${TASK_COUNTS[@]}"; do
        echo "  Configuring for $n tasks..."
        
        sed -i "s/uniform int numTasks = [0-9]*; \/\/ TASKS_HERE/uniform int numTasks = $n; \/\/ TASKS_HERE/g" $ISPC_SRC
        
        make clean > /dev/null
        EXTRA_FLAGS=$FLAG make > /dev/null
        if [ $? -ne 0 ]; then echo "Compile Error!"; exit 1; fi

        for i in $(seq 1 $ITERATIONS); do
            output=$(./sqrt)
            
            # 提取时间
            serial=$(echo "$output" | grep "serial]:" | sed -n 's/.*\[\(.*\)\].*/\1/p' | head -n1 | tr -d '\r\n ')
            ispc=$(echo "$output" | grep "ispc]:" | sed -n 's/.*\[\(.*\)\].*/\1/p' | head -n1 | tr -d '\r\n ')
            task_ispc=$(echo "$output" | grep "task ispc]:" | sed -n 's/.*\[\(.*\)\].*/\1/p' | head -n1 | tr -d '\r\n ')
            
            echo "$mode,$n,$serial,$ispc,$task_ispc" >> $LOG_DIR/sqrt_data.csv
        done
        echo "    TaskCount $n: Serial=${serial}ms, TaskISPC=${task_ispc}ms"
    done
done

echo "Benchmark Complete! Data saved in $LOG_DIR/sqrt_data.csv"