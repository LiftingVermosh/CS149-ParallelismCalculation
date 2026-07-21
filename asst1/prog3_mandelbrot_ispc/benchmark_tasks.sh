#!/bin/bash

LOG_DIR="./logs"
mkdir -p $LOG_DIR
ISPC_SRC="mandelbrot.ispc"
TASK_COUNTS=(2 4 8 16 32 64 128 256 512) # 任务规模
VIEWS=("1" "2")
ITERATIONS=3

# 备份
cp $ISPC_SRC ${ISPC_SRC}.bak

echo "TaskCount,View,Type,Time" > $LOG_DIR/task_data.csv

for n in "${TASK_COUNTS[@]}"; do
    echo "------------------------------------------------"
    echo "Testing Task Count: $n"
    
    # 使用 sed 修改 launch[x] 和 rowsPerTask = height / x
    sed -i "s/uniform int numTasks = [0-9]*/uniform int numTasks = $n/g" $ISPC_SRC
    sed -i "s/launch\[[0-9]*\]/launch[$n]/g" $ISPC_SRC
    
    # 重新编译
    make > /dev/null
    if [ $? -ne 0 ]; then echo "Compile Error!"; exit 1; fi

    for view in "${VIEWS[@]}"; do
        for i in $(seq 1 $ITERATIONS); do
            output=$(./mandelbrot_ispc --tasks --view $view)
            
            # 提取时间
            serial=$(echo "$output" | grep "serial\]:" | awk -F'[' '{print $3}' | awk -F']' '{print $1}')
            ispc_tasks=$(echo "$output" | grep "multicore ispc\]:" | awk -F'[' '{print $3}' | awk -F']' '{print $1}')
            
            
            echo "$n,$view,Serial,$serial" >> $LOG_DIR/task_data.csv
            echo "$n,$view,ISPC_Tasks,$ispc_tasks" >> $LOG_DIR/task_data.csv
        done
        echo "  View $view finished."
    done
done

# 还原代码并重新编译
mv ${ISPC_SRC}.bak $ISPC_SRC
make > /dev/null

echo "Benchmark Complete! Data saved in $LOG_DIR/task_data.csv"