#!/bin/bash

# Part 1 测评

LOG_DIR="./logs"
mkdir -p $LOG_DIR

# 测试视图
VIEWS=("1" "2")
ITERATIONS=5 # 跑5次取平均值以消除系统噪音

echo "Starting ISPC Mandelbrot Benchmark..."
echo "View,Type,Time" > $LOG_DIR/raw_data.csv

for view in "${VIEWS[@]}"; do
    echo "------------------------------------------------"
    echo "Testing View $view..."
    
    for i in $(seq 1 $ITERATIONS); do
        # 运行程序并获取输出
        output=$(./mandelbrot_ispc --view $view)
        
        # 提取时间 
        serial_time=$(echo "$output" | grep "serial]:" | sed -n 's/.*\[\(.*\)\].*/\1/p')
        ispc_time=$(echo "$output" | grep -v "multi-core" | grep "ispc]:" | sed -n 's/.*\[\(.*\)\].*/\1/p')
        
        echo "$view,Serial,$serial_time" >> $LOG_DIR/raw_data.csv
        echo "$view,ISPC,$ispc_time" >> $LOG_DIR/raw_data.csv
    done
    echo "  View $view completed."
done

echo "Benchmark Complete! Data saved in $LOG_DIR/raw_data.csv"