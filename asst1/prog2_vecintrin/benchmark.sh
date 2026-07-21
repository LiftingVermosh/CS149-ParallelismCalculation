#!/bin/bash

# 配置路径
LOG_DIR="./logs"
mkdir -p $LOG_DIR
HEADER_FILE="CS149intrin.h"

# 待测试的向量宽度 (通常是 2 的幂)
WIDTHS=(2 4 8 16 32 64)
SIZE=10000 # 测试数据大小

echo "Starting Vector Width Benchmark..."

for w in "${WIDTHS[@]}"; do
    echo "------------------------------------------------"
    echo "Testing VECTOR_WIDTH = $w"
    
    # 使用 sed 修改宏定义
    sed -i "s/#define VECTOR_WIDTH [0-9]\+/#define VECTOR_WIDTH $w/" $HEADER_FILE
    
    # 重新编译
    make clean > /dev/null
    make > /dev/null
    
    if [ $? -ne 0 ]; then
        echo "Compilation failed for width $w"
        continue
    fi

    # 运行并保存日志
    LOG_FILE="$LOG_DIR/width_${w}.log"
    echo "Running with size $SIZE, output to $LOG_FILE..."
    
    # 执行程序
    ./myexp -s $SIZE > $LOG_FILE
    
    # 提取关键统计数据展示在终端
    util=$(grep "Vector Utilization" $LOG_FILE | awk '{print $3}')
    instr=$(grep "Total Vector Instructions" $LOG_FILE | awk '{print $4}')
    echo "  Done. Utilization: $util, Total Instructions: $instr"
done

echo "Benchmark Complete! Data saved in $LOG_DIR"