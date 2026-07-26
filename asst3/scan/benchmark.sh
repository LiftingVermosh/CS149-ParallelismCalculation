#!/bin/bash

# 检查编译
make clean && make
if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

# 配置参数
LOG_DIR="./logs"
mkdir -p $LOG_DIR
TEST_TYPES=("scan" "find_repeats")
# 元素数量：1M, 10M, 20M, 40M
ELEMENT_COUNTS=(1000000 10000000 20000000 40000000)
ITERATIONS=3
CSV_FILE="$LOG_DIR/scan_perf_data.csv"


echo "TestType,ElementCount,StudentTime,RefTime" > $CSV_FILE

for type in "${TEST_TYPES[@]}"; do
    echo "------------------------------------------------"
    echo "Testing: $type"
    
    for n in "${ELEMENT_COUNTS[@]}"; do
        echo "  Running n=$n ..."
        
        for i in $(seq 1 $ITERATIONS); do
            # 运行学生版本并获取时间
            output_stu=$(./cudaScan -m $type -i random -n $n)
            stu_time=$(echo "$output_stu" | grep "Student GPU time:" | awk '{print $4}')
            
            # 运行参考版本
            if [ -f "./cudaScan_ref_x86" ]; then
                ref_bin="./cudaScan_ref_x86"
            else
                ref_bin="./cudaScan_ref"
            fi
            
            output_ref=$($ref_bin -m $type -i random -n $n)
            ref_time=$(echo "$output_ref" | grep "Student GPU time:" | awk '{print $4}')

            # 写入 CSV
            echo "$type,$n,$stu_time,$ref_time" >> $CSV_FILE
        done
        echo "    Done: Avg Student Time ≈ $(echo "$stu_time" | awk '{print $1}') ms"
    done
done

echo "Benchmark Complete! Data saved in $CSV_FILE"