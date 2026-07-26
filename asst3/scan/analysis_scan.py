import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置绘图风格
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['TimesSimSun', 'SimSun', 'Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

sns.set_theme(style="whitegrid")

def analyze_performance():
    # 加载数据
    try:
        df = pd.read_csv("./logs/scan_perf_data.csv")
    except FileNotFoundError:
        print("Error: CSV file not found. Run benchmark.sh first.")
        return

    # 数据预处理：按测试类型和规模计算平均值
    df_avg = df.groupby(['TestType', 'ElementCount']).mean().reset_index()

    # 计算性能指标
    # 效率比 (Efficiency Relative to Ref): RefTime / StudentTime
    df_avg['RelativeEfficiency'] = df_avg['RefTime'] / df_avg['StudentTime']

    # 绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 图表 1: 绝对执行时间对比
    for t_type in df_avg['TestType'].unique():
        data = df_avg[df_avg['TestType'] == t_type]
        ax1.plot(data['ElementCount'], data['StudentTime'], marker='o', label=f'Student {t_type}')
        ax1.plot(data['ElementCount'], data['RefTime'], marker='x', linestyle='--', label=f'Ref {t_type}')

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_title('Absolute Execution Time (Log-Log Scale)')
    ax1.set_xlabel('Element Count')
    ax1.set_ylabel('Time (ms)')
    ax1.legend()

    # 图表 2: 相对效率 (Student vs Ref)
    sns.barplot(data=df_avg, x='ElementCount', y='RelativeEfficiency', hue='TestType', ax=ax2)
    ax2.axhline(y=1.0, color='r', linestyle='--', label='Ref Performance')
    ax2.set_title('Performance Relative to Reference (Thrust/Ref)')
    ax2.set_ylabel('Ratio (Ref_Time / Student_Time)')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('./logs/scan_performance_analysis.png')
    plt.show()

    # 打印分析报告
    print("\n" + "="*40)
    print("      CUDA Scan Performance Summary")
    print("="*40)
    for t_type in df_avg['TestType'].unique():
        print(f"\nTest Type: {t_type}")
        subset = df_avg[df_avg['TestType'] == t_type]
        for _, row in subset.iterrows():
            status = "FASTER" if row['RelativeEfficiency'] > 1 else "SLOWER"
            print(f"  N={int(row['ElementCount']):10} | Efficiency: {row['RelativeEfficiency']:.2f}x of Ref ({status})")

if __name__ == "__main__":
    analyze_performance()