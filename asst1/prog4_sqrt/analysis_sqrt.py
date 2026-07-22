import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['TimesSimSun', 'SimSun', 'Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# 设置样式
sns.set_theme(style="whitegrid")
df = pd.read_csv("./logs/sqrt_data.csv")

# 计算平均值
df_avg = df.groupby(['Mode', 'TaskCount']).mean().reset_index()

# 计算加速比
df_avg['SIMD_Speedup'] = df_avg['Serial'] / df_avg['ISPC']
df_avg['Total_Speedup'] = df_avg['Serial'] / df_avg['TaskISPC']

# 绘图：总加速比 (SIMD + Multi-core)
plt.figure(figsize=(12, 7))
sns.lineplot(data=df_avg, x='TaskCount', y='Total_Speedup', hue='Mode', marker='o', linewidth=2)

plt.axvline(x=16, color='r', linestyle='--', label='16 P-Cores')
plt.axvline(x=32, color='g', linestyle='--', label='32 Threads')

plt.xscale('log', base=2)
plt.title('Prog4: Sqrt Speedup Analysis (i9-13900HX)', fontsize=15)
plt.xlabel('Number of Tasks (log scale)', fontsize=12)
plt.ylabel('Total Speedup over Serial (x)', fontsize=12)
plt.legend()
plt.savefig('./logs/sqrt_performance.png')
plt.show()

# 额外打印 SIMD 利用率分析数据
print("\n--- SIMD Efficiency Analysis ---")
for mode in df_avg['Mode'].unique():
    simd_s = df_avg[df_avg['Mode'] == mode]['SIMD_Speedup'].mean()
    print(f"Mode {mode:12}: Average SIMD Speedup = {simd_s:.2f}x")