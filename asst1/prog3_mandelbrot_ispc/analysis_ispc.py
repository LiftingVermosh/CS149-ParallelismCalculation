""" Analysis scripts for part 1 """

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['TimesSimSun', 'SimSun', 'Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_csv("./logs/raw_data.csv")

# 计算平均值
df_avg = df.groupby(['View', 'Type'])['Time'].mean().unstack().reset_index()

# 计算加速比 (Speedup = Serial / ISPC)
df_avg['Speedup'] = df_avg['Serial'] / df_avg['ISPC']

print("--- Benchmark Results ---")
print(df_avg)

# 加速比对比
plt.figure(figsize=(8, 6))
sns.barplot(x='View', y='Speedup', data=df_avg, palette='viridis', width=0.2)

# 理论上限 (8x) 
plt.axhline(y=8, color='r', linestyle='--', label='Theoretical Max (8x AVX2)')
plt.text(0.5, 8.1, 'Theoretical Limit (8x)', color='r', ha='center')

plt.title('ISPC SIMD Speedup: View 1 vs View 2')
plt.ylabel('Speedup (x)')
plt.ylim(0, 10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()

plt.savefig('./logs/ispc_analysis.png')
plt.show()