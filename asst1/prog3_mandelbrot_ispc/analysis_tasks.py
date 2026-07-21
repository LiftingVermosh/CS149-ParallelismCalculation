""" Analysis scripts for part 2 """

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['TimesSimSun', 'SimSun', 'Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv("./logs/task_data.csv")

# 计算平均时间
df_avg = df.groupby(['TaskCount', 'View', 'Type'])['Time'].mean().unstack().reset_index()

# 计算相对于串行的加速比
df_avg['Speedup'] = df_avg['Serial'] / df_avg['ISPC_Tasks']

plt.figure(figsize=(10, 6))
sns.lineplot(data=df_avg, x='TaskCount', y='Speedup', hue='View', marker='o', linewidth=2.5)

plt.axvline(x=8, color='red', linestyle='--', label='Physical Threads (8)')
plt.axvline(x=16, color='orange', linestyle='--', label='Physical Cores (16)')
plt.axvline(x=32, color='gray', linestyle='--', label='Physical Threads (32)')
plt.axvline(x=64, color='blue', linestyle='--', label='Physical Threads (64)')
plt.axvline(x=128, color='green', linestyle='--', label='Physical Threads (128)')
plt.axvline(x=256, color='pink', linestyle='--', label='Physical Threads (256)')
plt.axvline(x=512, color='yellow', linestyle='--', label='Physical Threads (512)')

plt.xscale('log', base=2) # 任务数通常按 2 的幂增长，用对数轴更清晰
plt.title('Speedup vs. Task Count (ISPC Tasks)')
plt.xlabel('Number of Tasks (log scale)')
plt.ylabel('Speedup (x)')
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()

plt.savefig('./logs/task_scaling_analysis.png')
plt.show()