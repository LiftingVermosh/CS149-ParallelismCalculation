import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['TimesSimSun', 'SimSun', 'Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# 数据
data = {
    'Implementation': ['Serial', 'ISPC (Single)', 'Task ISPC (Multi)', 'Task Stream (Opt)'],
    'Time (ms)': [10.536, 8.429, 4.717, 4.164],
    'Speedup': [1.0, 1.25, 2.24, 2.53] # 相对于 Serial 的加速比
}

df = pd.DataFrame(data)

sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 10))

colors = ["#BDC3C7", "#3498DB", "#2980B9", "#E74C3C"] # 灰色为基准，红色突出优化版
ax = sns.barplot(x='Implementation', y='Time (ms)', data=df, palette=colors, width=0.4)

for i, row in df.iterrows():
    # 标注耗时
    ax.text(i, row['Time (ms)'] + 0.05, f"{row['Time (ms)']:.3f} ms", 
            ha='center', va='bottom', fontsize=11, fontweight='bold', fontname='TimesSimSun')
    # 标注加速比
    if i > 0:
        ax.text(i, row['Time (ms)'] / 2, f"{row['Speedup']:.2f}x", 
                ha='center', va='center', color='white', fontsize=12, fontweight='bold', fontname='TimesSimSun')

plt.title('SAXPY Performance Comparison', fontsize=16, pad=20, fontname='TimesSimSun')
plt.ylabel('Execution Time (ms)', fontsize=13, fontname='TimesSimSun')
plt.xlabel('Implementation Type', fontsize=13, fontname='TimesSimSun')

plt.axhline(y=4.5, color='gray', linestyle='--', alpha=0.5)
plt.text(3.5, 4.6, 'Memory Bandwidth Saturation', ha='right', color='gray', fontsize=10, fontname='TimesSimSun')

plt.tight_layout()
plt.savefig('./logs/saxpy_performance_comparison.png', dpi=300)
plt.show()

print("\n--- SAXPY Optimization Summary ---")
print(f"Total speedup from Serial to Stream: {data['Speedup'][-1]:.2f}x")
print(f"Streaming Store Improvement: {((data['Time (ms)'][2] - data['Time (ms)'][3]) / data['Time (ms)'][2] * 100):.1f}% reduction in time")