import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['TimesSimSun', 'SimSun', 'Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

def parse_logs(log_dir):
    data = []
    # 匹配线程数、时间、加速比的正则表达式
    file_pattern = re.compile(r"mode_(?P<mode>\w+)_t(?P<threads>\d+).log")
    speedup_pattern = re.compile(r"\((?P<speedup>[\d.]+)x speedup")
    time_pattern = re.compile(r"\[mandelbrot thread\]:\s+\[(?P<time>[\d.]+)\] ms")

    for filename in os.listdir(log_dir):
        match = file_pattern.match(filename)
        if match:
            mode = match.group('mode')
            threads = int(match.group('threads'))
            
            with open(os.path.join(log_dir, filename), 'r') as f:
                content = f.read()
                speedups = [float(s) for s in speedup_pattern.findall(content)]
                times = [float(t) for t in time_pattern.findall(content)]
                
                for s, t in zip(speedups, times):
                    data.append({
                        "Mode": mode,
                        "Threads": threads,
                        "Speedup": s,
                        "Time": t
                    })
    return pd.DataFrame(data)

# 加载数据
log_path = "./logs"
df = parse_logs(log_path)

# 计算均值和标准差（用于误差棒）
stats = df.groupby(['Mode', 'Threads']).agg({'Speedup': ['mean', 'std'], 'Time': ['mean', 'std']}).reset_index()
stats.columns = ['Mode', 'Threads', 'Speedup_mean', 'Speedup_std', 'Time_mean', 'Time_std']

# 绘图：加速比对比图 (Scaling Plot)
plt.figure(figsize=(10, 6))
for mode in stats['Mode'].unique():
    mode_data = stats[stats['Mode'] == mode]
    plt.errorbar(mode_data['Threads'], mode_data['Speedup_mean'], 
                 yerr=mode_data['Speedup_std'], label=f'Mode: {mode}', 
                 marker='o', capsize=5)

# 画出理想加速参考线
max_t = stats['Threads'].max()
plt.plot([0, max_t], [0, max_t], 'k--', label='Ideal (Linear)', alpha=0.5)

plt.xlabel('Number of Threads')
plt.ylabel('Speedup (x)')
plt.title('Mandelbrot Parallel Performance: Interleaved vs Serial(Block)')
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.savefig('./logs/speedup_comparison.png')
plt.show()