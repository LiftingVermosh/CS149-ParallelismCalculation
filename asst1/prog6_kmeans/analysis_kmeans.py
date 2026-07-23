import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

# 设置绘图样式（沿用你的配置）
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['TimesSimSun', 'SimSun', 'Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid")

def parse_kmeans_log(file_path):
    data = []
    # 正则表达式匹配
    # 匹配线程数、模式、各项耗时和总耗时
    run_pattern = re.compile(r"Running with\s+(\d+)\s+threads(?:,\s+mode=([A-Za-z_]+))?")
    thread_pattern = re.compile(r"Threads=(\d+)")
    mode_pattern = re.compile(r"Mode=([A-Za-z_]+)")
    assign_pattern = re.compile(r"Assignments:\s+([\d\.]+)\s+s")
    centroid_pattern = re.compile(r"Centroids:\s+([\d\.]+)\s+s")
    cost_pattern = re.compile(r"Cost:\s+([\d\.]+)\s+s")
    total_pattern = re.compile(r"\[Total Time\]:\s+([\d\.]+)\s+ms")

    with open(file_path, 'r') as f:
        content = f.read()
        # 按分隔符切分每次运行的块
        blocks = content.split("------------------------------------------")
        
        for block in blocks:
            if "Running with" not in block: continue
            
            t_match = thread_pattern.search(block)
            run_match = run_pattern.search(block)
            mode_match = mode_pattern.search(block)
            a_match = assign_pattern.search(block)
            ce_match = centroid_pattern.search(block)
            co_match = cost_pattern.search(block)
            tt_match = total_pattern.search(block)
            
            if a_match and ce_match and co_match and tt_match and (t_match or run_match):
                threads = int(t_match.group(1)) if t_match else int(run_match.group(1))
                mode = "assignments"
                if mode_match:
                    mode = mode_match.group(1)
                elif run_match and run_match.group(2):
                    mode = run_match.group(2)

                data.append({
                    "Mode": mode,
                    "Threads": threads,
                    "Assignments": float(a_match.group(1)),
                    "Centroids": float(ce_match.group(1)),
                    "Cost": float(co_match.group(1)),
                    "TotalTime_ms": float(tt_match.group(1))
                })
    return pd.DataFrame(data)

# 读取并解析日志
log_path = "./logs/kmeans.log"
if not os.path.exists(log_path):
    print(f"Error: {log_path} not found!")
    exit()

df = parse_kmeans_log(log_path)

if df.empty:
    print(f"Error: no benchmark records parsed from {log_path}")
    exit()

# 计算加速比：每种模式各自以 1 线程为基准
baseline = df[df['Threads'] == 1].set_index('Mode')
missing_baseline = sorted(set(df['Mode']) - set(baseline.index))
if missing_baseline:
    print(f"Error: missing 1-thread baseline for modes: {missing_baseline}")
    exit()

df['Overall_Speedup'] = df.apply(
    lambda row: baseline.loc[row['Mode'], 'TotalTime_ms'] / row['TotalTime_ms'],
    axis=1
)
df['Assignments_Speedup'] = df.apply(
    lambda row: baseline.loc[row['Mode'], 'Assignments'] / row['Assignments'],
    axis=1
)

# 绘图
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
ax1 = axes[0, 0]
ax2 = axes[0, 1]
ax3 = axes[1, 0]
ax4 = axes[1, 1]

# 左上：整体加速比曲线
sns.lineplot(data=df, x='Threads', y='Overall_Speedup', hue='Mode',
             marker='o', ax=ax1, linewidth=2)

# 标记核心物理限制 (13900HX: 8 P-cores / 16 threads, 32 total threads)
ax1.axvline(x=8, color='gray', linestyle='--', label='8 P-Cores')
ax1.axvline(x=16, color='r', linestyle='--', label='16 P-Threads')
ax1.axvline(x=32, color='g', linestyle='--', label='32 Total Threads')

ax1.set_title('K-Means Speedup Analysis', fontsize=14)
ax1.set_xlabel('Thread Count', fontsize=12)
ax1.set_ylabel('Speedup (x)', fontsize=12)
ax1.legend()

# 右上：assignment 阶段自身加速比
sns.lineplot(data=df, x='Threads', y='Assignments_Speedup', hue='Mode',
             marker='s', ax=ax2, linewidth=2)
ax2.axvline(x=8, color='gray', linestyle='--', label='8 P-Cores')
ax2.axvline(x=16, color='r', linestyle='--', label='16 P-Threads')
ax2.axvline(x=32, color='g', linestyle='--', label='32 Total Threads')
ax2.set_title('Assignments Stage Speedup', fontsize=14)
ax2.set_xlabel('Thread Count', fontsize=12)
ax2.set_ylabel('Speedup (x)', fontsize=12)
ax2.legend()

def draw_time_breakdown(ax, mode, title):
    df_time = df[df['Mode'] == mode][['Threads', 'Assignments', 'Centroids', 'Cost']].copy()
    if df_time.empty:
        ax.set_visible(False)
        return

    df_time = df_time.sort_values('Threads')
    df_time['Assignments'] *= 1000
    df_time['Centroids'] *= 1000
    df_time['Cost'] *= 1000

    ax.stackplot(df_time['Threads'],
                 df_time['Assignments'], df_time['Centroids'], df_time['Cost'],
                 labels=['Assignments', 'Centroids', 'Cost'], alpha=0.7)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Thread Count', fontsize=12)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.legend(loc='upper right')

# 下方：按模式分别展示三阶段耗时
draw_time_breakdown(ax3, 'assignments', 'Execution Time Breakdown: Assignments Mode')
draw_time_breakdown(ax4, 'full', 'Execution Time Breakdown: Full Mode')

plt.tight_layout()
plt.savefig('./logs/kmeans_analysis.png')
plt.show()

# 打印分析报告
print("\n--- K-Means Scalability Analysis ---")
for mode, group in df.groupby('Mode'):
    max_idx = group['Overall_Speedup'].idxmax()
    assign_idx = group['Assignments_Speedup'].idxmax()
    print(f"[{mode}] Max Overall Speedup: {df.loc[max_idx, 'Overall_Speedup']:.2f}x "
          f"at {df.loc[max_idx, 'Threads']} threads")
    print(f"[{mode}] Max Assignments Speedup: {df.loc[assign_idx, 'Assignments_Speedup']:.2f}x "
          f"at {df.loc[assign_idx, 'Threads']} threads")
