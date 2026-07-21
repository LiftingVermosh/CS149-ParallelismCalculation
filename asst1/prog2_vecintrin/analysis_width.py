import os
import re
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['TimesSimSun', 'SimSun', 'Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

def parse_width_logs(log_dir):
    data = []
    # 匹配文件名
    file_pattern = re.compile(r"width_(?P<width>\d+).log")
    # 匹配日志内容
    util_pattern = re.compile(r"Vector Utilization:\s+(?P<util>[\d.]+)%")
    instr_pattern = re.compile(r"Total Vector Instructions:\s+(?P<instr>\d+)")

    for filename in os.listdir(log_dir):
        match = file_pattern.match(filename)
        if match:
            width = int(match.group('width'))
            with open(os.path.join(log_dir, filename), 'r') as f:
                content = f.read()
                util = float(util_pattern.search(content).group('util'))
                instr = int(instr_pattern.search(content).group('instr'))
                data.append({"Width": width, "Utilization": util, "Instructions": instr})
    return pd.DataFrame(data).sort_values("Width")

df = parse_width_logs("./logs")

# 绘图
fig, ax1 = plt.subplots(figsize=(10, 6))

# 左轴：利用率
color = 'tab:blue'
ax1.set_xlabel('Vector Width')
ax1.set_ylabel('Utilization (%)', color=color)
ax1.plot(df['Width'], df['Utilization'], marker='o', color=color, linewidth=2, label='Utilization')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.6)

# 右轴：指令数
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Total Instructions', color=color)
ax2.plot(df['Width'], df['Instructions'], marker='s', color=color, linestyle='--', label='Instructions')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Impact of Vector Width on Performance (Clamped Exp)')
fig.tight_layout()
plt.savefig('./logs/width_analysis.png')
plt.show()