import json
import matplotlib.pyplot as plt
import re
import os, sys

run_name = "CombatFSP-run-20251114-110706"

# 获取project目录
def get_current_file_dir():
    # 判断是否在 Jupyter Notebook 环境
    try:
        shell = get_ipython().__class__.__name__  # ← 误报，不用管
        if shell == 'ZMQInteractiveShell':  # Jupyter Notebook 或 JupyterLab
            # 推荐用 os.getcwd()，指向启动 Jupyter 的目录
            return os.getcwd()
        else:  # 其他 shell
            return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # 普通 Python 脚本
        return os.path.dirname(os.path.abspath(__file__))


current_dir = get_current_file_dir()
# 修正：当前目录是"其余测试"，它的父目录才是 project_root
project_root = os.path.dirname(current_dir)  # 只需要一层父目录
sys.path.append(project_root)

print("当前目录:", current_dir)
print("项目根目录:", project_root)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

pre_log_dir = os.path.join(project_root, "logs", "combat")

print("日志目录:", pre_log_dir)
file_path = os.path.join(pre_log_dir, run_name, "elo_ratings.json")

print("文件路径:", file_path)
print("文件是否存在:", os.path.exists(file_path))

# 读取 JSON 文件
with open(file_path, 'r', encoding='utf-8') as f:
    elo_data = json.load(f)

# 解析数据：提取编号和 ELO 分值
# 假设格式是 {"actor_rein1": 1200, "actor_rein2": 1250, ...}
actor_numbers = []
elo_scores = []

for key, value in elo_data.items():
    # 使用正则表达式提取编号
    match = re.search(r'actor_rein(\d+)', key)
    if match:
        number = int(match.group(1))
        actor_numbers.append(number)
        elo_scores.append(value)

# 按编号排序
sorted_data = sorted(zip(actor_numbers, elo_scores))
actor_numbers, elo_scores = zip(*sorted_data)

# 创建一个包含两个子图的画布
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# ========== 第一张图：ELO 分值随 Actor 编号的变化曲线 ==========
ax1.plot(actor_numbers, elo_scores, marker='o', linestyle='-', linewidth=2, markersize=6)
ax1.set_xlabel('Actor 编号', fontsize=12)
ax1.set_ylabel('ELO 分值', fontsize=12)
ax1.set_title('ELO 分值随 Actor 编号的变化曲线', fontsize=14)
ax1.grid(True, alpha=0.3)

# ========== 第二张图：ELO 分值分布直方图 ==========
# 使用 'auto' 自动确定分段数，也可以改为 'fd', 'scott', 'sturges' 或具体数字如 20
n, bins, patches = ax2.hist(elo_scores, bins='auto', edgecolor='black', alpha=0.7, color='skyblue')

# 添加均值线和标准差范围
import numpy as np
mean_elo = np.mean(elo_scores)
std_elo = np.std(elo_scores)
ax2.axvline(mean_elo, color='red', linestyle='--', linewidth=2, label=f'均值: {mean_elo:.2f}')
ax2.axvline(mean_elo - std_elo, color='orange', linestyle=':', linewidth=1.5, label=f'±1σ: [{mean_elo-std_elo:.2f}, {mean_elo+std_elo:.2f}]')
ax2.axvline(mean_elo + std_elo, color='orange', linestyle=':', linewidth=1.5)

ax2.set_xlabel('ELO 分值', fontsize=12)
ax2.set_ylabel('频数', fontsize=12)
ax2.set_title(f'ELO 分值分布直方图 (共 {len(bins)-1} 个分段)', fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()

# 显示图形
plt.show()

# 打印一些统计信息
print(f"\n========== 统计信息 ==========")
print(f"总共有 {len(actor_numbers)} 个 actor")
print(f"ELO 分值范围: {min(elo_scores):.2f} - {max(elo_scores):.2f}")
print(f"平均 ELO 分值: {mean_elo:.2f}")
print(f"标准差: {std_elo:.2f}")
print(f"中位数: {np.median(elo_scores):.2f}")
print(f"直方图分段数: {len(bins)-1}")
