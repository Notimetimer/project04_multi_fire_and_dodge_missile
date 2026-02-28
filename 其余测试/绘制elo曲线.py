import json
import matplotlib.pyplot as plt
import re

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 文件路径
file_path = r"D:\3_Machine_Learning_in_Python\project03_fire_and_dodge_missile\logs\combat\CombatFSP-run-20251119-112648\elo_ratings.json"

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

# 绘制曲线
plt.figure(figsize=(12, 6))
plt.plot(actor_numbers, elo_scores, marker='o', linestyle='-', linewidth=2, markersize=6)
plt.xlabel('Actor 编号', fontsize=12)
plt.ylabel('ELO 分值', fontsize=12)
plt.title('ELO 分值随 Actor 编号的变化曲线', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 显示图形
plt.show()

# 打印一些统计信息
print(f"总共有 {len(actor_numbers)} 个 actor")
print(f"ELO 分值范围: {min(elo_scores):.2f} - {max(elo_scores):.2f}")
print(f"平均 ELO 分值: {sum(elo_scores)/len(elo_scores):.2f}")
