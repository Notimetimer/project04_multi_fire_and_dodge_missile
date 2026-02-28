import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import os
from _context import * # 包含 project_root

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def draw_combat_matrix(csv_path, team_labels=None, 
                       title="Cross-Algorithm Final Agents Combat Matrix",
                       xlabel="Opponent Team (Column)",
                       ylabel="Evaluated Team (Row)",
                       cbar_label="Win Rate",
                       color_theme='blue'):
    """
    读取 CSV 并绘制博弈矩阵热力图。
    :param csv_path: CSV 文件路径
    :param team_labels: (Optional) 团队显示名称列表。如果不传，则使用 CSV 的列名。
    :param title: 图表标题
    :param xlabel: X轴标签
    :param ylabel: Y轴标签
    :param cbar_label: 颜色条标签
    :param color_theme: 'blue' (白蓝紫) 或 'red' (白红)
    """
    if not os.path.exists(csv_path):
        print(f"[Error] CSV file not found at {csv_path}")
        return

    # 读取 CSV (假设带有 header 和 index)
    try:
        df = pd.read_csv(csv_path, index_col=0)
    except Exception as e:
        print(f"[Error] Failed to read CSV: {e}")
        return

    results_matrix = df.values
    
    # 确定标签
    if team_labels is not None:
        if len(team_labels) != len(df.columns):
             print(f"[Warning] Provided team_labels length ({len(team_labels)}) "
                   f"does not match matrix dimension ({len(df.columns)}). Using CSV headers.")
             labels = df.columns.tolist()
        else:
            labels = team_labels
    else:
        labels = df.columns.tolist()

    num_teams = len(labels)

    # 4. 绘图部分
    # 动态调整图片大小
    fig_size = max(8, num_teams + 2)
    plt.figure(figsize=(fig_size + 2, fig_size))
    
    # 手动指定色彩空间向量（RGB 0-1）
    if color_theme == 'red':
        end_color = (0.6, 0.05, 0.05) # 深红色
    else:
        end_color = (0.1, 0.1, 0.44)  # 默认蓝紫色
    
    colors = [(1.0, 1.0, 1.0), end_color]
    
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)
    
    # 增加鲁棒性：如果数值完全一样（例如刚初始化的0.0），TwoSlopeNorm会报错
    vmin, vmax = results_matrix.min(), results_matrix.max()
    if vmin == vmax:
        norm = None
    else:
        norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)

    ax = sns.heatmap(
        results_matrix,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        norm=norm,
        xticklabels=labels,
        yticklabels=labels,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": cbar_label, "shrink": 0.8}
    )

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    # 使用传入的自定义文本
    plt.title(title, fontsize=14, pad=40)
    plt.xlabel(xlabel, fontsize=12, labelpad=15)
    plt.ylabel(ylabel, fontsize=12)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # # 绘制实验之间对比
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # test_path = os.path.join(current_dir, "outputs", "combat_matrix.csv")
    # print(f"Testing draw on: {test_path}")
    # draw_combat_matrix(test_path)
    
    # 绘制实验内自博弈对比
    name = 'IL_and_PFSP_分阶段_混规则对手_挑战-run-20260123-203921'
    log_dir = os.path.join(project_root, "结果展示", "logs", name)
    draw_combat_matrix(
        csv_path = os.path.join(project_root, "结果展示", "outputs", "history_combat_matrix.csv"), 
        team_labels = ['Progress 33%', 'Progress 67%', 'Final 100%'],
        title="Cross-Play Score Matrix: Training Progress Evaluation",
        xlabel="Opponent / Column",
        ylabel="Evaluated / Row",
        cbar_label="Score Rate",
    )