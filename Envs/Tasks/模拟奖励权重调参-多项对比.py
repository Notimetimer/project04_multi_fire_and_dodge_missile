# -*- coding: utf-8 -*-
"""奖励权重调参模拟：观察单回合奖励与蒙特卡洛回报曲线。"""

import re
import tkinter as tk
from tkinter import ttk, messagebox, font as tkfont

import numpy as np
import matplotlib

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

# 支持中文显示
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

defaults = {
        "总步数": "300",
        "密集奖励": "0.2",
        "偶发奖励": "15",
        "结果奖励": "50",
        "gamma": "0.988",
        "稀疏奖励位置": "-60,-120,-180",
    }

def simulate_episode(total_steps, dense_mag, occasional_mag, result_mag, gamma, event_indices):
    """生成一个回合的奖励、蒙特卡洛回报，以及各奖励成分的回报贡献。"""
    dense = np.full(total_steps, dense_mag, dtype=np.float64)

    # 稀疏奖励位置由用户指定，支持负索引（从总步数末尾倒数）
    sparse = np.zeros(total_steps, dtype=np.float64)
    if event_indices is not None and len(event_indices):
        event_indices = np.asarray(event_indices, dtype=np.int64)
        sparse[event_indices] += occasional_mag

    # 结果奖励放在最后一步
    result = np.zeros(total_steps, dtype=np.float64)
    result[-1] += result_mag

    rewards = dense + sparse + result

    def make_returns(arr):
        returns = np.empty_like(arr)
        g = 0.0
        for t in range(total_steps - 1, -1, -1):
            g = arr[t] + gamma * g
            returns[t] = g
        return returns

    returns = make_returns(rewards)
    dense_r = make_returns(dense)
    sparse_r = make_returns(sparse)
    result_r = make_returns(result)

    return rewards, returns, event_indices, dense_r, sparse_r, result_r


def generate_and_plot(entries):
    try:
        total_steps = int(entries["总步数"].get().strip())
        dense_mag = float(entries["密集奖励"].get().strip())
        occasional_mag = float(entries["偶发奖励"].get().strip())
        result_mag = float(entries["结果奖励"].get().strip())
        gamma = float(entries["gamma"].get().strip())

        pos_str = entries["稀疏奖励位置"].get().strip()
        # 兼容中英文逗号、分号、空格与中文负号
        pos_str = pos_str.replace("−", "-").replace("—", "-").replace("–", "-")
        positions = []
        for token in re.split(r"[,，;；\s]+", pos_str):
            token = token.strip()
            if not token:
                continue
            p = int(token)
            if p < 0:
                p = total_steps + p
            positions.append(p)
    except ValueError:
        messagebox.showerror("输入错误", "请检查所有参数和稀疏奖励位置均为有效数字")
        return
    if not (0.0 < gamma <= 1.0):
        messagebox.showerror("输入错误", "gamma 必须在 (0, 1] 之间")
        return
    if total_steps > 0:
        positions = [max(0, min(p, total_steps - 1)) for p in positions]
    event_indices = np.array(sorted(set(positions)), dtype=np.int64)
    rewards, returns, event_indices, dense_r, sparse_r, result_r = simulate_episode(
        total_steps, dense_mag, occasional_mag, result_mag, gamma, event_indices
    )
    steps = np.arange(-total_steps + 1, 1)

    fig, (ax_reward, ax_weight) = plt.subplots(
        2, 1, sharex=True, figsize=(10, 7)
    )

    # 奖励曲线
    ax_reward.plot(steps, rewards, color="steelblue", label="单步奖励")
    ax_reward.scatter(
        steps[event_indices],
        rewards[event_indices],
        color="red",
        zorder=5,
        label="偶发事件奖励",
    )
    ax_reward.scatter(
        [steps[-1]],
        [rewards[-1]],
        color="green",
        zorder=5,
        label="结果奖励",
    )

    # 标注事件步数和结果步数
    for idx in event_indices:
        ax_reward.annotate(
            str(idx),
            (steps[idx], rewards[idx]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=7,
            color="red",
        )
    ax_reward.annotate(
        str(total_steps - 1),
        (steps[-1], rewards[-1]),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        fontsize=8,
        color="green",
        fontweight="bold",
    )

    ax_reward.set_title(f"奖励曲线（总步数：{total_steps}）")
    ax_reward.set_ylabel("奖励")
    ax_reward.legend(loc="best")
    ax_reward.grid(True, linestyle=":", alpha=0.6)

    # 第二张图：各奖励成分对回报的权重占比（使用绝对值）
    total_abs = np.abs(dense_r) + np.abs(sparse_r) + np.abs(result_r)
    total_abs = np.where(total_abs == 0, 1, total_abs)
    ax_weight.plot(
        steps, np.abs(dense_r) / total_abs * 100, color="steelblue", label="密集奖励"
    )
    ax_weight.plot(
        steps, np.abs(sparse_r) / total_abs * 100, color="red", label="稀疏/偶发奖励"
    )
    ax_weight.plot(
        steps, np.abs(result_r) / total_abs * 100, color="green", label="结果奖励"
    )
    ax_weight.set_title(
        f"各奖励成分在回报中的权重占比（总步数：{total_steps}）"
    )
    ax_weight.set_xlabel("倒计时（步）")
    ax_weight.set_ylabel("占比（%）")
    ax_weight.legend(loc="best")
    ax_weight.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    plt.show(block=False)


def main():
    root = tk.Tk()

    # 让 tkinter 控件使用支持中文的字体
    try:
        default_font = tkfont.nametofont("TkDefaultFont")
        default_font.configure(family="Microsoft YaHei", size=10)
    except Exception:
        pass
    try:
        style = ttk.Style()
        style.configure(".", font=("Microsoft YaHei", 10))
    except Exception:
        pass

    root.title("奖励权重调参模拟")

    

    entries = {}
    for i, (label, default) in enumerate(defaults.items()):
        ttk.Label(root, text=label + ":").grid(
            row=i, column=0, padx=10, pady=5, sticky="e"
        )
        ent = ttk.Entry(root, width=15)
        ent.insert(0, default)
        ent.grid(row=i, column=1, padx=10, pady=5)
        entries[label] = ent

    ttk.Button(
        root,
        text="生成曲线",
        command=lambda: generate_and_plot(entries),
    ).grid(row=len(defaults), column=0, columnspan=2, pady=10)

    root.mainloop()


if __name__ == "__main__":
    main()
