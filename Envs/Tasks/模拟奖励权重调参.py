# -*- coding: utf-8 -*-
"""奖励权重调参模拟：观察单回合奖励与蒙特卡洛回报曲线（实时更新版）。"""

import re
import sys
import tkinter as tk
from tkinter import ttk, messagebox, font as tkfont

import numpy as np
import matplotlib

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 支持中文显示
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

defaults = {
    "总步数": "300",
    "gamma": "0.988",
    "偶发奖励": "15",
    "结果奖励": "50",
    "密集奖励": "0.2",
    "稀疏奖励位置": "-60,-120,-180",
}


def simulate_episode(total_steps, dense_mag, occasional_mag, result_mag, gamma, event_indices):
    """生成一个回合的奖励、蒙特卡洛回报，以及各奖励成分的回报贡献。"""
    # 密集奖励的波形设置
    t = np.arange(total_steps)
    # 常数
    dense = np.full(total_steps, dense_mag, dtype=np.float64)
    # # 正弦波
    # dense = dense_mag * np.sin(2 * np.pi * t / 50)  # 50步为一个周期
    # # 衰减正弦波
    # dense = dense_mag * np.exp(-t / (total_steps * 0.5)) * np.sin(2 * np.pi * t / 30)
    # # 带偏置的正弦波（始终为正但有起伏）：
    # dense = dense_mag * (1.0 + 0.5 * np.sin(2 * np.pi * t / 50))


    # 稀疏奖励位置由用户指定，支持负索引（从总步数末尾倒数）
    sparse = np.zeros(total_steps, dtype=np.float64)
    if event_indices is not None and len(event_indices):
        event_indices = np.asarray(event_indices, dtype=np.int64)
        sparse[event_indices] += occasional_mag

    # 结果奖励放在最后一步
    result = np.zeros(total_steps, dtype=np.float64)
    result[-1] += result_mag

    rewards = dense + sparse + result
    # 计算每个奖励成分的蒙特卡洛回报, 带符号计算
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


class RewardWeightTuner:
    def __init__(self, root):
        self.root = root
        self.root.title("奖励权重调参模拟")
        self.root.geometry("1100x750")

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

        # 布局：左侧控制面板，右侧绘图区
        self.ctrl_frame = ttk.Frame(root, width=300, padding=(10, 10))
        self.ctrl_frame.pack(side=tk.LEFT, fill=tk.Y)

        self.plot_frame = ttk.Frame(root)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 初始化 Matplotlib
        self.fig, (self.ax_reward, self.ax_weight) = plt.subplots(
            2, 1, sharex=True, figsize=(9, 7)
        )
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 确保关闭窗口正确退出
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.entries = {}
        self._build_control_panel()
        self.update_plot()  # 初次渲染

    def _build_control_panel(self):
        for i, (label, default) in enumerate(defaults.items()):
            ttk.Label(self.ctrl_frame, text=label + ":").grid(
                row=i, column=0, padx=10, pady=8, sticky="e"
            )
            ent = ttk.Entry(self.ctrl_frame, width=18)
            ent.insert(0, default)
            ent.grid(row=i, column=1, padx=10, pady=8)
            self.entries[label] = ent

            # 参数变化时实时更新
            ent.bind("<Return>", lambda e: self.update_plot())
            ent.bind("<FocusOut>", lambda e: self.update_plot())
            ent.bind("<KeyRelease>", lambda e: self._debounce_update())

    def _debounce_update(self):
        # 输入框键入时防抖，避免每次按键都重绘
        if hasattr(self, "_after_id") and self._after_id:
            self.root.after_cancel(self._after_id)
        self._after_id = self.root.after(300, self.update_plot)

    def _parse_inputs(self):
        try:
            total_steps = int(self.entries["总步数"].get().strip())
            dense_mag = float(self.entries["密集奖励"].get().strip())
            occasional_mag = float(self.entries["偶发奖励"].get().strip())
            result_mag = float(self.entries["结果奖励"].get().strip())
            gamma = float(self.entries["gamma"].get().strip())

            pos_str = self.entries["稀疏奖励位置"].get().strip()
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

            if not (0.0 < gamma <= 1.0):
                raise ValueError("gamma 必须在 (0, 1] 之间")
            if total_steps <= 0:
                raise ValueError("总步数必须大于 0")

            positions = [max(0, min(p, total_steps - 1)) for p in positions]
            event_indices = np.array(sorted(set(positions)), dtype=np.int64)

            return total_steps, dense_mag, occasional_mag, result_mag, gamma, event_indices
        except ValueError as e:
            # 输入不完整或非法时不重绘，保留上一次的合法状态
            return None

    def update_plot(self):
        parsed = self._parse_inputs()
        if parsed is None:
            return

        total_steps, dense_mag, occasional_mag, result_mag, gamma, event_indices = parsed

        rewards, returns, event_indices, dense_r, sparse_r, result_r = simulate_episode(
            total_steps, dense_mag, occasional_mag, result_mag, gamma, event_indices
        )
        steps = np.arange(-total_steps + 1, 1)

        # 奖励曲线
        self.ax_reward.clear()
        self.ax_reward.plot(steps, rewards, color="steelblue", label="单步奖励")
        self.ax_reward.scatter(
            steps[event_indices],
            rewards[event_indices],
            color="red",
            zorder=5,
            label="偶发事件奖励",
        )
        self.ax_reward.scatter(
            [steps[-1]],
            [rewards[-1]],
            color="green",
            zorder=5,
            label="结果奖励",
        )

        for idx in event_indices:
            self.ax_reward.annotate(
                str(idx),
                (steps[idx], rewards[idx]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=7,
                color="red",
            )
        self.ax_reward.annotate(
            str(total_steps - 1),
            (steps[-1], rewards[-1]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
            color="green",
            fontweight="bold",
        )

        self.ax_reward.set_title(f"奖励曲线（总步数：{total_steps}）")
        self.ax_reward.set_ylabel("奖励")
        self.ax_reward.legend(loc="best")
        self.ax_reward.grid(True, linestyle=":", alpha=0.6)

        # 第二张图：各奖励成分对回报的权重占比（使用绝对值）
        self.ax_weight.clear()
        # 先算的带符号蒙特卡洛回报再取的绝对值
        total_abs = np.abs(dense_r) + np.abs(sparse_r) + np.abs(result_r)
        total_abs = np.where(total_abs == 0, 1, total_abs)
        self.ax_weight.plot(
            steps, np.abs(dense_r) / total_abs * 100, color="steelblue", label="密集奖励"
        )
        self.ax_weight.plot(
            steps, np.abs(sparse_r) / total_abs * 100, color="red", label="稀疏/偶发奖励"
        )
        self.ax_weight.plot(
            steps, np.abs(result_r) / total_abs * 100, color="green", label="结果奖励"
        )
        self.ax_weight.set_title(
            f"各奖励成分在回报中的权重占比（总步数：{total_steps}）"
        )
        self.ax_weight.set_xlabel("倒计时（步）")
        self.ax_weight.set_ylabel("占比（%）")
        self.ax_weight.legend(loc="best")
        self.ax_weight.grid(True, linestyle=":", alpha=0.6)

        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _on_close(self):
        try:
            plt.close(self.fig)
        except Exception:
            pass
        try:
            self.root.quit()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass
        try:
            sys.exit(0)
        except SystemExit:
            try:
                import os
                os._exit(0)
            except Exception:
                pass


def main():
    root = tk.Tk()
    app = RewardWeightTuner(root)
    root.mainloop()


if __name__ == "__main__":
    main()
