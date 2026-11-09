import os
import sys
import csv
import torch
import argparse
import glob
import re
import numpy as np
import matplotlib.pyplot as plt

def plot_uav_mc_return(event_log_dir, uav_id=1, gamma=0.97, side_label='红方'):
    """读 uav_{id}_events.csv，按回合切分后分别计算三类奖励成分的折扣蒙特卡洛回报并画图。

    成分取自 cycle_events 的三个字段：密集(shaping) / 结果(end) / 事件(event)。
    每回合 G_t^c = r_t^c + gamma*r_{t+1}^c + gamma^2*r_{t+2}^c + ...，回合开头累积器归零
    （对应"每到一个新回合重置蒙特卡洛回报为 0"）。
    上图：各成分 G_t^c（保留正负号，红=密集，绿=结果，蓝=事件）；
    下图：该成分 |G_t^c| 在"三成分 |G_t| 之和"中的占比。
    """
    csv_path = os.path.join(event_log_dir, f'uav_{uav_id}_events.csv')
    if not os.path.exists(csv_path):
        print(f"[画图] 未找到 {csv_path}，跳过")
        return

    # (csv 字段, 图例名, 颜色)：红绿蓝分别对应密集/结果/事件
    components = [
        ('dense_reward', '密集奖励(shaping)', 'red'),
        ('end_reward',   '结果奖励(end)',     'green'),
        ('event_reward', '事件奖励(event)',   'blue'),
    ]

    # 逐行读取该机三类奖励，各自独立成列
    times = []
    rewards = {key: [] for key, _, _ in components}
    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            if int(row['id']) != uav_id:
                continue
            mm, ss = row['time'].strip('()').split(',')
            times.append(int(mm) * 60 + int(ss))
            for key, _, _ in components:
                rewards[key].append(float(row[key]))
    if not times:
        print(f"[画图] {csv_path} 中没有 id={uav_id} 的记录，跳过")
        return

    # 切分回合：time 每回合从 0 重新计时，出现回退即为新回合起点
    episodes, start = [], 0
    for i in range(1, len(times)):
        if times[i] < times[i - 1]:
            episodes.append((start, i))
            start = i
    episodes.append((start, len(times)))

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
    x_offset, boundaries = 0.0, []
    for ep_i, (s, e) in enumerate(episodes):
        x = np.asarray(times[s:e], dtype=float) + x_offset
        abs_all = []
        for key, name, color in components:
            r_ep = np.asarray(rewards[key][s:e], dtype=float)
            # 反向递推折扣累加：G_t = r_t + gamma * G_{t+1}，末位之后累积器为 0
            g_ep = np.zeros_like(r_ep)
            acc = 0.0
            for k in range(len(r_ep) - 1, -1, -1):
                acc = r_ep[k] + gamma * acc
                g_ep[k] = acc
            abs_all.append(np.abs(g_ep))
            # 各回合同色，图例只在首回合登记一次；上图保留正负号，仅下图用绝对值算比例
            axes[0].plot(x, g_ep, color=color, lw=1.2, label=name if ep_i == 0 else None)
            # 密集奖励叠加瞬时值：很浅的红色细线（每周期原始 r_shaping，未折扣累加）
            if key == 'dense_reward':
                axes[0].plot(x, r_ep, color='red', lw=0.6, alpha=0.35,
                             linestyle='-',
                             label=('密集奖励(瞬时)' if ep_i == 0 else None))
            # 原始 event/end 奖励用散点+竖线直接标出（未折扣、未相互抵消），只画非零行避免铺满 y=0
            if key in ('event_reward', 'end_reward'):
                nz = np.abs(r_ep) > 1e-9
                # 结果奖励即使为 0（平局）也要在每回合结束位置标出绿点
                if key == 'end_reward':
                    nz[-1] = True
                axes[0].vlines(x[nz], 0, r_ep[nz], colors=color, linestyles=':', lw=1.0, alpha=0.8)
                axes[0].scatter(x[nz], r_ep[nz], c=color, s=36, zorder=5,
                                edgecolors='k', linewidths=0.4,
                                label=(name + '(原始)') if ep_i == 0 else None)
        # 权重 = 该成分 |G_t| / 三成分 |G_t| 之和；和为 0 时无定义，置 NaN 而非 0
        abs_all = np.vstack(abs_all)
        total = abs_all.sum(axis=0)
        share = np.divide(abs_all, total,
                          out=np.full_like(abs_all, np.nan),
                          where=total > 1e-9)
        for (_, name, color), share_c in zip(components, share):
            axes[1].plot(x, share_c, color=color, lw=1.2, label=name if ep_i == 0 else None)
        if ep_i > 0:
            boundaries.append(x[0])
        x_offset = x[-1] + 1.0

    for b in boundaries:
        for ax in axes:
            ax.axvline(b, color='gray', ls='--', lw=0.8, alpha=0.7)
    axes[0].set_ylabel(f'G_t  (γ={gamma})')
    axes[1].set_ylabel('三成分 |G_t| 之间的占比')
    axes[1].set_xlabel('累计仿真时间 (s)')
    axes[0].set_title(f'{side_label} {uav_id} 号：各奖励成分的折扣蒙特卡洛回报（共 {len(episodes)} 回合）')
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    fig.tight_layout()
    out_path = os.path.join(event_log_dir, f'uav_{uav_id}_mc_return.png')
    fig.savefig(out_path, dpi=150)
    print(f"[画图] 已保存: {out_path}（{len(episodes)} 个回合）")

