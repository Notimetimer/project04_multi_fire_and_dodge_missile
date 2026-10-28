"同构基线测试 main：红方全队共享同一份 actor 权重（单份 state_dict），蓝方规则"
"""
复用 VsBaseline_MA.test_worker：其 _is_single_sd 检测到传入为单份 state_dict 时，
所有红机共享同一份权重。checkpoint 来自 CombatMAPPO_parall_homo：actor_rein{batch}.pt（无 _id 后缀）。
异构测试（actor_rein{batch}_{id}.pt）见 VsBaseline_MA.py 的 main。
"""

import os
import sys
import csv
import torch
import argparse
import glob
import re
import numpy as np
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from VsBaseline_MA import test_worker
from _context import load_init_states

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


if __name__=='__main__':
    print("\n根目录为：", project_root, "\n")
    parser = argparse.ArgumentParser("UAV swarm confrontation")
    args = parser.parse_args()

    # 测试配置
    RULES_NUM = 2      # 蓝方规则编号
    N_EPISODES = 3    # 测试场次  10
    TEMPERATURE = 0.3  # 采样温度：<1更贪婪（机动/目标选取决策趋于确定）

    args.max_episode_len = 15*60
    args.R_cage = 60.3e3

    # 构建场地边界，南北长54km，东西宽100km的长方形边界
    vertices = load_init_states()['vertices']

    # 加载同构训练目录下的单份 actor（CombatMAPPO_parall_homo 产出：actor_rein{batch}.pt）
    upper_log_dir = r"D:\3_Machine_Learning_in_Python\project04_multi_fire_and_dodge_missile\logs\combat"
    LOG_DIR = os.path.join(upper_log_dir,
                           "MAPPO_homo1v1-run-20261027-235710" # "MAPPO_homo-run-20261025-105746"
                           )
    actor_files = glob.glob(os.path.join(LOG_DIR, "actor_rein*.pt"))
    if not actor_files:
        raise FileNotFoundError(f"未在 {LOG_DIR} 找到 actor_rein*.pt")
    # 只保留同构文件名 actor_rein{batch}.pt；异构 actor_rein{batch}_{id}.pt 因 _id 后缀不匹配
    homo = {}
    for f in actor_files:
        m = re.match(r'actor_rein(\d+)\.pt', os.path.basename(f))
        if m:
            homo[int(m.group(1))] = f
    if not homo:
        raise FileNotFoundError(f"未在 {LOG_DIR} 找到同构 actor_rein{{batch}}.pt（该目录只有异构命名）")
    latest_batch = max(homo.keys())

    # 加载的版本号
    loaded_version = latest_batch  # 可手动指定，如 10

    # 单份 state_dict：test_worker 检测到后所有红机共享同一份权重
    model_state_dict = torch.load(homo[loaded_version], map_location='cpu')
    print(f"加载同构 actor：batch={loaded_version}（单份，全队共享）")

    rules_num, result, wins, loses, draws, perish_togethers = test_worker(
        model_state_dict, RULES_NUM, args, num_runs=N_EPISODES, vertices=vertices,
        save_record=True, temperature=TEMPERATURE)

    print(f"\n红方(神经网络) vs 蓝方(规则{rules_num})，共{N_EPISODES}场：")
    print(f"胜 {wins} / 负 {loses} / 平 {draws}，综合得分率 {result:.2%}，死光 {perish_togethers} 场")

    # 红、蓝方 1 号奖励的折扣蒙特卡洛回报曲线（数据来自 test_worker 落盘的 uav_{id}_events.csv）
    # plt.subplots 每次调用都会新建窗口：红方 1 号是 Figure 1，蓝方 1 号（id 从 201 起）是 Figure 2
    event_log_dir = os.path.join(project_root, 'acmi_records')
    plot_uav_mc_return(event_log_dir, uav_id=1, side_label='红方')
    plot_uav_mc_return(event_log_dir, uav_id=201, side_label='蓝方')
    
    plt.show()
