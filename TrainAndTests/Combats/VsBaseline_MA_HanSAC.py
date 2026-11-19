"""同构 HanSAC 纯部署测试 main：只加载 HanWrapper actor（不含 Q / 温度），红方全队共享同一份权重。

与 VsBaseline_MA_with_homo_test.py 的 MAPPO 版本对应，区别在于：
1. checkpoint 来自 CombatHanSAC_parall_homo 落盘的 actor_rein{batch}.pt（HanWrapper.state_dict）；
2. 网络按部署规模构建：敌机槽位=蓝方机数、队友槽位=红方机数-1，因此可以用与训练不同的规模；
3. 决策为联合分布 argmax + 开火命中概率阈值 0.8，全程不计算 Q。
"""

import os
import sys
import glob
import re
import argparse
import torch

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from CombatHanSAC_parall_homo import test_worker_han
from _context import load_init_states

if __name__ == '__main__':
    print("\n根目录为：", project_root, "\n")
    parser = argparse.ArgumentParser("UAV swarm confrontation")
    args = parser.parse_args()

    # 测试配置
    RULES_NUM = 3            # 蓝方规则编号
    N_EPISODES = 3           # 测试场次
    RED_INIT_NUM = None      # 红方机数；None 表示用 init_states.json
    BLUE_INIT_NUM = None     # 蓝方机数；None 表示用 init_states.json（可与训练规模不同）
    USE_TACVIEW = 2          # 1 动态显示，2 静默写入 acmi_records，0 关闭
    DT_MANEUVER = 0.2
    ACTION_CYCLE_MULTIPLIER = 10  # 与训练 dt_decide/dt_maneuver 一致
    LOADED_VERSION = None    # actor_rein 的 batch 号；None 表示取目录内最新
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    args.max_episode_len = 15 * 60
    args.R_cage = 62.00e3

    # 场地边界与默认机数/挂载统一来自 init_states.json
    init_cfg = load_init_states()
    vertices = init_cfg['vertices']
    red_init_num = init_cfg['red_init_num'] if RED_INIT_NUM is None else int(RED_INIT_NUM)
    blue_init_num = init_cfg['blue_init_num'] if BLUE_INIT_NUM is None else int(BLUE_INIT_NUM)
    print(f"部署规模：红 {red_init_num} vs 蓝 {blue_init_num}（网络按该规模构建）")

    # 加载同构 actor（CombatHanSAC_parall_homo 产出：actor_rein{batch}.pt）
    upper_log_dir = r"D:\3_Machine_Learning_in_Python\project04_multi_fire_and_dodge_missile\logs\combat"
    LOG_DIR = os.path.join(upper_log_dir,
                           "HanSAC_homo_JPO-run-20261115-163430",  # TODO 改成实际训练目录
                           )
    actor_files = glob.glob(os.path.join(LOG_DIR, "actor_rein*.pt"))
    if not actor_files:
        raise FileNotFoundError(f"未在 {LOG_DIR} 找到 actor_rein*.pt")
    homo = {}
    for f in actor_files:
        m = re.match(r'actor_rein(\d+)\.pt', os.path.basename(f))
        if m:
            homo[int(m.group(1))] = f
    if not homo:
        raise FileNotFoundError(f"未在 {LOG_DIR} 找到同构 actor_rein{{batch}}.pt")
    loaded_version = max(homo.keys()) if LOADED_VERSION is None else LOADED_VERSION

    # 单份 state_dict：HanWrapper 参数与敌机/队友数量无关，可跨规模加载
    model_state_dict = torch.load(homo[loaded_version], map_location='cpu')
    print(f"加载 HanSAC actor：batch={loaded_version}（单份，全队共享）")

    stats = test_worker_han(
        model_state_dict, RULES_NUM, args, device_name=DEVICE, num_runs=N_EPISODES,
        vertices=vertices, dt_move=0.07, dt_maneuver=DT_MANEUVER,
        action_cycle_multiplier=ACTION_CYCLE_MULTIPLIER,
        red_init_num=red_init_num, blue_init_num=blue_init_num,
        red_init_ammo=init_cfg['red_init_ammo'], blue_init_ammo=init_cfg['blue_init_ammo'],
        temperature=None, use_tacview=USE_TACVIEW)

    print(f"\n红方(HanSAC) vs 蓝方(规则{stats['rules_num']})，共{N_EPISODES}场：")
    print(f"胜 {stats['wins']} / 负 {stats['loses']} / 平 {stats['draws']}，"
          f"综合得分率 {stats['result']:.2%}，死光 {stats['perish_togethers']} 场")
    print(f"累计击落 {stats['kill_count']} 架，累计损失 {stats['die_count']} 架")

    # ------------------------ 红方开火概率曲线 ------------------------
    # 曲线数据来自 test_worker_han：每个红机按决策点记录 (全局时间 s, 开火概率)；
    # 多场合并在同一条时间轴上，用竖向虚线 + 交替底色标出回合边界。
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    series = stats.get('fire_prob_series', {})
    spans = stats.get('episode_spans', [])
    red_ids = sorted(series.keys()) if series else list(range(1, int(red_init_num) + 1))
    n_red = max(len(red_ids), 1)

    fig, axes = plt.subplots(n_red, 1, sharex=True,
                             figsize=(12, 2.0 * n_red + 1.0), squeeze=False)
    for i, (ax, rid) in enumerate(zip(axes[:, 0], red_ids)):
        # 回合底色：奇数回合淡灰、偶数回合留白；回合起点再用竖向虚线标出
        for k, (t0, t1) in enumerate(spans):
            if k % 2 == 0:
                ax.axvspan(t0, t1, color='0.88', alpha=0.5, lw=0)
        for k in range(1, len(spans)):
            ax.axvline(spans[k][0], color='0.45', linestyle='--', linewidth=0.8)
        pts = series.get(rid) or []
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    color='#c0392b', linewidth=1.0)
        else:
            ax.text(0.5, 0.5, '本场无决策点', transform=ax.transAxes,
                    ha='center', va='center', color='0.5')
        ax.set_ylim(0.0, 1.02)
        ax.set_ylabel(f'红机 {rid}\n开火概率', fontsize=9)
        ax.grid(alpha=0.25, linewidth=0.5)
        if i == 0:
            # 回合编号标在最上方子图的绘图区顶部（clip_on=False 允许写到轴外）
            for k, (t0, t1) in enumerate(spans):
                ax.text((t0 + t1) / 2.0, 1.01, f'第{k + 1}回合',
                        transform=ax.get_xaxis_transform(),
                        ha='center', va='bottom', fontsize=8, color='0.35',
                        clip_on=False)
    axes[-1, 0].set_xlabel('时间 (s)')
    fig.suptitle(f'红方开火概率（HanSAC vs 规则{stats["rules_num"]}，'
                 f'batch={loaded_version}，{N_EPISODES} 场）', fontsize=11)
    fig.tight_layout()
    plt.show()
