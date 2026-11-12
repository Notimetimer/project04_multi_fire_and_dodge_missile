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
    args.R_cage = 60.3e3

    # 场地边界与默认机数/挂载统一来自 init_states.json
    init_cfg = load_init_states()
    vertices = init_cfg['vertices']
    red_init_num = init_cfg['red_init_num'] if RED_INIT_NUM is None else int(RED_INIT_NUM)
    blue_init_num = init_cfg['blue_init_num'] if BLUE_INIT_NUM is None else int(BLUE_INIT_NUM)
    print(f"部署规模：红 {red_init_num} vs 蓝 {blue_init_num}（网络按该规模构建）")

    # 加载同构 actor（CombatHanSAC_parall_homo 产出：actor_rein{batch}.pt）
    upper_log_dir = r"D:\3_Machine_Learning_in_Python\project04_multi_fire_and_dodge_missile\logs\combat"
    LOG_DIR = os.path.join(upper_log_dir,
                           "HanSAC_homo_JPO-run-20261111-224913",  # TODO 改成实际训练目录
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
        temperature=None, enm_chosen_visable=0, use_tacview=USE_TACVIEW)

    print(f"\n红方(HanSAC) vs 蓝方(规则{stats['rules_num']})，共{N_EPISODES}场：")
    print(f"胜 {stats['wins']} / 负 {stats['loses']} / 平 {stats['draws']}，"
          f"综合得分率 {stats['result']:.2%}，死光 {stats['perish_togethers']} 场")
    print(f"累计击落 {stats['kill_count']} 架，累计损失 {stats['die_count']} 架")
