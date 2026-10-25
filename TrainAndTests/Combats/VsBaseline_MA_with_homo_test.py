"同构基线测试 main：红方全队共享同一份 actor 权重（单份 state_dict），蓝方规则"
"""
复用 VsBaseline_MA.test_worker：其 _is_single_sd 检测到传入为单份 state_dict 时，
所有红机共享同一份权重。checkpoint 来自 CombatMAPPO_parall_homo：actor_rein{batch}.pt（无 _id 后缀）。
异构测试（actor_rein{batch}_{id}.pt）见 VsBaseline_MA.py 的 main。
"""

import os
import sys
import torch
import argparse
import glob
import re

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from VsBaseline_MA import test_worker

if __name__=='__main__':
    print("\n根目录为：", project_root, "\n")
    parser = argparse.ArgumentParser("UAV swarm confrontation")
    args = parser.parse_args()

    # 测试配置
    RULES_NUM = 5      # 蓝方规则编号
    N_EPISODES = 3    # 测试场次  10
    TEMPERATURE = 0.3  # 采样温度：<1更贪婪（机动/目标选取决策趋于确定）

    args.max_episode_len = 15*60
    args.R_cage = 60.3e3

    # 构建场地边界，南北长54km，东西宽100km的长方形边界
    vertices = [[50e3, 50e3], [-50e3, 50e3], [-50e3, -50e3], [50e3, -50e3]]

    # 加载同构训练目录下的单份 actor（CombatMAPPO_parall_homo 产出：actor_rein{batch}.pt）
    upper_log_dir = r"D:\3_Machine_Learning_in_Python\project04_multi_fire_and_dodge_missile\logs\combat"
    LOG_DIR = os.path.join(upper_log_dir,
                           "MAPPO_homo-run-20261024-234937"
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
