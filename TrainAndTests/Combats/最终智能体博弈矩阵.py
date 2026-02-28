import os
import json
import re
import time
import torch
import numpy as np
import argparse
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from _context import * # 包含 project_root
from Envs.Tasks.ChooseStrategyEnv2_2 import ChooseStrategyEnv
from Algorithms.PPOHybrid23_0 import PolicyNetHybrid, HybridActorWrapper
from Envs.battle6dof1v1_missile0919 import launch_missile_immediately
# --- 2. 辅助函数 ---
from Utilities.LocateDirAndAgents2 import get_latest_log_dir

# --- 1. 配置参数 ---
TOTAL_ROUNDS = 10 # 100    # 每对任务之间对抗 100 场
TEAM_SIZE = 5 # 50        # 每队从 Elo 排行中取前 50 名
action_cycle_multiplier = 30
dt_maneuver = 0.2

# --- 2. 核心辅助函数 ---

def get_top_elo_agents(log_dir, top_n=50):
    """从 elo_ratings.json 中筛选 Elo 分数最高的 top_n 个非规则智能体"""
    elo_file = os.path.join(log_dir, 'elo_ratings.json')
    if not os.path.exists(elo_file):
        raise FileNotFoundError(f"在目录 {log_dir} 中未找到 elo_ratings.json")
        
    with open(elo_file, 'r') as f:
        elo_data = json.load(f)
    
    # 筛选规则：必须以 actor_rein 开头，排除 Rule_ 和 __CURRENT_MAIN__
    agent_list = []
    for key, val in elo_data.items():
        if key.startswith('actor_rein'):
            agent_list.append({'name': f"{key}.pt", 'elo': val})
    
    # 按 Elo 分数从高到低排序
    agent_list.sort(key=lambda x: x['elo'], reverse=True)
    
    # 提取完整路径
    top_agents_paths = [os.path.join(log_dir, item['name']) for item in agent_list[:top_n]]
    
    if len(top_agents_paths) < top_n:
        print(f"注意：{log_dir} 仅找到 {len(top_agents_paths)} 个有效智能体，不足 {top_n} 个。")
        
    return top_agents_paths

def run_battle(env, blue_wrapper, red_wrapper, device):
    """仿真逻辑 (保持与文件 1 一致)"""
    env.reset(red_init_ammo=6, blue_init_ammo=6)
    done = False
    r_label, b_label = 0, 0
    
    for count in range(3000):
        if done: break
        if count % action_cycle_multiplier == 0:
            r_obs, r_check = env.obs_1v1('r', pomdp=1)
            b_obs, b_check = env.obs_1v1('b', pomdp=1)
            with torch.no_grad():
                r_act, _, _, _ = red_wrapper.get_action(r_obs, explore={'cat':0,'bern':1}, check_obs=r_check)
                b_act, _, _, _ = blue_wrapper.get_action(b_obs, explore={'cat':0,'bern':1}, check_obs=b_check)
            if r_act['bern'][0]: launch_missile_immediately(env, 'r')
            if b_act['bern'][0]: launch_missile_immediately(env, 'b')
            r_label, b_label = r_act['cat'][0], b_act['cat'][0]

        r_maneuver = env.maneuver14LR(env.RUAV, r_label)
        b_maneuver = env.maneuver14LR(env.BUAV, b_label)
        env.step(r_maneuver, b_maneuver)
        done, _, _, _ = env.combat_terminate_and_reward('b', b_label, b_act['bern'][0], action_cycle_multiplier)
    
    if env.win: return 1.0   # 蓝胜
    if env.lose: return 0.0  # 红胜
    return 0.5               # 平局

# --- 3. 主程序 ---
if __name__ == "__main__":
    # --- [在此处修改输入列表] ---
    mission_names = [
        'IL_and_PFSP_2元奖励_小alpha_清理老顽固-run-20260111-125315', # 任务1
        'IL_and_PFSP_2元奖励_更小alpha_清理老顽固-run-20260111-125324', # 任务2 (示例重复)
        'IL_and_PFSP_2元奖励_分阶段学习-run-20260110-173920', # 任务3
        'IL_and_PFSP_2元奖励_分阶段-run-20260109-201911'  # 任务4
    ]
    
    team_labels = [
        'Algo_A',
        'Algo_B',
        'Algo_C',
        'Algo_D',
    ]
    
    # 强制校验长度
    if len(mission_names) != len(team_labels):
        raise ValueError(f"输入错误：任务目录数量({len(mission_names)}) 与 标签数量({len(team_labels)}) 不一致！")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logs_root_dir = os.path.join(project_root, "logs/combat")
    
    # 1. 准备各算法的 Top 50 精英队
    teams = []
    print("正在准备各任务精英智能体...")
    for name in mission_names:
        # log_dir = get_latest_log_dir(logs_root_dir, name)
        log_dir = os.path.join(logs_root_dir, name)
        if not log_dir:
            raise FileNotFoundError(f"未找到任务目录: {name}")
        teams.append(get_top_elo_agents(log_dir, TEAM_SIZE))

    num_teams = len(teams)
    results_matrix = np.zeros((num_teams, num_teams))
    np.fill_diagonal(results_matrix, 0.5)

    # 2. 初始化环境和模型结构
    env = ChooseStrategyEnv(argparse.Namespace(max_episode_len=600, R_cage=55e3), tacview_show=0)
    state_dim, action_dims = env.obs_dim, {'cont':0, 'cat':env.fly_act_dim, 'bern':env.fire_dim}
    
    actor_blue = HybridActorWrapper(PolicyNetHybrid(state_dim, [128,128,128], action_dims), action_dims, None, device).to(device)
    actor_red = HybridActorWrapper(PolicyNetHybrid(state_dim, [128,128,128], action_dims), action_dims, None, device).to(device)

    # 3. 跨任务博弈计算
    print(f"\n开始跨任务博弈矩阵计算 ({num_teams}x{num_teams})...")
    start_time = time.time()
    
    for i in range(num_teams):      # 行 i 为蓝方 (Evaluated)
        for j in range(num_teams):  # 列 j 为红方 (Opponent)
            if i == j: continue
            
            # 同样使用对称性，只跑 i > j
            if i > j:
                print(f"正在对抗: [Row]{team_labels[i]} (Blue) vs [Col]{team_labels[j]} (Red)...")
                total_score = 0
                for r in range(TOTAL_ROUNDS):
                    # 从各自的 Top 50 中随机抽样
                    blue_path = random.choice(teams[i])
                    red_path = random.choice(teams[j])
                    
                    # 加载并设置为评估模式
                    actor_blue.load_state_dict(torch.load(blue_path, map_location=device, weights_only=True))
                    actor_red.load_state_dict(torch.load(red_path, map_location=device, weights_only=True))
                    actor_blue.eval(); actor_red.eval()
                    
                    total_score += run_battle(env, actor_blue, actor_red, device)
                
                win_rate = total_score / TOTAL_ROUNDS
                results_matrix[i, j] = win_rate       # i 打赢 j 的胜率
                results_matrix[j, i] = 1.0 - win_rate # j 打赢 i 的胜率 (即 i 作为蓝方输掉的概率)
                print(f"  -> {team_labels[i]} 对阵 {team_labels[j]} 胜率: {win_rate:.2f}")

    print(f"\n矩阵计算完成！总耗时: {time.time() - start_time:.2f}s")

    # 4. 绘图部分 (PRGn 配色 + X轴置顶)
    plt.figure(figsize=(num_teams + 4, num_teams + 2))
    ax = sns.heatmap(
        results_matrix, 
        annot=True, 
        fmt=".2f", 
        cmap="PRGn", 
        center=0.5,
        xticklabels=team_labels, 
        yticklabels=team_labels,
        square=True, 
        linewidths=0.5,
        cbar_kws={"label": "Blue Team Win Rate", "shrink": 0.8}
    )
    
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    
    plt.title("Cross-Algorithm Final Agents Combat Matrix", fontsize=14, pad=40)
    plt.xlabel("Red Team (Opponent / Column)", fontsize=12, labelpad=15)
    plt.ylabel("Blue Team (Evaluated / Row)", fontsize=12)
    
    plt.tight_layout()
    plt.show()