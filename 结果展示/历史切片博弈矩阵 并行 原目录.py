import os
import json
import re
import time
import torch
import numpy as np
import argparse
import random
import pandas as pd
from multiprocessing import Pool, cpu_count 

from _context import * # 包含 project_root
from Envs.Tasks.ChooseStrategyEnv2_2 import ChooseStrategyEnv
from Algorithms.PPOHybrid23_0 import PolicyNetHybrid, HybridActorWrapper
from Envs.battle6dof1v1_missile0919 import launch_missile_immediately
from Utilities.LocateDirAndAgents2 import get_latest_log_dir
from read_n_draw_inter_experiment_tests import draw_combat_matrix

# --- 1. 参数配置 ---
action_cycle_multiplier = 30
dt_maneuver = 0.2
TOTAL_ROUNDS = 100 # 每两队之间打100场
TEAM_SIZE = 25     # 每队成员数
using_explore_maneuver = 1  # 是否在实验间测试的时候允许动作有随机性

def get_agent_teams(log_dir):
    """根据文件编号划分三个进度的队伍"""
    files = [f for f in os.listdir(log_dir) if f.startswith('actor_rein') and f.endswith('.pt')]
    # 提取编号并排序
    agents = []
    for f in files:
        match = re.search(r'actor_rein(\d+)', f)
        if match:
            agents.append({'id': int(match.group(1)), 'path': os.path.join(log_dir, f)})
    
    agents.sort(key=lambda x: x['id'])
    total_count = len(agents)
    
    # 定义进度索引 (1/3, 2/3, 1.0)
    indices = [total_count // 3, (2 * total_count) // 3, total_count - 1]
    teams = []
    
    for idx in indices:
        # 在该进度点附近进行间隔采样
        start = max(0, idx - 10)
        end = min(total_count, idx + 10)
        sample_pool = agents[start:end]
        # 确保每队只要 TEAM_SIZE 个
        team = sample_pool[::max(1, len(sample_pool)//TEAM_SIZE)][:TEAM_SIZE]
        teams.append(team)
    
    return teams

# --- 保持原样 ---
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
                r_act, _, _, _ = red_wrapper.get_action(r_obs, explore={'cat':using_explore_maneuver,'bern':1})
                b_act, _, _, _ = blue_wrapper.get_action(b_obs, explore={'cat':using_explore_maneuver,'bern':1})
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

# --- 并行工作函数 ---
def worker_process_battle(args_pack):
    """
    子进程执行函数
    """
    blue_path, red_path = args_pack
    
    # 强制在 Worker 中使用 CPU
    device = torch.device("cpu")
    torch.set_num_threads(1) 
    
    # 1. 初始化环境
    # 注意：这里假设 Namespace 参数是固定的，如果需要动态传参需修改 args_pack
    env = ChooseStrategyEnv(argparse.Namespace(max_episode_len=600, R_cage=55e3), tacview_show=0)
    state_dim, action_dims = env.obs_dim, {'cont':0, 'cat':env.fly_act_dim, 'bern':env.fire_dim}
    
    # 2. 初始化模型
    blue_wrapper = HybridActorWrapper(PolicyNetHybrid(state_dim, [128,128,128], action_dims), action_dims, None, device).to(device)
    red_wrapper = HybridActorWrapper(PolicyNetHybrid(state_dim, [128,128,128], action_dims), action_dims, None, device).to(device)
    
    # 3. 加载权重
    try:
        blue_wrapper.load_state_dict(torch.load(blue_path, map_location=device, weights_only=True))
        red_wrapper.load_state_dict(torch.load(red_path, map_location=device, weights_only=True))
    except Exception as e:
        print(f"模型加载出错: {e}")
        return 0.5
    
    blue_wrapper.eval()
    red_wrapper.eval()
    
    # 4. 调用原始函数
    return run_battle(env, blue_wrapper, red_wrapper, device)

# --- 主程序 ---
if __name__ == "__main__":
    name = 'IL_and_PFSP_带自模仿_混规则对手_平衡_并行-run-20260129-214607'


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    log_dir = os.path.join(project_root, "logs","combat", name)
    
    # 1. 准备队伍
    teams = get_agent_teams(log_dir)
    team_labels = ['Progress 33%', 'Progress 67%', 'Final 100%']
    results_matrix = np.zeros((3, 3))
    np.fill_diagonal(results_matrix, 0.5)

    # 并行配置
    num_processes = min(cpu_count(), 20)
    print(f"\nStart Parallel Simulation ({num_processes} cores)...")
    total_start_time = time.time()

    # 2. 并行计算
    with Pool(processes=num_processes) as pool:
        for i in range(3):      # Blue (Row)
            for j in range(3):  # Red (Col)
                if i == j: continue
                
                # 只计算下三角 (i > j)，即进度靠后的打进度靠前的
                if i > j:
                    print(f"Testing Blue:{team_labels[i]} vs Red:{team_labels[j]}...")
                    
                    # 准备任务列表
                    tasks = []
                    for _ in range(TOTAL_ROUNDS):
                        # 注意：teams[i] 是一个包含字典的列表，需要取 ['path']
                        blue_agent = random.choice(teams[i])['path']
                        red_agent = random.choice(teams[j])['path']
                        tasks.append((blue_agent, red_agent))
                    
                    # 并行执行
                    # map 会返回一个结果列表 [1.0, 0.0, 0.5 ...]
                    results = pool.map(worker_process_battle, tasks)
                    
                    win_rate = sum(results) / len(results)
                    
                    # 填充矩阵
                    results_matrix[i, j] = win_rate
                    results_matrix[j, i] = 1.0 - win_rate
                    
                    print(f"  -> Result: {win_rate:.2f}")

    total_elapsed = time.time() - total_start_time
    print(f"\nTotal Time: {total_elapsed:.2f}s")

    # 3. 保存结果 CSV
    os.makedirs(os.path.join(project_root, "结果展示", "outputs"), exist_ok=True)
    csv_path = os.path.join(project_root, "结果展示", "outputs", "history_combat_matrix.csv")
    df = pd.DataFrame(results_matrix, index=team_labels, columns=team_labels)
    df.to_csv(csv_path, float_format="%.4f", encoding="utf-8-sig")
    print(f"博弈矩阵已保存到: {csv_path}")

    # 4. [修改] 调用公共绘图函数
    print("正在绘图...")
    draw_combat_matrix(
        csv_path, 
        team_labels, 
        title="Cross-Play Score Matrix: Training Progress Evaluation",
        xlabel="Opponent / Column",
        ylabel="Evaluated / Row",
        cbar_label="Score Rate",
    )