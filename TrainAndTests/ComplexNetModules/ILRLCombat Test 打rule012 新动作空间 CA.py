import os
import sys
import numpy as np
import torch
import argparse
import glob
import re
from math import pi
import time
import datetime

# # --- 1. 项目路径和模块导入 ---

# project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(project_root)

from _context import *

from BasicRules import basic_rules
# from Envs.Tasks.ChooseStrategyEnv20 import ChooseStrategyEnv
from Envs.Tasks.ChooseStrategyEnv2_2 import * # 1218-104003
from Envs.battle6dof1v1_missile0919 import launch_missile_immediately
from Algorithms.PPOHybrid23_0_1 import PolicyNetHybrid, HybridActorWrapper # 纯MLP
# from Algorithms.PPOHybrid23_2 import PPOHybrid, ValueNet, PolicyNetHybrid, HybridActorWrapper # 带通道注意力

# --- [修正] 在此处直接定义缺失的常量 ---
action_cycle_multiplier = 30
dt_maneuver = 0.2
# -----------------------------------------

# --- 2. 辅助函数 ---
from Utilities.LocateDirAndAgents2 import get_latest_log_dir, find_latest_agent_path

def create_initial_state():
    """创建固定的初始状态"""
    blue_height, red_height = 9000, 9000
    red_psi, blue_psi = -pi / 2, pi / 2
    red_N, red_E = 0, 45e3
    blue_N, blue_E = 0, -45e3
    DEFAULT_RED_BIRTH_STATE = {'position': np.array([red_N, red_height, red_E]), 'psi': red_psi}
    DEFAULT_BLUE_BIRTH_STATE = {'position': np.array([blue_N, blue_height, blue_E]), 'psi': blue_psi}
    return DEFAULT_RED_BIRTH_STATE, DEFAULT_BLUE_BIRTH_STATE

# --- 3. 主程序 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser("RL/IL Combat Test")
    parser.add_argument("--agent-id", type=int, default=None, help="Specific agent ID to test. If None, loads the latest.")
    parser.add_argument("--mission-name", type=str, default='RL_combat_PFSP_简单熵_区分左右_CA_变熵', help="Mission name to find the log directory.")
    args = parser.parse_args()

    'RL_combat_PFSP_简单熵_区分左右'
    'IL_RL_combat_PFSP_简单熵_区分左右'
    'RL_combat_PFSP_简单熵_区分左右_无淘汰机制'
    '打莽夫_左右_noCA_4epochs'
    '打莽夫_左右_CA_4epochs'
    
    
    # --- 环境和模型参数 (必须与训练时一致) ---
    env_args = argparse.Namespace(max_episode_len=10*60, R_cage=55e3)
    hidden_dim = [128, 128, 128]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 初始化环境 ---
    env = ChooseStrategyEnv(env_args, tacview_show=0)
    
    state_dim = env.obs_dim
    action_dims_dict = {'cont': 0, 'cat': env.fly_act_dim, 'bern': env.fire_dim}

    # --- 查找并加载模型 ---
    logs_root_dir = os.path.join(project_root, "logs/combat")
    latest_log_dir = get_latest_log_dir(logs_root_dir, args.mission_name)
    
    # 如果要硬编码为本地绝对路径，使用原始字符串并检查存在性
    # hardcoded = r'D:\3_Machine_Learning_in_Python\project03_fire_and_dodge_missile\logs\combat\RL_combat_PFSP-run-20251215-175820'
    # if os.path.exists(hardcoded):
    #     latest_log_dir = hardcoded
    
    if not latest_log_dir:
        raise FileNotFoundError(f"No log directory found for mission '{args.mission_name}' in '{logs_root_dir}'")
    
    agent_path = find_latest_agent_path(latest_log_dir, args.agent_id)
    if not agent_path:
        raise FileNotFoundError(f"No agent file found in '{latest_log_dir}' (ID: {args.agent_id or 'latest'})")

    print()
    print(f"Found log directory: {latest_log_dir}")
    print(f"Loading agent weights from: {agent_path}")
    print()

    # 实例化模型结构并加载权重
    actor_net = PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict).to(device)
    # 注意：测试时只需要 Actor Wrapper，不需要完整的 PPO agent
    actor_wrapper = HybridActorWrapper(actor_net, action_dims_dict, None, device).to(device)
    actor_wrapper.load_state_dict(torch.load(agent_path, map_location=device, weights_only=1), strict=False)
    actor_wrapper.eval() # **非常重要**：设置为评估模式

    env = ChooseStrategyEnv(env_args, tacview_show=1)
    env.shielded = 1
    env.no_out = 0
    
    # --- 循环测试 ---
    rule_opponents = [0, 1, 2]
    t_bias = 0

    try:
        for rule_num in rule_opponents:
            print("\n" + "="*50)
            print(f"--- Starting Test: Loaded Agent vs Rule_{rule_num} ---")
            print("="*50)

            # 重置环境
            DEFAULT_RED_BIRTH_STATE, DEFAULT_BLUE_BIRTH_STATE = create_initial_state()
            env.reset(red_birth_state=DEFAULT_RED_BIRTH_STATE, blue_birth_state=DEFAULT_BLUE_BIRTH_STATE)

            done = False
            last_r_action_label = 0
            last_b_action_label = 0
            b_action_label = 0

            # 回合仿真循环
            for count in range(round(env_args.max_episode_len / dt_maneuver)):
                if not env.running or done:
                    break

                r_obs, r_check_obs = env.obs_1v1('r', pomdp=1)
                b_obs, b_check_obs = env.obs_1v1('b', pomdp=1)

                # 决策
                if count % action_cycle_multiplier == 0:
                    # 红方 (规则智能体)
                    r_state_check = env.unscale_state(r_check_obs)
                    r_action_label, r_fire = basic_rules(r_state_check, rule_num, last_action=last_r_action_label)
                    last_r_action_label = r_action_label
                    if r_fire:
                        launch_missile_immediately(env, 'r')

                    # 蓝方 (RL 智能体)
                    # -- 规则
                    # b_state_check = env.unscale_state(b_check_obs)
                    # b_action_label, b_fire = basic_rules(b_state_check, rule_num, last_action=last_b_action_label)
                    # last_b_action_label = b_action_label
                    # -- 训练
                    with torch.no_grad():
                        # **修正点：使用正确的、已加载权重的 actor_wrapper**
                        b_action_exec, _, _, b_action_check = actor_wrapper.get_action(b_obs, \
                                    explore={'cont':0, 'cat':0, 'bern':1}, check_obs=b_check_obs, bern_threshold=0.38) # , bern_threshold= 1e-4
                        
                    b_action_label = b_action_exec['cat'][0]
                    b_fire = b_action_exec['bern'][0]
                    print("开火概率", b_action_check['bern'][0])
                    
                    # b_action_label = np.random.choice([4,6,13])

                    if b_fire:
                        launch_missile_immediately(env, 'b', tabu=0)
                    
                    # print(f"Time: {env.t:.1f}s, Blue Action Probs: Maneuver={b_action_check['cat']}, Fire={b_action_check['bern'][0]:.2f}")

                # 执行机动并步进
                r_maneuver = env.maneuver14(env.RUAV, r_action_label)
                b_maneuver = env.maneuver14LR(env.BUAV, b_action_label)
                env.step(r_maneuver, b_maneuver)
                done, b_rew_event, b_rew_constraint, b_rew_shaping = env.combat_terminate_and_reward('b', b_action_label, b_fire, action_cycle_multiplier)
                done = done
                env.render(t_bias=t_bias)

            # 报告结果
            result = "Draw"
            if env.win: result = "Win"
            elif env.lose: result = "Lose"
            print(f"\n--- Test Finished. Result for Blue (Loaded Agent): {result} ---")
            
            env.clear_render(t_bias=t_bias)
            t_bias += env.t
            
            # input("Press Enter to continue to the next test...")

    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    finally:
        env.end_render()
        print("\nAll tests completed.")

