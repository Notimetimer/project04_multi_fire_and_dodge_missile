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
import copy

# --- 1. 项目路径和模块导入 ---
from _context import *

from Envs.Tasks.ChooseStrategyEnv2_2 import ChooseStrategyEnv
from TrainAndTests.Combats.BasicRules_new import basic_rules
from Envs.battle6dof1v1_missile0919 import launch_missile_immediately
from Algorithms.PPOHybrid23_0 import PolicyNetHybrid, HybridActorWrapper
# from Algorithms.PPOHybrid23 import PolicyNetHybrid, HybridActorWrapper

# --- [修正] 在此处直接定义缺失的常量 ---
action_cycle_multiplier = 30
dt_maneuver = 0.2
# -----------------------------------------

# --- 2. 辅助函数 ---
from Utilities.LocateDirAndAgents2 import get_latest_log_dir, find_latest_agent_path

def create_initial_state(randomized=0):
    """创建固定的初始状态"""
    blue_height, red_height = 9000, 9000
    red_psi, blue_psi = pi / 2, -pi / 2
    init_North = np.random.uniform(-30e3, 30e3) * int(randomized)
    red_N, red_E = init_North, -45e3
    blue_N, blue_E = init_North, 45e3
    DEFAULT_RED_BIRTH_STATE = {'position': np.array([red_N, red_height, red_E]), 'psi': red_psi}
    DEFAULT_BLUE_BIRTH_STATE = {'position': np.array([blue_N, blue_height, blue_E]), 'psi': blue_psi}
    return DEFAULT_RED_BIRTH_STATE, DEFAULT_BLUE_BIRTH_STATE

# --- 3. 主程序 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser("RL/IL Combat Test")
    parser.add_argument("--agent-id", type=int, default=None, help="Specific agent ID to test. If None, loads the latest.")
    parser.add_argument("--mission-name", type=str, default='RL_combat_no_shoot', help="Mission name to find the log directory.")
    args = parser.parse_args()



    # --- 环境和模型参数 (必须与训练时一致) ---
    env_args = argparse.Namespace(max_episode_len=10*60, R_cage=45e3)
    hidden_dim = [128, 128, 128]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 初始化环境 ---
    env = ChooseStrategyEnv(env_args, tacview_show=0)
    
    state_dim = env.obs_dim
    action_dims_dict = {'cont': 0, 'cat': env.fly_act_dim, 'bern': env.fire_dim}

    simulation_name = 'IL_and_PFSP_分阶段_混规则对手_中上-run-20260123-204031'
    
    latest_log_dir = logs_root_dir = os.path.join(project_root, "logs/combat", simulation_name)
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
    
    enm_wrapper = copy.deepcopy(actor_wrapper)

    env = ChooseStrategyEnv(env_args, tacview_show=1)
    env.shielded = 1
    
    # --- 循环测试 ---
    rule_opponents = [-1, 1, 2, 3, 4]
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
            last_b_action_label = 0
            last_r_action_label = 0
            r_action_label = 0

            # 回合仿真循环
            for count in range(round(env_args.max_episode_len / dt_maneuver)):
                if not env.running or done:
                    break

                b_obs, b_check_obs = env.obs_1v1('b', pomdp=1)
                r_obs, r_check_obs = env.obs_1v1('r', pomdp=1)

                # 决策
                if count % action_cycle_multiplier == 0:
                    # 蓝方 (对手决策)
                    if rule_num >= 0:
                        # [Fix] 传入选定的 rule_num
                        b_state_check = env.unscale_state(b_check_obs)
                        b_action_label, b_fire = basic_rules(b_state_check, rule_num, last_action=last_b_action_label)
                        b_action_exec = {'cat': None, 'bern': None}
                        b_action_exec['cat'] = np.array([b_action_label])
                        b_action_exec['bern'] = np.array([b_fire], dtype=np.float32)
                    else:
                        # [Fix] NN 对手决策
                        # 1. 使用网络获取机动动作
                        with torch.no_grad():
                            b_action_exec, _, _, b_action_check = enm_wrapper.get_action(b_obs, 
                                        explore={'cont':0, 'cat':0, 'bern':1}, )
                                        # check_obs=b_check_obs, 
                                        # bern_threshold=0.33)
                        b_action_label = b_action_exec['cat'][0]
                        b_fire = b_action_exec['bern'][0]
                        
                        # # 2. 使用规则获取开火决策 (这里使用 Rule 1 作为开火逻辑)
                        # _, b_fire = basic_rules(b_state_check, 1, last_action=last_b_action_label)
                        # last_b_action_label = b_action_label # 更新上一机动，供规则使用

                    if b_fire:
                        launch_missile_immediately(env, 'b', tabu=1)
                        # launch_missile_immediately(env, 'r')
                    last_b_action_label = b_action_label

                    # 红方 (RL 智能体)
                    r_state_check = env.unscale_state(r_check_obs)
                    
                    # 1. 使用网络获取机动动作
                    with torch.no_grad():
                        r_action_exec, _, _, r_action_check = actor_wrapper.get_action(r_obs, 
                                    explore={'cont':0, 'cat':0, 'bern':1}, )
                                    # check_obs=r_check_obs, 
                                    # bern_threshold=0.33)
                    r_action_label = r_action_exec['cat'][0]
                    r_fire = r_action_exec['bern'][0]
                    
                    # # 2. 使用规则获取开火决策 (这里使用 Rule 1 作为开火逻辑)
                    # _, r_fire = basic_rules(r_state_check, 1, last_action=last_r_action_label)
                    # last_r_action_label = r_action_label # 更新上一机动，供规则使用

                    if r_fire:
                        launch_missile_immediately(env, 'r', tabu=1)
                    
                    # print(f"Time: {env.t:.1f}s, Blue Action: Maneuver={r_action_label}, Fire Rule Active={r_fire}")

                # 执行机动并步进
                b_maneuver = env.maneuver14LR(env.BUAV, b_action_label)
                r_maneuver = env.maneuver14LR(env.RUAV, r_action_label)
                env.step(r_maneuver, b_maneuver)
                done, _, _, _ = env.combat_terminate_and_reward('r', r_action_label, r_fire)
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

