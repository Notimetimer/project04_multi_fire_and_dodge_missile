# 1、目录
import argparse
import time
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from math import pi
import numpy as np
import matplotlib
import json
import glob
import copy
import socket
import threading
from send2tacview import *
from Algorithms.Rules import decision_rule
from Math_calculates.CartesianOnEarth import *
from Math_calculates.sub_of_angles import *
from torch.distributions import Normal
import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
from Envs.UAVmodel6d import UAVModel
from Math_calculates.CartesianOnEarth import NUE2LLH, LLH2NUE
from Visualize.tacview_visualize import *
from Visualize.tensorboard_visualize import *

from Algorithms.PPObernouli import *

# 2、读取teacher智能体
from TrainAndTests.Attacks.PPOShoot__Train import *
from Utilities.LocateDirAndAgents import *
from Algorithms.Utils import compute_advantage, compute_monte_carlo_returns

# 读取teacher_agent
agent = PPO_bernouli(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                lmbda, epochs, eps, gamma, device)

# pre_log_dir = os.path.join("./logs")
pre_log_dir = os.path.join(project_root, "logs/shoot")
log_dir = get_latest_log_dir(pre_log_dir, mission_name=mission_name)

# 固定初始teacher策略
log_dir = os.path.join(pre_log_dir, "Shoot-run-20251125-173400")

# 用新函数加载 actor：若想强制加载编号为 990 的模型，传入 number=990
actor_path = load_actor_from_log(log_dir, number=None)
if not actor_path:
    print(f"No actor checkpoint found in {log_dir}")
else:
    sd = th.load(actor_path, map_location=device, weights_only=True)
    agent.actor.load_state_dict(sd)
    print(f"Loaded actor for test from: {actor_path}")

t_bias = 0

env = ShootTrainEnv(args, tacview_show=0)


# 出生状态和动作空间需要按shoot的环境指定
# load shoot 的模型参数、
# 给保存的pkl改名


# 3、随机初始化环境存储100个episode的il_transition_dict
transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
il_transition_dict = {'states':[], 'actions': [], 'returns': []}

if __name__ == "__main__":
    episode_count = 0
    # 计时用
    total_start_time = time.time()
    episode_times = []

    for i_episode in range(10):
        test_run = 1
        episode_return = 0
        episode_count += 1

        # 飞机出生状态指定
        red_E = 45e3 # random.uniform(30e3, 40e3)  # 20, 60 特意训练一个近的，测试一个远的
        blue_E= -red_E
        red_height = random.uniform(3e3, 12e3)
        blue_height = random.uniform(3e3, 12e3)

        DEFAULT_RED_BIRTH_STATE = {'position': np.array([0.0, red_height, red_E]),
                                    'psi': -pi/2
                                    }
        DEFAULT_BLUE_BIRTH_STATE = {'position': np.array([0.0, blue_height, blue_E]),
                                    'psi': pi/2
                                    }
        env.reset(red_birth_state=DEFAULT_RED_BIRTH_STATE, blue_birth_state=DEFAULT_BLUE_BIRTH_STATE,
                    red_init_ammo=0, blue_init_ammo=6)
        RUAV_set_speed = random.uniform(0.7, 1.5) * 340
        RUAV_set_height = red_height
        BUAV_set_speed = random.uniform(0.7, 1.5) * 340
        
        last_launch_time = - np.inf
        random_theta_plus = 0
        random_psi_plus = 0
        done = False

        actor_grad_list = []
        critc_grad_list = []
        actor_loss_list = []
        critic_loss_list = []
        entropy_list = []
        ratio_list = []

        r_action_list = []
        b_action_list = []

        episode_start = time.time()
        # 环境运行一轮的情况
        for count in range(round(args.max_episode_len / dt_maneuver)):
            # print(f"time: {env.t}")  # 打印当前的 count 值
            # 回合结束判断
            # print(env.running)
            current_t = count * dt_maneuver
            if env.running == False or done:  # count == round(args.max_episode_len / dt_maneuver) - 1:
                # print('回合结束，时间为：', env.t, 's')
                break
            # 获取观测信息
            r_obs_n, r_obs_check = env.attack_obs('r')
            b_obs_n, b_obs_check = env.attack_obs('b')

            # 在这里将观测信息压入记忆
            env.RUAV.obs_memory = r_obs_check.copy()
            env.BUAV.obs_memory = b_obs_check.copy()

            b_obs = np.squeeze(b_obs_n)

            cos_delta_psi = b_obs_check["target_information"][0]
            sin_delta_psi = b_obs_check["target_information"][1]
            delta_psi = atan2(sin_delta_psi, cos_delta_psi)

            distance = b_obs_check["target_information"][3]*10e3
            alpha = b_obs_check["target_information"][4]
            AA_hor = b_obs_check["target_information"][6]
            launch_interval = b_obs_check["weapon"]*120
            missile_in_mid_term = b_obs_check["missile_in_mid_term"]

            
            # 发射导弹判决
            u, _ = agent.take_action(b_obs_n, explore=0) # 0 1
            ut = u[0]
            at = ut

            # Shield
            at, _ = shoot_action_shield(at, distance, alpha, AA_hor, launch_interval)

            if at == 1:
                last_launch_time = env.t
                launch_missile_immediately(env, side='b')                  


            # 机动决策
            r_action_n = decision_rule(ego_pos_=env.RUAV.pos_, ego_psi=env.RUAV.psi,
                                        enm_pos_=env.BUAV.pos_, distance=distance,
                                        ally_missiles=env.Rmissiles, enm_missiles=env.Bmissiles,
                                        o00=o00, R_cage=env.R_cage, wander=0,
                                        set_height=RUAV_set_height, set_speed=RUAV_set_speed
                                        )

            random_theta_plus = generate_ar1_value(random_theta_plus, 0.9, 0.1)
            random_psi_plus = generate_ar1_value(random_psi_plus, 0.9, 0.1)
            

            b_action_n = np.array([env.RUAV.alt-env.BUAV.alt + 1000 * random_theta_plus, 
                                    delta_psi + pi/4 * random_psi_plus, 
                                    BUAV_set_speed])

            r_action_list.append(r_action_n)
            b_action_list.append(b_action_n)

            env.step(r_action_n, b_action_n)  # 2、环境更新并反馈
            done, b_reward, _ = env.get_terminate_and_reward('b', u)
            next_b_obs, _ = env.attack_obs('b')  # 子策略的训练不要用get_obs
            env.BUAV.act_memory = b_action_n.copy()  # 存储上一步动作
  
            transition_dict['states'].append(b_obs_n)
            transition_dict['actions'].append(at)  # 模仿学习可以不on-policy
            transition_dict['next_states'].append(next_b_obs)
            transition_dict['rewards'].append(b_reward)
            transition_dict['dones'].append(done)

        # 记录本 episode 耗时（秒）
        ep_dur = time.time() - episode_start
        episode_times.append(ep_dur)
        print(f"Episode {episode_count} done, duration {ep_dur:.2f}s")

        print("累计收集次数", episode_count, "/10")
 
    pass
    # 计算蒙特卡洛回报
    il_transition_dict['states'] = transition_dict['states']
    il_transition_dict['actions'] = transition_dict['actions']
    il_transition_dict['returns'] = compute_monte_carlo_returns(gamma, \
                                                                transition_dict['rewards'], \
                                                                transition_dict['dones'])
    
    # 保存到当前脚本所在目录（只保存 pickle，且同时保存 transition_dict 以便后续分析）
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    il_pkl_path = os.path.join(cur_dir, "il_transitions_shoot.pkl")
    trans_pkl_path = os.path.join(cur_dir, "transition_dict_shoot.pkl")
    import pickle
    # 保存示范轨迹（IL 用）
    with open(il_pkl_path, "wb") as f:
        pickle.dump(il_transition_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    # 保存原始 transition_dict（包含 next_states, rewards, dones 等）
    with open(trans_pkl_path, "wb") as f:
        pickle.dump(transition_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved il_transitions.pkl to: {il_pkl_path}")
    print(f"Saved transition_dict.pkl to: {trans_pkl_path}")
 
    total_dur = time.time() - total_start_time
    avg_ep = (sum(episode_times) / len(episode_times)) if episode_times else 0.0
    print(f"Timing: total {total_dur:.2f}s, episodes {len(episode_times)}, avg per-episode {avg_ep:.2f}s")
 
    # 4、模仿学习
    '''换一个文件来实现'''




