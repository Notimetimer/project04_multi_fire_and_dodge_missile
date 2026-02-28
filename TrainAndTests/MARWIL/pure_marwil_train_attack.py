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

from Algorithms.PPOcontinues_std_no_state import *
from Algorithms.PPOcontinues_std_no_state import PPOContinuous

# 2、读取teacher智能体
from TrainAndTests.Attacks.PPOAttack__Train import *
from Utilities.LocateDirAndAgents import *
from Algorithms.Utils import compute_advantage, compute_monte_carlo_returns

# 读取teacher_agent
agent = PPOContinuous(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                lmbda, epochs, eps, gamma, device)

# pre_log_dir = os.path.join("./logs")
pre_log_dir = os.path.join(project_root, "logs/attack")
log_dir = get_latest_log_dir(pre_log_dir, mission_name=mission_name)
# log_dir = os.path.join(pre_log_dir, "Attack-run-20251031-094218")

# 用新函数加载 actor：若想强制加载编号为 990 的模型，传入 number=990
actor_path = load_actor_from_log(log_dir, number=None)
if not actor_path:
    print(f"No actor checkpoint found in {log_dir}")
else:
    sd = th.load(actor_path, map_location=device, weights_only=True)
    agent.actor.load_state_dict(sd)
    print(f"Loaded actor for test from: {actor_path}")

t_bias = 0

env = AttackTrainEnv(args, tacview_show=0)

# 3、随机初始化环境存储100个episode的il_transition_dict
transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
il_transition_dict = {'states':[], 'actions': [], 'returns': []}

# 飞机出生状态应该是均匀的，100条示范轨迹，初始ATA分为10个，初始高度分为10个组合
ATA_list = np.linspace(0, 9, 10)/10 *2*pi

ego_psi_list = [0] # np.zeros_like(ATA_list)
enm_beta_list = [0] # ATA_list
enm_psi_list = [pi] #  pi + ATA_list

ego_height_list = [7000] # np.linspace(5000, 10000, 5)
enm_height_list = np.linspace(8000, 12000, 2)

if __name__ == "__main__":
    episode_count = 0
    # 计时用
    total_start_time = time.time()
    episode_times = []
    for i, enm_beta in enumerate(enm_beta_list):
        for blue_height in ego_height_list:
            for red_height in enm_height_list:
                episode_count += 1
                episode_start = time.time()
                # 飞机出生状态指定
                init_distance = 90e3
                red_R_ = init_distance/2 # random.uniform(20e3, 60e3)
                blue_R_ = init_distance/2
                red_beta = enm_beta
                red_psi = enm_psi_list[i]
                red_N = red_R_*cos(red_beta)
                red_E = red_R_*sin(red_beta)

                DEFAULT_RED_BIRTH_STATE = {'position': np.array([red_N, red_height, red_E]),
                                        'psi': red_psi
                                        }
                DEFAULT_BLUE_BIRTH_STATE = {'position': np.array([blue_R_, blue_height, 0.0]),
                                            'psi': 0.0
                                            }
                
                env.reset(red_birth_state=DEFAULT_RED_BIRTH_STATE, blue_birth_state=DEFAULT_BLUE_BIRTH_STATE,
                        red_init_ammo=0, blue_init_ammo=0)
                env.dt_maneuver = dt_maneuver

                done = False
                hist_b_action = np.zeros(3)

                while not done:
                    # print(env.t)
                    r_obs_n, r_obs_check = env.attack_obs('r')
                    b_obs_n, b_obs_check = env.attack_obs('b')
                    # 在这里将观测信息压入记忆
                    env.RUAV.obs_memory = r_obs_check.copy()
                    env.BUAV.obs_memory = b_obs_check.copy()
                    state = np.squeeze(b_obs_n)
                    distance = norm(env.RUAV.pos_ - env.BUAV.pos_)
                    r_action_n = decision_rule(ego_pos_=env.RUAV.pos_, ego_psi=env.RUAV.psi,
                                            enm_pos_=env.BUAV.pos_, distance=distance,
                                            ally_missiles=env.Rmissiles, enm_missiles=env.Bmissiles,
                                            o00=o00, R_cage=env.R_cage, wander=1
                                            )
                    # r_action_n, u_r = agent.take_action(r_obs_n, action_bounds=action_bound, explore=False)
                    b_action_n, u = agent.take_action(b_obs_n, action_bounds=action_bound, explore=False)
                    
                    # # 动作平滑（实验性）
                    # b_action_n = action_eps*hist_b_action+(1-action_eps)*b_action_n
                    # hist_b_action = b_action_n

                    # ### 发射导弹，这部分不受10step约束
                    # distance = norm(env.RUAV.pos_ - env.BUAV.pos_)
                    # # 发射导弹判决
                    # if distance <= 80e3 and distance >= 5e3:  # 在合适的距离范围内每0.2s判决一次导弹发射
                    #     launch_time_count = 0
                    #     launch_missile_with_basic_rules(env, side='r')
                    #     launch_missile_with_basic_rules(env, side='b')

                    env.step(r_action_n, b_action_n)  # 2、环境更新并反馈
                    done, b_reward, _ = env.get_terminate_and_reward('b')
                    next_b_obs, _ = env.attack_obs('b')  # 子策略的训练不要用get_obs
                    env.BUAV.act_memory = b_action_n.copy()  # 存储上一步动作
                    
                    transition_dict['states'].append(b_obs_n)
                    transition_dict['actions'].append(u)
                    transition_dict['next_states'].append(next_b_obs)
                    transition_dict['rewards'].append(b_reward)
                    transition_dict['dones'].append(done)

                    env.render(t_bias=t_bias)
                
                env.clear_render(t_bias=t_bias)
                t_bias += env.t
                # 记录本 episode 耗时（秒）
                ep_dur = time.time() - episode_start
                episode_times.append(ep_dur)
                print(f"Episode {episode_count} done, duration {ep_dur:.2f}s")

                print("累计收集次数", episode_count, "/100")
 
    pass
    # 计算蒙特卡洛回报
    il_transition_dict['states'] = transition_dict['states']
    il_transition_dict['actions'] = transition_dict['actions']
    il_transition_dict['returns'] = compute_monte_carlo_returns(gamma, \
                                                                transition_dict['rewards'], \
                                                                transition_dict['dones'])
    
    # 保存到当前脚本所在目录（只保存 pickle，且同时保存 transition_dict 以便后续分析）
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    il_pkl_path = os.path.join(cur_dir, "il_transitions_chase.pkl")
    trans_pkl_path = os.path.join(cur_dir, "transition_dict.pkl")
    import pickle
    # 保存示范轨迹（IL 用）
    with open(il_pkl_path, "wb") as f:
        pickle.dump(il_transition_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    # 保存原始 transition_dict（包含 next_states, rewards, dones 等）
    with open(trans_pkl_path, "wb") as f:
        pickle.dump(transition_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved il_transitions_chase.pkl to: {il_pkl_path}")
    print(f"Saved transition_dict.pkl to: {trans_pkl_path}")
 
    total_dur = time.time() - total_start_time
    avg_ep = (sum(episode_times) / len(episode_times)) if episode_times else 0.0
    print(f"Timing: total {total_dur:.2f}s, episodes {len(episode_times)}, avg per-episode {avg_ep:.2f}s")
 
    # 4、模仿学习
    '''换一个文件来实现'''




