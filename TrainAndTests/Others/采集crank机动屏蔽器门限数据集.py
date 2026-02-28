'''
训练左crank策略：
红方纯追踪蓝方，蓝方一开始就发射一枚导弹，然后练习crank机动

1、目标初始化前首先计算导弹可发射区范围，然后将目标置于可发射区内、不可逃逸区外，对我机纯追踪
2、目标速度和高度为随机数,与我机同高度
3、蓝方只有一枚导弹，开始就发射导弹，后续需保持雷达照射

todo 把这个从回合数限制改成步数限制的，回合结束的太快，1000回合不够用了!!!!!

'''


import argparse
import time
import sys
import os
sys.path.append(project_root)
from Envs.Tasks.CrankManeuverEnv import *
# from Envs.battle6dof1v1_missile0919 import *
#   battle3dof1v1_proportion battle3dof1v1_missile0812 battle3dof1v1_missile0901
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
from Algorithms.Wasted.PPOcontinues_dual_a_out import *
# from Algorithms.SquashedPPOcontinues_std_no_state import *
# from tqdm import tqdm #  停用tqdm
from LaunchZone.calc_DLZ import *
import multiprocessing as mp
from multiprocessing import Pool

use_tacview = 1  # 是否可视化

# matplotlib.use('TkAgg')  # 'TkAgg' 或 'Qt5Agg'
if matplotlib.get_backend() != 'TkAgg':
    matplotlib.use('TkAgg')
    plt.switch_backend('TkAgg')


parser = argparse.ArgumentParser("UAV swarm confrontation")
# Environment
parser.add_argument("--max-episode-len", type=float, default=180,  # 8 * 60,
                    help="maximum episode time length")  # test 真的中远距空战可能会持续20分钟那么长
parser.add_argument("--R-cage", type=float, default=170e3,  # 8 * 60,
                    help="")

args = parser.parse_args()


env = CrankTrainEnv(args, tacview_show=use_tacview)
# env = Battle(args, tacview_show=use_tacview)
r_action_spaces, b_action_spaces = env.r_action_spaces, env.b_action_spaces
action_bound0 = np.array([[-5000, 5000], [-pi, pi], [200, 600]])
action_bound = copy.deepcopy(action_bound0)
state_dim = 1+8+7+4+2  # 35 # len(b_obs_spaces)
action_dim = b_action_spaces[0].shape[0]

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 整一下高度-攻击区离散表（使用有限的选择）
# 水平距离20km~80km, 我机高度 env.min_alt_safe 到 env.max_alt_safe 按 2e3 间隔划分
# 目标高度 = 我机高度-2e3, 0, 2e3, 双方速度均为1.2Ma，最后构建一个[我机高度, 目标高度, 不可逃逸区边界，最大边界]的查询表

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(project_root, "data")
os.makedirs(data_dir, exist_ok=True)
save_path = os.path.join(data_dir, "Crankinitial_states.npy")
# initial_states = np.load(save_path)
# print("读取的数据\n", initial_states)


if __name__=="__main__":
    
    transition_dict_threshold = env.args.max_episode_len//env.dt_maneuver + 1

    out_range_count = 0
    return_list = []
    win_list = []
    steps_count = 0

    total_steps = 0
    i_episode = 0

    training_start_time = time.time()
    launch_time_count = 0

    t_bias = 0
    steps_since_update = 0

    # 准备数据集结构

    # v: 0.9*340到2*340，5个等间隔
    v = np.linspace(0.9 * 340, 2 * 340, 5)

    # alt: 3000到12000，10个等间隔
    alt = np.linspace(3000, 12000, 10)

    # d0: 20e3到80e3，7个等间隔
    d0 = np.linspace(20e3, 80e3, 7)

    # ATA: 40到60，步长1
    ata0 = 60


    # 组合所有参数
    from itertools import product

    data = []
    for vv in v:
        for aa in alt:
            for dd in d0:
                data.append([vv, aa, dd, ata0])

    data_array = np.array(data)
    print(data_array.shape)  # (7*5*10*7*21, 7)


    # D: 0到2pi（不含2pi），7个等间隔
    D = np.linspace(0, 2 * np.pi, 8)[:-1]  # shape=(7,)
    Delta_theta = np.pi / 2 * np.cos(D)
    Delta_psi = np.pi / 2 * np.sin(D)
    # ATA: 40到60，步长1
    ATA = np.arange(40, 61, 1)

    # 采集样本
    for i1 in range(data_array.shape[0]):
        initial_v = data_array[0]
        initial_height = data_array[1]
        initial_d = data_array[2]
        for i_D in range(len(D)):
            # ata_list = []
            max_ATA = 0
            for ata in ATA:
                # 初始化，仿真，找出ata的介入边界

                blue_height = random.uniform(3e3, 12e3)
                red_height = np.clip(blue_height+random.uniform(-2e3, 2e3), 3e3, 12e3)
                blue_psi = pi/2
                red_psi = -pi/2
                red_N = 0 # random.choice([-54e3, 54e3]) # red_N = random.uniform(-50e3, 50e3)
                red_E = 35e3
                blue_N = red_N
                blue_E = -35e3

                DEFAULT_RED_BIRTH_STATE = {'position': np.array([red_N, red_height, red_E]),
                                        'psi': red_psi}
                DEFAULT_BLUE_BIRTH_STATE = {'position': np.array([blue_N, blue_height, blue_E]),
                                            'psi': blue_psi}
                
                env.reset(red_birth_state=DEFAULT_RED_BIRTH_STATE, blue_birth_state=DEFAULT_BLUE_BIRTH_STATE,
                        red_init_ammo=0, blue_init_ammo=0) # 1
                DEFAULT_RED_BIRTH_STATE = {'position': np.array([red_N, red_height, red_E]),
                                        'psi': red_psi}
                DEFAULT_BLUE_BIRTH_STATE = {'position': np.array([blue_N, blue_height, blue_E]),
                                            'psi': blue_psi}
                
                env.reset(red_birth_state=DEFAULT_RED_BIRTH_STATE, blue_birth_state=DEFAULT_BLUE_BIRTH_STATE,
                        red_init_ammo=0, blue_init_ammo=0) # 1

                done = False

                env.dt_maneuver = dt_maneuver
                
                episode_start_time = time.time()

                last_alpha = None

                # 环境运行一轮的情况
                for count in range(round(args.max_episode_len / dt_maneuver)):
                    # print(f"time: {env.t}")  # 打印当前的 count 值
                    # 回合结束判断
                    # print(env.running)
                    current_t = count * dt_maneuver
                    if env.running == False or done: # count == round(args.max_episode_len / dt_maneuver) - 1:
                        # print('回合结束，时间为：', env.t, 's')
                        break
                    # 获取观测信息
                    r_obs_n, r_obs_check = env.crank_obs('r')
                    b_obs_n, b_obs_check = env.crank_obs('b')

                    # 在这里将观测信息压入记忆
                    env.RUAV.obs_memory = r_obs_check.copy()
                    env.BUAV.obs_memory = b_obs_check.copy()
                    # 在这里将观测信息压入记忆
                    env.RUAV.obs_memory = r_obs_check.copy()
                    env.BUAV.obs_memory = b_obs_check.copy()

                    b_obs = np.squeeze(b_obs_n)
                    b_obs = np.squeeze(b_obs_n)

                    distance = norm(env.RUAV.pos_ - env.BUAV.pos_)
                    
                    # 开局就发射一枚导弹
                    if env.BUAV.ammo>0:
                        new_missile = env.BUAV.launch_missile(env.RUAV, env.t, missile_class)
                        env.BUAV.ammo -= 1
                        new_missile.side = 'blue'
                        env.Bmissiles.append(new_missile)
                        env.missiles = env.Rmissiles + env.Bmissiles

                    height_ego = env.BUAV.alt
                    delta_psi = b_obs_check["target_information"][1]
                    height_ego = env.BUAV.alt
                    delta_psi = b_obs_check["target_information"][1]

                    # 机动决策
                    r_action_n = decision_rule(ego_pos_=env.RUAV.pos_, ego_psi=env.RUAV.psi,
                                            enm_pos_=env.BUAV.pos_, distance=distance,
                                            ally_missiles=env.Rmissiles, enm_missiles=env.Bmissiles,
                                            o00=o00, R_cage=env.R_cage, wander=1
                                            )
                    if np.isnan(b_obs).any() or np.isinf(b_obs).any():
                        print('b_obs', b_obs_check)
                        print()

                    # 每10个回合测试一次，测试回合不统计步数，不采集经验，不更新智能体，训练回合不回报胜负
                    b_action_n = np.array([-4000, pi/2, 300])
                    if env.BUAV.alt < 3000:
                        b_action_n[0] = 0
                    # 每10个回合测试一次，测试回合不统计步数，不采集经验，不更新智能体，训练回合不回报胜负
                    b_action_n = np.array([-4000, pi/2, 300])
                    if env.BUAV.alt < 3000:
                        b_action_n[0] = 0

                    # b_action_n = decision_rule(ego_pos_=env.BUAV.pos_, ego_psi=env.BUAV.psi,
                    #                         enm_pos_=env.RUAV.pos_, distance=distance,
                    #                         ally_missiles=env.Bmissiles, enm_missiles=env.Rmissiles,
                    #                         o00=o00, R_cage=env.R_cage, wander=0
                    #                         )
                    
                    # Shield介入
                    delta_theta = b_obs_check["target_information"][2]
                    alpha = b_obs_check["target_information"][4]
                    if last_alpha is None:
                        alpha_dot = 0
                    else:
                        alpha_dot = (alpha-last_alpha)/dt_maneuver
                    if alpha_dot > 0:
                        t_last = (ata-alpha)/alpha_dot
                    else:
                        t_last = np.inf

                    last_alpha = alpha
                    last_alpha = alpha

                    if t_last < 2 or alpha > alpha_trigger:
                        b_action_n[0] = 2 * delta_theta/pi*2*5000 # env.RUAV.alt - env.BUAV.alt
                        b_action_n[1] = 2 * delta_psi
                        print()
                    if t_last < 2 or alpha > alpha_trigger:
                        b_action_n[0] = 2 * delta_theta/pi*2*5000 # env.RUAV.alt - env.BUAV.alt
                        b_action_n[1] = 2 * delta_psi
                        print()

                    # b_action_n[0] = np.clip(b_action_n[0], env.min_alt_safe-height_ego, env.max_alt_safe-height_ego)
                    # if delta_psi>0:
                    #     b_action_n[1] = max(sub_of_radian(delta_psi-50*pi/180, 0), b_action_n[1])
                    # else:
                    #     b_action_n[1] = min(sub_of_radian(delta_psi+50*pi/180, 0), b_action_n[1])
                    # b_action_n[0] = np.clip(b_action_n[0], env.min_alt_safe-height_ego, env.max_alt_safe-height_ego)
                    # if delta_psi>0:
                    #     b_action_n[1] = max(sub_of_radian(delta_psi-50*pi/180, 0), b_action_n[1])
                    # else:
                    #     b_action_n[1] = min(sub_of_radian(delta_psi+50*pi/180, 0), b_action_n[1])

                    env.step(r_action_n, b_action_n)  # 2、环境更新并反馈
                    done, b_reward, b_event_reward = env.left_crank_terminate_and_reward('b')
                    next_b_obs, next_b_obs_check = env.crank_obs('b')  # 子策略的训练不要用get_obs
                    env.step(r_action_n, b_action_n)  # 2、环境更新并反馈
                    done, b_reward, b_event_reward = env.left_crank_terminate_and_reward('b')
                    next_b_obs, next_b_obs_check = env.crank_obs('b')  # 子策略的训练不要用get_obs

                    
                    if next_b_obs_check["target_information"][4]>60*pi/180:
                        out_angle = 1
                    else:
                        out_angle = 0
                    
                    if next_b_obs_check["target_information"][4]>60*pi/180:
                        out_angle = 1
                    else:
                        out_angle = 0

                    done = done
                    done = done

                    '''显示运行轨迹'''
                    # 可视化
                    env.render(t_bias=t_bias)
                    '''显示运行轨迹'''
                    # 可视化
                    env.render(t_bias=t_bias)

                    time.sleep(0.01)
                
                # print(t_bias)
                env.clear_render(t_bias=t_bias)
                t_bias += env.t

            

