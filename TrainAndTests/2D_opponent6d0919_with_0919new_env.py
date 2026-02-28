'''
写超视距对手策略和靶机对战，靶机只追踪，对手策略分为4个阶段：
1、速度保持1.5ma，40km外只追踪，40km处立即发射导弹并以45°crank转2（雷达范围±60°）
2、保持45°偏角进行中制导，到达20km处若收到导弹告警，规避转3否则转4
3、若不再受导弹威胁，调头攻击转4，否则进行水平置尾机动，速度保持1.6Ma
4、速度保持1.2Ma，其余同1
'''

'''
训练事项：
1、所有动作缩放统一改在action_bound里面进行
2、随机初始化我机和目标机位置
3、禁止发射导弹，测试进攻相关的观测、结束判读和奖励能否完整运行 √
4、高度指令改为相对高度 √
5、tacview 文件头命令改在env init时候加入, √
    加入clear_render方法清空所有在可视化的实体, √
    在render方法中使用t_bias处理多次仿真可视化的起点问题 √
'''


import argparse
import time
import sys
import os

sys.path.append(project_root)
from Envs.battle6dof1v1_missile0919 import *
#   battle3dof1v1_proportion battle3dof1v1_missile0812 battle3dof1v1_missile0901
from math import pi
import numpy as np
import matplotlib
import socket
import threading
from send2tacview import *
from Algorithms.Rules import decision_rule
from Math_calculates.CartesianOnEarth import *
from Math_calculates.sub_of_angles import *

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd

use_tacview = 1  # 是否可视化

# matplotlib.use('TkAgg')  # 'TkAgg' 或 'Qt5Agg'
if matplotlib.get_backend() != 'TkAgg':
    matplotlib.use('TkAgg')
    plt.switch_backend('TkAgg')


parser = argparse.ArgumentParser("UAV swarm confrontation")
# Environment
parser.add_argument("--max-episode-len", type=float, default=120,  # 8 * 60,
                    help="maximum episode time length")  # test 真的中远距空战可能会持续20分钟那么长
parser.add_argument("--R-cage", type=float, default=70e3,  # 8 * 60,
                    help="")

# parser.add_argument("--num-RUAVs", type=int, default=1, help="number of red UAVs")
# parser.add_argument("--num-BUAVs", type=int, default=1, help="number of blue UAVs")
args = parser.parse_args()

# if visualize_needed:
#     tacview = Tacview()

# def main():
# 将红蓝双方运行的速度和位置保存
red_pos_list = np.empty((0, 3))
blue_pos_list = np.empty((0, 3))
red_vel_list = np.empty((0, 3))
blue_vel_list = np.empty((0, 3))


env = Battle(args, tacview_show=use_tacview)
r_obs_spaces = 34
b_obs_spaces = 34
# r_obs_spaces, b_obs_spaces = env.get_obs_spaces()
r_action_spaces, b_action_spaces = env.r_action_spaces, env.b_action_spaces
'''滚动时域优化开始'''

r_action = []
b_action = []


start_time = time.time()
launch_time_count = 0

t_list = []
R_controll_type = []
B_controll_type = []
R_controll_check_switch1 = []
B_controll_check_switch1 = []
R_controll_check_switch2 = []
B_controll_check_switch2 = []
r_ammo = []
b_ammo = []

t_bias = 0

for i in range(10):
    print("轮次", i+1)
    episode_return = 0
    # 飞机出生状态指定
    red_R_ = random.uniform(20e3, 60e3)
    red_beta = random.uniform(0, 2*pi)
    red_psi = random.uniform(0, 2*pi)
    red_height = random.uniform(3e3, 10e3)
    red_N = red_R_*cos(red_beta)
    red_E = red_R_*sin(red_beta)
    blue_height = random.uniform(3e3, 10e3)

    DEFAULT_RED_BIRTH_STATE = {'position': np.array([red_N, red_height, red_E]),
                               'psi': red_psi
                               }
    DEFAULT_BLUE_BIRTH_STATE = {'position': np.array([0.0, blue_height, 0.0]),
                                'psi': pi
                                }
    env.reset(red_birth_state=DEFAULT_RED_BIRTH_STATE, blue_birth_state=DEFAULT_BLUE_BIRTH_STATE,
              red_init_ammo=6, blue_init_ammo=6)

    a1 = env.BUAV.pos_  # 58000,7750,20000
    a2 = env.RUAV.pos_  # 2000,7750,20000
    b1 = env.UAVs[0].pos_
    b2 = env.UAVs[1].pos_
    done = False
    r_action_list = []
    b_action_list = []
    Rtrajectory = []
    Btrajectory = []

    # 环境运行一轮的情况
    for count in range(round(args.max_episode_len / dt_maneuver)):
        # print(f"time: {env.t}")  # 打印当前的 count 值
        # 回合结束判断
        # print(env.running)
        current_t = count * dt_maneuver
        if env.running == False or count == round(args.max_episode_len / dt_maneuver) - 1:
            # print('回合结束，时间为：', env.t, 's')
            break
        # 获取观测信息
        r_obs_n = env.get_state('r')
        b_obs_n = env.get_state('b')

        # 在这里将观测信息压入记忆


        state = np.squeeze(r_obs_n)

        # todo 可发射判据添加解算，且更换if_possible和距离判据的顺序（发射时间间隔过了之后再解算攻击区会更有效率）
        distance = norm(env.RUAV.pos_ - env.BUAV.pos_)
        # 发射导弹判决
        if distance <= 40e3 and distance >= 5e3 and count % 1 == 0:  # 在合适的距离范围内每0.2s判决一次导弹发射
            launch_time_count = 0
            launch_missile_with_basic_rules(env, side='r')
            launch_missile_with_basic_rules(env, side='b')

        # 机动决策
        # r_action_n = decision_rule(ego_pos_=env.RUAV.pos_, ego_psi=env.RUAV.psi,
        #                            enm_pos_=env.BUAV.pos_, distance=distance,
        #                            ally_missiles=env.Rmissiles, enm_missiles=env.Bmissiles,
        #                            o00=o00, R_cage=env.R_cage, wander=0 #1
        #                            )
        
        L_ = env.BUAV.pos_ - env.RUAV.pos_
        q_beta = atan2(L_[2], L_[0])
        L_h = np.sqrt(L_[0] ** 2 + L_[2] ** 2)
        L_v = L_[1]
        q_epsilon = atan2(L_v, L_h)
        delta_psi = sub_of_radian(q_beta, env.RUAV.psi)
        r_action_n_0 = np.clip(env.BUAV.pos_[1], env.min_alt_safe, env.max_alt_safe)-env.RUAV.pos_[1]
        r_action_n_1 = delta_psi
        r_action_n_2 = 340
        r_action_n = [r_action_n_0, r_action_n_1, r_action_n_2]

        # b_action_n = decision_rule(ego_pos_=env.BUAV.pos_, ego_psi=env.BUAV.psi,
        #                            enm_pos_=env.RUAV.pos_, distance=distance,
        #                            ally_missiles=env.Bmissiles, enm_missiles=env.Rmissiles,
        #                            o00=o00, R_cage=env.R_cage, wander=0
        #                            )
        
        b_states = env.base_obs('b', pomdp=1)
        distance = b_states["target_information"][3] * 10e3
        warning = b_states["warning"]
        enm_delta_psi = b_states["target_information"][1]
        cos_threat_delta_psi = b_states["threat"][0]
        sin_threat_delta_psi = b_states["threat"][1]
        threat_delta_psi = atan2(sin_threat_delta_psi, cos_threat_delta_psi)

        # b_action_n = env.right_crank_behavior(env.BUAV.alt, enm_delta_psi)

        b_action_n = env.decision_rule(
            env.BUAV.pos_, env.BUAV.psi, enm_delta_psi, distance, 
            warning, threat_delta_psi, env.Bmissiles, wander=0
        )
        
        r_action_list.append(r_action_n)
        b_action_list.append(b_action_n)

        r_reward_n, b_reward_n, r_dones, b_dones, terminate = env.step(r_action_n, b_action_n)  # 2、环境更新并反馈

        '''显示运行轨迹'''
        # # print("红方位置：", env.RUAV.pos_)
        # # print("蓝方位置：", env.BUAV.pos_)
        # # # 如果导弹已发射
        # # print(f"当前发射的导弹数量：{len(env.Rmissiles) + len(env.Bmissiles)}")
        
        # 可视化
        env.render(t_bias=t_bias)
        t_list.append(current_t)
        # 红
        position = env.RUAV.pos_
        angle = np.array([env.RUAV.gamma, env.RUAV.theta, env.RUAV.psi]) * 180 / pi
        Rtrajectory.append(np.hstack((position, angle)))
        R_controll_type.append(env.RUAV.PIDController.type)
        R_controll_check_switch1.append(env.RUAV.PIDController.check_switch1)
        R_controll_check_switch2.append(env.RUAV.PIDController.check_switch2)
        r_ammo.append(env.RUAV.ammo)

        # 蓝
        position = env.BUAV.pos_
        angle = np.array([env.BUAV.gamma, env.BUAV.theta, env.BUAV.psi]) * 180 / pi
        Btrajectory.append(np.hstack((position, angle)))
        B_controll_type.append(env.BUAV.PIDController.type)
        B_controll_check_switch1.append(env.BUAV.PIDController.check_switch1)
        B_controll_check_switch2.append(env.BUAV.PIDController.check_switch2)
        b_ammo.append(env.BUAV.ammo)

        if terminate == True:
            break
    
    print(t_bias)
    env.clear_render(t_bias=t_bias)
    t_bias += env.t

end_time = time.time()  # 记录结束时间
print(f"程序运行时间: {end_time - start_time} 秒")

r_action_list = np.array(r_action_list)
b_action_list = np.array(b_action_list)

R_controll_check_switch1 = np.array(R_controll_check_switch1)
R_controll_check_switch2 = np.array(R_controll_check_switch2)
B_controll_check_switch1 = np.array(B_controll_check_switch1)
B_controll_check_switch2 = np.array(B_controll_check_switch2)

# import matplotlib.pyplot as plt

# # Create a figure and 10 subplots in 2 columns
# fig, axs = plt.subplots(5, 2, figsize=(14, 24), sharex=True)

# axs[0, 0].plot(t_list, R_controll_type, label='R_controll_type', color='red')
# axs[0, 0].legend()
# axs[0, 0].grid()

# axs[0, 1].plot(t_list, B_controll_type, label='B_controll_type', color='blue')
# axs[0, 1].legend()
# axs[0, 1].grid()

# axs[1, 0].plot(t_list, r_action_list[:, 1] * 180 / pi, label='r_action_list[1]', color='green')
# axs[1, 0].legend()
# axs[1, 0].grid()

# axs[1, 1].plot(t_list, b_action_list[:, 1] * 180 / pi, label='b_action_list[1]', color='orange')
# axs[1, 1].set_xlabel('Time (s)')
# axs[1, 1].legend()
# axs[1, 1].grid()

# axs[2, 0].plot(t_list, R_controll_check_switch1, label='R_controll_check_switch1', color='purple')
# axs[2, 0].legend()
# axs[2, 0].grid()

# axs[2, 1].plot(t_list, R_controll_check_switch2, label='R_controll_check_switch2', color='brown')
# axs[2, 1].legend()
# axs[2, 1].grid()

# axs[3, 0].plot(t_list, B_controll_check_switch1, label='B_controll_check_switch1', color='pink')
# axs[3, 0].legend()
# axs[3, 0].grid()

# axs[3, 1].plot(t_list, B_controll_check_switch2, label='B_controll_check_switch2', color='cyan')
# axs[3, 1].set_xlabel('Time (s)')
# axs[3, 1].legend()
# axs[3, 1].grid()

# # 新增弹药数曲线
# axs[4, 0].plot(t_list, r_ammo, label='r_ammo', color='darkred')
# axs[4, 0].legend()
# axs[4, 0].grid()
# axs[4, 0].set_ylabel('Red Ammo')

# axs[4, 1].plot(t_list, b_ammo, label='b_ammo', color='navy')
# axs[4, 1].legend()
# axs[4, 1].grid()
# axs[4, 1].set_ylabel('Blue Ammo')
# axs[4, 1].set_xlabel('Time (s)')

# plt.tight_layout()
# plt.show()
