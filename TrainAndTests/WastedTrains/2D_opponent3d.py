'''
写超视距对手策略和靶机对战，靶机只追踪，对手策略分为4个阶段：
1、速度保持1.5ma，40km外只追踪，40km处立即发射导弹并以45°crank转2（雷达范围±60°）
2、保持45°偏角进行中制导，到达20km处若收到导弹告警，规避转3否则转4
3、若不再受导弹威胁，调头攻击转4，否则进行水平置尾机动，速度保持1.6Ma
4、速度保持1.2Ma，其余同1
'''

import argparse
import time
import sys
import os
sys.path.append(project_root)
from Envs.battle3dof1v1_missile0812 import *
#   battle3dof1v1_proportion battle3dof1v1_missile0812 battle3dof1v1_missile0901
from math import pi
import numpy as np
import matplotlib
import socket
import threading
from send2tacview import *

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd

dt_refer = dt

# matplotlib.use('TkAgg')  # 'TkAgg' 或 'Qt5Agg'
if matplotlib.get_backend() != 'TkAgg':
    matplotlib.use('TkAgg')
    plt.switch_backend('TkAgg')


def get_args():
    parser = argparse.ArgumentParser("UAV swarm confrontation")
    # Environment
    parser.add_argument("--max-episode-len", type=float, default=8 * 60,  # * 60.0,
                        help="maximum episode time length")  # test 真的中远距空战可能会持续20分钟那么长
    # parser.add_argument("--num-RUAVs", type=int, default=1, help="number of red UAVs")
    # parser.add_argument("--num-BUAVs", type=int, default=1, help="number of blue UAVs")
    args = parser.parse_args()
    return args


tacview = Tacview()

# def main():
# 将红蓝双方运行的速度和位置保存
red_pos_list = np.empty((0, 3))
blue_pos_list = np.empty((0, 3))
red_vel_list = np.empty((0, 3))
blue_vel_list = np.empty((0, 3))

args = get_args()
env = Battle(args)

r_obs_spaces, b_obs_spaces = env.get_obs_spaces()
r_action_spaces, b_action_spaces = env.r_action_spaces, env.b_action_spaces
'''滚动时域优化开始'''

r_action = []
b_action = []


def strong_rolling_optim(env, side='r'):
    # 7种离散动作：
    action_list = [
        [[0, 0, 0]],
        # [[1, 0, 0]],
        # [[-1, 0, 0]],
        [[0, 1, 0]],
        [[0, 1, 1]],
        [[0, 0, 0.4]],
        [[0, 0, -0.4]]
    ]
    reward_list = []
    # 遍历7种动作并记录对应的奖励信息
    for i in range(len(action_list)):
        if side == 'r':
            r_action_n = action_list[i]
            b_action_n = [[0, 0, 0]]  # 假设对面没有动作
        elif side == 'b':
            b_action_n = action_list[i]
            r_action_n = [[0, 0, 0]]  # 假设对面没有动作
        else:
            raise RuntimeError("请输入正确的阵营")
        # 执行动作并记录奖励信息
        r_reward_n, b_reward_n, r_dones, b_dones, terminate = env.step(r_action_n, b_action_n, assume=True)
        if side == 'r':
            reward_list.append(r_reward_n[0])  # 获取当前动作状态奖励信息
        if side == 'b':
            reward_list.append(b_reward_n[0])  # 获取当前动作状态奖励信息
    chosen_action = reward_list.index(max(reward_list))
    global r_action, b_action
    if side == 'r':
        r_action.append(chosen_action)
    if side == 'b':
        b_action.append(chosen_action)
    return action_list[chosen_action]
    # return action_list[0]


def launch_missile_if_possible(env, side='r'):
    """
    根据条件判断是否发射导弹
    """
    if side == 'r':
        uav = env.RUAV
        target = env.BUAV
    else:  # side == 'b'
        uav = env.BUAV
        target = env.RUAV

    # 判断是否可以发射导弹
    if uav.can_launch_missile(target, env.t):
        # 发射导弹
        new_missile = uav.launch_missile(target, env.t)
        uav.ammo -= 1
        new_missile.side = 'red' if side == 'r' else 'blue'
        if side == 'r':
            env.Rmissiles.append(new_missile)
        else:
            env.Bmissiles.append(new_missile)
        env.missiles = env.Rmissiles + env.Bmissiles
        print(f"{'红方' if side == 'r' else '蓝方'}发射导弹")


start_time = time.time()
launch_time_count = 0

for i in range(1):
    episode_return = 0
    # DEFAULT_RED_BIRTH_STATE = {'position': np.array([-38000.0, 8000.0, -1700.0]),
    #                            'psi': 0
    #                            }
    # DEFAULT_BLUE_BIRTH_STATE = {'position': np.array([38000.0, 8000.0, -1700.0]),
    #                             'psi': pi
    #                             }
    # env.reset(red_birth_state=DEFAULT_RED_BIRTH_STATE, blue_birth_state=DEFAULT_BLUE_BIRTH_STATE,
    #           red_init_ammo=6, blue_init_ammo=6)

    # env.reset(red_birth_state=None, blue_birth_state=None,
    #           red_init_ammo=6, blue_init_ammo=6)
    env.reset(red_birth_state=None, blue_birth_state=None, red_init_ammo=6, blue_init_ammo=6)
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
    for count in range(round(args.max_episode_len / dt_refer)):
        # print(f"time: {env.t}")  # 打印当前的 count 值
        # 回合结束判断
        # print(env.running)
        if env.running == False or count == round(args.max_episode_len / dt_refer) - 1:
            # print('回合结束，时间为：', env.t, 's')
            break
        # 获取观测信息
        r_obs_n, b_obs_n = env.get_obs()
        state = np.squeeze(r_obs_n)

        # todo 可发射判据添加解算，且更换if_possible和距离判据的顺序（发射时间间隔过了之后再解算攻击区会更有效率）
        distance = norm(env.RUAV.pos_ - env.BUAV.pos_)
        # 是否可发射导弹
        if distance <= 40e3 and distance >= 5e3:  # 在合适的距离范围内
            launch_time_count = 0
            launch_missile_if_possible(env, side='r')
            launch_missile_if_possible(env, side='b')
        r_has_missile_in_the_air = 0
        # 是否有导弹可用
        for missile in env.Rmissiles:
            if not missile.dead:
                r_has_missile_in_the_air = 1
                break
        # 是否被敌机导弹锁定（存在敌机导弹而且距离在20km内，此时导弹雷达可以锁定）
        should_escape = 0
        rel_psi_m = None
        for missile in env.Bmissiles:
            if not missile.dead:
                line_m_ = missile.pos_ - env.RUAV.pos_
                dist_mt = norm(line_m_)
                if dist_mt <= 20e3:  # 假设为导弹雷达的锁定距离
                    should_escape = 1
                    rel_psi_m = sub_of_radian(atan2(line_m_[2], line_m_[0]), env.RUAV.psi)
                    break  # trick：先发射的导弹被当做距离最近的，没有在所有导弹中判断距离 todo 判断距离

        # 执行动作
        # b_action_n = [[-1, 0, 0]]  # for i in range(args.num_BUAVs)]
        # r_action_n = [[1,0,0]]
        b_action_n = strong_rolling_optim(env, side='b')  # 蓝方

        r_action_n = [[0, 0, 0]]

        # print('距离', distance/1e3, 'km')
        # print('是否需要逃', should_escape)
        # print('是否已发射导弹', r_has_missile_in_the_air)

        # 1 追踪
        if distance > 40e3:
            # print('追')
            r_action_n = strong_rolling_optim(env, side='r')  # 短视优化
            r_action_n[0][0] = np.sign(1.5-env.RUAV.mach)  # 速度往1.5Ma去
        # 2 crank
        elif not should_escape and r_has_missile_in_the_air:
            # print('crank')
            # 计算蓝相对红的夹角
            line_rb_ = env.BUAV.pos_ - env.RUAV.pos_
            rel_psi = sub_of_radian(atan2(line_rb_[2], line_rb_[0]), env.RUAV.psi)
            if 0 <= rel_psi:
                temp = 0.4 * (rel_psi - pi / 4) / (pi / 4) * 2
                r_action_n[0][2] = np.clip(temp, -0.4, 0.4)
            if rel_psi < 0:
                temp = 0.4 * (rel_psi + pi / 4) / (pi / 4) * 2
                r_action_n[0][2] = np.clip(temp, -0.4, 0.4)
            r_action_n[0][0] = np.sign(1.1-env.RUAV.mach)  # 速度往1.1Ma去
        # 3 escape
        elif should_escape:
            # print('逃')
            # 水平置尾机动
            temp = 0.4 * sub_of_radian(rel_psi_m, pi) / pi * 4
            r_action_n[0][2] = np.clip(temp, -0.4, 0.4)
            r_action_n[0][0] = np.sign(1.5 - env.RUAV.mach)  # 速度往1.5Ma去

        # 4 attack again
        else:
            # print('回击')
            r_action_n = strong_rolling_optim(env, side='r')  # 短视优化
            r_action_n[0][0] = np.sign(1.2 - env.RUAV.mach)  # 速度往1.2Ma去

        # print('b_action_n=',b_action_n)
        # print('r_action_n=',r_action_n)
        r_reward_n, b_reward_n, r_dones, b_dones, terminate = env.step(r_action_n, b_action_n, assume=False,
                                                                       record=True)  # 2、环境更新并反馈

        '''显示运行轨迹'''
        # print("红方位置：", env.RUAV.pos_)
        # print("蓝方位置：", env.BUAV.pos_)
        # # 如果导弹已发射
        # print(f"当前发射的导弹数量：{len(env.Rmissiles) + len(env.Bmissiles)}")
        # 遍历导弹列表
        for missile in env.Rmissiles:
            if hasattr(missile, 'dead') and missile.dead:
                continue
            # 记录导弹的位置
            missile_pos = missile.pos_  # 假设导弹对象有 pos_ 属性表示位置
            # print("红方导弹位置：", missile_pos)
        for missile in env.Bmissiles:
            if hasattr(missile, 'dead') and missile.dead:
                continue
            # 记录导弹的位置
            missile_pos = missile.pos_  # 假设导弹对象有 pos_ 属性表示位置
            # print("蓝方导弹位置：", missile_pos)

        send_t = env.t
        name_R = env.RUAV.id
        name_B = env.BUAV.id
        loc_r = ENU2LLH(mark, env.RUAV.pos_)
        loc_b = ENU2LLH(mark, env.BUAV.pos_)
        data_to_send = ''
        if not env.RUAV.dead:
            data_to_send += f"#{send_t:.2f}\n{name_R},T={loc_r[0]:.6f}|{loc_r[1]:.6f}|{loc_r[2]:.6f},Name=F16,Color=Red\n"
        else:
            data_to_send += f"#{send_t:.2f}\n-{name_R}\n"
        if not env.BUAV.dead:
            data_to_send += f"#{send_t:.2f}\n{name_B},T={loc_b[0]:.6f}|{loc_b[1]:.6f}|{loc_b[2]:.6f},Name=F16,Color=Blue\n"
        else:
            data_to_send += f"#{send_t:.2f}\n-{name_B}\n"

        for j, missile in enumerate(env.missiles):
            if hasattr(missile, 'dead') and missile.dead:
                data_to_send += f"#{send_t:.2f}\n-{missile.id}\n"
            else:
                # 记录导弹的位置
                loc_m = ENU2LLH(mark, missile.pos_)
                if missile.side == 'red':
                    color = 'Orange'
                else:
                    color = 'Green'
                data_to_send += f"#{send_t:.2f}\n{missile.id},T={loc_m[0]:.6f}|{loc_m[1]:.6f}|{loc_m[2]:.6f}," \
                                f"Name=AIM-120C,Color={color}\n"

        # print("data_to_send", data_to_send)
        tacview.send_data_to_client(data_to_send)
        time.sleep(0.01)
        # 红
        position = env.RUAV.pos_
        angle = np.array([env.RUAV.gamma, env.RUAV.theta, env.RUAV.psi]) * 180 / pi
        Rtrajectory.append(np.hstack((position, angle)))
        # 蓝
        position = env.BUAV.pos_
        angle = np.array([env.BUAV.gamma, env.BUAV.theta, env.BUAV.psi]) * 180 / pi
        Btrajectory.append(np.hstack((position, angle)))

        if terminate == True:
            break

# 补充显示
loc_o = ENU2LLH(mark, np.zeros(3))
data_to_send = ''
data_to_send = f"#{send_t + dt:.2f}\n{900},T={loc_o[0]:.6f}|{loc_o[1]:.6f}|{loc_o[2]:.6f},Name=Game Over, Color=Black\n"
# print("data_to_send", data_to_send)
tacview.send_data_to_client(data_to_send)

data_to_send = f"#{send_t + dt * 10:.2f}\n{900},T={loc_o[0]:.6f}|{loc_o[1]:.6f}|{loc_o[2]:.6f},Name=Game Over, Color=Black\n"
# print("data_to_send", data_to_send)
tacview.send_data_to_client(data_to_send)

end_time = time.time()  # 记录结束时间
print(f"程序运行时间: {end_time - start_time} 秒")
