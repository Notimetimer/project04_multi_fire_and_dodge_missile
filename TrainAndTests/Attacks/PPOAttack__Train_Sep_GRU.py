'''
训练进攻策略:

红方闲庭信步
蓝方追赶红方
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
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from Envs.Tasks.AttackManeuverEnv import *
# from Envs.battle6dof1v1_missile0919 import *
#   battle3dof1v1_proportion battle3dof1v1_missile0812 battle3dof1v1_missile0901
from math import pi
import numpy as np
from collections import deque
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


from Algorithms.PPOcontinues_std_no_state_with_truncs_with_seperated_GRU import *

# from Algorithms.SquashedPPOcontinues_dual_a_out import *

# 实验性 AMPPO
# from Algorithms.SquashedPPOcontinues_dual_a_AM import *

# GRU-MLP ppo
from Algorithms.SharedLayers import *

# from tqdm import tqdm

use_tacview = 0  # 是否可视化

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

# 超参数
actor_lr = 1e-4  # 1e-4 1e-6  # 2e-5 警告，学习率过大会出现"nan"
critic_lr = actor_lr * 5  # *10 为什么critic学习率大于一都不会梯度爆炸？ 为什么设置成1e-5 也会爆炸？ chatgpt说要actor的2~10倍
# num_episodes = 1000 # 10000 1000
max_steps = 65e4
hidden_dim = [128, 128, 128]  # 128
gamma = 0.9
lmbda = 0.9
epochs = 10  # 10
eps = 0.2
pre_train_rate = 0  # 0.05 # 0.25 # 0.25
k_entropy = 0.01  # 熵系数
mission_name = 'AttackWithGRU3'

env = AttackTrainEnv(args, tacview_show=use_tacview)

r_action_spaces, b_action_spaces = env.r_action_spaces, env.b_action_spaces
action_bound = np.array([[-5000, 5000], [-pi, pi], [200, 600]])

state_dim = 8+7+2  # 35 # len(b_obs_spaces)
action_dim = 3
gru_hidden_size = state_dim
gru_num_layers = 1
middle_dim = state_dim
head_hidden_dims = hidden_dim

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# def launch_missile_if_possible(env, side='r'):
#     """
#     根据条件判断是否发射导弹
#     """
#     if side == 'r':
#         uav = env.RUAV
#         ally_missiles = env.Rmissiles
#         target = env.BUAV
#     else:  # side == 'b'
#         uav = env.BUAV
#         ally_missiles = env.Bmissiles
#         target = env.RUAV

#     waite = False
#     for missile in ally_missiles:
#         if not missile.dead:
#             waite = True
#             break
        
#     if not waite:
#         # 判断是否可以发射导弹
#         if uav.can_launch_missile(target, env.t):
#             # 发射导弹
#             new_missile = uav.launch_missile(target, env.t, missile_class)
#             uav.ammo -= 1
#             new_missile.side = 'red' if side == 'r' else 'blue'
#             if side == 'r':
#                 env.Rmissiles.append(new_missile)
#             else:
#                 env.Bmissiles.append(new_missile)
#             env.missiles = env.Rmissiles + env.Bmissiles
#             print(f"{'红方' if side == 'r' else '蓝方'}发射导弹")

def save_meta_once(path, state_dict):
    if os.path.exists(path):
        return
    meta = {k: list(v.shape) for k, v in state_dict.items()}
    with open(path, "w") as f:
        json.dump(meta, f)


if __name__ == "__main__":

    agent = PPOContinuous(state_dim, gru_hidden_size, gru_num_layers, middle_dim,
                          head_hidden_dims, action_dim, actor_lr, critic_lr,
                          lmbda, epochs, eps, gamma, device, 
                          critic_max_grad=2, actor_max_grad=2) # 2,2
    
    # SEQ_LEN = 10
    # --- 仅保存一次网络形状（meta json），如果已存在则跳过
    # log_dir = "./logs"
    from datetime import datetime

    # log_dir = os.path.join("./logs", "run-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    # log_dir = os.path.join("./logs", f"{mission_name}-run-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    logs_dir = os.path.join(project_root, "logs/attack")
    log_dir = os.path.join(logs_dir, f"{mission_name}-run-" + datetime.now().strftime("%Y%m%d-%H%M%S"))

    os.makedirs(log_dir, exist_ok=True)
    actor_meta_path = os.path.join(log_dir, "actor.meta.json")
    critic_meta_path = os.path.join(log_dir, "critic.meta.json")

    save_meta_once(actor_meta_path, agent.actor.state_dict())
    save_meta_once(critic_meta_path, agent.critic.state_dict())

    from Math_calculates.ScaleLearningRate import scale_learning_rate

    # 根据参数数量缩放学习率
    actor_lr = scale_learning_rate(actor_lr, agent.actor)
    critic_lr = scale_learning_rate(critic_lr, agent.critic)
    agent.set_learning_rate(actor_lr=actor_lr, critic_lr=critic_lr)

    from Visualize.tensorboard_visualize import TensorBoardLogger

    out_range_count = 0
    return_list = []
    win_list = []
    # steps_count = 0
    total_steps = 0
    i_episode = 0

    logger = TensorBoardLogger(log_root=log_dir, host="127.0.0.1", port=6006, use_log_root=True, auto_show=False)
    training_start_time = time.time()
    launch_time_count = 0
    t_bias = 0

    try:
        rl_steps = 0
        return_list = []
        win_list = []
        # with tqdm(total=int(num_episodes*(1-pre_train_rate)), desc='Iteration') as pbar:  # 进度条
        # for i_episode in range(int(num_episodes*(1-pre_train_rate))):
        while total_steps < int(max_steps * (1 - pre_train_rate)):
            i_episode += 1
            episode_return = 0
            transition_dict = {'states': [], 'actions': [], 'next_values': [], 'rewards': [], 'dones': [], 'h0as': [], 'h0cs': []}

            # 飞机出生状态指定
            red_R_ = random.uniform(30e3, 40e3)  # 20, 60 特意训练一个近的，测试一个远的
            red_beta = random.uniform(0, 2 * pi)
            red_psi = random.uniform(0, 2 * pi)
            red_height = random.uniform(3e3, 10e3)
            red_N = red_R_ * cos(red_beta)
            red_E = red_R_ * sin(red_beta)
            blue_height = random.uniform(3e3, 10e3)

            DEFAULT_RED_BIRTH_STATE = {'position': np.array([red_N, red_height, red_E]),
                                       'psi': red_psi
                                       }
            DEFAULT_BLUE_BIRTH_STATE = {'position': np.array([0.0, blue_height, 0.0]),
                                        'psi': pi
                                        }
            env.reset(red_birth_state=DEFAULT_RED_BIRTH_STATE, blue_birth_state=DEFAULT_BLUE_BIRTH_STATE,
                      red_init_ammo=0, blue_init_ammo=0)

            # # === 修改：初始化一个空的观测序列缓冲区 ===
            # # 不再用0填充
            # obs_sequence_deque = deque(maxlen=SEQ_LEN)

            # # === 新增：回合内步数计数器 ===
            # # 用于判断何时开始记录经验
            # episode_steps = 0

            # a1 = env.BUAV.pos_  # 58000,7750,20000
            # a2 = env.RUAV.pos_  # 2000,7750,20000
            # b1 = env.UAVs[0].pos_
            # b2 = env.UAVs[1].pos_
            done = False
            agent.reset_hidden_state_a()
            agent.reset_hidden_state_c()

            actor_grad_list = []
            critc_grad_list = []
            actor_loss_list = []
            critic_loss_list = []
            entropy_list = []
            ratio_list = []

            r_action_list = []
            b_action_list = []

            episode_start_time = time.time()

            # 环境运行一轮的情况
            for count in range(round(args.max_episode_len / dt_maneuver)):
                # print(f"time: {env.t}")  # 打印当前的 count 值
                # 0.回合结束判断
                # print(env.running)
                current_t = count * dt_maneuver
                if env.running == False or done:  # count == round(args.max_episode_len / dt_maneuver) - 1:
                    # print('回合结束，时间为：', env.t, 's')
                    break
                # 1.获取观测信息
                _, r_obs_check = env.attack_obs('r')
                b_obs_n, b_obs_check = env.attack_obs('b')

                # 在这里将观测信息压入记忆
                env.RUAV.obs_memory = r_obs_check.copy()
                env.BUAV.obs_memory = b_obs_check.copy()

                b_obs = np.squeeze(b_obs_n)
                
                h_a = agent.get_current_hidden_state_a()
                h_c = agent.get_current_hidden_state_c()

                # # 2. 更新观测序列
                # obs_sequence_deque.append(b_obs)

                # # 更新回合步数计数器
                # episode_steps += 1

                # # === 重要修改：准备模型输入 ===
                # # 即使序列不足10，也要为决策准备输入
                # # 我们用最新的观测值来填充序列的前面部分
                # current_obs_list = list(obs_sequence_deque)
                # if len(current_obs_list) < SEQ_LEN:
                #     # 如果长度不足，用第一个有效观测值填充到满10个
                #     padding = [current_obs_list[0]] * (SEQ_LEN - len(current_obs_list))
                #     current_sequence_state = np.array(padding + current_obs_list)
                # else:
                #     current_sequence_state = np.array(current_obs_list)


                distance = norm(env.RUAV.pos_ - env.BUAV.pos_)
                # 发射导弹判决
                if distance <= 40e3 and distance >= 5e3 and count % 1 == 0:  # 在合适的距离范围内每0.2s判决一次导弹发射
                    launch_time_count = 0
                    launch_missile_if_possible(env, side='r')
                    launch_missile_if_possible(env, side='b')

                # 机动决策
                r_action_n = decision_rule(ego_pos_=env.RUAV.pos_, ego_psi=env.RUAV.psi,
                                           enm_pos_=env.BUAV.pos_, distance=distance,
                                           ally_missiles=env.Rmissiles, enm_missiles=env.Bmissiles,
                                           o00=o00, R_cage=env.R_cage, wander=1
                                           )
                b_action_n, u, h_a_next = agent.take_action(state=b_obs, h_0_a=h_a, action_bounds=action_bound, explore=True)

                # b_action_n = decision_rule(ego_pos_=env.BUAV.pos_, ego_psi=env.BUAV.psi,
                #                         enm_pos_=env.RUAV.pos_, distance=distance,
                #                         ally_missiles=env.Bmissiles, enm_missiles=env.Rmissiles,
                #                         o00=o00, R_cage=env.R_cage, wander=0
                #                         )

                r_action_list.append(r_action_n)
                b_action_list.append(b_action_n)

                env.step(r_action_n, b_action_n)  # 2、环境更新并反馈
                done, b_reward, _ = env.get_terminate_and_reward('b')
                next_b_obs, _ = env.attack_obs('b')  # 子策略的训练不要用get_obs
                env.BUAV.act_memory = b_action_n.copy()  # 存储上一步动作
                total_steps += 1
                h_c_next = agent.get_current_hidden_state_c()
                next_value, _ = agent.get_value(next_b_obs, h_c_next)

                transition_dict['states'].append(b_obs)
                transition_dict['actions'].append(u)
                transition_dict['next_values'].append(next_value)
                transition_dict['rewards'].append(b_reward)
                transition_dict['dones'].append(done)
                transition_dict['h0as'].append(h_a)
                transition_dict['h0cs'].append(h_c)
                # state = next_state
                episode_return += b_reward * env.dt_maneuver

                '''显示运行轨迹'''
                # 可视化
                env.render(t_bias=t_bias)

            episode_end_time = time.time()  # 记录结束时间
            # print(f"回合时长: {episode_end_time - episode_start_time} 秒")

            if env.lose == 1:
                out_range_count += 1
            return_list.append(episode_return)
            win_list.append(1 - env.lose)
            agent.update(transition_dict)

            # print(t_bias)
            env.clear_render(t_bias=t_bias)
            t_bias += env.t
            r_action_list = np.array(r_action_list)
            b_action_list = np.array(b_action_list)

            # --- 保存模型（强化学习阶段：actor_rein + i_episode，critic 每次覆盖）
            os.makedirs(log_dir, exist_ok=True)
            # critic overwrite
            critic_path = os.path.join(log_dir, "critic.pt")
            th.save(agent.critic.state_dict(), critic_path)
            # actor RL snapshot
            if i_episode % 10 == 0:
                actor_name = f"actor_rein{i_episode}.pt"
                actor_path = os.path.join(log_dir, actor_name)
                th.save(agent.actor.state_dict(), actor_path)

            # # tqdm 训练进度显示
            # if (i_episode + 1) >= 10:
            #     pbar.set_postfix({'episode': '%d' % (i_episode + 1),
            #                     'return': '%.3f' % np.mean(return_list[-10:])})
            # pbar.update(1)
            # 训练进度显示
            if (i_episode) >= 10:
                print(
                    f"episode {i_episode}, 进度: {total_steps / max_steps:.3f}, return: {np.mean(return_list[-10:]):.3f}")
            else:
                print(f"episode {i_episode}, total_steps {total_steps}")

            # tensorboard 训练进度显示
            logger.add("train/1 episode_return", episode_return, total_steps)
            logger.add("train/2 win", env.win, total_steps)

            actor_grad_norm = agent.actor_grad
            actor_pre_clip_grad = agent.pre_clip_actor_grad
            critic_grad_norm = agent.critic_grad
            critic_pre_clip_grad = agent.pre_clip_critic_grad
            # 梯度监控
            logger.add("train/3 actor_grad_norm", actor_grad_norm, total_steps)
            logger.add("train/5 actor_pre_clip_grad", actor_pre_clip_grad, total_steps)
            logger.add("train/4 critic_grad_norm", critic_grad_norm, total_steps)
            logger.add("train/6 critic_pre_clip_grad", critic_pre_clip_grad, total_steps)
            # 损失函数监控
            logger.add("train/7 actor_loss", agent.actor_loss, total_steps)
            logger.add("train/8 critic_loss", agent.critic_loss, total_steps)
            # 强化学习actor特殊项监控
            logger.add("train/9 entropy", agent.entropy_mean, total_steps)
            logger.add("train/10 ratio", agent.ratio_mean, total_steps)

        training_end_time = time.time()  # 记录结束时间
        elapsed = training_end_time - training_start_time
        from datetime import timedelta

        td = timedelta(seconds=elapsed)
        d = td.days
        h, rem = divmod(td.seconds, 3600)
        m, s = divmod(rem, 60)
        print(f"总训练时长: {d}天 {h}小时 {m}分钟 {s}秒")


    except KeyboardInterrupt:
        print("\n检测到 KeyboardInterrupt，正在关闭 logger ...")
    finally:
        logger.close()
        print(f"日志已保存到：{logger.run_dir}")