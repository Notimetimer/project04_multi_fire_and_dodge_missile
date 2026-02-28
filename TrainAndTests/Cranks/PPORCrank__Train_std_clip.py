'''
待续：使用带方差裁剪的算法训练crank策略
'''

import argparse
import time
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
from Algorithms.PPOcontinues_std_no_state_with_truncs import *
# from tqdm import tqdm #  停用tqdm
from LaunchZone.calc_DLZ import *
import multiprocessing as mp
from multiprocessing import Pool

use_tacview = 0  # 是否可视化

# matplotlib.use('TkAgg')  # 'TkAgg' 或 'Qt5Agg'
if matplotlib.get_backend() != 'TkAgg':
    matplotlib.use('TkAgg')
    plt.switch_backend('TkAgg')


parser = argparse.ArgumentParser("UAV swarm confrontation")
# Environment
parser.add_argument("--max-episode-len", type=float, default=180,  # 8 * 60,
                    help="maximum episode time length")  # test 真的中远距空战可能会持续20分钟那么长
parser.add_argument("--R-cage", type=float, default=70e3,  # 8 * 60,
                    help="")

# parser.add_argument("--num-RUAVs", type=int, default=1, help="number of red UAVs")
# parser.add_argument("--num-BUAVs", type=int, default=1, help="number of blue UAVs")
args = parser.parse_args()

# 超参数
dt_maneuver = 0.2 # 0.2 2
actor_lr = 1e-4 # 1e-4 1e-6  # 2e-5 警告，学习率过大会出现"nan"
critic_lr = actor_lr * 5  # *10 为什么critic学习率大于一都不会梯度爆炸？ 为什么设置成1e-5 也会爆炸？ chatgpt说要actor的2~10倍
# max_episodes = 1000 # 1000
max_steps = 120e4 # 65e4
hidden_dim = [128, 128, 128]  # 128
gamma = 0.95 # 0.9
lmbda = 0.95 # 0.9
epochs = 10  # 10
eps = 0.2
pre_train_rate = 0  # 0.25 # 0.25
k_entropy = 0.01 # 熵系数
mission_name = 'RCrank_with_std_clipping'

env = CrankTrainEnv(args, tacview_show=use_tacview)
# env = Battle(args, tacview_show=use_tacview)
r_action_spaces, b_action_spaces = env.r_action_spaces, env.b_action_spaces
action_bound0 = np.array([[-5000, 5000], [-pi, pi], [200, 600]])
action_bound = copy.deepcopy(action_bound0)
state_dim = 1+8+7+4+2  # len(b_obs_spaces)
action_dim = b_action_spaces[0].shape[0]

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 整一下高度-攻击区离散表（使用有限的选择）
# 水平距离20km~80km, 我机高度 env.min_alt_safe 到 env.max_alt_safe 按 2e3 间隔划分
# 目标高度 = 我机高度-2e3, 0, 2e3, 双方速度均为1.2Ma，最后构建一个[我机高度, 目标高度, 不可逃逸区边界，最大边界]的查询表


data_dir = os.path.join(project_root, "data")
os.makedirs(data_dir, exist_ok=True)
save_path = os.path.join(data_dir, "Crankinitial_states.npy")
# initial_states = np.load(save_path)
# print("读取的数据\n", initial_states)

# --- 仅保存一次网络形状（meta json），如果已存在则跳过
# log_dir = "./logs"
from datetime import datetime
# log_dir = os.path.join("./logs", "run-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
# log_dir = os.path.join("./logs", f"{mission_name}-run-" + datetime.now().strftime("%Y%m%d-%H%M%S"))

logs_dir = os.path.join(project_root, "logs/rcrank")
log_dir = os.path.join(logs_dir, f"{mission_name}-run-" + datetime.now().strftime("%Y%m%d-%H%M%S"))


def save_meta_once(path, state_dict):
    if os.path.exists(path):
        return
    meta = {k: list(v.shape) for k, v in state_dict.items()}
    with open(path, "w") as f:
        json.dump(meta, f)

def softplus(x):
    """
    计算softplus函数值
    
    参数:
    x -- 输入值（可以是标量、列表或NumPy数组）
    
    返回:
    计算后的softplus值，与输入形状相同
    """
    x_np = np.asarray(x)  # 确保输入是NumPy数组
    return np.log(1 + np.exp(x_np))

if __name__=="__main__":
    
    transition_dict_threshold = env.args.max_episode_len//env.dt_maneuver + 1

    agent = PPOContinuous(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                        lmbda, epochs, eps, gamma, device, critic_max_grad=2, actor_max_grad=2) # 2,2

    # # --- 仅保存一次网络形状（meta json），如果已存在则跳过
    # # log_dir = "./logs"
    # from datetime import datetime
    # # log_dir = os.path.join("./logs", "run-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    # # log_dir = os.path.join("./logs", f"{mission_name}-run-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    
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
    steps_count = 0

    logger = TensorBoardLogger(log_root=log_dir, host="127.0.0.1", port=6006, use_log_root=True, auto_show=False)

    total_steps = 0
    i_episode = 0

    training_start_time = time.time()
    launch_time_count = 0

    t_bias = 0
    steps_since_update = 0

    test_interval = 10 # 50

    try:
        # 强化学习训练
        # for i_episode in range(int(max_episodes*(1-pre_train_rate))):
        while total_steps < int(max_steps*(1-pre_train_rate)):
            i_episode += 1
            if i_episode % test_interval == test_interval-1:
                test_run = 1
            else:
                test_run = 0
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'action_bounds': [], 'max_stds': []}
            
            # # 飞机出生状态指定
            # init_case = np.random.randint(initial_states.shape[0])

            blue_height = random.uniform(3e3, 12e3)
            red_height = np.clip(blue_height+random.uniform(-2e3, 2e3), 3e3, 12e3)
            blue_psi = pi/2
            red_psi = -pi/2
            red_N = 0  # random.choice([-54e3, 54e3]) # red_N = random.uniform(-50e3, 50e3)
            red_E = 35e3
            blue_N = red_N
            blue_E = -35e3

            DEFAULT_RED_BIRTH_STATE = {'position': np.array([red_N, red_height, red_E]),
                                    'psi': red_psi}
            DEFAULT_BLUE_BIRTH_STATE = {'position': np.array([blue_N, blue_height, blue_E]),
                                        'psi': blue_psi}
            
            env.reset(red_birth_state=DEFAULT_RED_BIRTH_STATE, blue_birth_state=DEFAULT_BLUE_BIRTH_STATE,
                    red_init_ammo=0, blue_init_ammo=0) # 1

            done = False

            env.dt_maneuver = dt_maneuver
            actor_grad_list = []
            critc_grad_list = []
            actor_loss_list = []
            critic_loss_list = []
            entropy_list = []
            ratio_list = []

            # r_action_list = []
            # b_action_list = []
            
            episode_start_time = time.time()

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
                cos_delta_psi = b_obs_check["target_information"][0]
                sin_delta_psi = b_obs_check["target_information"][1]
                delta_psi = atan2(sin_delta_psi, cos_delta_psi)
                # 机动决策
                r_action_n = decision_rule(ego_pos_=env.RUAV.pos_, ego_psi=env.RUAV.psi,
                                        enm_pos_=env.BUAV.pos_, distance=distance,
                                        ally_missiles=env.Rmissiles, enm_missiles=env.Bmissiles,
                                        o00=o00, R_cage=env.R_cage, wander=1
                                        )
                if np.isnan(b_obs).any() or np.isinf(b_obs).any():
                    print('b_obs', b_obs_check)
                    print()

                # 方差裁剪
                max_std = 0.3
                # temp1 = min(height_ego-env.min_alt, env.max_alt-height_ego)
                if height_ego<env.min_alt_safe+1e3 or height_ego>env.max_alt_safe-1e3:
                    max_std = 0.1

                if delta_psi>50*pi/180 or delta_psi<-50*pi/180:
                    max_std = softplus(0.05+(60-abs(delta_psi)*180/pi)*0.25/10)


                # 每10个回合测试一次，测试回合不统计步数，不采集经验，不更新智能体，训练回合不回报胜负
                if not test_run:
                    b_action_n, u = agent.take_action(b_obs, action_bounds=action_bound, explore=True, max_std=max_std)
                else:
                    b_action_n, u = agent.take_action(b_obs, action_bounds=action_bound, explore=False, max_std=max_std)

                # b_action_n = decision_rule(ego_pos_=env.BUAV.pos_, ego_psi=env.BUAV.psi,
                #                         enm_pos_=env.RUAV.pos_, distance=distance,
                #                         ally_missiles=env.Bmissiles, enm_missiles=env.Rmissiles,
                #                         o00=o00, R_cage=env.R_cage, wander=0
                #                         )
                
                # # 动作裁剪
                # b_action_n[0] = np.clip(b_action_n[0], env.min_alt_safe-height_ego, env.max_alt_safe-height_ego)
                # if delta_psi>0:
                #     b_action_n[1] = max(sub_of_radian(delta_psi-50*pi/180, 0), b_action_n[1])
                # else:
                #     b_action_n[1] = min(sub_of_radian(delta_psi+50*pi/180, 0), b_action_n[1])

                env.step(r_action_n, b_action_n)  # 2、环境更新并反馈
                done, b_reward, b_event_reward = env.right_crank_terminate_and_reward('b')
                next_b_obs, _ = env.crank_obs('b')  # 子策略的训练不要用get_obs

                done = done
                if not test_run:
                    total_steps += 1
                    transition_dict['states'].append(b_obs)
                    transition_dict['actions'].append(u)
                    transition_dict['next_states'].append(next_b_obs)
                    transition_dict['rewards'].append(b_reward)
                    transition_dict['dones'].append(done)
                    transition_dict['action_bounds'].append(action_bound)
                    transition_dict['max_stds'].append(max_std)
                # state = next_state
                episode_return += b_reward * env.dt_maneuver
                # steps_since_update += 1

                # if steps_since_update >= transition_dict_threshold:
                #     steps_since_update = 0
                #     agent.update(transition_dict, adv_normed=0)
                #     transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'action_bounds': []}
                #     actor_grad_norm = agent.actor_grad
                #     actor_pre_clip_grad = agent.pre_clip_actor_grad
                #     critic_grad_norm = agent.critic_grad
                #     critic_pre_clip_grad = agent.pre_clip_critic_grad
                #     logger.add("train/1 episode_return", np.mean(episode_return), total_steps)
                #     # 梯度监控
                #     logger.add("train/3 actor_grad_norm", actor_grad_norm, total_steps)
                #     logger.add("train/5 actor_pre_clip_grad", actor_pre_clip_grad, total_steps)
                #     logger.add("train/4 critic_grad_norm", critic_grad_norm, total_steps)
                #     logger.add("train/6 critic_pre_clip_grad", critic_pre_clip_grad, total_steps)
                #     # 损失函数监控
                #     logger.add("train/7 actor_loss", agent.actor_loss, total_steps)
                #     logger.add("train/8 critic_loss", agent.critic_loss, total_steps)
                #     # 强化学习actor特殊项监控
                #     logger.add("train/9 entropy", agent.entropy_mean, total_steps)
                #     logger.add("train/10 ratio", agent.ratio_mean, total_steps)

                '''显示运行轨迹'''
                # 可视化
                env.render(t_bias=t_bias)
            
            episode_end_time = time.time()  # 记录结束时间
            # print(f"回合时长: {episode_end_time - episode_start_time} 秒")

            if env.lose==1:
                out_range_count+=1
            return_list.append(episode_return)

            if test_run == 1:
                logger.add("train/2 win", np.mean(env.win), total_steps)
                logger.add("train/2 lose", np.mean(env.lose), total_steps)

            # tensorboard 训练进度显示
            if test_run == 0:
                agent.update(transition_dict, adv_normed=False)
                actor_grad_norm = agent.actor_grad
                actor_pre_clip_grad = agent.pre_clip_actor_grad
                critic_grad_norm = agent.critic_grad
                critic_pre_clip_grad = agent.pre_clip_critic_grad
                logger.add("train/1 episode_return", np.mean(episode_return), total_steps)
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

                # print(t_bias)
                env.clear_render(t_bias=t_bias)
                t_bias += env.t

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
            
            # 训练进度显示
            if (i_episode) >= 10:
                print(f"episode {i_episode}, 进度: {total_steps/max_steps:.3f}, return: {np.mean(return_list[-10:]):.3f}")
            else:
                print(f"episode {i_episode}, total_steps {total_steps}")


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
