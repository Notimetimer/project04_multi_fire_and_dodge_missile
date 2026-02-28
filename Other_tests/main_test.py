"""
对手为离散规则决策，我方为引导-控制结构
新增导弹和飞机不等步长解算
"""

import keyboard
import argparse
import time
from battle3dof1v1_missile import *  # battle3dof1v1_proportion
from math import pi
import numpy as np
import matplotlib
from torch.optim.lr_scheduler import OneCycleLR
import copy
import os
import pickle

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import torch
from torch import nn
import torch.nn.functional as F
import collections
import rl_utils
from tqdm import tqdm
import pandas as pd
from torch.distributions import Normal
import random
import gym
from guid2contr_abs import guided_control
import seaborn as sns  # 导入模块

stop_flag = False  # 控制退出的全局变量

# 设置随机种子

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


# torch.cuda.manual_seed(0)
# torch.cuda.manual_seed_all(0)

def stop_loop():
    global stop_flag
    stop_flag = True
    print("检测到 'esc' 键，准备退出所有循环！")


keyboard.add_hotkey('esc', stop_loop)  # 监听 'esc' 键，按下时执行 `stop_loop()`

# matplotlib.use('TkAgg')  # 'TkAgg' 或 'Qt5Agg'
if matplotlib.get_backend() != 'TkAgg':
    matplotlib.use('TkAgg')
    plt.switch_backend('TkAgg')


def get_args():
    parser = argparse.ArgumentParser("UAV swarm confrontation")
    # Environment
    parser.add_argument("--max-episode-len", type=float, default=8 * 60,  # * 60.0,
                        help="maximum episode time length")  # 对局时长：真的中远距空战可能会持续20分钟那么长
    # parser.add_argument("--num-RUAVs", type=int, default=1, help="number of red UAVs")
    # parser.add_argument("--num-BUAVs", type=int, default=1, help="number of blue UAVs")
    args = parser.parse_args()
    return args


def sub_of_radian(input1, input2):
    # 计算两个弧度的差值，范围为[-pi, pi]
    diff = input1 - input2
    diff = (diff + pi) % (2 * pi) - pi
    return diff


# 将红蓝双方运行的速度和位置保存
red_pos_list = np.empty((0, 3))
blue_pos_list = np.empty((0, 3))
red_vel_list = np.empty((0, 3))
blue_vel_list = np.empty((0, 3))

args = get_args()
env = Battle(args)
env.reset()
r_obs_spaces, b_obs_spaces = env.r_obs_spaces, env.b_obs_spaces
r_action_spaces, b_action_spaces = env.r_action_spaces, env.b_action_spaces

'超参数设置'
actor_lr0 = 2e-4  # 3e-4 # 归一化后学习率
critic_lr0 = actor_lr0 * 8  # 8,10 3e-3 critic 学习率
num_episodes = 500  # 1000  # 500  120
hidden_dim = [128, 128]  # [128, 128, 128]  # 32  # 隐藏层维度
gamma = 0.95  # 0.98
alpha_lr = actor_lr0  # 3e-4  # * actor_lr0  # 3e-4 归一化后学习率
tau = 0.005  # 0.005  # 软更新参数
buffer_size = 4000  # 8000 对于0.2s的dt这个大小足够，但是对于0.02s的dt就不太够了
minimal_size = 240  # 480
batch_size = minimal_size  # 64
epsilon0 = 0.9  # 0.1  # 随机策略概率

best_actor_state_dict = None

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def one_cycle_lr(base_lr, max_lr, current_episode, total_episodes):
    """实现one-cycle学习率调整策略"""
    # 前10%的episodes学习率从base_lr上升到max_lr
    if current_episode < total_episodes * 0.1:
        return base_lr + (max_lr - base_lr) * (current_episode / (total_episodes * 0.1))
    # 剩余90%的episodes使用余弦退火从max_lr降到base_lr
    else:
        progress = (current_episode - total_episodes * 0.1) / (total_episodes * 0.9)
        return max_lr - (max_lr - base_lr) * (1 + np.cos(np.pi * progress)) / 2


# todo 随机种子
replay_buffer = rl_utils.ReplayBuffer(buffer_size)  # 经验回放池
# actor网络输入层维度
r_obs_dim = np.shape(env.r_obs_spaces)[0]
# actor网络输出层维度
r_action_dim = 3  # np.shape(env.r_action_spaces[0])[0]
action_bound = 1  # env.r_action_spaces[0].high  # array([1., 1.], dtype=float32) 动作输出最大值


#
# def strong_rolling_optim(env, side='r'):
#     # 7种离散动作：
#     action_list = [
#         [[0, 0, 0]],
#         [[1, 0, 0]],
#         [[-1, 0, 0]],
#         [[0, 1, 0]],
#         [[0, 1, 1]],
#         [[0, 0, 0.4]],
#         [[0, 0, -0.4]]
#     ]
#     reward_list = []
#     # 遍历7种动作并记录对应的奖励信息
#     for i in range(len(action_list)):
#         if side == 'r':
#             r_action_n = action_list[i]
#             b_action_n = [[0, 0, 0]]  # 假设对面没有动作
#         elif side == 'b':
#             b_action_n = action_list[i]
#             r_action_n = [[0, 0, 0]]  # 假设对面没有动作
#         else:
#             raise RuntimeError("请输入正确的阵营")
#         # 执行动作并记录奖励信息
#         r_reward_n, b_reward_n, r_dones, b_dones, terminate = env.step(r_action_n, b_action_n, assume=True)
#         if side == 'r':
#             reward_list.append(r_reward_n)  # 获取当前动作状态奖励信息
#         if side == 'b':
#             reward_list.append(b_reward_n)  # 获取当前动作状态奖励信息
#     chosen_action = reward_list.index(max(reward_list))
#     # global r_action, b_action
#     # if side == 'r':
#     #     r_action.append(chosen_action)
#     # if side == 'b':
#     #     b_action.append(chosen_action)
#     return action_list[chosen_action]
#     # return action_list[0]


def weak_rolling_optim(env, side='r'):
    # 7种离散动作：
    action_list = [
        [[0, 0, 0]],
        [[0.2, 0, 0]],
        [[-0.2, 0, 0]],
        [[0, 0.2, 0]],
        [[0, 0.2, 1]],
        [[0, 0, 0.1]],
        [[0, 0, -0.1]]
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
    # global r_action, b_action
    # if side == 'r':
    #     r_action.append(chosen_action)
    # if side == 'b':
    #     b_action.append(chosen_action)
    return action_list[chosen_action]
    # return action_list[0]


'actor-critic网络'


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)


class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNetContinuous, self).__init__()
        self.action_bound = action_bound
        self.prelu1 = torch.nn.PReLU()
        self.action_dim = action_dim
        layers = []
        prev_size = state_dim
        for layer_size in hidden_dim:
            layers.append(nn.Linear(prev_size, layer_size))
            layers.append(self.prelu1)

            # layers.append(nn.ReLU())
            # layers.append(nn.LayerNorm(layer_size))  # test 添加LayerNorm归一化层， 效果不太好
            prev_size = layer_size
        # layers.append(nn.Linear(prev_size, action_dim))
        self.net = nn.Sequential(*layers)
        self.fc_mu = torch.nn.Linear(prev_size, action_dim)
        self.fc_std = torch.nn.Linear(prev_size, action_dim)
        # 固定神经网络初始化参数
        torch.nn.init.xavier_normal_(self.fc_mu.weight, gain=0.01)
        torch.nn.init.xavier_normal_(self.fc_std.weight, gain=0.01)

    def forward(self, x, sample=True):
        x = self.net(x)
        mu = self.fc_mu(x)
        if sample:
            std = F.softplus(self.fc_std(x)) + 1e-6
            dist = Normal(mu, std)
            normal_sample = dist.rsample()  # rsample()是重参数化采样
            log_prob = dist.log_prob(normal_sample)
            action = torch.tanh(normal_sample)
            # 计算tanh_normal分布的对数概率密度
            log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        else:
            action = torch.tanh(mu)
            log_prob = None  # 测试阶段不计算对数概率

        action = action * torch.tensor(self.action_bound)

        return action, log_prob


class QValueNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNetContinuous, self).__init__()
        # self.action_bound = action_bound
        self.prelu1 = torch.nn.PReLU()
        self.action_dim = action_dim
        layers = []
        prev_size = state_dim + action_dim
        for layer_size in hidden_dim:
            layers.append(nn.Linear(prev_size, layer_size))
            layers.append(self.prelu1)

            # layers.append(nn.ReLU())
            # layers.append(nn.LayerNorm(layer_size))  # test 添加LayerNorm归一化层， 效果不太好
            prev_size = layer_size
        # layers.append(nn.Linear(prev_size, action_dim))
        self.net = nn.Sequential(*layers)
        self.fc_out = torch.nn.Linear(prev_size, 1)

    def forward(self, x, a):
        # print('x',x.shape)
        # print('a',a.shape)
        cat = torch.cat([x, a], dim=1)
        y = self.net(cat)
        return self.fc_out(y)


class SACContinuous:
    ''' 处理连续动作的SAC算法 '''

    def __init__(self, state_dim, hidden_dim, action_dim, action_bound,
                 actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma,
                 device):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim, action_bound).to(device)  # 策略网络
        self.critic_1 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)  # 第一个Q网络
        self.critic_2 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)  # 第二个Q网络
        self.target_critic_1 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)  # 第一个目标Q网络
        self.target_critic_2 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)  # 第二个目标Q网络
        # 令目标Q网络的初始参数和Q网络一样
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, betas=(0.9, 0.999))  # betas参数默认值
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr, betas=(0.9, 0.999))
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr, betas=(0.9, 0.999))
        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)  # 注意，这的alpha也是可变的
        self.target_entropy = target_entropy  # 目标熵的大小
        self.gamma = gamma
        self.tau = tau
        self.device = device

    # test 更改学习率
    def update_actor_lr(self, new_lr):
        """
        更新Actor网络优化器的学习率
        :param new_lr: 新的学习率
        """
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = new_lr

    def update_critic_lr(self, new_lr):
        """
        更新Critic网络优化器的学习率
        :param new_lr: 新的学习率
        """
        for param_group in self.critic_1_optimizer.param_groups:
            param_group['lr'] = new_lr
        for param_group in self.critic_2_optimizer.param_groups:
            param_group['lr'] = new_lr

    def take_action(self, state, epsilon=0, explore=True):
        # state = torch.tensor([state], dtype=torch.float).to(self.device)
        temp = [state]
        temp = np.array(temp)
        state = torch.tensor(temp, dtype=torch.float).to(self.device)
        action = self.actor(state, sample=explore)[0].detach().cpu().numpy().flatten()  # cuda 需要.cpu()
        return action  # [action.item()]

    def calc_target(self, rewards, next_states, dones):  # 计算目标Q值
        next_actions, log_prob = self.actor(next_states)
        # entropy = -log_prob # 这里不对，应该把三维变成一维的
        entropy = -log_prob.sum(dim=1, keepdim=True)
        q1_value = self.target_critic_1(next_states, next_actions)
        q2_value = self.target_critic_2(next_states, next_actions)
        next_value = torch.min(q1_value, q2_value) + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        if td_target.shape[1] > 1:
            print('error')
            raise ValueError
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        temp = np.array(transition_dict['actions'])
        actions = torch.tensor(temp,
                               dtype=torch.float).to(self.device)  # squeeze
        # actions = torch.tensor(transition_dict['actions'],
        #                        dtype=torch.float).to(self.device)  # squeeze
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        # 更新两个Q网络
        td_target = self.calc_target(rewards, next_states, dones)

        # #  确保td_target的尺寸和Q网络输出的尺寸一致
        # td_target = td_target.view(-1, 1)  # 如果是一维的控制，需要添加这一行
        # print('states:',states.shape)
        # print('actions:',actions.shape)
        # print('td_target',td_target.shape)
        # print('self.critic_1(states, actions)', self.critic_1(states, actions).shape)
        # print('self.critic_2(states, actions)', self.critic_2(states, actions).shape)

        critic_1_loss = torch.mean(F.mse_loss(self.critic_1(states, actions), td_target.detach()))
        critic_2_loss = torch.mean(F.mse_loss(self.critic_2(states, actions), td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # 更新策略网络
        new_actions, log_prob = self.actor(states)
        entropy = -log_prob
        q1_value = self.critic_1(states, new_actions)
        q2_value = self.critic_2(states, new_actions)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy -
                                torch.min(q1_value, q2_value))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新alpha值
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)


'''
chap4 训练红方智能体
'''
if __name__ == '__main__':
    replay_buffer = ReplayBuffer(buffer_size)  # 经验回放
    # action_bound = env.action_space.high[0]  # 动作最大值
    target_entropy = -env.r_action_spaces[0].shape[0]

    agent = SACContinuous(r_obs_dim, hidden_dim, r_action_dim, action_bound,
                          actor_lr0, critic_lr0, alpha_lr, target_entropy, tau,
                          gamma, device)

    # train_off_policy_agent
    return_list = []
    start_time = time.time()

    '''训练'''
    # agent.actor.load_state_dict(torch.load('2dimwith_rule_actor.pth'))
    # agent.critic_1.load_state_dict(torch.load('2dimwith_rule_c1.pth'))
    # agent.critic_2.load_state_dict(torch.load('2dimwith_rule_c2.pth'))

    best_eps_return = -np.inf

    steps = 0
    last_action = np.zeros(3)

    with tqdm(total=int(num_episodes), desc='Iteration') as pbar:
        'num/10 start'
        for i_episode in range(int(num_episodes)):  # range(2):  # range(int(num_episodes / 10)):  # 每1/10训练回合

            steps_of_this_episode = 0
            epsilon = epsilon0 * 0.43 ** (-1 + (1 + i_episode) / num_episodes * 10)

            episode_return = 0
            env.reset()
            done = False
            # 环境运行一轮的情况
            episode_length = round(args.max_episode_len / dt_refer)
            # if i_episode<num_episodes/2:
            #     episode_length = round(episode_length/10)  # test

            'episode start'
            for count in range(episode_length):
                # 01
                steps += 1
                steps_of_this_episode += 1
                # 回合结束判断
                done = env.running == False or count == round(args.max_episode_len / dt_refer) - 1
                if done:
                    # print('回合结束，时间为：', env.t, 's')
                    break

                # 03
                # 获取观测信息
                r_obs_n, b_obs_n = env.get_obs()
                mask = np.zeros(13)  # test 屏蔽暂时顾不了的观测信息
                mask[[7, 8]] = 1  # 4, 5, 7, 8
                # mask[3] = 1 / 100  # 速率特征缩放
                r_obs_n = r_obs_n * mask
                r_obs = r_obs_n
                # r_obs = np.squeeze(r_obs_n)
                # 执行动作
                # r_action_n = [[-1, 0.0, 0.001]]  # 机动动作输入结构，期望nx,ny和gamma，范围[-1,1]
                b_action_n = weak_rolling_optim(env, side='b')  # 短视优化

                # 红方使用SAC执行动作得到环境反馈
                # 训练使用原有神经网络
                if i_episode + 1 < num_episodes:
                    red_required_plus = agent.take_action(np.squeeze(r_obs_n), explore=True)
                # 验证使用最优策略(cancel)
                else:
                    red_required_plus = agent.take_action(np.squeeze(r_obs_n), explore=False)
                    # # 1. 重建一个神经网络并传入参数
                    # # new_agent = SACContinuous(r_obs_dim, hidden_dim, r_action_dim, action_bound,
                    # #                           actor_lr0, critic_lr0, alpha_lr, target_entropy, tau,
                    # #                           gamma, device)
                    # # agent1 = new_agent
                    #
                    # # 2. 复制原神经网络的结构
                    # agent1 = copy.deepcopy(agent)
                    #
                    # # 3. 加载最优参数
                    # # 读内存
                    # if best_actor_state_dict is not None:
                    #     agent1.actor.load_state_dict(best_actor_state_dict)
                    # # # 读盘
                    # # agent1.actor.load_state_dict(torch.load('best_actor_temp.pth'))
                    #
                    # # 测试使用最优策略
                    # red_required_plus = agent1.take_action(np.squeeze(r_obs_n))

                # test epsilon-random策略
                if np.random.random() < epsilon:
                    red_required_plus = torch.randn(r_action_dim) * 0.3
                    red_required_plus = torch.clamp(red_required_plus, -1, 1).tolist()

                # # test 调整学习率

                # 计算当前学习率
                current_actor_lr = one_cycle_lr(1e-4, 3e-3, i_episode, num_episodes)
                current_critic_lr = current_actor_lr * 5  # 保持critic学习率是actor的5倍
                # 更新学习率
                agent.update_actor_lr(current_actor_lr)
                agent.update_critic_lr(current_critic_lr)

                # # 固定学习率
                # new_actor_lr = actor_lr0
                # new_critic_lr = critic_lr0
                #         agent.update_actor_lr(new_actor_lr)
                # agent.update_actor_lr(new_critic_lr)

                '叠加当前角度和速度'
                # 红方引导式控制
                current_R_velocity = env.RUAV.vel_
                current_R_speed = np.linalg.norm(env.RUAV.vel_)
                current_R_psi = np.arctan2(env.RUAV.vel_[2], env.RUAV.vel_[0])
                # current_R_theta = np.arctan2(env.RUAV.vel_[1], np.sqrt(env.RUAV.vel_[0] ** 2 + env.RUAV.vel_[2] ** 2))
                thetaR_req = red_required_plus[0] * np.pi / 2
                psiR_req = red_required_plus[1] * np.pi + current_R_psi  # red_required_plus[1]*np.pi # + current_psi
                vR_req = (0.5 + 0.5 * red_required_plus[2]) * (env.RUAV.speed_max - env.RUAV.speed_min) + \
                         env.RUAV.speed_min  # 340  # current_R_speed + 1/2*(1+red_required_plus[0])*340
                vR_req = np.clip(vR_req, env.RUAV.speed_min, env.RUAV.speed_max)

                # 04 引导式控制
                r_action_n = guided_control(env.RUAV, thetaR_req, psiR_req, vR_req)

                # 执行动作并获取环境反馈
                if i_episode + 1 < num_episodes:
                    r_reward_n, b_reward_n, r_dones, b_dones, terminate = env.step(r_action_n, b_action_n)  # 2、环境更新并反馈
                else:
                    agent.actor.eval()  # 设置模型为评估模式
                    with torch.no_grad():  # 禁止反向传播，反之：agent.actor.train()  # 恢复训练模式
                        r_reward_n, b_reward_n, r_dones, b_dones, terminate = env.step(r_action_n, b_action_n,
                                                                                       record=True)
                    # r_reward_n, b_reward_n, r_dones, b_dones, terminate = env.step(r_action_n, b_action_n, record=True)
                    # print(theta_req * 180 / pi)
                    # 记录当前环境中飞机的运动状态
                    red_pos_list = np.vstack((red_pos_list, env.RUAV.pos_))
                    blue_pos_list = np.vstack((blue_pos_list, env.BUAV.pos_))
                done = r_dones or b_dones

                # 05
                r_obs_n, b_obs_n = env.get_obs()  # 后state
                r_next_obs = r_obs_n  # np.squeeze(r_obs_n)
                r_next_obs = r_next_obs * mask

                # 06
                # 运行记录添加回放池
                replay_buffer.add(r_obs, red_required_plus, r_reward_n[0], r_next_obs, done)

                # 从经验回放池采样更新智能体
                if replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r,
                                       'dones': b_d}
                    agent.update(transition_dict)

                episode_return += r_reward_n[1]
                if stop_flag:
                    break
            'episode end'

            episode_return_per_step = episode_return / (count + 1)  # test
            return_list.append(episode_return_per_step)  # 平均回合奖励除以步数，以免存活越久看上去奖励越低
            # best_eps_return1 = max(best_eps_return, episode_return_per_step)
            # if best_eps_return1 > best_eps_return:
            #     best_eps_return = best_eps_return1
            #     # # 存盘
            #     # torch.save(agent.actor.state_dict(), 'best_actor_temp.pth')
            #     # 存内存
            #     best_actor_state_dict = copy.deepcopy(agent.actor.state_dict())  # 使用copy()创建深拷贝

            # 显示训练进度
            # if (i_episode + 1) % 10 == 0:
            if (i_episode + 1) >= 10:
                pbar.set_postfix({'episode': '%d' % (i_episode + 1),
                                  'episode_return': '%.3f' % np.mean(return_list[-10:])})
            pbar.update(1)

            # print('steps_of_this_episode:', steps_of_this_episode)
            print('period:', env.t)
            # print('end_speed', env.RUAV.speed)
            if stop_flag:
                break

    episodes_list = list(range(len(return_list)))

    from datetime import datetime
    # 获取当前日期和时间
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    df = pd.DataFrame({
        'Episodes': episodes_list,
        'Returns': return_list
    })
    # 在文件名中加入日期和时间
    df.to_csv(f'training_results_{current_time}.csv', index=False)

    if not stop_flag:
        # 训练曲线可视化
        plt.figure(1)
        plt.plot(episodes_list, return_list)
        plt.xlabel('Episodes')
        plt.ylabel('Returns')
        env_name = 'test-targetshooting'
        plt.title('SAC on {}'.format(env_name))
        # plt.show()

    end_time = time.time()  # 记录结束时间
    print(f"程序运行时间: {end_time - start_time} 秒")

    if not stop_flag:
        # 最后一幕轨迹显示
        red_pos_list = env.RUAV.trajectory
        blue_pos_list = env.BUAV.trajectory
        # 可视化
        red_p_show_show = np.array(red_pos_list).T
        blue_p_show_show = np.array(blue_pos_list).T

        plt.figure(2)
        from show_trajectory import show_trajectory

        show_trajectory(red_pos_list, blue_pos_list, min_east, min_north, min_height, max_east, max_north, max_height,
                        r_show=1, b_show=1)

        # dataframe = pd.DataFrame({'N': red_pos_list[:, 0], 'U': red_pos_list[:, 1], 'E': red_pos_list[:, 2]})
        # dataframe.to_csv("test.csv", index=False, sep=',')
        # print(steps)




        # plt.show()
    # # todo 保存网络参数（必须手动保存）
    # torch.save(agent.actor.state_dict(), '3dim_rule_actor_0.2s.pth')
    # torch.save(agent.critic_1.state_dict(), '3dim_rule_c1_0.2s.pth')
    # torch.save(agent.critic_2.state_dict(), '3dim_rule_c2_0.2s.pth')

