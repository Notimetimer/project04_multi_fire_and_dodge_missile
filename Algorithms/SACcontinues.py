import random
import gym
import numpy as np
from tqdm import tqdm
import collections
import torch
from torch import nn
import torch.nn.functional as F

from gym import spaces
from numpy.linalg import norm
from torch.distributions import Normal


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
    def __init__(self, state_dim, hidden_dims, action_dim):
        super(PolicyNetContinuous, self).__init__()
        self.prelu1 = torch.nn.PReLU()
        self.action_dim = action_dim
        layers = []
        prev_size = state_dim
        for layer_size in hidden_dims:
            layers.append(nn.Linear(prev_size, layer_size))
            layers.append(self.prelu1)
            # layers.append(nn.ReLU())
            prev_size = layer_size
        # layers.append(nn.Linear(prev_size, action_dim))
        self.net = nn.Sequential(*layers)
        self.fc_mu = torch.nn.Linear(prev_size, action_dim)
        self.fc_std = torch.nn.Linear(prev_size, action_dim)
        # torch.nn.init.xavier_normal_(self.fc_mu.weight, gain=0.01)
        # torch.nn.init.xavier_normal_(self.fc_std.weight, gain=0.01)

    def forward(self, x, action_bound=1, explore=True):
        x = self.net(x)
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x)) + 1e-6
        if explore:
            dist = Normal(mu, std)
            normal_sample = dist.rsample()  # rsample()是重参数化采样
            log_prob = dist.log_prob(normal_sample)
            action = torch.tanh(normal_sample)
            # 计算tanh_normal分布的对数概率密度
            log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
            action = action * action_bound
            return action, log_prob
        else:
            action = torch.tanh(mu) * action_bound
            return action, None

class QValueNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dims, action_dim):
        super(QValueNetContinuous, self).__init__()
        self.prelu1 = torch.nn.PReLU()
        self.action_dim = action_dim
        layers = []
        prev_size = state_dim + action_dim
        for layer_size in hidden_dims:
            layers.append(nn.Linear(prev_size, layer_size))
            layers.append(self.prelu1)
            # layers.append(nn.ReLU())
            prev_size = layer_size
        # layers.append(nn.Linear(prev_size, action_dim))
        self.net = nn.Sequential(*layers)
        self.fc_out = torch.nn.Linear(prev_size, 1)

        # self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dims)
        # self.fc2 = torch.nn.Linear(hidden_dims, hidden_dims)
        # self.fc_out = torch.nn.Linear(hidden_dims, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        y = self.net(cat)
        return self.fc_out(y)

        # x = F.relu(self.fc1(cat))
        # x = F.relu(self.fc2(x))
        # return self.fc_out(x)


class SACContinuous:
    ''' 处理连续动作的SAC算法 '''

    def __init__(self, state_dim, hidden_dims, action_dim,
                 actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma,
                 device):
        self.actor = PolicyNetContinuous(state_dim, hidden_dims, action_dim).to(device)  # 策略网络
        self.critic_1 = QValueNetContinuous(state_dim, hidden_dims, action_dim).to(device)  # 第一个Q网络
        self.critic_2 = QValueNetContinuous(state_dim, hidden_dims, action_dim).to(device)  # 第二个Q网络
        self.target_critic_1 = QValueNetContinuous(state_dim, hidden_dims, action_dim).to(device)  # 第一个目标Q网络
        self.target_critic_2 = QValueNetContinuous(state_dim, hidden_dims, action_dim).to(device)  # 第二个目标Q网络
        # 令目标Q网络的初始参数和Q网络一样
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)
        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
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

    def take_action(self, state, action_bound=1, explore=True):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action, _ = self.actor(state, action_bound=action_bound, explore=explore)
        return action.detach().numpy().flatten()

        # action, _ = self.actor(state)[0].detach().numpy().flatten()  # self.actor(state)[0]
        # return action  # [action.item()]

    def calc_target(self, rewards, next_states, dones):  # 计算目标Q值
        next_actions, log_prob = self.actor(next_states, explore=True)
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
        actions = torch.tensor(transition_dict['actions'],
                               dtype=torch.float).to(self.device)
        # actions = torch.tensor(transition_dict['actions'],
        #                        dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        # 更新两个Q网络
        td_target = self.calc_target(rewards, next_states, dones)

        critic_1_loss = torch.mean(F.mse_loss(self.critic_1(states, actions), td_target.detach()))
        critic_2_loss = torch.mean(F.mse_loss(self.critic_2(states, actions), td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # 更新策略网络
        new_actions, log_prob = self.actor(states, explore=True)
        entropy = -log_prob.sum(dim=1, keepdim=True)
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
