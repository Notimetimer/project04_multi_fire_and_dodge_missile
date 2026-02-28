# 相比书籍原版，新增了列表定义多层神经网络形状的方法

import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
# import rl_utils
from tqdm import tqdm
from torch import nn
import sys
import os
sys.path.append(project_root)
from Algorithms.PPOdiscrete import *


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def compute_advantage(gamma, lmbda, td_delta, dones):
    td_delta = td_delta.detach().cpu().numpy()
    dones = dones.detach().cpu().numpy() # [新增] 转为 numpy
    advantage_list = []
    advantage = 0.0
    
    # [修改] 同时遍历 delta 和 done
    for delta, done in zip(td_delta[::-1], dones[::-1]):
        # 如果当前是 done，说明这是序列的最后一步（或者该步之后没有未来），
        # 此时不应该加上一步（时间上的未来）的 advantage。
        # 注意：这里的 advantage 变量存的是“下一步的优势”，所以要乘 (1-done)
        advantage = delta + gamma * lmbda * advantage * (1 - done)
        advantage_list.append(advantage)
        
    advantage_list.reverse()
    return torch.tensor(np.array(advantage_list), dtype=torch.float)


# 加入新的 仿真环境：石头剪刀布，每回合10场。对手策略：初始随机，
# 若对手赢则保持不变；若对手输則按 rock->paper, paper->scissors, scissors->rock 旋转（opp = (opp+1)%3）
class RPS_Env(gym.Env):
    """
    Observation: 4-dim vector:
      - prev opponent move (one-hot, 3)
      - prev result scalar: win=1.0, tie=0.0, loss=-1.0 (1 dim)
    Action: 0=rock,1=paper,2=scissors
    Reward: +1 win, -1 loss, 0 tie
    Episode length: rounds_per_episode (内部计数，不作为观测)
    Opponent rule: initial random; if opponent loses (agent wins) -> rotate opp move (0->1->2->0),
                   if opponent wins or tie -> keep same.
    """
    metadata = {'render.modes': []}
    def __init__(self, rounds_per_episode=10, seed=None):
        super().__init__()
        self.rounds_per_episode = rounds_per_episode
        self.action_space = gym.spaces.Discrete(3)
        # prev opp one-hot (3) + prev result scalar (1)
        self.observation_space = gym.spaces.Box(low=np.array([0.0,0.0,0.0,-1.0], dtype=np.float32),
                                                high=np.array([1.0,1.0,1.0, 1.0], dtype=np.float32),
                                                shape=(4,), dtype=np.float32)
        self._rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        self.opp_move = 0          # opponent's current move (used this step)
        self.prev_opp_move = 0     # opponent move from previous step (for observation)
        self.prev_result = 0.0     # previous result scalar: 1.0 win, 0.0 tie, -1.0 loss
        self.round_idx = 0

    def seed(self, s=None):
        self._rng = np.random.RandomState(s)

    def reset(self):
        # initialize opponent current move
        self.opp_move = int(self._rng.randint(0, 3))
        # previous info for the first step should be random per要求
        self.prev_opp_move = int(self._rng.randint(0, 3))
        self.prev_result = float(self._rng.choice([1.0, 0.0, -1.0]))
        self.round_idx = 0
        return self._get_obs()

    def step(self, action):
        assert self.action_space.contains(action)
        agent = int(action)
        opp = int(self.opp_move)            # opponent's action used this round (current)
        # outcome from agent perspective: (agent - opp) mod 3 -> 1 win, 2 lose, 0 tie
        diff = (agent - opp) % 3
        if diff == 1:
            reward = 1.0
            # opponent lost -> update opponent move for next round by rotating
            next_opp = (self.opp_move + 1) % 3
            result_scalar = 1.0   # agent win
        elif diff == 2:
            reward = -1.0
            next_opp = self.opp_move  # opponent won -> keep same
            result_scalar = -1.0  # agent loss
        else:
            reward = 0.0
            next_opp = self.opp_move  # tie -> keep same
            result_scalar = 0.0   # tie

        # set previous info for next state's observation:
        self.prev_opp_move = opp
        self.prev_result = result_scalar

        # update opponent for next round
        self.opp_move = next_opp

        self.round_idx += 1
        done = (self.round_idx >= self.rounds_per_episode)
        obs = self._get_obs()
        info = {'opp_move': int(self.opp_move), 'prev_result': float(self.prev_result)}
        return obs, float(reward), bool(done), info

    def _get_obs(self):
        # prev opponent one-hot
        onehot_prev_opp = np.zeros(3, dtype=np.float32)
        onehot_prev_opp[self.prev_opp_move] = 1.0
        # prev result scalar as one float
        res_scalar = np.array([self.prev_result], dtype=np.float32)
        return np.concatenate([onehot_prev_opp, res_scalar])

# 超参数
actor_lr = 1e-3
critic_lr = 1e-2
num_episodes = 200 # 500
hidden_dim = [128]
gamma = 0.98
lmbda = 0.95
epochs = 10
eps = 0.2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env = RPS_Env(rounds_per_episode=10, seed=0)
env.seed(0)
torch.manual_seed(0)
state_dim = env.observation_space.shape[0]  # 应为4
action_dim = env.action_space.n  # 应为3
action_dims_dict = {'cont':0, 'cat':3, 'bern':0}
action_bound = None

agent = PPO_discrete(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
                     epochs, eps, gamma, device)

return_list = []
for i in range(10):
    with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes/10)):
            episode_return = 0
            transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
            state = env.reset()
            done = False
            while not done:
                
                _, u = agent.take_action(state, explore=1)
                action = u # ['cat'][0]

                next_state, reward, done, _ = env.step(action)
                transition_dict['states'].append(state)
                transition_dict['actions'].append(u)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                state = next_state
                episode_return += reward

            return_list.append(episode_return)
            agent.update(transition_dict, adv_normed=0)
            if (i_episode+1) % 10 == 0:
                pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
            pbar.update(1)

# 画图展示收敛过程
episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
# plt.title('PPO on {}'.format(env_name))
plt.show()

mv_return = moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
# plt.title('PPO on {}'.format(env_name))
plt.show()


state = env.reset()
done = False
steps = 0
while not done:
    steps += 1
    action, _ = agent.take_action(state, explore=0)
    action = action['cat']
    next_state, reward, done, _ = env.step(action)
    # 拆分 next_state：前3维为 one-hot（对手动作），第4维为上一步胜负标量
    onehot = np.asarray(next_state[:3], dtype=np.float32)
    # 如果 one-hot 全 0，返回 None，否则 argmax +1 -> 1,2,3
    if onehot.sum() == 0:
        opp_move = None
    else:
        opp_move = int(np.argmax(onehot))
    prev_result = float(next_state[3])
    state = next_state
    print("第", steps, "步 动作", action, "对手动作", opp_move, "当前步胜负", prev_result)

