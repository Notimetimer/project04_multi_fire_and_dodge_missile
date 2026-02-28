# 游标训练环境
import random
import matplotlib
import matplotlib.pyplot as plt
import gym
from gym import spaces
from numpy.linalg import norm
from torch.distributions import Normal
import random
import numpy as np
from tqdm import tqdm
import collections
import torch
from torch import nn
import torch.nn.functional as F

# matplotlib.use('Qt5Agg')  # 使用Qt5作为后端

class Env:
    def __init__(self):
        self.min_pos = -10
        self.max_pos = 10
        self.position = None
        self.out_range = None
        self.dt = 1

    def reset(self):
        self.position = np.array([random.randint(self.min_pos, self.max_pos)],dtype='float64')
        self.v = np.array([0],dtype='float64')
        self.a = np.array([0],dtype='float64')
        self.steps=0
        self.out_range = 0
        return self.get_obs()

    def get_obs(self):
        return self.position

    def step(self, action_input):
        self.a = action_input
        self.v += self.a * self.dt
        self.position += self.v * self.dt  # + np.random.normal()
        self.position = np.clip(self.position, self.min_pos, self.max_pos) # 栏杆

        self.steps+=1
        done = self.get_done()
        reward = self.get_reward()
        # print(reward)
        return self.get_obs(), reward, done

    def get_done(self):
        done=0
        if self.position == self.min_pos or self.position == self.max_pos:
            self.a[0] = 0 
            self.v[0] = 0
        if self.position < self.min_pos or self.position > self.max_pos:
            done=1
            self.out_range = 1
        if self.steps>=20:
            done=1
        return done

    def get_reward(self):
        if self.min_pos <= self.position <= self.max_pos:
            reward1 = 1 - np.linalg.norm(self.position - (self.max_pos + self.min_pos) / 2) / 10
        else:
            reward1 = 0 # -3
        return reward1

# 改进算法

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()

        layers = []
        prev_size = state_dim
        for layer_size in hidden_dim:
            layers.append(torch.nn.Linear(prev_size, layer_size))
            layers.append(nn.ReLU())
            prev_size = layer_size
        self.net = nn.Sequential(*layers)
        self.fc_out = torch.nn.Linear(prev_size, 1)

    def forward(self, x):
        y = self.net(x)
        return self.fc_out(y)


class SquashedNormal:
    """带 tanh 压缩的高斯分布。

    采样：u ~ N(mu, std)（使用 rsample 支持 reparam），a = tanh(u)
    log_prob：基于 u 的 normal.log_prob(u) 并加上 tanh 的 Jacobian 修正项：-sum log(1 - tanh(u)^2)
    注意：外部需要把动作缩放到环境动作空间（仿射变换）。
    """

    def __init__(self, mu, std, eps=1e-6):
        self.mu = mu
        self.std = std
        self.normal = Normal(mu, std)
        self.eps = eps

    def sample(self):
        # rsample 以支持 reparameterization 重参数化采样, 结果是可导的
        u = self.normal.rsample()
        a = torch.tanh(u)
        return a, u

    def log_prob(self, a, u):
        # a: tanh(u)
        # log_prob(u) - sum log(1 - tanh(u)^2)
        # normal.log_prob 返回每个维度的 log_prob，需要 sum
        # 为数值稳定性添加小量
        log_prob_u = self.normal.log_prob(u)
        # jacobian term
        jacobian = torch.log(1 - a.pow(2) + self.eps) # fixme 应该+还是-？
        # sum over action dim, keep dims consistent: return (N, 1)
        return (log_prob_u - jacobian).sum(-1, keepdim=True)

    def entropy(self):
        # 近似：使用 base normal 的熵之和（不考虑 tanh 的修正）
        # 这在实践中通常足够，若需精确熵可用采样估计
        ent = self.normal.entropy().sum(-1)
        return ent


class PolicyNetContinuous(torch.nn.Module):
    """输出未压缩（pre-squash）的 mu 和 std。

    网络输出的 mu 是未经过 tanh 的原始均值，std 用 softplus 保证正值。
    不在网络内部做 action scaling，统一在采样/执行阶段处理。
    """

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetContinuous, self).__init__()
        layers = []
        prev_size = state_dim
        for layer_size in hidden_dim:
            layers.append(nn.Linear(prev_size, layer_size))
            layers.append(nn.ReLU())
            prev_size = layer_size
        self.net = nn.Sequential(*layers)
        self.fc_mu = torch.nn.Linear(prev_size, action_dim)
        self.fc_std = torch.nn.Linear(prev_size, action_dim)

    def forward(self, x, min_std=1e-3):
        x = self.net(x)
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        std = torch.clamp(std, min=min_std)
        return mu, std


class PPOContinuous:
    ''' 处理连续动作的PPO算法，支持时变动作区间（每步 amin/amax 不同）。

    设计说明（必须注意）：
    - 如果环境的动作约束随状态变化（amin/amax 为时变），则经验回放需保存当时的
      amin/amax（请把它放到 transition_dict['action_bounds']，形状为 (N, 2) 或每步的 (amin, amax)）。
    - 如果 action_bounds 在训练时始终恒定（标量或单个区间），也可以直接把 action_bound
      作为常数传入 update()。
    - 在本实现中，策略内部输出的是标准化前的 mu 和 std（即对 u 的分布参数）。
      对应的执行动作为：a = tanh(u)  -> normalized in (-1,1)
      最后缩放到真实区间： a_exec = amin + (a+1)/2 * (amax-amin)
    - update() 中会把存储的 a_exec "反归一化" 回 normalized a（[-1,1]），以便计算 log_prob。
    '''

    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device

    def _scale_action_to_exec(self, a, action_bounds):
        """把 normalized action a (in [-1,1]) 缩放到环境区间。

        action_bounds 可以是：
        - 单个数值：action_bound，表示对称区间 [-action_bound, action_bound]
        - 长度为 2 的元组/列表 (amin, amax)
        - 每步的数组，形状 (N, 2)
        """
        if isinstance(action_bounds, (int, float)):
            # 对称区间
            amin = -float(action_bounds)
            amax = float(action_bounds)
        else:
            amin, amax = action_bounds
        amin = torch.as_tensor(amin, dtype=a.dtype, device=a.device)
        amax = torch.as_tensor(amax, dtype=a.dtype, device=a.device)
        # a in (-1,1) -> scale to [amin, amax]
        return amin + (a + 1.0) * 0.5 * (amax - amin)

    def _unscale_exec_to_normalized(self, a_exec, action_bounds):
        """把执行动作 a_exec 反向归一化到 [-1,1]。

        如果 action_bounds 是标量，视作对称区间 [-b, b]。
        返回 normalized action (in (-1,1)).
        """
        if isinstance(action_bounds, (int, float)):
            amin = -float(action_bounds)
            amax = float(action_bounds)
        else:
            amin, amax = action_bounds
        amin = torch.as_tensor(amin, dtype=a_exec.dtype, device=a_exec.device)
        amax = torch.as_tensor(amax, dtype=a_exec.dtype, device=a_exec.device)
        # 防止除以零
        span = (amax - amin)
        span = torch.where(span == 0, torch.tensor(1e-6, device=span.device, dtype=span.dtype), span)
        a = 2.0 * (a_exec - amin) / span - 1.0
        # numerical stability
        return a.clamp(-0.999999, 0.999999)

    def take_action(self, state, action_bounds=1.0, explore=True):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        mu, std = self.actor(state)
        dist = SquashedNormal(mu, std)
        if explore:
            a_norm, u = dist.sample()
        else:
            # use mean action: tanh(mu)
            u = mu
            a_norm = torch.tanh(u)

        # # scale to actual action bounds
        # if isinstance(action_bounds, (int, float)):
        #     amin = -float(action_bounds)
        #     amax = float(action_bounds)
        # else:
        #     amin, amax = action_bounds
        # amin = torch.as_tensor(amin, dtype=a_norm.dtype, device=a_norm.device)
        # a_exec = amin + (a_norm + 1.0) * 0.5 * (float(amax) - float(amin))
        # return a_exec[0].cpu().detach().numpy().flatten()
        # 调用 _scale_action_to_exec 进行动作缩放

        a_exec = self._scale_action_to_exec(a_norm, action_bounds)
        return a_exec[0].cpu().detach().numpy().flatten()
    
    def update(self, transition_dict, action_bounds=None):
        """更新函数兼容以下几种调用方式：
        - 如果 action_bounds 是 None：期望 transition_dict 中包含 'action_bounds'，其形状为 (N,2) 或每步 (amin,amax)
        - 如果 action_bounds 是标量/二元元组/数组：作为全局固定区间使用

        transition_dict 必须包含 keys: 'states','actions','rewards','next_states','dones'
        当动作区间随步变化时，必须包含 'action_bounds' 与之对应。
        存储的 'actions' 应当是环境执行动作 a_exec（未归一化）。
        """
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions_exec = torch.tensor(transition_dict['actions'], dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # # 获取每步的 action_bounds，如果外部传入了 action_bounds 参数则优先使用
        # if action_bounds is None:
        #     if 'action_bounds' not in transition_dict:
        #         action_bounds = 1.0
        #         # raise KeyError("action_bounds 未提供：当动作约束为时变时，transition_dict 必须包含 'action_bounds'")
        #     # action_bounds_arr = transition_dict['action_bounds']
        # else:
        #     # 将固定 action_bounds 扩展为每步相同的序列，方便后续计算
        #     action_bounds_arr = [action_bounds] * len(transition_dict['actions'])

        if action_bounds is None:
            if 'action_bounds' in transition_dict:
                action_bounds = transition_dict['action_bounds']
            else:
                action_bounds = 1.0  # 默认值

        # 将 action_bounds 处理为每步的数组
        if isinstance(action_bounds, (int, float)):
            # 对称区间，扩展为每步相同的区间
            # action_bounds_arr = [action_bounds] * len(transition_dict['actions'])
            amin_list = [-float(action_bounds)] * len(transition_dict['actions'])
            amax_list = [float(action_bounds)] * len(transition_dict['actions'])
        elif isinstance(action_bounds, (tuple, list, np.ndarray)) and len(action_bounds) == 2:
            # 二元元组或列表，扩展为每步相同的 min 和 max
            amin_list = [float(action_bounds[0])] * len(transition_dict['actions'])
            amax_list = [float(action_bounds[1])] * len(transition_dict['actions'])
        else:
            # 每步不同的区间，直接解包
            amin_list = [float(ab[0]) if isinstance(ab, (tuple, list, np.ndarray)) else -float(ab) for ab in action_bounds]
            amax_list = [float(ab[1]) if isinstance(ab, (tuple, list, np.ndarray)) else float(ab) for ab in action_bounds]

        # 转换为张量
        amin_tensor = torch.tensor(amin_list, dtype=actions_exec.dtype, device=self.device).unsqueeze(-1)
        amax_tensor = torch.tensor(amax_list, dtype=actions_exec.dtype, device=self.device).unsqueeze(-1)

        # # 将 action_bounds 处理为每步的数组
        # if isinstance(action_bounds, (int, float)):
        #     action_bounds_arr = [action_bounds] * len(transition_dict['actions'])
        # else:
        #     action_bounds_arr = action_bounds

        # 将 action_bounds_arr 变为 tensor，形状 (N, 2)
        # 支持传入每步为 (amin,amax) 或 标量
        # amin_list = []
        # amax_list = []
        # for ab in action_bounds_arr:
        #     if isinstance(ab, (int, float)):
        #         amin_list.append(-float(ab))
        #         amax_list.append(float(ab))
        #     else:
        #         amin_list.append(float(ab[0]))
        #         amax_list.append(float(ab[1]))
        # amin_tensor = torch.tensor(amin_list, dtype=actions_exec.dtype, device=self.device).unsqueeze(-1)
        # amax_tensor = torch.tensor(amax_list, dtype=actions_exec.dtype, device=self.device).unsqueeze(-1)

        # 计算 td_target, advantage
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu(), dones.cpu()).to(self.device)

        # 策略输出（未压缩的 mu,std）
        mu, std = self.actor(states)
        # 构造 SquashedNormal 并计算 old_log_probs
        dist = SquashedNormal(mu.detach(), std.detach())

        # 将执行动作反向归一化到 [-1,1]，以便计算 log_prob
        # actions_normalized = 2.0 * (actions_exec - amin_tensor) / (amax_tensor - amin_tensor + 1e-8) - 1.0
        # actions_normalized = actions_normalized.clamp(-0.999999, 0.999999)
        actions_normalized = self._unscale_exec_to_normalized(actions_exec, (amin_tensor, amax_tensor))
        
        # 反算 u = atanh(a)
        u_old = torch.atanh(actions_normalized)
        old_log_probs = dist.log_prob(actions_normalized, u_old)

        if torch.isnan(old_log_probs).any():
            raise ValueError("old_log_probs 包含 NaN，检查 action_bounds 或 actions 的合法性")

        for _ in range(self.epochs):
            mu, std = self.actor(states)
            if torch.isnan(mu).any() or torch.isnan(std).any():
                raise ValueError("NaN in Actor outputs in loop")
            critic_values = self.critic(states)
            if torch.isnan(critic_values).any():
                raise ValueError("NaN in Critic outputs in loop")

            dist = SquashedNormal(mu, std)
            # 计算当前策略对历史执行动作的 log_prob（使用同一个 u_old）
            log_probs = dist.log_prob(actions_normalized, u_old)

            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = -torch.min(surr1, surr2).mean() - 0.1 * dist.entropy().mean()

            critic_loss = F.mse_loss(self.critic(states), td_target.detach())
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()

            # 梯度裁剪
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=2)
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=2)

            self.actor_optimizer.step()
            self.critic_optimizer.step()


# 注意：为了兼容原来的训练循环，请在构造 transition_dict 时保证：
# - 'actions' 存储的是环境实际执行的动作 a_exec（未归一化）
# - 如果动作区间随步变化，则 transition_dict['action_bounds'] 应为长度为步数的序列，
#   其中每个元素是 (amin, amax) 或 标量 b（表示对称区间 [-b,b]）。
# 示例：
# transition = {
#   'states': [...],
#   'actions': [...],  # 执行到环境的动作
#   'rewards': [...],
#   'next_states': [...],
#   'dones': [...],
#   'action_bounds': [(amin0,amax0), (amin1,amax1), ...]  # 可选
# }

# 以上实现把 action_bounds 纳入了 update 的计算链路，从而保证了 log_prob 的计算与当时执行动作的一致性，
# 避免了因动作裁剪/投影导致的策略-执行分布不匹配，从根本上缓解了你提到的 update 中梯度爆炸问题。


# 超参数
actor_lr = 1e-3 /10 # 1e-4 1e-6  # 2e-5 警告，学习率过大会出现"nan"
critic_lr = actor_lr * 10  # 1e-3  9e-3  5e-3 为什么critic学习率大于一都不会梯度爆炸？ 为什么设置成1e-5 也会爆炸？ chatgpt说要actor的2~10倍
num_episodes = 400 # fixme 如果不限制最小方差，500 的时候会梯度爆炸, 限制后1000 也会爆炸
hidden_dims = [128]  # 128 fixme 层数大时actor梯度也会爆炸
gamma = 0.9
lmbda = 0.9
epochs = 10  # 10 # fixme 4的时候也会梯度爆炸
eps = 0.2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# env_name = 'testEnv'
env = Env()
random.seed(0)
np.random.seed(0)
# env.seed(0)
torch.manual_seed(0)
state_dim = 1
action_dim = 1




action_bound = 3  # 动作最大值

agent = PPOContinuous(state_dim, hidden_dims, action_dim, actor_lr, critic_lr,
                      lmbda, epochs, eps, gamma, device)
out_range_count = 0
return_list = []
clear_batch_flag=1
with tqdm(total=int(num_episodes), desc='Iteration') as pbar:  # 进度条
    for i_episode in range(int(num_episodes)):  # 每个1/10的训练轮次
        episode_return = 0
        if clear_batch_flag:
            transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
            clear_batch_flag=0
        state = env.reset()
        done = False
        while not done:  # 每个训练回合
            # state_check=state
            # 1.执行动作得到环境反馈
            action = agent.take_action(state, action_bounds=action_bound, explore=True)
            next_state, reward, done = env.step(action)  # pendulum中的action一定要是ndarray才能输入吗？
            # print(reward)
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            state = next_state
            episode_return += reward

        if env.out_range==1:
            out_range_count+=1

        return_list.append(episode_return)
        if 1: # len(transition_dict['dones'])>20: # 逐batch更新
            agent.update(transition_dict, action_bound)
            clear_batch_flag=1
        if (i_episode + 1) >= 10:
            pbar.set_postfix({'episode': '%d' % (i_episode + 1),
                              'return': '%.3f' % np.mean(return_list[-10:])})
        pbar.update(1)
    # return return_list

# return_list = train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size)

# %matplotlib inline

episodes_list = list(range(len(return_list)))
plt.figure()
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
# plt.title('PPO on {}'.format(env_name))


mv_return = moving_average(return_list, 9)
plt.figure()
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
# plt.title('PPO on {}'.format(env_name))

plt.show()

print("出界次数：", out_range_count)