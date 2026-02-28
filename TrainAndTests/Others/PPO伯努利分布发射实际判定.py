import gymnasium as gymn
from gymnasium import spaces
import numpy as np
import pygame
import math
import time
import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
# import rl_utils
from tqdm import tqdm
from torch import nn

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


class ArcherEnv(gymn.Env):
    """
    枪手射击移动目标的环境。

    - **观测空间 (Observation Space)**: 3维Box
        1. [0] 与目标的距离 (m)
        2. [1] 距离上次射击的时间 (s)
        3. [2] 剩余箭的数量

    - **动作空间 (Action Space)**: 离散(2)
        - 0: 不发射
        - 1: 发射

    - **奖励 (Reward)**:
        - 获胜: +20
        - 失败: -20
        

    - **终止条件 (Termination)**:
        - 射中5箭 (获胜)
        - 箭用完 (失败)
        - 目标距离小于2m (失败)
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()

        # --- 环境参数 ---
        self.initial_distance = 100.0  # 初始距离 (m)
        self.target_speed = 4.0        # 目标速度 (m/s)
        self.min_safe_distance = 2.0   # 枪手失败的最小距离 (m)
        self.total_arrows = 10         # 初始箭数
        self.hits_to_win = 5           # 获胜所需命中数

        # --- 时间步长 ---
        self.decision_dt = 0.5        # 决策步长 (s)
        self.physics_dt = self.decision_dt # 0.1    # 物理模拟步长 (s)
        self.physics_steps_per_decision = int(self.decision_dt / self.physics_dt)

        # --- 命中率参数 ---
        self.guaranteed_hit_range = 20.0 # 必中范围 (m)
        # 根据 P(d) = exp(-k * (d - 20)) 和 P(50)=0.5 计算衰减系数k
        self.hit_decay_k = -math.log(0.5) / (50.0 - self.guaranteed_hit_range)

        # --- 定义观测空间和动作空间 ---
        # [距离, 上次射击间隔, 剩余箭数]
        obs_low = np.array([0, 0, 0], dtype=np.float32)
        obs_high = np.array([self.initial_distance + 10, 100, self.total_arrows], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # [不发射, 发射]
        # 动作为单维 0/1，使用 float32 的 Box 更兼容外部传入的标量或数组
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        # --- Pygame 可视化 ---
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.font = None

    def get_obs(self):
        return np.array([
            self.target_pos / 10,
            self.time_since_last_shot,
            self.arrows_left
        ], dtype=np.float32)

    def _get_info(self):
        return {
            "distance": self.target_pos,
            "arrows_left": self.arrows_left,
            "hits": self.hits
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 初始化状态
        self.target_pos = self.initial_distance
        self.arrows_left = self.total_arrows
        self.hits = 0
        self.time_since_last_shot = 0.0 # 可设为较大值表示从未射击

        # 用于可视化的箭矢列表
        self.projectiles = []

        if self.render_mode == "human":
            self._render_frame()

        return self.get_obs(), self._get_info()

    def step(self, action):
        # 兼容外部可能传入的标量 0/1 或形如 (1,) 的数组
        a = np.asarray(action)
        if a.ndim == 0:
            a = np.array([a])
        # 转为 float32 以匹配 action_space.dtype，再裁剪到 [0,1]
        a = np.clip(a.astype(np.float32), 0.0, 1.0)
        if not self.action_space.contains(a):
            # 保险起见打印并抛出更明确的错误
            raise ValueError(f"Action {action!r} is not within the action space {self.action_space}")
        # 最终用整数 0/1 表示是否开火
        action = int(a.flat[0])

        reward = 0.0
        terminated = False

        # --- 1. 处理射击动作 ---
        shoot_action = (action == 1)
        if shoot_action and self.arrows_left > 0:
            # a. 计算射击奖励
            # reward -= 0.5  # 每射一箭的固定惩罚 ###
            # if self.time_since_last_shot < 5.0:
            #     reward += (self.time_since_last_shot) / 5.0

            # b. 判定是否命中
            distance_at_shot = self.target_pos
            hit_prob = 0.0
            if distance_at_shot <= self.guaranteed_hit_range:
                hit_prob = 1.0
            else:
                hit_prob = math.exp(-self.hit_decay_k * (distance_at_shot - self.guaranteed_hit_range))
            # if distance_at_shot <= 70:
            #     hit_prob = 1.0
            # else:
            #     hit_prob = 0.0
            
            if self.np_random.random() < hit_prob: # 如果命中
                self.hits += 1
                reward += 2 # * np.clip(distance_at_shot/100, 0, 1)
                # 可视化：添加一个表示命中的标记
                if self.render_mode == "human":
                    self.projectiles.append({"pos": distance_at_shot, "hit": True})
            else:
                if self.render_mode == "human":
                    self.projectiles.append({"pos": distance_at_shot, "hit": False})

            # c. 更新状态
            self.arrows_left -= 1
            self.time_since_last_shot = 0.0
        
        # --- 2. 模拟世界演进 ---
        # # 一个决策步包含多个物理步
        # for _ in range(self.physics_steps_per_decision):
        #     if not terminated: # 如果已结束，则不再移动目标
        #         self.target_pos -= self.target_speed * self.physics_dt
        #         self.target_pos = max(0, self.target_pos) # 防止穿过枪手
        self.target_pos -= self.target_speed * self.decision_dt

        self.time_since_last_shot += self.decision_dt

        # --- 3. 检查终止条件 ---
        if self.hits >= self.hits_to_win:
            reward += 20  # 获胜奖励
            terminated = True
        elif self.arrows_left <= 0 and self.hits < self.hits_to_win:
            reward -= 20  # 失败奖励
            terminated = True
        elif self.target_pos <= self.min_safe_distance:
            reward -= 20  # 失败奖励
            terminated = True

        if self.render_mode == "human":
            self._render_frame()

        return self.get_obs(), reward, terminated, False, self._get_info()

    def render(self):
        if self.render_mode == "human":
            return self._render_frame()

    def _render_frame(self):
        if self.screen is None:
            pygame.init()
            pygame.display.set_caption("Archer vs Target")
            self.screen = pygame.display.set_mode((800, 200))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)

        # 绘制背景
        self.screen.fill((210, 210, 210)) # 浅灰色背景

        # 坐标转换：将米映射到像素
        def to_px(pos_m):
            return int(50 + (pos_m / self.initial_distance) * 700)

        # 绘制枪手 (绿色方块)
        archer_px = to_px(0)
        pygame.draw.rect(self.screen, (0, 150, 0), (archer_px - 10, 100 - 20, 20, 40))

        # 绘制目标 (红色方块)
        target_px = to_px(self.target_pos)
        pygame.draw.rect(self.screen, (150, 0, 0), (target_px - 10, 100 - 15, 20, 30))

        # 绘制箭矢和命中效果
        for proj in self.projectiles:
            proj_px = to_px(proj["pos"])
            if proj["hit"]:
                # 命中：在目标位置画一个黄色星星
                pygame.draw.circle(self.screen, (255, 255, 0), (proj_px, 100), 8)
            else:
                # 未命中：在目标位置画一个灰色叉
                pygame.draw.line(self.screen, (50, 50, 50), (proj_px - 5, 95), (proj_px + 5, 105), 2)
                pygame.draw.line(self.screen, (50, 50, 50), (proj_px - 5, 105), (proj_px + 5, 95), 2)
        
        # 清理旧的箭矢效果（只显示一小段时间）
        if len(self.projectiles) > 0:
            self.projectiles.pop(0)

        # 绘制状态信息
        info_text = (
            f"Distance: {self.target_pos:.1f}m | "
            f"Arrows: {self.arrows_left}/{self.total_arrows} | "
            f"Hits: {self.hits}/{self.hits_to_win}"
        )
        text_surface = self.font.render(info_text, True, (0, 0, 0))
        self.screen.blit(text_surface, (10, 10))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None
            self.font = None




# 计算并记录 actor / critic 的梯度范数（L2）
def model_grad_norm(model):
    total_sq = 0.0
    found = False
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach().cpu()
            total_sq += float(g.norm(2).item()) ** 2
            found = True
    return float(total_sq ** 0.5) if found else float('nan')

def check_weights_bias_nan(model, model_name="model", place=None):
    """检查模型中名为 weight/bias 的参数是否包含 NaN，发现则抛出异常。
    参数:
      model: torch.nn.Module
      model_name: 用于错误消息中标识模型（如 "actor"/"critic"）
      place: 字符串，调用位置/上下文（如 "update_loop","pretrain_step"），用于更明确的错误报告
    """
    for name, param in model.named_parameters():
        if ("weight" in name) or ("bias" in name):
            if param is None:
                continue
            if torch.isnan(param).any():
                loc = f" at {place}" if place else ""
                raise ValueError(f"NaN detected in {model_name} parameter '{name}'{loc}")


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


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dims):
        super(ValueNet, self).__init__()
        # self.prelu = torch.nn.PReLU()

        layers = []
        prev_size = state_dim
        for layer_size in hidden_dims:
            layers.append(torch.nn.Linear(prev_size, layer_size))
            # layers.append(self.prelu)
            layers.append(nn.ReLU())
            prev_size = layer_size
        self.net = nn.Sequential(*layers)
        self.fc_out = torch.nn.Linear(prev_size, 1)

        # # 添加参数初始化
        # for layer in self.net:
        #     if isinstance(layer, nn.Linear):
        #         torch.nn.init.xavier_normal_(layer.weight, gain=0.01)
        # torch.nn.init.xavier_normal_(self.fc_out.weight, gain=0.01)

    def forward(self, x):
        y = self.net(x)
        return self.fc_out(y)


class PolicyNetDiscrete(torch.nn.Module):
    def __init__(self, state_dim, hidden_dims, action_dim):
        super(PolicyNetDiscrete, self).__init__()
        self.prelu = torch.nn.PReLU()
        layers = []
        prev_size = state_dim
        for layer_size in hidden_dims:
            layers.append(nn.Linear(prev_size, layer_size))
            # layers.append(self.prelu)
            layers.append(nn.ReLU())
            prev_size = layer_size
        self.net = nn.Sequential(*layers)
        self.fc_out = torch.nn.Linear(prev_size, action_dim)

        # # 固定神经网络初始化参数
        # torch.nn.init.xavier_normal_(self.fc_out.weight, gain=0.01)

    def forward(self, x):
        x = self.net(x)
        return F.softmax(self.fc_out(x), dim=1)


class PolicyNetBernouli(torch.nn.Module):
    def __init__(self, state_dim, hidden_dims, action_dim):
        super(PolicyNetBernouli, self).__init__()
        self.prelu = torch.nn.PReLU()
        layers = []
        prev_size = state_dim
        for layer_size in hidden_dims:
            layers.append(nn.Linear(prev_size, layer_size))
            # layers.append(self.prelu)
            layers.append(nn.ReLU())
            prev_size = layer_size
        self.net = nn.Sequential(*layers)
        self.fc_out = torch.nn.Linear(prev_size, action_dim)  # 输出 action_dim 个概率值

        # # 固定神经网络初始化参数
        # torch.nn.init.xavier_normal_(self.fc_out.weight, gain=0.01)

    def forward(self, x):
        x = self.net(x)
        # 使用 sigmoid 激活函数将输出限制在 0 到 1 之间，表示伯努利分布的概率
        return torch.sigmoid(self.fc_out(x))


class PPO_bernouli:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, state_dim, hidden_dims, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device, k_entropy=0.01, critic_max_grad=2, actor_max_grad=2):
        self.actor = PolicyNetBernouli(state_dim, hidden_dims, action_dim).to(device)  # 使用 PolicyNetBernouli
        self.critic = ValueNet(state_dim, hidden_dims).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device
        self.k_entropy = k_entropy
        self.critic_max_grad=critic_max_grad
        self.actor_max_grad=actor_max_grad

    def set_learning_rate(self, actor_lr=None, critic_lr=None):
        """动态设置 actor 和 critic 的学习率"""
        if actor_lr is not None:
            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] = actor_lr
        if critic_lr is not None:
            for param_group in self.critic_optimizer.param_groups:
                param_group['lr'] = critic_lr    

    # take action
    def take_action(self, state, explore=True, mask=None):
        # state -> tensor (1, state_dim)
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        probs = self.actor(state)  # (1, action_dim)

        if explore:
            sampled = torch.bernoulli(probs)  # (1, action_dim)
        else:
            sampled = (probs >= 0.5).float()

        # to numpy arrays (flatten for probs, keep action as 1-D ndarray)
        probs_np = probs.detach().cpu().numpy().flatten()           # shape (action_dim,)
        actions_np = sampled.detach().cpu().numpy().reshape(-1)     # shape (action_dim,)

        # 如果 action_dim == 1，保持返回形状为 (1,) 而不是 python int
        return actions_np, probs_np

    def update(self, transition_dict, adv_normed=False, clip_vf=False, clip_range=0.2):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        # actions 必须为 float，用于计算 log_prob
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).to(self.device)
        # # 统一 actions 形状为 (N, action_dim)
        # if actions.dim() == 1:
        #     actions = actions.unsqueeze(1)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        
        probs = self.actor(states)  # 获取动作概率
        log_probs = torch.log(probs) * actions + torch.log(1 - probs) * (1 - actions)
        log_probs = log_probs.sum(dim=1, keepdim=True)  # 对所有动作维度求和
        
        # 添加Actor NaN检查
        if torch.isnan(log_probs).any():
            raise ValueError("NaN in Actor outputs")
        # 添加Critic NaN检查
        critic_values = self.critic(states)
        if torch.isnan(critic_values).any():
            raise ValueError("NaN in Critic outputs")

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu(), dones.cpu()).to(self.device)
        
        # 优势归一化
        if adv_normed:
            adv_mean, adv_std = advantage.detach().mean(), advantage.detach().std(unbiased=False) 
            advantage = (advantage - adv_mean) / (adv_std + 1e-8)

        # 提前计算一次旧的 value 预测（用于 value clipping）
        v_pred_old = self.critic(states).detach()  # (N,1)

        old_probs = self.actor(states).detach()
        old_log_probs = torch.log(old_probs) * actions + torch.log(1 - old_probs) * (1 - actions)
        old_log_probs = old_log_probs.sum(dim=1, keepdim=True).detach()

        actor_grad_list = []
        actor_loss_list = []
        critic_grad_list = []
        post_clip_actor_grad = []
        post_clip_critic_grad = []
        critic_loss_list = []
        entropy_list = []
        ratio_list = []

        for _ in range(self.epochs):
            probs = self.actor(states)
            log_probs = torch.log(probs) * actions + torch.log(1 - probs) * (1 - actions)
            log_probs = log_probs.sum(dim=1, keepdim=True)

            # 添加Actor NaN检查
            if torch.isnan(log_probs).any():
                raise ValueError("NaN in Actor outputs in loop")
            # 添加Critic NaN检查
            critic_values = self.critic(states)
            if torch.isnan(critic_values).any():
                raise ValueError("NaN in Critic outputs in loop")

            # 权重/偏置 NaN 检查（在每次前向后、反向前检查参数）
            check_weights_bias_nan(self.actor, "actor", "update循环中")
            check_weights_bias_nan(self.critic, "critic", "update循环中")

            ratio = torch.exp(log_probs - old_log_probs) # (N,1)
            surr1 = torch.clamp(ratio, -20, 20) * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage

            # 计算熵
            entropy = - probs * torch.log(probs) - (1 - probs) * torch.log(1 - probs)
            entropy_factor = entropy.sum(dim=1).mean()

            actor_loss = torch.mean(-torch.min(surr1, surr2)) - self.k_entropy * entropy_factor # 标量

            # 计算 critic_loss：支持可选的 value clipping（PPO 风格）
            if clip_vf:
                v_pred = self.critic(states)                                  # 当前预测 (N,1)
                v_pred_clipped = torch.clamp(v_pred, v_pred_old - clip_range, v_pred_old + clip_range)
                vf_loss1 = (v_pred - td_target.detach()).pow(2)               # (N,1)
                vf_loss2 = (v_pred_clipped - td_target.detach()).pow(2)       # (N,1)
                critic_loss = torch.max(vf_loss1, vf_loss2).mean()
            else:
                # critic_loss = F.mse_loss(self.critic(states), td_target.detach()) # 原有
                critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
            
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            
            # 裁剪前梯度
            post_clip_actor_grad.append(model_grad_norm(self.actor))
            post_clip_critic_grad.append(model_grad_norm(self.critic))  

            # 梯度裁剪
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.actor_max_grad)
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.critic_max_grad)

            self.actor_optimizer.step()
            self.critic_optimizer.step()

            # # 保存用于日志/展示的数值（断开计算图并搬到 CPU）
            actor_grad_list.append(model_grad_norm(self.actor))
            actor_loss_list.append(actor_loss.detach().cpu().item())
            critic_grad_list.append(model_grad_norm(self.critic))            
            critic_loss_list.append(critic_loss.detach().cpu().item())
            entropy_list.append(entropy_factor.detach().cpu().item())
            ratio_list.append(ratio.mean().detach().cpu().item())
        
        self.actor_loss = np.mean(actor_loss_list)
        self.actor_grad = np.mean(actor_grad_list)
        self.critic_loss = np.mean(critic_loss_list)
        self.critic_grad = np.mean(critic_grad_list)
        self.entropy_mean = np.mean(entropy_list)
        self.ratio_mean = np.mean(ratio_list)
        self.post_clip_critic_grad = np.mean(post_clip_critic_grad)
        self.post_clip_actor_grad = np.mean(post_clip_actor_grad)
        self.advantage = advantage.abs().mean().detach().cpu().item()
        # 权重/偏置 NaN 检查（在每次前向后、反向前检查参数）
        check_weights_bias_nan(self.actor, "actor", "update后")
        check_weights_bias_nan(self.critic, "critic", "update后")


if __name__ == '__main__':
    # --- 环境测试代码 ---
    # 使用随机动作策略来运行环境并观察效果
    env = ArcherEnv(render_mode="other")
    state_dim = env.observation_space.shape[0]  # 应为4
    action_dim = 1  # 应为1
    agent = PPO_bernouli(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
                     epochs, eps, gamma, device)
    
    return_list = []
    transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}    
    for ep in range(num_episodes):
        episode_return = 0
        
        obs, info = env.reset()
        terminated = False
        done = terminated
        total_reward = 0
        step_count = 0
        
        print(f"\n--- Episode {ep + 1} ---")
        
        while not terminated:
            obs = env.get_obs()
            # 随机策略：有15%的几率射击
            action, _ = agent.take_action(obs, explore=1)
            next_obs, reward, terminated, truncated, info = env.step(action)
            transition_dict['states'].append(obs)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_obs)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(terminated)
            total_reward += reward
            step_count += 1
            
            # # 打印每一步的信息
            # print(
            #     f"Step {step_count}: Action={action}, "
            #     f"Dist={info['distance']:.1f}, "
            #     f"Reward={reward:.2f}, "
            #     f"Total Reward={total_reward:.2f}"
            # )
            episode_return += reward
        return_list.append(episode_return)

        if 1: # ep % 1 == 0:
            agent.update(transition_dict, adv_normed=0)
            transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}    

        print(f"Episode finished after {step_count} steps. Final reward: {total_reward:.2f}")
        print(f"Final state: Hits={info['hits']}, Arrows Left={info['arrows_left']}")

    env.close()

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

    env = ArcherEnv(render_mode="human")
    obs, info = env.reset()
    terminated = False
    steps = 0
    total_reward = 0
    while not terminated:
        steps += 1
        obs = env.get_obs()
        action, _ = agent.take_action(obs, explore=0)
        next_obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(
                f"Step {steps}: Action={action}, "
                f"Dist={info['distance']:.1f}, "
                f"Reward={reward:.2f}"
            )
        time.sleep(0.5)
    env.close()
    print(f"Episode finished after {steps} steps. Final reward: {total_reward:.2f}")
    print(f"Final state: Hits={info['hits']}, Arrows Left={info['arrows_left']}")

