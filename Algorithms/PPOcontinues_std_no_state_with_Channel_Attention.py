'''
注意：take action 有两个输出，即actor原始输出u和tanh和scale后的action_exec
有监督预训练时经验池存 action_exec, 强化学习训练时经验池存 u
'''

from torch.distributions import Normal
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import os, sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from Algorithms.SharedLayers import ChannelAttention
from Algorithms.Utils import SquashedNormal, moving_average, check_weights_bias_nan, model_grad_norm, compute_advantage
from Algorithms.MLP_heads import PolicyNetContinuous, ValueNet

class PolicyNetContinuousCA(torch.nn.Module):
    """输出未压缩（pre-squash）的 mu，std 为与状态无关的可训练参数（每个动作维度一小段）。
    使用组合：ChannelAttention 作为前处理，PolicyNetContinuous 作为 head（复用已有实现）。
    """
    def __init__(self, state_dim, reduction_ratio, hidden_dim, action_dim, init_std=0.5):
        super(PolicyNetContinuousCA, self).__init__()
        # 保留 ChannelAttention 前处理模块
        self.channel_attention = ChannelAttention(state_dim, reduction_ratio)
        # 复用已有的 PolicyNetContinuous 作为 head（负责 net, fc_mu, log_std_param 等）
        self.head_net = PolicyNetContinuous(state_dim=state_dim,
                                            hidden_dim=hidden_dim,
                                            action_dim=action_dim,
                                            init_std=init_std)

    def forward(self, x, min_std=1e-6, max_std=0.4):
        # ChannelAttention 先处理输入特征
        x, _ = self.channel_attention(x)
        # head_net 返回 mu, std（PolicyNetContinuous 负责 clamp / broadcast）
        mu, std = self.head_net(x, min_std=min_std, max_std=max_std)
        return mu, std


class ValueNetCA(torch.nn.Module):
    """ValueNet with ChannelAttention 前处理 + ValueNet head（组合形式）。"""
    def __init__(self, state_dim, reduction_ratio, hidden_dim):
        super(ValueNetCA, self).__init__()
        # ChannelAttention 前处理
        self.channel_attention = ChannelAttention(state_dim, reduction_ratio)
        # 复用已有的 ValueNet 作为 head
        self.head_net = ValueNet(state_dim=state_dim, hidden_dim=hidden_dim)

    def forward(self, x):
        x, _ = self.channel_attention(x)
        return self.head_net(x)


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

    def __init__(self, state_dim, reduction_ratio, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device, k_entropy=0.01, critic_max_grad=2, actor_max_grad=2, max_std=0.3):
        self.actor = PolicyNetContinuousCA(state_dim, reduction_ratio, hidden_dim, action_dim).to(device)
        self.critic = ValueNetCA(state_dim, reduction_ratio, hidden_dim).to(device)
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
        self.max_std = max_std

        # [新增] MARWIL 专用：优势函数归一化因子的平方 (c^2)
        # 初始化为 1.0，用于动态追踪 (R_t - V)^2 的移动平均值
        self.c_sq = torch.tensor(1.0, dtype=torch.float).to(device)
    
    def set_learning_rate(self, actor_lr=None, critic_lr=None):
        """动态设置 actor 和 critic 的学习率"""
        if actor_lr is not None:
            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] = actor_lr
        if critic_lr is not None:
            for param_group in self.critic_optimizer.param_groups:
                param_group['lr'] = critic_lr    

    def _scale_action_to_exec(self, a, action_bounds):
        """把 normalized action a (in [-1,1]) 缩放到环境区间。

        action_bounds: 形状为 (action_dim, 2) 的二维 NumPy 数组，
                       每行对应 amin 和 amax。
        """
        action_bounds = torch.as_tensor(action_bounds, dtype=a.dtype, device=a.device)
        if action_bounds.dim() == 2:
            # 处理二维张量 (action_dim, 2)
            amin = action_bounds[:, 0]
            amax = action_bounds[:, 1]
        elif action_bounds.dim() == 3:
            # 处理三维张量 (batch, action_dim, 2)
            amin = action_bounds[:, :, 0]
            amax = action_bounds[:, :, 1]
        else:
            raise ValueError("action_bounds 的维度必须是 2 或 3")
        
        # a in (-1,1) -> scale to [amin, amax]
        return amin + (a + 1.0) * 0.5 * (amax - amin)

    def _unscale_exec_to_normalized(self, a_exec, action_bounds):
        """把执行动作 a_exec 反向归一化到 [-1,1]。

        action_bounds: 形状为 (action_dim, 2) 的二维 NumPy 数组，
                       每行对应 amin 和 amax。
        """
        action_bounds = torch.as_tensor(action_bounds, dtype=a_exec.dtype, device=a_exec.device)
        if action_bounds.dim() == 2:
            # 处理二维张量 (action_dim, 2)
            amin = action_bounds[:, 0]
            amax = action_bounds[:, 1]
        elif action_bounds.dim() == 3:
            # 处理三维张量 (batch, action_dim, 2)
            amin = action_bounds[:, :, 0]
            amax = action_bounds[:, :, 1]
        else:
            raise ValueError("action_bounds 的维度必须是 2 或 3")
        
        # 防止除以零
        span = (amax - amin)
        span = torch.where(span == 0, torch.tensor(1e-6, device=span.device, dtype=span.dtype), span)
        a = 2.0 * (a_exec - amin) / span - 1.0
        # numerical stability
        return a.clamp(-0.999999, 0.999999)
    
    # 对外接口（保证numpy输入和numpy输出）
    def unscale_exec_to_normalized(self, a_exec, action_bounds):
        """Public wrapper: accepts numpy or torch, returns numpy on CPU."""
        a_exec_t = torch.as_tensor(a_exec, dtype=torch.float, device=self.device)
        action_bounds_t = torch.as_tensor(action_bounds, dtype=torch.float, device=self.device)
        a_norm_t = self._unscale_exec_to_normalized(a_exec_t, action_bounds_t)
        return a_norm_t.cpu().numpy()
    
    # take action
    def take_action(self, state, action_bounds, explore=True, max_std=None):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        # 检查state中是否存在nan
        if torch.isnan(state).any() or torch.isinf(state).any():
            print('state', state)
        # 检查actor参数中是否存在nan
        check_weights_bias_nan(self.actor, "actor", "take action中")
        if max_std is None:
            max_action_std = self.max_std
        else:
            max_action_std = max_std
        mu, std = self.actor(state, min_std=1e-6, max_std=max_action_std)
        # 检查mu, std是否含有nan
        if torch.isnan(mu).any() or torch.isnan(std).any() or torch.isinf(mu).any() or torch.isinf(std).any():
            print('mu', mu)
            print('std', std)
            raise ValueError(
                f"NaN/Inf detected in actor outputs: mu_nan={torch.isnan(mu).any().item()}, "
                f"std_nan={torch.isnan(std).any().item()}, mu_inf={torch.isinf(mu).any().item()}, "
                f"std_inf={torch.isinf(std).any().item()}"
            ) 

        dist = SquashedNormal(mu, std)
        if explore:
            a_norm, u = dist.sample()
        else:
            # use mean action: tanh(mu)
            u = mu
            a_norm = torch.tanh(u)

        a_exec = self._scale_action_to_exec(a_norm, action_bounds)
        return a_exec[0].cpu().detach().numpy().flatten(), u[0].cpu().detach().numpy().flatten()
    

    def update(self, transition_dict, adv_normed=False, clip_vf=False, clip_range=0.2):
        """更新函数兼容以下几种调用方式：
        - 如果 action_bounds 是 None: 期望 transition_dict 中包含 'action_bounds'，其形状为 (N,2) 或每步 (amin,amax)
        - 如果 action_bounds 是标量/二元元组/数组：作为全局固定区间使用

        transition_dict 必须包含 keys: 'states','actions','rewards','next_states','dones'
        当动作区间随步变化时，必须包含 'action_bounds' 与之对应。
        存储的 'actions' 应当是环境执行动作 (a_exec 未归一化）。
        """
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        u_s = torch.tensor(np.array(transition_dict['actions']), dtype=torch.float).to(self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']), dtype=torch.float).view(-1, 1).to(self.device)
        action_bounds = torch.tensor(np.array(transition_dict['action_bounds']), dtype=torch.float).to(self.device)
        if 'max_stds' in transition_dict:
            max_stds = torch.tensor(np.array(transition_dict['max_stds']), dtype=torch.float).view(-1, 1).to(self.device)
        else:
            max_stds = self.max_std
        # 计算 td_target, advantage
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu(), dones.cpu()).to(self.device)
        
        # 优势归一化
        if adv_normed:
            adv_mean, adv_std = advantage.detach().mean(), advantage.detach().std(unbiased=False) 
            advantage = (advantage - adv_mean) / (adv_std + 1e-8)
            # advantage = torch.clamp((advantage - adv_mean) / (adv_std + 1e-8) -10.0, 10.0)

        # 提前计算一次旧的 value 预测（用于 value clipping）
        v_pred_old = self.critic(states).detach()  # (N,1)

        # 策略输出（未压缩的 mu,std）
        mu, std = self.actor(states, min_std=1e-6, max_std=max_stds) # self.max_std
        # 构造 SquashedNormal 并计算 old_log_probs
        dist = SquashedNormal(mu.detach(), std.detach())

        # # 将执行动作反向归一化到 [-1,1]，以便计算 log_prob
        # actions_normalized = self._unscale_exec_to_normalized(actions_exec, action_bounds)
        
        # # 反算 u = atanh(a)
        u_old = u_s
        # old_log_probs = dist.log_prob(0, u_old) # (N,1)

        # 提前在action_dim维度求和
        old_log_probs = dist.log_prob(0, u_old).sum(-1, keepdim=True)    # -> (N,1)

        if torch.isnan(old_log_probs).any():
            raise ValueError("old_log_probs 包含 NaN，检查 action_bounds 或 actions 的合法性")

        actor_grad_list = []
        actor_loss_list = []
        critic_grad_list = []
        pre_clip_actor_grad = []
        pre_clip_critic_grad = []
        critic_loss_list = []
        entropy_list = []
        ratio_list = []

        for _ in range(self.epochs):
            mu, std = self.actor(states, min_std=1e-6, max_std=self.max_std)
            if torch.isnan(mu).any() or torch.isnan(std).any():
                raise ValueError("NaN in Actor outputs in loop")
            critic_values = self.critic(states)
            if torch.isnan(critic_values).any():
                raise ValueError("NaN in Critic outputs in loop")

            # 权重/偏置 NaN 检查（在每次前向后、反向前检查参数）
            check_weights_bias_nan(self.actor, "actor", "update循环中")
            check_weights_bias_nan(self.critic, "critic", "update循环中")

            dist = SquashedNormal(mu, std)
            # 计算当前策略对历史执行动作的 log_prob（使用同一个 u_old）
            # log_probs = dist.log_prob(0, u_old) # (N,1)

            # 提前在action_dim维度求和
            log_probs = dist.log_prob(0, u_old).sum(-1, keepdim=True)   # -> (N,1)

            ratio = torch.exp(log_probs - old_log_probs) # (N,1)
            # surr1 = ratio * advantage
            # clamp surr1
            surr1 = torch.clamp(ratio, -20, 20) * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            # 可选：对surr1用一个很大的范围去clamp防止出现一个很负的数
            entropy_factor = dist.entropy().mean()  # torch.clamp(dist.entropy().mean(), -20, 70) # -20, 7 e^2
            actor_loss_reward_term = -torch.min(surr1, surr2).sum(-1).mean()
            actor_loss = actor_loss_reward_term - self.k_entropy * entropy_factor

            # ↑如果求和之和还要保留原先的张量维度，用torch.sum(torch.min(surr1,surr2),dim=-1,keepdim=True)

            # 计算 critic_loss：支持可选的 value clipping（PPO 风格）
            if clip_vf:
                v_pred = self.critic(states)                                  # 当前预测 (N,1)
                v_pred_clipped = torch.clamp(v_pred, v_pred_old - clip_range, v_pred_old + clip_range)
                vf_loss1 = (v_pred - td_target.detach()).pow(2)               # (N,1)
                vf_loss2 = (v_pred_clipped - td_target.detach()).pow(2)       # (N,1)
                critic_loss = torch.max(vf_loss1, vf_loss2).mean()
            else:
                critic_loss = F.mse_loss(self.critic(states), td_target.detach())

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            
            # 裁剪前梯度
            pre_clip_actor_grad.append(model_grad_norm(self.actor))
            pre_clip_critic_grad.append(model_grad_norm(self.critic))  

            # 梯度裁剪
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.actor_max_grad)
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.critic_max_grad)

            # self.actor_optimizer.step()
            # self.critic_optimizer.step()

            # # 保存用于日志/展示的数值（断开计算图并搬到 CPU）
            actor_grad_list.append(model_grad_norm(self.actor))
            actor_loss_list.append(actor_loss.detach().cpu().item())
            critic_grad_list.append(model_grad_norm(self.critic))            
            critic_loss_list.append(critic_loss.detach().cpu().item())
            entropy_list.append(dist.entropy().mean().detach().cpu().item())
            ratio_list.append(ratio.mean().detach().cpu().item())

            self.actor_optimizer.step()
            self.critic_optimizer.step()
        
        self.actor_loss = np.mean(actor_loss_list)
        self.actor_grad = np.mean(actor_grad_list)
        self.critic_loss = np.mean(critic_loss_list)
        self.critic_grad = np.mean(critic_grad_list)
        self.entropy_mean = np.mean(entropy_list)
        self.ratio_mean = np.mean(ratio_list)
        self.pre_clip_critic_grad = np.mean(pre_clip_critic_grad)
        self.pre_clip_actor_grad = np.mean(pre_clip_actor_grad)
        self.advantage = advantage.abs().mean().detach().cpu().item()
        # 权重/偏置 NaN 检查（在每次前向后、反向前检查参数）
        check_weights_bias_nan(self.actor, "actor", "update后")
        check_weights_bias_nan(self.critic, "critic", "update后")



    def MARWIL_update(self, il_transition_dict, beta=1.0, batch_size=64, alpha=1.0, c_v=1.0, shuffled=1, max_weight=100):
        """
        MARWIL 离线更新函数 (连续动作空间版 - 修正版)
        逻辑：直接使用存储的 u (pre-tanh) 计算 log_prob，无需反向归一化。
        
        参数:
            il_transition_dict (dict): 必须包含:
                - 'states': 状态
                - 'actions': 未经 tanh 和缩放的原始输出 u (pre-tanh actions)
                - 'returns': 预先计算好的蒙特卡洛回报 (R_t)
                - 'max_stds' (可选): 专家数据的 max_std
                # 注意：此时不再强制需要 'action_bounds'，除非用于其他用途，因为计算 log_prob(u) 不需要 bounds
            beta (float): 优势指数系数 建议的起始值通常是 介于 0.25 到 1.0 之间
            batch_size (int): Mini-batch 大小
            alpha (float): 模仿学习权重
            c_v (float): 价值损失权重
            shuffled (bool): 是否打乱数据
            
        返回:
            avg_actor_loss, avg_critic_loss, current_c
        """
        # 1. 提取全量数据并转为 Tensor
        states_all = torch.tensor(np.array(il_transition_dict['states']), dtype=torch.float).to(self.device)
        # 修正：这里的 actions 直接就是 u
        u_all = torch.tensor(np.array(il_transition_dict['actions']), dtype=torch.float).to(self.device)
        returns_all = torch.tensor(np.array(il_transition_dict['returns']), dtype=torch.float).view(-1, 1).to(self.device)
        
        # 2. 准备 Batch 索引
        total_size = states_all.size(0)
        indices = np.arange(total_size)
        if shuffled:
            np.random.shuffle(indices)

        total_actor_loss = 0
        total_critic_loss = 0
        batch_count = 0

        self.actor.set_fixed_std() # 监督学习期间锁定标准差

        # 3. Mini-batch 循环
        for start in range(0, total_size, batch_size):
            end = min(start + batch_size, total_size)
            batch_indices = indices[start:end]
            
            s_batch = states_all[batch_indices]
            u_batch = u_all[batch_indices] # 直接取出 u
            r_batch = returns_all[batch_indices]

            # ----------------------------------------------------
            # A. 计算优势 (Advantage) 和 权重 (Weights)
            # ----------------------------------------------------
            with torch.no_grad():
                values = self.critic(s_batch)
                residual = r_batch - values
                
                # 动态更新 c^2
                batch_mse = (residual ** 2).mean().item()
                self.c_sq = self.c_sq + 1e-8 * (batch_mse - self.c_sq)
                c = torch.sqrt(self.c_sq)
                
                # 归一化优势
                advantage = residual / (c + 1e-8)
                
                # 计算指数权重
                # weights = torch.exp(beta * advantage)
                # 修改为:
                # 1. 计算原始权重
                raw_weights = torch.exp(beta * advantage)
                # 2. 截断权重，例如最大不超过 100.0 (e^4.6)
                weights = torch.clamp(raw_weights, max=max_weight)

            # ----------------------------------------------------
            # B. 计算 Actor Loss (模仿学习部分)
            # ----------------------------------------------------
            # 1. 获取当前策略分布参数 mu, std
            mu, std = self.actor(s_batch, min_std=1e-6)
            dist = SquashedNormal(mu, std)
            
            # 2. 计算 Log Probability
            # 直接使用存储的 u 计算 log_prob(u)
            # 注意：PPO update 中通常也是 dist.log_prob(0, u_old)
            log_probs = dist.log_prob(0, u_batch).sum(-1, keepdim=True) # (Batch, 1)
            
            # 3. 计算 Loss
            actor_loss = -torch.mean(alpha * weights.detach() * log_probs)

            # ----------------------------------------------------
            # C. 计算 Critic Loss
            # ----------------------------------------------------
            v_pred = self.critic(s_batch)
            critic_loss = F.mse_loss(v_pred, r_batch) * c_v

            # ----------------------------------------------------
            # D. 反向传播
            # ----------------------------------------------------
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            
            actor_loss.backward()
            critic_loss.backward()
            
            # 梯度裁剪
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.actor_max_grad)
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.critic_max_grad)
            
            self.actor_optimizer.step()
            self.critic_optimizer.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            batch_count += 1

        # 返回平均 Loss
        avg_actor_loss = total_actor_loss / batch_count
        avg_critic_loss = total_critic_loss / batch_count
        
        return avg_actor_loss, avg_critic_loss, c.item()


# 仅调用actor读取输出
def _scale_action_to_exec_standalone(a, action_bounds, device):
    """(独立函数) 把 normalized action a (in [-1,1]) 缩放到环境区间。"""
    action_bounds = torch.as_tensor(action_bounds, dtype=a.dtype, device=device)
    if action_bounds.dim() == 2:
        amin = action_bounds[:, 0]
        amax = action_bounds[:, 1]
    elif action_bounds.dim() == 3:
        amin = action_bounds[:, :, 0]
        amax = action_bounds[:, :, 1]
    else:
        raise ValueError("action_bounds 的维度必须是 2 或 3")
    return amin + (a + 1.0) * 0.5 * (amax - amin)

def take_action_from_policy(policy_net, state, action_bounds, device, max_std=0.3):
    """独立于PPO类的推理函数，仅使用策略网络。"""
    state = torch.tensor(np.array([state]), dtype=torch.float).to(device)
    # 确保网络处于评估模式
    policy_net.eval()
    with torch.no_grad():
        mu, _ = policy_net(state, min_std=1e-6, max_std=max_std)
        # 确定性动作：使用 mu 的 tanh 作为标准化动作
        a_norm = torch.tanh(mu)
        # 缩放到实际动作区间
        a_exec = _scale_action_to_exec_standalone(a_norm, action_bounds, device)
    return a_exec[0].cpu().numpy().flatten()

