'''
相对改动：
- Actor和Critic现在各自使用独立的GRU backbone实例。
- 优化器相应地调整为分别管理整个Actor网络和整个Critic网络。
- 隐藏状态管理（hidden_state, get_h0等）被分裂为actor和critic两套独立的方法。
'''

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import os, sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from Algorithms.SharedLayers import GruMlp
from Algorithms.Utils import model_grad_norm, moving_average, check_weights_bias_nan, compute_advantage, SquashedNormal
from Algorithms.SharedLayers import SharedGruBackbone
from Algorithms.MLP_heads import PolicyNetContinuous, ValueNet

class PolicyNetContinuousGRU(nn.Module):
    """Actor: 组合形式 — backbone (GRU) + PolicyNetContinuous 作为 head。"""
    def __init__(self, state_dim, gru_hidden_size, gru_num_layers=1, middle_dim=35,
                 head_hidden_dims=[128], action_dim=3, init_std=0.5, batch_first=True):
        super(PolicyNetContinuousGRU, self).__init__()
        # 独立 backbone，输出特征维度为 middle_dim（gru/MLP 输出）
        self.backbone = SharedGruBackbone(state_dim, gru_hidden_size, gru_num_layers,
                                          output_dim=middle_dim, batch_first=batch_first)
        # 复用已有的 PolicyNetContinuous 作为 head（输入维度 middle_dim）
        self.head_net = PolicyNetContinuous(state_dim=middle_dim,
                                            hidden_dim=head_hidden_dims,
                                            action_dim=action_dim,
                                            init_std=init_std)

    def forward(self, x, h_0=None, min_std=1e-6, max_std=0.4):
        # 支持 (B, D) -> treat as (B, 1, D) 或 (B, S, D) 输入
        if x.dim() == 2:
            x = x.unsqueeze(1)
        # backbone 返回每步特征与最后隐藏态
        features_per_step, h_n = self.backbone(x, h_0)
        # 使用序列最后一步特征作为汇聚表示
        features = features_per_step[:, -1, :]
        # head 输出 mu, std（PolicyNetContinuous 自行做 batch 广播）
        mu, std = self.head_net(features, min_std=min_std, max_std=max_std)
        return mu, std, h_n


class ValueNetGRU(nn.Module):
    """Critic: backbone (GRU) + ValueNet head (组合形式)。"""
    def __init__(self, state_dim, gru_hidden_size, gru_num_layers, middle_dim, head_hidden_dims, batch_first=True):
        super(ValueNetGRU, self).__init__()
        # 独立 backbone（与 Actor 可各自持有或外部传入共享实例）
        self.backbone = SharedGruBackbone(state_dim, gru_hidden_size, gru_num_layers,
                                          output_dim=middle_dim, batch_first=batch_first)
        # 使用已有的 ValueNet 作为 head（输入维度 middle_dim）
        self.head_net = ValueNet(state_dim=middle_dim, hidden_dim=head_hidden_dims)

    def forward(self, x, h_0=None):
        # 支持 (B, D) -> (B,1,D) 或 (B, S, D)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        features_per_step, h_n = self.backbone(x, h_0)
        features = features_per_step[:, -1, :]
        value = self.head_net(features)
        return value, h_n


class PPOContinuous:
    def __init__(self, state_dim, gru_hidden_size, gru_num_layers, middle_dim,
                 head_hidden_dims, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device,
                 k_entropy=0.01, critic_max_grad=2, actor_max_grad=2, max_std=0.3, batch_first=True):

        # 1. 创建 Actor, 它内部包含自己独立的 backbone
        self.actor = PolicyNetContinuousGRU(state_dim, gru_hidden_size, gru_num_layers, middle_dim,
                                            head_hidden_dims, action_dim, batch_first=batch_first).to(device)



        # 2. 创建 Critic, 它现在也创建并拥有自己独立的 backbone
        self.critic = ValueNetGRU(state_dim, gru_hidden_size, gru_num_layers, middle_dim,
                                  head_hidden_dims, batch_first=batch_first).to(device)



        # 3. 设置优化器
        # Actor 优化器负责更新整个 Actor 网络 (backbone + head)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        # Critic 优化器负责更新整个 Critic 网络 (backbone + head)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)


        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device
        self.k_entropy = k_entropy
        self.critic_max_grad = critic_max_grad
        self.actor_max_grad = actor_max_grad
        self.max_std = max_std

        # 添加隐藏状态管理 (分裂为 actor 和 critic)
        self.gru_num_layers = gru_num_layers
        self.gru_hidden_size = gru_hidden_size
        self.hidden_state_a = None
        self.hidden_state_c = None

    def reset_hidden_state_a(self):
        """重置Actor的隐藏状态"""
        self.hidden_state_a = self.get_h0_a(batch_size=1)
        
    def reset_hidden_state_c(self):
        """重置Critic的隐藏状态"""
        self.hidden_state_c = self.get_h0_c(batch_size=1)

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
    def take_action(self, state, h_0_a=None, action_bounds=None, explore=True, max_std=None):
        # === 修改：处理序列输入 ===
        # state 现在的形状是 (SeqLen, StateDim), e.g., (10, 35)
        # 需要增加一个 batch 维度 -> (1, 10, 35)
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        # 如果外部传入了以 CPU 存储的 hidden（或来自 get_current_hidden_state_*），
        # 在送进模型前需要移动到 device
        if h_0_a is not None:
            h_0_a = self._maybe_move_h0_to_device(h_0_a)
        # 检查state中是否存在nan
        if torch.isnan(state).any() or torch.isinf(state).any():
            print('state', state)
        # 检查actor参数中是否存在nan
        check_weights_bias_nan(self.actor, "actor", "take action中")
        if max_std is None:
            max_action_std = self.max_std
        else:
            max_action_std = max_std

        mu, std, h_n_a = self.actor(state, h_0_a, min_std=1e-6, max_std=max_action_std)

        # 更新Actor的隐藏状态
        # 输出的 hidden 应保存在 CPU 上，便于外部存储/传递；但后续再次作为输入前需调用 _maybe_move_h0_to_device
        self.hidden_state_a = h_n_a.detach().cpu()

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
        # 返回动作、未压缩动作 u 和Actor的隐藏状态
        return a_exec[0].cpu().detach().numpy().flatten(), u[0].cpu().detach().numpy().flatten(), self.hidden_state_a

    def get_value(self, state, h_0_c):
        # 仅在采集经验的时候使用，外部传入为一维向量 (StateDim,)
        # 构造形状 (1,1,StateDim) 并传入 critic
        state_input = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        # 确保 h_0_c 在同一 device（如果有）
        if h_0_c is not None:
            h_0_c = self._maybe_move_h0_to_device(h_0_c)

        value, h_n_c = self.critic(state_input, h_0_c)
        # 返回：value 为 numpy (detached on CPU)，h_n_c 为 detached CPU tensor
        return value.squeeze(0).detach().cpu().numpy(), h_n_c.detach().cpu()
    
    
    def get_h0_a(self, batch_size=1):
        """
        获取 Actor GRU 的初始隐藏状态 h0。
        输出：一个 detach 并转移到 CPU 的 tensor。
        """
        h0 = torch.zeros(self.gru_num_layers, batch_size, self.gru_hidden_size, dtype=torch.float32, device=self.device)
        return h0.detach().cpu()

    def get_h0_c(self, batch_size=1):
        """
        获取 Critic GRU 的初始隐藏状态 h0。
        输出：一个 detach 并转移到 CPU 的 tensor。
        """
        h0 = torch.zeros(self.gru_num_layers, batch_size, self.gru_hidden_size, dtype=torch.float32, device=self.device)
        return h0.detach().cpu()

    def get_current_hidden_state_a(self):
        """
        获取 Actor 当前隐藏状态。
        """
        return self.hidden_state_a

    def get_current_hidden_state_c(self):
        """
        获取 Critic 当前隐藏状态。
        """
        return self.hidden_state_c

    def _maybe_move_h0_to_device(self, h0):
        """
        如果 h0 是 Tensor，则把它移动到 self.device（in-place 不做，返回新 tensor）。
        如果 h0 为 None 或非 Tensor，直接返回原值。
        目的：外部/存储的 hidden 通常以 CPU tensor 存储，使用前要移动到模型所在 device。
        """
        if h0 is None:
            return None
        if torch.is_tensor(h0):
            # 保证是同一 device（避免出现 input 在 cuda:0 而 hidden 在 cpu 的错误）
            if h0.device != self.device:
                return h0.to(self.device)
            return h0
        # 如果是 numpy 等，先转为 tensor 再移动
        try:
            return torch.as_tensor(h0, device=self.device)
        except Exception:
            return h0

    def update(self, transition_dict, adv_normed=False, clip_vf=False, clip_range=0.2, shuffled=0):
        # states 张量现在的形状是 (Batch, SeqLen, StateDim)
        # 在 update 方法开头添加 shuffle 功能

        # 首先按原始顺序把数据载入（不要一开始就打乱）
        N = len(transition_dict['states'])
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        u_s = torch.tensor(np.array(transition_dict['actions']), dtype=torch.float).to(self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float).view(-1, 1).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']), dtype=torch.float).view(-1, 1).to(self.device)
        # h0s 以样本为第一维堆叠，后面按需转换到 (num_layers, B, hidden_size)
        # 分别读取 actor 和 critic 的隐藏状态
        h0as_stack = torch.stack([transition_dict['h0as'][i] for i in range(N)]).to(self.device)  # (N, num_layers, 1, hidden)
        h0cs_stack = torch.stack([transition_dict['h0cs'][i] for i in range(N)]).to(self.device)
        # next_values 也按原顺序读取
        next_values = torch.tensor(np.array(transition_dict['next_values']), dtype=torch.float).to(self.device)

        # 为 critic 准备 h0s 的形状 (num_layers, B, hidden_size)
        # 分别修正 actor 和 critic 隐藏状态的形状
        h0as_for_actor = h0as_stack.squeeze(2).permute(1, 0, 2).contiguous()
        h0cs_for_critic = h0cs_stack.squeeze(2).permute(1, 0, 2).contiguous()

        if 'max_stds' in transition_dict:
            max_stds = torch.tensor(np.array(transition_dict['max_stds']), dtype=torch.float).view(-1, 1).to(self.device)
        else:
            max_stds = self.max_std

        # 计算 td_target, advantage（在 shuffle 之前保持原始顺序）
        td_target = rewards + self.gamma * next_values * (1 - dones)
        td_delta = td_target - self.critic(states, h0cs_for_critic)[0]
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu(), dones.cpu()).to(self.device)

        # 现在如果要求打乱，则统一对所有输入（包括 advantage）进行相同的 shuffle
        indices = np.arange(N)
        if shuffled:
            np.random.shuffle(indices)
            states = states[indices]
            u_s = u_s[indices]
            rewards = rewards[indices]
            dones = dones[indices]
            next_values = next_values[indices]
            advantage = advantage[indices]
            max_stds = max_stds[indices] if isinstance(max_stds, torch.Tensor) and max_stds.shape[0] == N else max_stds
            # 重新整理 h0s
            h0as_stack = h0as_stack[indices]
            h0cs_stack = h0cs_stack[indices]
            h0as = h0as_stack.squeeze(2).permute(1, 0, 2).contiguous()
            h0cs = h0cs_stack.squeeze(2).permute(1, 0, 2).contiguous()
        else:
            # 未打乱时直接使用之前为 critic 准备好的 h0s_for_critic
            h0as = h0as
            h0cs = h0cs
        # 优势归一化
        if adv_normed:
            adv_mean, adv_std = advantage.detach().mean(), advantage.detach().std(unbiased=False)
            advantage = (advantage - adv_mean) / (adv_std + 1e-8)
            # advantage = torch.clamp((advantage - adv_mean) / (adv_std + 1e-8) -10.0, 10.0)

        # 提前计算一次旧的 value 预测（用于 value clipping）
        v_pred_old = self.critic(states, h0cs)[0].detach()  # (N,1)

        # 策略输出（未压缩的 mu,std）
        mu, std, _ = self.actor(states, h0as, min_std=1e-6, max_std=max_stds) # self.max_std
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
            mu, std, _ = self.actor(states, h0as, min_std=1e-6, max_std=self.max_std)
            if torch.isnan(mu).any() or torch.isnan(std).any():
                raise ValueError("NaN in Actor outputs in loop")
            critic_values, _ = self.critic(states, h0cs)
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
                v_pred, _ = self.critic(states, h0cs)                                  # 当前预测 (N,1)
                v_pred_clipped = torch.clamp(v_pred, v_pred_old - clip_range, v_pred_old + clip_range)
                vf_loss1 = (v_pred - td_target.detach()).pow(2)               # (N,1)
                vf_loss2 = (v_pred_clipped - td_target.detach()).pow(2)       # (N,1)
                critic_loss = torch.max(vf_loss1, vf_loss2).mean()
            else:
                current_values, _ = self.critic(states, h0cs)
                critic_loss = F.mse_loss(current_values, td_target.detach())

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