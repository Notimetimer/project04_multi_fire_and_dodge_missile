from torch.distributions import Normal
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import os, sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from Algorithms.Utils import model_grad_norm, moving_average, check_weights_bias_nan, compute_advantage, SquashedNormal
from Algorithms.MLP_heads import ValueNet, PolicyNetDiscrete

class PPO_discrete:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, state_dim, hidden_dims, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device, k_entropy=0.01, actor_max_grad=2, critic_max_grad=2,
                 init_temperature=1.0, temp_decay=0.995):
        self.actor = PolicyNetDiscrete(state_dim, hidden_dims, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dims).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device
        self.k_entropy = k_entropy
        self.actor_max_grad=actor_max_grad
        self.critic_max_grad=critic_max_grad

        # [新增] MARWIL 专用：优势函数归一化因子的平方 (c^2)
        # 初始化为 1.0，用于动态追踪 (R_t - V)^2 的移动平均值
        self.c_sq = torch.tensor(1.0, dtype=torch.float).to(device)

        self.temperature = init_temperature  # 初始温度设为 2.0 或更高，视过拟合程度而定
        self.min_temperature = 1.0
        self.temp_decay = temp_decay # 每个 episode 或 update 后衰减
        self.action_dim = action_dim


    def set_learning_rate(self, actor_lr=None, critic_lr=None):
        """动态设置 actor 和 critic 的学习率"""
        if actor_lr is not None:
            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] = actor_lr
        if critic_lr is not None:
            for param_group in self.critic_optimizer.param_groups:
                param_group['lr'] = critic_lr    

    def take_action(self, state, explore=True):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        probs = self.actor(state, temperature=self.temperature)
        action_dist = torch.distributions.Categorical(probs) # 离散的输出为类别分布
        if explore:
            action = action_dist.sample()
        else:
            action = torch.argmax(probs)
        # 返回动作索引与对应的概率分布（numpy array）
        probs_np = probs.detach().cpu().numpy()[0].copy() # [0]是batch维度
        return probs_np, action.item()

    def update(self, transition_dict, adv_normed=False, clip_vf=False, clip_range=0.2):
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        # actions 必须为 long 用于 gather 索引
        actions = torch.tensor(np.array(transition_dict['actions']), dtype=torch.long).view(-1, 1).to(self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']), dtype=torch.float).view(-1, 1).to(self.device)
        
        log_probs = torch.log(self.actor(states, temperature=self.temperature).gather(1, actions))
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
            # advantage = torch.clamp((advantage - adv_mean) / (adv_std + 1e-8) -10.0, 10.0)
            
            # adv_mean, adv_std = advantage.mean(), advantage.std(unbiased=False) 
            advantage = (advantage - adv_mean) / (adv_std + 1e-8)

        # 提前计算一次旧的 value 预测（用于 value clipping）
        v_pred_old = self.critic(states).detach()  # (N,1)

        old_log_probs = torch.log(self.actor(states, temperature=self.temperature).gather(1, actions)).detach()

        actor_grad_list = []
        actor_loss_list = []
        critic_grad_list = []
        pre_clip_actor_grad = []
        pre_clip_critic_grad = []
        critic_loss_list = []
        entropy_list = []
        ratio_list = []

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states, temperature=self.temperature).gather(1, actions))
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

            log_probs = torch.log(self.actor(states, temperature=self.temperature).gather(1, actions)) # (N,1)
            ratio = torch.exp(log_probs - old_log_probs) # (N,1)
            surr1 = torch.clamp(ratio, -20, 20) * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage

            probs = self.actor(states, temperature=self.temperature)
            action_dist = torch.distributions.Categorical(probs)

            entropy_factor = action_dist.entropy().mean() # torch.clamp(dist.entropy().mean(), -20, 70) # -20, 7 e^2

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
            pre_clip_actor_grad.append(model_grad_norm(self.actor))
            pre_clip_critic_grad.append(model_grad_norm(self.critic))  

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
            entropy_list.append(action_dist.entropy().mean().detach().cpu().item())
            ratio_list.append(ratio.mean().detach().cpu().item())
        
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

        self.temperature = max(self.min_temperature, self.temperature * self.temp_decay)


def MARWIL_update(self, il_transition_dict, beta, batch_size=64, alpha=1.0, c_v=1.0, shuffled=1, max_weight=100, label_smoothing=0.1):
        """
        MARWIL 离线更新函数 (执行一个 Epoch)
        逻辑：接收全量专家数据 -> 打乱(可选) -> 按 batch_size 拆分 -> 逐个 mini-batch 更新
        
        参数:
            il_transition_dict (dict): 包含 'states', 'actions', 'returns' 的字典
            beta (float): MARWIL 优势指数系数 (exp(beta * A))
            batch_size (int): Mini-batch 的大小
            alpha (float): 模仿学习损失的权重 (用于平衡 Actor Loss)
            c_v (float): 价值网络损失的权重 (用于平衡 Critic Loss)
            shuffled (bool): 是否在分批前打乱数据
            
        返回:
            avg_actor_loss, avg_critic_loss, current_c
        
        [改进版] MARWIL 离线更新函数 (Discrete)
        新增参数:
            label_smoothing (float): 标签平滑系数 (0.0~1.0). 建议值 0.05 ~ 0.1.
                                     0.0 表示使用原始硬标签 (One-hot).
        """
        # 1. 提取全量数据并转为 Tensor (一次性转移到 Device)
        # 注意：这里需要 il_transition_dict 包含 'returns' (即计算好的 R_t)，而不是原始 rewards
        states_all = torch.tensor(np.array(il_transition_dict['states']), dtype=torch.float).to(self.device)
        actions_all = torch.tensor(np.array(il_transition_dict['actions']), dtype=torch.long).to(self.device) # 离散动作需为 long
        returns_all = torch.tensor(np.array(il_transition_dict['returns']), dtype=torch.float).view(-1, 1).to(self.device)
        
        # 确保 actions 形状适配 gather (N, 1)
        if actions_all.dim() == 1:
            actions_all = actions_all.view(-1, 1)

        # 2. 准备 Batch 索引
        total_size = states_all.size(0)
        indices = np.arange(total_size)
        if shuffled:
            np.random.shuffle(indices)

        # 用于记录本轮 Epoch 的平均 Loss
        total_actor_loss = 0
        total_critic_loss = 0
        batch_count = 0

        # 3. Mini-batch 循环
        for start in range(0, total_size, batch_size):
            end = min(start + batch_size, total_size)
            batch_indices = indices[start:end]
            
            # 切片获取 Mini-batch 数据
            # torch.tensor(batch_indices) 可能会带来额外的 CPU->GPU 开销，
            # 如果 indices 是 numpy，直接切片 tensor 也是可以的，但在 GPU tensor 上索引通常建议传入 tensor 或 list
            # 这里为了通用性，直接使用 numpy 索引
            s_batch = states_all[batch_indices]
            a_batch = actions_all[batch_indices]
            r_batch = returns_all[batch_indices]

            # ----------------------------------------------------
            # A. 计算优势 (Advantage) 和 权重 (Weights)
            # ----------------------------------------------------
            with torch.no_grad():
                values = self.critic(s_batch)
                # 计算残差: A_hat = R_t - V(s)
                residual = r_batch - values
                
                # --- 动态更新归一化因子 c ---
                # 论文脚注: c^2 <- c^2 + 10^-8 * (residual^2 - c^2)
                batch_mse = (residual ** 2).mean().item()
                self.c_sq = self.c_sq + 1e-8 * (batch_mse - self.c_sq)
                c = torch.sqrt(self.c_sq)
                
                # 归一化优势
                advantage = residual / (c + 1e-8)
                
                # 计算指数权重: w = exp(beta * A)
                # 1. 计算原始权重
                raw_weights = torch.exp(beta * advantage)
                # 2. 截断权重，例如最大不超过 100.0 (e^4.6)
                weights = torch.clamp(raw_weights, max=max_weight)

            # ----------------------------------------------------
            # B. 计算 Actor Loss (模仿学习部分)
            # ----------------------------------------------------
            # 获取当前策略对 batch 状态的概率分布
            all_probs = self.actor(s_batch) # (Batch, Action_Dim)
            
            # 【关键修正1】这里计算的是所有动作的 Log 概率，后续要一直用这个变量
            log_all_probs = torch.log(all_probs + 1e-10)

            # 新的拟合目标，人为加大熵防止过拟合
            if label_smoothing > 0:
                # 1. 构建 One-hot 目标
                # a_batch shape: (Batch, 1) -> scatter 需要 index
                one_hot = torch.zeros_like(all_probs).scatter_(1, a_batch, 1.0) # 填充一个全0向量，在下方构造概率目标
                
                # 2. 构建平滑目标: y_ls = (1 - epsilon) * y_hot + epsilon / K
                smooth_target = one_hot * (1.0 - label_smoothing) + (label_smoothing / self.action_dim)
                
                # 3. 计算 Soft Cross Entropy: - sum( target * log_p )
                # 【关键修正2】使用 log_all_probs 进行加权求和，而不是不存在的 log_probs
                ce_loss = -torch.sum(smooth_target * log_all_probs, dim=1, keepdim=True)
            else:
                # 原始逻辑: 只取专家动作的 log_prob
                # 【关键修正3】从 log_all_probs 中提取对应动作的概率
                chosen_log_probs = log_all_probs.gather(1, a_batch) # 只收集“参考答案”所在的概率值
                ce_loss = -chosen_log_probs
            
            # 加权 Loss
            actor_loss = torch.mean(alpha * weights.detach() * ce_loss)

            # ----------------------------------------------------
            # C. 计算 Critic Loss (监督学习拟合 R_t)
            # ----------------------------------------------------
            v_pred = self.critic(s_batch)
            critic_loss = F.mse_loss(v_pred, r_batch) * c_v * alpha

            # ----------------------------------------------------
            # D. 反向传播与更新
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

            # 记录数据
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            batch_count += 1

        # 返回本轮 Epoch 的平均 Loss
        avg_actor_loss = total_actor_loss / batch_count
        avg_critic_loss = total_critic_loss / batch_count
        
        return avg_actor_loss, avg_critic_loss, self.c_sq.sqrt().item()


# 仅调用actor
def take_action_from_policy_discrete(policy_net, state, device, explore=False):
    """
    独立的离散策略推理函数。
    输入:
      policy_net: PolicyNetDiscrete 实例
      state: 可被 np.array 接受的状态（标量或一维特征向量）
      device: torch.device
      explore: True 则从分布采样，False 则选择 argmax（确定性）
    返回:
      action: int 动作索引
    """
    policy_net.eval()
    state_t = torch.tensor(np.array([state]), dtype=torch.float).to(device)
    with torch.no_grad():
        probs = policy_net(state_t)  # (1, action_dim)
        if explore:
            dist = torch.distributions.Categorical(probs)
            action = int(dist.sample().item())
        else:
            action = int(torch.argmax(probs, dim=1).item())
    return action
