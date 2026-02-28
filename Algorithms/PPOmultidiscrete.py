from torch.distributions import Normal
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import os, sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from Algorithms.Utils import model_grad_norm, moving_average, check_weights_bias_nan, compute_advantage, SquashedNormal
# [修改] 引入 PolicyNetMultiDiscrete
from Algorithms.MLP_heads import ValueNet, PolicyNetMultiDiscrete

class PPO_multi_discrete:
    ''' PPO算法, 适配 Multi-Discrete (多重离散) 动作空间 '''
    def __init__(self, state_dim, hidden_dims, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device, k_entropy=0.01, actor_max_grad=2, critic_max_grad=2):
        # [修改] 使用 PolicyNetMultiDiscrete, action_dim 应当是一个 list [dim_1, dim_2, ...]
        self.actor = PolicyNetMultiDiscrete(state_dim, hidden_dims, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dims).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device
        self.k_entropy = k_entropy
        self.actor_max_grad = actor_max_grad
        self.critic_max_grad = critic_max_grad

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

    def take_action(self, state, explore=True):
        """
        [修改] 适配多重离散动作
        返回:
            probs_list: list of numpy arrays (每个维度的概率分布)
            action: list of ints (每个维度的动作索引)
        """
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        
        # PolicyNetMultiDiscrete 返回的是一个 list，包含每个维度的 (batch, dim_actions) 概率
        probs_list_t = self.actor(state)
        
        actions = []
        probs_np_list = []

        # 遍历每个维度的输出分布
        for probs in probs_list_t:
            action_dist = torch.distributions.Categorical(probs)
            if explore:
                action = action_dist.sample()
            else:
                action = torch.argmax(probs, dim=1)
            
            actions.append(action.item())
            probs_np_list.append(probs.detach().cpu().numpy()[0]) # [0] batch dim

        return probs_np_list, actions

    def _get_log_probs_and_entropy(self, states, actions):
        """
        [新增辅助函数] 计算联合对数概率和总熵
        state: (Batch, State_Dim)
        actions: (Batch, Action_Dims_Count)
        Returns:
            log_prob_sum: (Batch, 1) 所有维度 log_prob 之和
            entropy_sum: Scalar, 所有维度熵的平均值之和 (用于 Loss)
        """
        probs_list = self.actor(states) # List of (Batch, Num_Actions_i)
        
        log_prob_list = []
        entropy_sum = 0
        
        # 遍历每个动作维度
        for i, probs in enumerate(probs_list):
            dist = torch.distributions.Categorical(probs)
            
            # 取出当前维度对应的动作索引 (Batch,)
            # actions[:, i] 
            current_action = actions[:, i]
            
            # 计算该维度的 log_prob
            log_prob_list.append(dist.log_prob(current_action))
            
            # 计算熵 (Batch,) -> Mean
            entropy_sum += dist.entropy().mean()

        # 堆叠所有维度的 log_prob: (Batch, Num_Dims)
        log_prob_stack = torch.stack(log_prob_list, dim=1)
        
        # 联合概率的 log 等于各独立概率 log 之和: (Batch, 1)
        log_prob_sum = log_prob_stack.sum(dim=1, keepdim=True)
        
        return log_prob_sum, entropy_sum

    def update(self, transition_dict, adv_normed=False, clip_vf=False, clip_range=0.2):
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        
        # [修改] actions 形状调整为 (Batch, Num_Dims)，不再强制 view(-1, 1)
        actions = torch.tensor(np.array(transition_dict['actions']), dtype=torch.long).to(self.device)
        if actions.dim() == 1:
            actions = actions.view(-1, 1) # 兼容单维度情况，变成 (N, 1)

        rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']), dtype=torch.float).view(-1, 1).to(self.device)
        
        # 1. 初始计算
        # 计算旧的 log_probs
        with torch.no_grad():
            old_log_probs, _ = self._get_log_probs_and_entropy(states, actions)
            
            # NaN 检查
            if torch.isnan(old_log_probs).any():
                raise ValueError("NaN in Old Actor outputs")
            
            critic_values = self.critic(states)
            if torch.isnan(critic_values).any():
                raise ValueError("NaN in Critic outputs")
            
            # 提前计算旧的 value 预测（用于 value clipping）
            v_pred_old = critic_values.detach()

        # 2. 优势函数计算
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu(), dones.cpu()).to(self.device)
        
        # 优势归一化
        if adv_normed:
            adv_mean, adv_std = advantage.detach().mean(), advantage.detach().std(unbiased=False) 
            advantage = (advantage - adv_mean) / (adv_std + 1e-8)

        actor_grad_list = []
        actor_loss_list = []
        critic_grad_list = []
        pre_clip_actor_grad = []
        pre_clip_critic_grad = []
        critic_loss_list = []
        entropy_list = []
        ratio_list = []

        for _ in range(self.epochs):
            # [修改] 使用新的辅助函数获取 log_probs 和 entropy
            log_probs, entropy_mean_step = self._get_log_probs_and_entropy(states, actions)
            
            if torch.isnan(log_probs).any():
                raise ValueError("NaN in Actor outputs in loop")

            # 权重/偏置 NaN 检查
            check_weights_bias_nan(self.actor, "actor", "update循环中")
            check_weights_bias_nan(self.critic, "critic", "update循环中")

            # Ratio 计算
            ratio = torch.exp(log_probs - old_log_probs) # (N,1)
            surr1 = torch.clamp(ratio, -20, 20) * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage

            # Entropy
            entropy_factor = entropy_mean_step

            # Actor Loss
            actor_loss = torch.mean(-torch.min(surr1, surr2)) - self.k_entropy * entropy_factor

            # Critic Loss
            if clip_vf:
                v_pred = self.critic(states)
                v_pred_clipped = torch.clamp(v_pred, v_pred_old - clip_range, v_pred_old + clip_range)
                vf_loss1 = (v_pred - td_target.detach()).pow(2)
                vf_loss2 = (v_pred_clipped - td_target.detach()).pow(2)
                critic_loss = torch.max(vf_loss1, vf_loss2).mean()
            else:
                critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
            
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            
            pre_clip_actor_grad.append(model_grad_norm(self.actor))
            pre_clip_critic_grad.append(model_grad_norm(self.critic))  

            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.actor_max_grad)
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.critic_max_grad)

            self.actor_optimizer.step()
            self.critic_optimizer.step()

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
        self.pre_clip_critic_grad = np.mean(pre_clip_critic_grad)
        self.pre_clip_actor_grad = np.mean(pre_clip_actor_grad)
        self.advantage = advantage.abs().mean().detach().cpu().item()
        
        check_weights_bias_nan(self.actor, "actor", "update后")
        check_weights_bias_nan(self.critic, "critic", "update后")

def MARWIL_update(self, il_transition_dict, beta, batch_size=64, alpha=1.0, c_v=1.0, shuffled=1, max_weight=100, label_smoothing=0.1):
        """
        MARWIL 离线更新函数 (适配 Multi-Discrete)
        新增参数:
            label_smoothing (float): 标签平滑系数. 
        """
        states_all = torch.tensor(np.array(il_transition_dict['states']), dtype=torch.float).to(self.device)
        actions_all = torch.tensor(np.array(il_transition_dict['actions']), dtype=torch.long).to(self.device)
        returns_all = torch.tensor(np.array(il_transition_dict['returns']), dtype=torch.float).view(-1, 1).to(self.device)
        
        # 确保 actions 形状 (N, Dims)
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
            
            s_batch = states_all[batch_indices]
            a_batch = actions_all[batch_indices]
            r_batch = returns_all[batch_indices]

            # ----------------------------------------------------
            # A. 计算优势 (Advantage) 和 权重 (Weights)
            # ----------------------------------------------------
            with torch.no_grad():
                values = self.critic(s_batch)
                residual = r_batch - values
                
                # --- 动态更新归一化因子 c ---
                batch_mse = (residual ** 2).mean().item()
                self.c_sq = self.c_sq + 1e-8 * (batch_mse - self.c_sq)
                c = torch.sqrt(self.c_sq)
                
                # 归一化优势 & 计算权重
                advantage = residual / (c + 1e-8)
                raw_weights = torch.exp(beta * advantage)
                weights = torch.clamp(raw_weights, max=max_weight)

            # ----------------------------------------------------
            # B. 计算 Actor Loss (模仿学习部分)
            # ----------------------------------------------------
            probs_list = self.actor(s_batch) # List of (Batch, Dim_i_Actions)
            
            # 我们需要累加所有维度的 CrossEntropy Loss
            joint_ce_loss = 0 
            
            # 遍历每一个动作维度
            for i, probs in enumerate(probs_list):
                # 1. 获取该维度下的动作索引 (Batch, 1)
                action_idx = a_batch[:, i].view(-1, 1)
                
                # 2. 计算该维度所有动作的 log_prob (Batch, Num_Classes_i)
                log_all_probs = torch.log(probs + 1e-10)
                
                if label_smoothing > 0:
                    # === Label Smoothing 分支 ===
                    # A. 构建 One-hot
                    one_hot = torch.zeros_like(probs).scatter_(1, action_idx, 1.0)
                    
                    # B. 构建 Soft Target
                    # 注意: probs.size(1) 是当前维度的类别数 K
                    n_classes = probs.size(1)
                    smooth_target = one_hot * (1.0 - label_smoothing) + (label_smoothing / n_classes)
                    
                    # C. 计算 Soft Cross Entropy (Batch, 1)
                    ce_loss_dim = -torch.sum(smooth_target * log_all_probs, dim=1, keepdim=True)
                else:
                    # === 标准 Gather 分支 ===
                    # 只提取专家动作的 log_prob
                    chosen_log_probs = log_all_probs.gather(1, action_idx)
                    # 变成正的 loss
                    ce_loss_dim = -chosen_log_probs
                
                # 3. 累加维度的 Loss
                # 联合概率 P(a, b) = P(a) * P(b) -> log P(a,b) = log P(a) + log P(b)
                # 所以 CrossEntropy(Joint) = sum(CrossEntropy(Dim_i))
                joint_ce_loss = joint_ce_loss + ce_loss_dim
            
            # 4. 加权并求平均
            # joint_ce_loss 已经是正数形式的 Loss，所以直接 mean
            actor_loss = torch.mean(alpha * weights.detach() * joint_ce_loss)

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
            
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.actor_max_grad)
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.critic_max_grad)

            self.actor_optimizer.step()
            self.critic_optimizer.step()

            # 记录数据
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            batch_count += 1

        avg_actor_loss = total_actor_loss / batch_count
        avg_critic_loss = total_critic_loss / batch_count
        
        return avg_actor_loss, avg_critic_loss, self.c_sq.sqrt().item()

# 仅调用actor的推理函数
def take_action_from_policy_multi_discrete(policy_net, state, device, explore=False):
    """
    [修改] 适配多重离散策略推理。
    返回:
      action: list of ints (每个维度的动作索引)
    """
    policy_net.eval()
    state_t = torch.tensor(np.array([state]), dtype=torch.float).to(device)
    with torch.no_grad():
        probs_list = policy_net(state_t)  # List of (1, action_dim_i)
        
        actions = []
        for probs in probs_list:
            if explore:
                dist = torch.distributions.Categorical(probs)
                action = int(dist.sample().item())
            else:
                action = int(torch.argmax(probs, dim=1).item())
            actions.append(action)
            
    return actions