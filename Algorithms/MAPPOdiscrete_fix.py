from torch.distributions import Normal, Categorical
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import os, sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from Algorithms.Utils import model_grad_norm, moving_average, check_weights_bias_nan, compute_advantage, SquashedNormal
from Algorithms.MLP_heads import PolicyNetDiscrete

# ================================================================= #
#                         网络结构定义                                #
# ================================================================= #

# class PolicyNetDiscrete(torch.nn.Module):
#     """
#     [修改] 输出 Logits 而不是 Probabilities
#     """
#     def __init__(self, obs_dim, hidden_dims, action_dim):
#         super(PolicyNetDiscrete, self).__init__()
#         layers = []
#         prev_size = obs_dim
#         for layer_size in hidden_dims:
#             layers.append(nn.Linear(prev_size, layer_size))
#             layers.append(nn.ReLU())
#             prev_size = layer_size
#         self.net = nn.Sequential(*layers)
#         self.fc_out = torch.nn.Linear(prev_size, action_dim)

#     def forward(self, x):
#         x = self.net(x)
#         # [重要修改] 这里直接返回 logits (未经过 softmax)
#         # 这样结合 Categorical(logits=...) 使用时数值更稳定，避免 log(0)=-inf
#         return self.fc_out(x)


class CentralizedValueNet(torch.nn.Module):
    """
    中心化 Critic 网络 (价值网络)
    输入:
      - global_state: 所有智能体观测拼接成的全局状态
      - agent_id_one_hot: 标记当前智能体的 one-hot 向量
    输出:
      - 对应智能体的状态价值
    """
    def __init__(self, global_state_dim, back_hidden_dims, front_hidden_dims, num_agents):
        super(CentralizedValueNet, self).__init__()
        
        # 躯干网络 (Backbone)
        back_layers = []
        prev_size = global_state_dim
        for layer_size in back_hidden_dims:
            back_layers.append(torch.nn.Linear(prev_size, layer_size))
            back_layers.append(nn.ReLU())
            prev_size = layer_size
        self.back_net = nn.Sequential(*back_layers)

        # 头部网络 (Heads)
        self.front_heads = nn.ModuleList()
        for _ in range(num_agents):
            front_layers = []
            head_prev_size = prev_size # 从backbone的输出开始
            for layer_size in front_hidden_dims:
                front_layers.append(torch.nn.Linear(head_prev_size, layer_size))
                front_layers.append(nn.ReLU())
                head_prev_size = layer_size
            front_layers.append(torch.nn.Linear(head_prev_size, 1))
            self.front_heads.append(nn.Sequential(*front_layers))

    def forward(self, global_state, agent_id_one_hot):
        # 1. 全局状态通过共享的躯干网络
        back_output = self.back_net(global_state)
        
        # 2. 计算所有头部的输出
        head_outputs = [head(back_output) for head in self.front_heads]
        all_values = torch.cat(head_outputs, dim=1) # shape: (batch_size, num_agents)
        
        # 3. 根据 one-hot id 选择对应的头部输出
        # agent_id_one_hot shape: (batch_size, num_agents)
        value = (all_values * agent_id_one_hot).sum(dim=1, keepdim=True) # shape: (batch_size, 1)
        
        return value

# ================================================================= #
#                      MAPPO 主算法                                 #
# ================================================================= #

class MAPPO:
    ''' MAPPO 算法, 采用 PPO 截断方式 '''
    def __init__(self, obs_dim, action_dim, num_agents, 
                 actor_hidden_dims, critic_hidden_dims_back, critic_hidden_dims_front,
                 actor_lr, critic_lr, lmbda, epochs, eps, gamma, device, 
                 k_entropy=0.01, actor_max_grad=2, critic_max_grad=2):
        
        self.num_agents = num_agents
        
        # Actor 网络 (去中心化)
        self.actor = PolicyNetDiscrete(obs_dim, actor_hidden_dims, action_dim).to(device)
        
        # Critic 网络 (中心化)
        global_state_dim = obs_dim * num_agents
        self.critic = CentralizedValueNet(
            global_state_dim, critic_hidden_dims_back, critic_hidden_dims_front, num_agents
        ).to(device)
        
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

    def set_learning_rate(self, actor_lr=None, critic_lr=None):
        """动态设置 actor 和 critic 的学习率"""
        if actor_lr is not None:
            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] = actor_lr
        if critic_lr is not None:
            for param_group in self.critic_optimizer.param_groups:
                param_group['lr'] = critic_lr    

    def take_action(self, obs, explore=True):
        """根据单个智能体的观测 obs 采取行动"""
        state = torch.tensor(np.array([obs]), dtype=torch.float).to(self.device)
        # [修改] actor 返回的是 logits
        logits = self.actor(state, logits=1)
        
        if explore:
            # Categorical 内部会处理 softmax，数值更稳定
            action_dist = Categorical(logits=logits)
            action = action_dist.sample()
            # 为了返回概率分布给外部（如果需要），手动做 softmax
            probs = F.softmax(logits, dim=1)
        else:
            action = torch.argmax(logits)
            probs = F.softmax(logits, dim=1)
        # 返回动作索引与对应的概率分布（numpy array）
        probs_np = probs.detach().cpu().numpy()[0].copy() # [0]是batch维度
        return probs_np, action.item()

    def update(self, transition_dict, adv_normed=False, clip_vf=False, clip_range=0.2, shuffled=True, mini_batches=4):
        # 从 transition_dict 中提取数据
        obs = torch.tensor(np.array(transition_dict['obs']), dtype=torch.float).to(self.device)
        global_states = torch.tensor(np.array(transition_dict['global_states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions']), dtype=torch.long).view(-1, 1).to(self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float).view(-1, 1).to(self.device)
        next_global_states = torch.tensor(np.array(transition_dict['next_global_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']), dtype=torch.float).view(-1, 1).to(self.device)
        agent_ids = torch.tensor(np.array(transition_dict['agent_ids']), dtype=torch.long).view(-1, 1).to(self.device)
        agent_ids_one_hot = F.one_hot(agent_ids.squeeze(), num_classes=self.num_agents).float().to(self.device)
        
        # [新增] 提取 active_masks
        active_masks = torch.tensor(np.array(transition_dict['active_masks']), dtype=torch.float).view(-1, 1).to(self.device)

        # --- 计算优势函数 ---
        with torch.no_grad(): # 显式关闭梯度计算
            v_current = self.critic(global_states, agent_ids_one_hot)
            v_next = self.critic(next_global_states, agent_ids_one_hot)
            td_target = rewards + self.gamma * v_next * (1 - dones)
            td_delta = td_target - v_current
            
            # [修改] 传入 dones 以处理跨 episode 截断
            advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu(), dones.cpu()).to(self.device)
            

        if adv_normed:
            # 归一化时只考虑有效数据的均值和标准差，以获得更准确的缩放
            active_advantage = advantage[active_masks.squeeze(-1).bool()]
            # [安全措施] 防止 active_advantage 为空导致 std 为 NaN
            if active_advantage.numel() > 1:
                adv_mean = active_advantage.mean()
                adv_std = active_advantage.std()
                advantage = (advantage - adv_mean) / (adv_std + 1e-8)
            else:
                # 如果数据太少，不归一化或仅中心化
                pass 

        # --- 提前计算旧策略的对数概率 ---
        # [修改] 使用 logits 计算 log_prob，避免 log(0) = -inf
        with torch.no_grad():
            old_logits = self.actor(obs, logits=1)
            old_dist = Categorical(logits=old_logits)
            old_log_probs = old_dist.log_prob(actions.squeeze()).view(-1, 1) # 注意 squeeze 配合 Categorical
            v_pred_old = self.critic(global_states, agent_ids_one_hot)

        if torch.isnan(old_log_probs).any():
            # 如果这里出现 NaN，说明网络权重已经损坏或者输入 obs 有 NaN
            raise ValueError("Error: NaN detected in old_log_probs calculation.")


        # --- 初始化日志列表 ---
        actor_grad_list, critic_grad_list = [], []
        actor_loss_list, critic_loss_list = [], []
        pre_clip_actor_grad, pre_clip_critic_grad = [], []
        entropy_list, ratio_list = [], []

        batch_size = obs.shape[0]
        if mini_batches <= 0:
            mini_batch_size = batch_size
        else:
            mini_batch_size = batch_size // mini_batches
        
        # --- 开始多轮 Epoch 训练 ---
        for _ in range(self.epochs):
            if shuffled:
                indices = torch.randperm(batch_size)
            else:
                indices = torch.arange(batch_size)
            
            for i in range(0, batch_size, mini_batch_size):
                mb_indices = indices[i:i + mini_batch_size]
                
                # --- 从整个 batch 中切分出 mini-batch 数据 (包括 active_mask) ---
                mb_obs = obs[mb_indices]
                mb_global_states = global_states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_agent_ids_one_hot = agent_ids_one_hot[mb_indices]
                mb_advantage = advantage[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_td_target = td_target[mb_indices]
                mb_v_pred_old = v_pred_old[mb_indices]
                mb_active_masks = active_masks[mb_indices] # [新增]

                # --- Actor 更新 ---
                # [修改] 再次获取 logits，构建分布
                logits = self.actor(mb_obs, logits=1)
                
                # [重要安全检查] 如果输入是垃圾数据，logits 可能是极大的数值
                # 我们可以简单地检查一下 logits 是否包含 NaN
                if torch.isnan(logits).any():
                     raise ValueError("Actor logits contain NaN in loop (likely due to invalid inputs for dead agents)")

                action_dist = Categorical(logits=logits)
                log_probs = action_dist.log_prob(mb_actions.squeeze()).view(-1, 1)
                
                # 计算 ratio
                # log_probs 和 mb_old_log_probs 现在是稳定的，不会有 -inf
                ratio = torch.exp(log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantage
                surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * mb_advantage
                
                # [修改] 计算每个样本的损失，然后使用 mask 进行加权平均
                surrogate_loss_per_sample = -torch.min(surr1, surr2)
                
                # [修改] 确保分母不为0，使用小常数避免除零（保持 tensor 类型）
                eps = 1e-5
                active_sum = mb_active_masks.sum()  # tensor on device
                actor_loss = (surrogate_loss_per_sample * mb_active_masks).sum() / (active_sum + eps)
 
                entropy_per_sample = action_dist.entropy().view(-1, 1)
                entropy_factor = (entropy_per_sample * mb_active_masks).sum() / (active_sum + eps)

                actor_loss = actor_loss - self.k_entropy * entropy_factor

                # --- Critic 更新 ---
                current_values = self.critic(mb_global_states, mb_agent_ids_one_hot)
                if torch.isnan(current_values).any():
                    raise ValueError("Critic在循环里Nan")

                if clip_vf:
                    v_pred_clipped = torch.clamp(current_values, mb_v_pred_old - clip_range, mb_v_pred_old + clip_range)
                    vf_loss1 = (current_values - mb_td_target.detach()).pow(2)
                    vf_loss2 = (v_pred_clipped - mb_td_target.detach()).pow(2)
                    critic_loss_per_sample = torch.max(vf_loss1, vf_loss2)
                else:
                    critic_loss_per_sample = F.mse_loss(current_values, mb_td_target.detach(), reduction='none')
                
                # [修改] 使用 mask 对 critic loss 进行加权平均
                critic_loss = (critic_loss_per_sample * mb_active_masks).sum() / (active_sum + eps)
                
                # --- 优化器步骤 ---
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                # 处理 loss 可能是 tensor(0.0) 的情况
                if isinstance(actor_loss, torch.Tensor):
                    actor_loss.backward()
                if isinstance(critic_loss, torch.Tensor):
                    critic_loss.backward()
                
                # 记录梯度前的 norm，用于调试
                pre_clip_actor_grad.append(model_grad_norm(self.actor))
                pre_clip_critic_grad.append(model_grad_norm(self.critic))  
                # Clip Grad
                nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.actor_max_grad)
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.critic_max_grad)

                self.actor_optimizer.step()
                self.critic_optimizer.step()

                # --- 记录日志 ---
                actor_grad_list.append(model_grad_norm(self.actor))
                actor_loss_list.append(actor_loss.item() if isinstance(actor_loss, torch.Tensor) else actor_loss)
                critic_grad_list.append(model_grad_norm(self.critic))            
                critic_loss_list.append(critic_loss.item() if isinstance(critic_loss, torch.Tensor) else critic_loss)
                entropy_list.append(entropy_factor.item() if isinstance(entropy_factor, torch.Tensor) else entropy_factor)
                ratio_list.append(ratio.mean().item())
        
        # --- 保存平均指标 ---
        self.actor_loss = np.mean(actor_loss_list)
        self.actor_grad = np.mean(actor_grad_list)
        self.critic_loss = np.mean(critic_loss_list)
        self.critic_grad = np.mean(critic_grad_list)
        self.entropy_mean = np.mean(entropy_list)
        self.ratio_mean = np.mean(ratio_list)
        self.pre_clip_critic_grad = np.mean(pre_clip_critic_grad)
        self.pre_clip_actor_grad = np.mean(pre_clip_actor_grad)
        
        # [修改] 日志记录的优势函数也使用 mask
        active_sum_total = active_masks.sum().item()
        if active_sum_total > 0:
            self.advantage = (advantage.abs() * active_masks).sum().item() / active_sum_total
        else:
            self.advantage = 0

        check_weights_bias_nan(self.actor, "actor", "update后")
        check_weights_bias_nan(self.critic, "critic", "update后")


def take_action_from_policy_discrete(policy_net, obs, device, explore=False):
    """
    独立的离散策略推理函数。
    输入:
      policy_net: PolicyNetDiscrete 实例
      obs: 单个智能体的观测 (可被 np.array 接受的状态)
      device: torch.device
      explore: True 则从分布采样，False 则选择 argmax（确定性）
    返回:
      action: int 动作索引
    """
    policy_net.eval()
    state_t = torch.tensor(np.array([obs]), dtype=torch.float).to(device)
    with torch.no_grad():
        logits = policy_net(state_t)  # (1, action_dim)
        if explore:
            dist = Categorical(logits=logits)
            action = int(dist.sample().item())
        else:
            action = int(torch.argmax(logits, dim=1).item())
    return action