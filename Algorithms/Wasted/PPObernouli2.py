'''
构造联合损失函数让MARWIL和PPO之间存在过渡
有严重的梯度爆炸问题，得不偿失，废弃。
'''

from torch.distributions import Normal
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import os, sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from Algorithms.Utils import model_grad_norm, check_weights_bias_nan, moving_average, compute_advantage
from Algorithms.MLP_heads import PolicyNetBernouli, ValueNet

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
        self.critic_max_grad = critic_max_grad
        self.actor_max_grad = actor_max_grad


        # [新增] 用于 MARWIL 优势归一化的平方范数估计值
        # 初始值通常设置为 1.0 或一个小的正数
        self.c_sq = torch.tensor(1.0, dtype=torch.float).to(device)

    def set_learning_rate(self, actor_lr=None, critic_lr=None):
        """动态设置 actor 和 critic 的学习率"""
        if actor_lr is not None:
            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] = actor_lr
        if critic_lr is not None:
            for param_group in self.critic_optimizer.param_groups:
                param_group['lr'] = critic_lr    

    def get_action_probs(self, state):
        """
        获取动作概率分布。
        :param state: 输入状态 (tensor)
        :return: 动作概率 (tensor)
        """
        logits = self.actor(state)  # 获取未经过 sigmoid 的 logits
        probs = torch.sigmoid(logits)  # 应用 sigmoid 激活函数
        return probs

    # take action
    def take_action(self, state, explore=True, mask=None):
        # state -> tensor (1, state_dim)
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        probs = self.get_action_probs(state)  # 调用 get_action_probs 获取概率 (1,action_dim)

        if explore:
            sampled = torch.bernoulli(probs)  # (1, action_dim)
        else:
            sampled = (probs >= 0.5).float()

        # to numpy arrays (flatten for probs, keep action as 1-D ndarray)
        probs_np = probs.detach().cpu().numpy().flatten()           # shape (action_dim,)
        actions_np = sampled.detach().cpu().numpy().reshape(-1)     # shape (action_dim,)

        # mask = np.array([[0,1]] * probs_np.shape[-1]) ###
        if mask is not None:
            actions_np = np.clip(actions_np, mask[:, 0], mask[:, 1])
            # for i, action in enumerate(actions_np):
            #     actions_np[i] = np.clip(action, mask[i][0], mask[i][1])

        # 如果 action_dim == 1，保持返回形状为 (1,) 而不是 python int
        return actions_np, probs_np

    def update(self, transition_dict, adv_normed=False, clip_vf=False, clip_range=0.2, shuffled=False):
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions']), dtype=torch.float).to(self.device)
        # # 统一 actions 形状为 (N, action_dim)
        # if actions.dim() == 1:
        #     actions = actions.unsqueeze(1)
        rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']), dtype=torch.float).view(-1, 1).to(self.device)
        
        # 基础检查
        if torch.isnan(states).any(): raise ValueError("NaN in input states")
        probs = self.get_action_probs(states)  # 调用 get_action_probs 获取概率
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

        old_probs = self.get_action_probs(states).detach()  # 调用 get_action_probs 获取概率
        old_log_probs = torch.log(old_probs) * actions + torch.log(1 - old_probs) * (1 - actions)
        old_log_probs = old_log_probs.sum(dim=1, keepdim=True).detach()

        # [新增] 打乱顺序 (Shuffling)
        if shuffled:
            # 生成随机索引
            perm_indices = torch.randperm(states.size(0)).to(self.device)
            states = states[perm_indices]
            actions = actions[perm_indices]
            advantage = advantage[perm_indices]
            td_target = td_target[perm_indices]
            old_log_probs = old_log_probs[perm_indices]
            v_pred_old = v_pred_old[perm_indices]
            # rewards, next_states, dones 在 update 循环内部不直接使用（只用了计算出的 adv 和 target），
            # 但为了保持一致性也可以打乱，不过这里关键变量已全部打乱。

        actor_grad_list = []
        actor_loss_list = []
        critic_grad_list = []
        pre_clip_actor_grad = []
        pre_clip_critic_grad = []
        critic_loss_list = []
        entropy_list = []
        ratio_list = []

        for _ in range(self.epochs):
            probs = self.get_action_probs(states)  # 调用 get_action_probs 获取概率
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
        # 权重/偏置 NaN 检查（在每次前向后、反向前检查参数）
        check_weights_bias_nan(self.actor, "actor", "update后")
        check_weights_bias_nan(self.critic, "critic", "update后")

    def MARWIL_update(self, il_transition_dict, beta, batch_size=64, alpha=1.0, c_v=1.0, shuffled=1, max_weight=100):
        """
        MARWIL 离线更新函数 (适配 Multi-Discrete)
        """
        states_all = torch.tensor(np.array(il_transition_dict['states']), dtype=torch.float).to(self.device)
        actions_all = torch.tensor(np.array(il_transition_dict['actions']), dtype=torch.float).to(self.device)
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
            # [修改] 对多维动作，Loss = - mean( weights * sum(log_probs_per_dim) )
            
            probs = self.get_action_probs(s_batch)
            probs = torch.clamp(probs, 1e-10, 1.0 - 1e-10)

            # 计算 log_prob (针对 Bernoulli 分布)
            # log_probs = log(p)*a + log(1-p)*(1-a)
            log_probs = torch.log(probs) * a_batch + torch.log(1 - probs) * (1 - a_batch)
            log_probs = log_probs.sum(dim=-1, keepdim=True)
            
            # Loss = - mean( alpha * weights * log_pi_joint )
            actor_loss = -torch.mean(alpha * weights.detach() * log_probs)

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
    


    def mixed_update(self, transition_dict, il_transition_dict, beta,
                     batch_size=64, alpha=1.0, c_v=1.0, shuffled=True,
                     max_weight=100, adv_normed=False, clip_vf=False, clip_range=0.2,
                     il_sample_ratio=1.0):
        """
        PPO 与 MARWIL 的混合更新函数 (Joint Optimization)
        
        参数:
            transition_dict (dict): 在线 PPO 经验
            il_transition_dict (dict): 离线 MARWIL 专家经验
            beta (float): MARWIL 优势指数系数
            batch_size (int): 这里的 batch_size 主要用于 MARWIL 数据的切分 (或者控制总的计算量)
            alpha (float): 模仿学习损失的权重
            c_v (float): 价值损失权重
            shuffled (bool): 是否打乱
            max_weight (float): MARWIL 权重截断
            il_sample_ratio (float): 从全量专家数据中抽样的比例 (0.0 ~ 1.0)
                                     例如 0.1 表示只使用 10% 的专家数据参与本轮更新
        """
        # ======================================================================
        # 1. 准备 PPO 数据 (On-policy Data)
        # ======================================================================
        # 注意：PPO 的数据量通常较小（一个 Episode 或几个 Episode），我们通常直接全量使用
        states_ppo = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        actions_ppo = torch.tensor(np.array(transition_dict['actions']), dtype=torch.float).to(self.device)
        rewards_ppo = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float).view(-1, 1).to(self.device)
        next_states_ppo = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        dones_ppo = torch.tensor(np.array(transition_dict['dones']), dtype=torch.float).view(-1, 1).to(self.device)

        # 计算 PPO 的 Target 和 Advantage
        with torch.no_grad():
            td_target_ppo = rewards_ppo + self.gamma * self.critic(next_states_ppo) * (1 - dones_ppo)
            td_delta_ppo = td_target_ppo - self.critic(states_ppo)
            advantage_ppo = compute_advantage(self.gamma, self.lmbda, td_delta_ppo.cpu(), dones_ppo.cpu()).to(self.device)
            
            if adv_normed:
                adv_mean, adv_std = advantage_ppo.mean(), advantage_ppo.std()
                advantage_ppo = (advantage_ppo - adv_mean) / (adv_std + 1e-8)
            
            # PPO 旧策略概率 (用于计算 Ratio)
            old_probs_ppo = self.get_action_probs(states_ppo)
            old_log_probs_ppo = torch.log(old_probs_ppo + 1e-10) * actions_ppo + \
                                torch.log(1 - old_probs_ppo + 1e-10) * (1 - actions_ppo)
            old_log_probs_ppo = old_log_probs_ppo.sum(dim=-1, keepdim=True)

        # ======================================================================
        # 2. 准备 MARWIL 数据 (Offline Data)
        # ======================================================================
        states_il_all = torch.tensor(np.array(il_transition_dict['states']), dtype=torch.float).to(self.device)
        actions_il_all = torch.tensor(np.array(il_transition_dict['actions']), dtype=torch.float).to(self.device)
        returns_il_all = torch.tensor(np.array(il_transition_dict['returns']), dtype=torch.float).view(-1, 1).to(self.device)
        
        if actions_il_all.dim() == 1:
            actions_il_all = actions_il_all.view(-1, 1)

        total_il_size = states_il_all.size(0)
        
        # 确定本轮混合更新要使用的 IL 数据量
        num_il_samples = int(total_il_size * il_sample_ratio)
        if num_il_samples < 1: num_il_samples = 1 # 至少采一个
        
        # 生成 IL 数据的随机索引
        il_indices = np.random.choice(total_il_size, num_il_samples, replace=False)
        
        # 将 PPO 数据的索引也准备好 (为了能够分 Batch 训练)
        # 策略：我们希望每个 Batch 里既有 PPO 数据也有 IL 数据，或者交替训练
        # 简单起见：我们在 PPO 的 epochs 循环里，每次随机采一个 IL Batch
        
        # ======================================================================
        # 3. 联合优化循环 (PPO Epochs)
        # ======================================================================
        # 为了平衡，我们以 PPO 的 epochs 为主循环
        # 在每个 Step 中，我们使用全量 PPO 数据 (因为 PPO 数据少) + 一个 Batch 的 IL 数据
        
        actor_grad_list = []
        actor_loss_list = []
        critic_grad_list = []
        pre_clip_actor_grad = []
        pre_clip_critic_grad = []
        critic_loss_list = []
        entropy_list = []
        ratio_list = []
        
        ppo_indices = np.arange(states_ppo.size(0)) # PPO 全量索引

        for _ in range(self.epochs):
            # 打乱 PPO 数据 (如果需要 Mini-batch PPO，可以在这里切分，但通常 PPO 数据量小直接全量)
            if shuffled:
                np.random.shuffle(ppo_indices)
                
            # 我们将 IL 数据分 Batch 喂入，为了避免 IL 数据过多淹没 PPO，
            # 我们每次 PPO 更新只采 *一个* 或 *少数几个* IL Batch
            # 这里逻辑：每次 PPO Step 对应一个随机采样的 IL Batch
            
            # 从选定的 num_il_samples 中随机采一个 Batch 用于本次计算
            # (注意：如果 il_sample_ratio 很大，这里其实是随机重采，相当于 SGD)
            batch_il_idx = np.random.choice(il_indices, batch_size, replace=True) 
            
            # --- 提取 IL Batch 数据 ---
            s_il = states_il_all[batch_il_idx]
            a_il = actions_il_all[batch_il_idx]
            r_il = returns_il_all[batch_il_idx]

            # ==========================
            # A. 计算 MARWIL Loss (IL)
            # ==========================
            with torch.no_grad():
                v_il = self.critic(s_il)
                residual = r_il - v_il
                # 动态 c 更新
                batch_mse = (residual ** 2).mean().item()
                self.c_sq = self.c_sq + 1e-8 * (batch_mse - self.c_sq)
                c = torch.sqrt(self.c_sq)
                adv_il = residual / (c + 1e-8)
                raw_w = torch.exp(beta * adv_il)
                w_il = torch.clamp(raw_w, max=max_weight)

            probs_il = self.get_action_probs(s_il)
            probs_il = torch.clamp(probs_il, 1e-6, 1.0 - 1e-6) # 强力数值保护 1.0=1e-6
            log_probs_il = torch.log(probs_il) * a_il + torch.log(1 - probs_il) * (1 - a_il)
            log_probs_il = log_probs_il.sum(dim=-1, keepdim=True)
            
            loss_actor_il = -torch.mean(w_il.detach() * log_probs_il)
            loss_critic_il = F.mse_loss(self.critic(s_il), r_il)

            # ==========================
            # B. 计算 PPO Loss (RL)
            # ==========================
            # 使用全量 PPO 数据 (或者也可以切 Batch)
            # 为简单起见，这里假设 PPO 数据量不大，直接用全量
            s_ppo = states_ppo
            a_ppo = actions_ppo
            adv_ppo = advantage_ppo
            target_ppo = td_target_ppo
            old_lp_ppo = old_log_probs_ppo

            probs_ppo = self.get_action_probs(s_ppo)
            probs_ppo = torch.clamp(probs_ppo, 1e-6, 1.0 - 1e-6)
            log_probs_ppo = torch.log(probs_ppo) * a_ppo + torch.log(1 - probs_ppo) * (1 - a_ppo)
            log_probs_ppo = log_probs_ppo.sum(dim=-1, keepdim=True)

            ratio = torch.exp(log_probs_ppo - old_lp_ppo)
            surr1 = ratio * adv_ppo
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * adv_ppo
            
            # Entropy
            entropy = - probs_ppo * torch.log(probs_ppo) - (1 - probs_ppo) * torch.log(1 - probs_ppo)
            entropy_mean = entropy.sum(dim=-1).mean()

            loss_actor_ppo = -torch.min(surr1, surr2).mean() - self.k_entropy * entropy_mean
            
            # if clip_vf: # 带上MARWIL损失函数之后critic_loss裁剪变得不好做了
            #     v_pred = self.critic(s_ppo)
            #     v_pred_clipped = torch.clamp(v_pred, v_pred_old - clip_range, v_pred_old + clip_range)
            #     vf_loss1 = (v_pred - target_ppo).pow(2)
            #     vf_loss2 = (v_pred_clipped - target_ppo).pow(2)
            #     loss_critic_ppo = torch.max(vf_loss1, vf_loss2).mean()
            # else:
            loss_critic_ppo = F.mse_loss(self.critic(s_ppo), target_ppo)

            # ==========================
            # C. 联合反向传播 (Joint Backward)
            # ==========================
            # 核心公式: L_total = L_ppo + alpha * L_marwil
            
            total_actor_loss = loss_actor_ppo + alpha * loss_actor_il
            total_critic_loss = loss_critic_ppo + alpha * c_v * loss_critic_il

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            
            total_actor_loss.backward()
            total_critic_loss.backward()

            # 裁剪前梯度
            pre_clip_actor_grad.append(model_grad_norm(self.actor))
            pre_clip_critic_grad.append(model_grad_norm(self.critic))  
            
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_max_grad)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.critic_max_grad)
            
            self.actor_optimizer.step()
            self.critic_optimizer.step()

            # 记录 (仅供参考)
            # # 保存用于日志/展示的数值（断开计算图并搬到 CPU）
            actor_grad_list.append(model_grad_norm(self.actor))
            actor_loss_list.append(total_actor_loss.detach().cpu().item())
            critic_grad_list.append(model_grad_norm(self.critic))            
            critic_loss_list.append(total_critic_loss.detach().cpu().item())
            entropy_list.append(entropy_mean.detach().cpu().item())
            ratio_list.append(ratio.mean().detach().cpu().item())

        self.actor_loss = np.mean(actor_loss_list)
        self.actor_grad = np.mean(actor_grad_list)
        self.critic_loss = np.mean(critic_loss_list)
        self.critic_grad = np.mean(critic_grad_list)
        self.entropy_mean = np.mean(entropy_list)
        self.ratio_mean = np.mean(ratio_list)
        self.pre_clip_critic_grad = np.mean(pre_clip_critic_grad)
        self.pre_clip_actor_grad = np.mean(pre_clip_actor_grad)
        self.advantage = adv_ppo.abs().mean().detach().cpu().item()
        # 权重/偏置 NaN 检查（在每次前向后、反向前检查参数）
        check_weights_bias_nan(self.actor, "actor", "update后")
        check_weights_bias_nan(self.critic, "critic", "update后")