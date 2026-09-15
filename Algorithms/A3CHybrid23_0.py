'''
A3C 消融实验
'''

import numpy as np
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, Bernoulli
import copy
import os, sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from Algorithms.Utils import model_grad_norm, check_weights_bias_nan, compute_advantage, SquashedNormal
from Algorithms.MLP_heads import ValueNet

def sigmoid(x):
    return 1/(1+np.exp(-x))

# =============================================================================
# 1. 神经网络定义 (统一从 PPOHybrid23_0 导入，保证全局策略网络与掩码逻辑一致)
# =============================================================================
from Algorithms.PPOHybrid23_0 import PolicyNetHybrid, HybridActorWrapper

# =============================================================================
# 2. Actor 适配器 (Wrapper) - 从 PPOHybrid23_0 继承
# =============================================================================

# =============================================================================
# 3. A3C/A2C 算法类 (精简版)
# =============================================================================

class A3CHybrid:
    def __init__(self, actor, critic, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device, 
                 k_entropy={'cont':0.01, 'cat':0.005, 'bern':0.05}, critic_max_grad=2, actor_max_grad=2, max_std=0.7, # ):
                 rnd_state_dim=None, rnd_lr=3e-4, rnd_output_dim=128, rnd_hidden_dim=256):
        
        self.actor = actor # 这是一个 HybridActorWrapper 实例
        self.critic = critic
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device
        
        # [修改] 解析 k_entropy，支持字典输入
        if isinstance(k_entropy, dict):
            self.k_entropy = k_entropy
        else:
            self.k_entropy = {'cont': k_entropy, 'cat': k_entropy, 'bern': k_entropy}
        
        # [新增] SAC 风格的可学习 Cat 熵系数 (初始值对应原配置)
        init_k_cat = k_entropy.get('cat', k_entropy)
        self.log_k_cat = torch.nn.Parameter(torch.log(torch.tensor(init_k_cat, device=device)))
        # [新增] 为自适应熵系数配备独立的优化器 (学习率通常与 Actor 保持一致或略大)
        self.k_cat_optim = torch.optim.Adam([self.log_k_cat], lr=actor_lr)
        
        self.critic_max_grad = critic_max_grad
        self.actor_max_grad = actor_max_grad
        self.max_std = max_std
        
        # 记录指标
        self.actor_loss = 0
        self.critic_loss = 0
        self.actor_grad = 0
        self.critic_grad = 0
        self.entropy_mean = 0
        self.ratio_mean = 0
        self.pre_clip_actor_grad = 0
        self.pre_clip_critic_grad = 0
        
        #  额外的监控指标
        self.approx_kl = 0        # 近似 KL 散度 (判断策略变化幅度)
        self.clip_frac = 0        # 裁剪触发比例 (判断 eps 或 lr 是否合适)
        self.explained_var = 0    # 解释方差 (判断 Critic 拟合程度)
        #  分项 Entropy 监控
        self.entropy_cat = 0
        self.entropy_bern = 0
        self.entropy_cont = 0
        
        # [新增] 监控指标
        self.td_error_var = 0     # TD error 的分布方差
        self.grad_norm_ratio = 0  # actor 梯度与 critic 梯度的范数比

    def set_learning_rate(self, actor_lr=None, critic_lr=None):
        if actor_lr is not None:
            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] = actor_lr
        if critic_lr is not None:
            for param_group in self.critic_optimizer.param_groups:
                param_group['lr'] = critic_lr  


    # 重置优化器，这里仍然是adam，目前只有PBT在使用这个方法
    def reset_optimizer(self, reset_type='both'):
        """
        清除 optimizer 的动量等 state，但保留当前学习率/其他 param_group 设置。
        通过重建 optimizer（保留 param_group 的 lr/betas/eps/weight_decay）来达到清空 state 的目的。
        param_groups 里只存配置参数（lr、betas、eps、weight_decay 等），不存动量状态。
        PyTorch 的动量信息（Adam 的 exp_avg/exp_avg_sq、RMSprop 的 square_avg 等）都放在 old_optim.state 这个独立字典里
        """
        def _recreate_from(old_optim, params):
            optim_cls = type(old_optim)
            pg = old_optim.param_groups[0]
            kwargs = {k: v for k, v in pg.items() if k != 'params'}
            return optim_cls(params, **kwargs)

        if reset_type=='both':
            # 重新创建 optim（以清除内部 state），但使用当前的 lr/betas/eps/weight_decay
            self.actor_optimizer = _recreate_from(self.actor_optimizer, self.actor.parameters())
            self.critic_optimizer = _recreate_from(self.critic_optimizer, self.critic.parameters())
            # 清除可能残留的梯度
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
        elif reset_type=='actor':
            self.actor_optimizer = _recreate_from(self.actor_optimizer, self.actor.parameters())
            self.actor_optimizer.zero_grad()
        elif reset_type=='critic':
            self.critic_optimizer = _recreate_from(self.critic_optimizer, self.critic.parameters())
            self.critic_optimizer.zero_grad()
        else:
            pass
    
    def take_action(self, state, h0=None, explore=True, max_std=None, check_obs=None, temperature=1.0, mask_on=0):
        # 委托给 Actor Wrapper
        max_s = max_std if max_std is not None else self.max_std
        
        # [修改] 透传 check_obs
        actions_exec, actions_raw, h_state, actions_dist_check = self.actor.get_action(
            state, h=h0, explore=explore, max_std=max_s, check_obs=check_obs, temperature=temperature, mask_on=mask_on
        )
        #  保持原有的返回两个字典的接口，或者根据需要返回 diagnostic output
        return actions_exec, actions_raw, h_state, actions_dist_check

    def update(self, transition_dict, adv_normed=False, 
                clip_vf=False, clip_range=0.2, shuffled=1, 
                mini_batch_size=None, alpha_logit_reg=0.05,
                v_trace=None, target_p1=0.65, target_p1_b=0.8, 
                k_nonlinear=0.89, mask_on=0, actor_frozen=0, bern_max_logits=4.0): 
                # [新增] target_p1 默认“一超”概率，剩下来的留给“多强”)
                # [修改] 增加 target_p1_b 参数，对应开火控制的“笃定程度”

        # RL 更新阶段：确保所有分布参数都参与梯度更新
        if hasattr(self.actor.net, 'log_std_cont'):
            self.actor.net.log_std_cont.requires_grad = True
        # if hasattr(self.actor.net, 'log_temp_cat'):
        #     self.actor.net.log_temp_cat.requires_grad = False
        # if hasattr(self.actor.net, 'log_temp_bern'):
        #     self.actor.net.log_temp_bern.requires_grad = False

        #  6. 智能数据转换：如果已经是 np.ndarray (来自 HybridReplayBuffer)，直接转 Tensor
        # 否则 (来自 list append)，先转 np 再转 Tensor
        def to_tensor(x, dtype):
            if isinstance(x, np.ndarray):
                return torch.tensor(x, dtype=dtype).to(self.device)
            else:
                return torch.tensor(np.array(x), dtype=dtype).to(self.device)

        states = to_tensor(transition_dict['states'], torch.float)
        
        dones = to_tensor(transition_dict['dones'], torch.float).view(-1, 1)
        rewards = to_tensor(transition_dict['rewards'], torch.float).view(-1, 1)

        #  处理 active_masks (可选输入)
        # 如果 transition_dict 中没有 active_masks，则默认所有样本均有效
        if 'active_masks' in transition_dict:
            active_masks = to_tensor(transition_dict['active_masks'], torch.float).view(-1, 1)
        else:
            # 创建与 dones 形状一致的全 1 张量
            active_masks = torch.ones_like(dones)

        # 处理 obs (如果存在)
        if 'obs' in transition_dict:
            actor_inputs = to_tensor(transition_dict['obs'], torch.float)
            critic_inputs = states
        else:
            actor_inputs = states
            critic_inputs = states
        
        # 1. 准备动作数据
        actions_from_buffer = transition_dict['actions']
        
        # todo action_mask 防止“死后动作”干扰决策
        # todo truncs
        # todo global states，适配集中式Critic

        # 1. 准备动作数据 (转 Tensor)
        actions_on_device = {}
        
        # Buffer 传来的 actions 已经是 dict of arrays
        if isinstance(actions_from_buffer, dict):
            for key, val in actions_from_buffer.items():
                if key == 'cat':
                    actions_on_device[key] = to_tensor(val, torch.long)
                else:
                    actions_on_device[key] = to_tensor(val, torch.float)
        else:
            # 兼容旧代码 (list of dicts)
            # 旧逻辑：List of Dicts (较慢)
            all_keys = actions_from_buffer[0].keys()
            for key in all_keys:
                vals = [d[key] for d in actions_from_buffer]
                if key == 'cat':
                    actions_on_device[key] = torch.tensor(np.array(vals), dtype=torch.long).to(self.device)
                else:
                    actions_on_device[key] = torch.tensor(np.array(vals), dtype=torch.float).to(self.device)


        # 2. 获取 Advantage (优先使用 Buffer 算好的)
        if 'advantages' in transition_dict and 'td_targets' in transition_dict:
            advantage = to_tensor(transition_dict['advantages'], torch.float).view(-1, 1)
            td_target = to_tensor(transition_dict['td_targets'], torch.float).view(-1, 1)
        else:
            # 现场计算 GAE (不推荐用于并行展平后的数据)
            
            #  处理 truncs
            if 'truncs' in transition_dict:
                truncs = to_tensor(transition_dict['truncs'], torch.float).view(-1, 1)
            else:
                truncs = None # 或者 torch.zeros_like(dones)

            # 以下为公共部分
            # 如果没有预计算，则现场计算 (注意：如果是并行数据直接展平进来的，这里计算会有偏差)
            with torch.no_grad():
                # [修改] 优先使用预计算的 next_values
                if 'next_values' in transition_dict:
                    next_vals = to_tensor(transition_dict['next_values'], torch.float).view(-1, 1)
                else:
                    # Critic 使用全局 next_states 计算 Target
                    # 注意：对于截断的步，next_value 应该是 V(s_t+1) 而不是 0
                    next_states = to_tensor(transition_dict['next_states'], torch.float)
                    # Critic 使用全局 next_states 计算 Target
                    # 注意：对于截断的步，next_value 应该是 V(s_t+1) 而不是 0
                    next_vals = self.critic(next_states)
                
                # td_target的计算不应考虑truncs。仅当dones=1时，next_value才为0。
                # truncs的影响由compute_advantage函数内部处理。
                td_target = rewards + self.gamma * next_vals * (1 - dones)

                # Critic 使用全局 states 计算当前 Value
                td_delta = td_target - self.critic(critic_inputs)
                advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu(), dones.cpu(), truncs.cpu() if truncs is not None else None).to(self.device)
                
        # 3. 计算旧策略的 log_probs (使用 Wrapper)
        with torch.no_grad():
            v_pred_old = self.critic(critic_inputs)
            
        # --- [2. 优势归一化 - 适配 active_masks] ---
        if adv_normed:
            #  仅使用 active 的数据计算统计量
            active_adv = advantage[active_masks.squeeze(-1).bool()]
            
            if active_adv.numel() > 1: # 防止 active 数据过少导致 NaN
                adv_mean = active_adv.mean()
                adv_std = active_adv.std(unbiased=False)
                advantage = (advantage - adv_mean) / (adv_std + 1e-8)
            else:
                # 降级策略：如果有效数据太少，不进行归一化或者只做去中心化
                pass
        else:
            # 推荐: 即使不归一化，也建议减去均值 (Centering)
            # 这有助于降低方差，且不改变梯度的方向
            #  使用 mask 计算均值
            active_adv = advantage[active_masks.squeeze(-1).bool()]
            if active_adv.numel() > 0:
                adv_mean = active_adv.mean()
                advantage = advantage - adv_mean

        # =====================================================================
        # [新增] 计算 Cat 分布的动态目标熵 (Target Entropy)
        # =====================================================================
        target_entropy_cat_total = 0.0
        if 'cat' in self.actor.action_dims and sum(self.actor.action_dims['cat']) > 0:
            for n_actions in self.actor.action_dims['cat']:
                if n_actions > 1:
                    # “一超”概率为 target_p1，其余 (n_actions - 1) 个动作平分剩余概率
                    p_rest = (1.0 - target_p1) / (n_actions - 1)
                    # 香农熵: -p1*ln(p1) - (N-1) * p_rest*ln(p_rest)
                    h_dim = - target_p1 * np.log(target_p1 + 1e-8) - (n_actions - 1) * p_rest * np.log(p_rest + 1e-8)
                    target_entropy_cat_total += h_dim
        target_entropy_cat_tensor = torch.tensor(target_entropy_cat_total, device=self.device)
        
        # =====================================================================
        # [新增] 计算 Bernoulli 分布的动态目标熵 (Target Entropy)
        # =====================================================================
        target_entropy_bern_total = 0.0
        if 'bern' in self.actor.action_dims and self.actor.action_dims['bern'] > 0:
            # 伯努利熵公式: -p*ln(p) - (1-p)*ln(1-p)
            p = target_p1_b
            h_bern_dim = - p * np.log(p + 1e-8) - (1 - p) * np.log(1 - p + 1e-8)
            # 伯努利熵按动作维度累加
            target_entropy_bern_total = h_bern_dim * self.actor.action_dims['bern']
        target_entropy_bern_tensor = torch.tensor(target_entropy_bern_total, device=self.device)

        # 4. A3C Update Loop
        actor_loss_list, critic_loss_list, entropy_list = [], [], []
        actor_grad_list, critic_grad_list = [], []
        pre_clip_actor_grad, pre_clip_critic_grad = [], []

        #  监控列表
        kl_list = []
        #  分项 Entropy 列表
        entropy_cat_list = []
        entropy_bern_list = []
        entropy_cont_list = []
        grad_norm_ratio_list = [] # [新增] 范数比列表
        max_fire_prob_list = [] # [新增] 记录未被mask的bern最大触发概率
        min_fire_prob_list = [] # [新增] 记录未被mask的bern最小触发概率
        
        # [新增] 初始化样本统计计数器 (包含重复更新累加)
        ppo_samples_total = 0
        ppo_valid_samples_total = 0

        mask_eps = 1e-5
        
        num_samples = actor_inputs.size(0)
        if mini_batch_size is None:
            mini_batch_size = num_samples

        for _ in range(self.epochs):
            
            # --- [Shuffle 逻辑移入 Epoch 循环] ---
            # 每个 epoch 生成新的随机索引，确保数据打乱
            if shuffled:
                idx = torch.randperm(num_samples, device=self.device)
            else:
                idx = torch.arange(num_samples, device=self.device)
            
            # --- [引入 Mini-Batch 循环] ---
            for start in range(0, num_samples, mini_batch_size):
                end = min(start + mini_batch_size, num_samples)
                batch_idx = idx[start:end]
                
                # 切片 mini-batch 数据
                mb_actor_inputs = actor_inputs[batch_idx]
                mb_critic_inputs = critic_inputs[batch_idx]
                mb_advantage = advantage[batch_idx]
                mb_td_target = td_target[batch_idx]
                mb_active_masks = active_masks[batch_idx]
                # v_pred_old_batch = v_pred_old[batch_idx] # 如果需要用到旧 Value
                
                # 切片 Actions (Dict 结构)
                mb_actions = {}
                for k, v in actions_on_device.items():
                    mb_actions[k] = v[batch_idx]

                # 计算当前策略的 log_probs 和 entropy (使用 Wrapper)
                #  接收 entropy_details 和 actor_outputs
                log_probs, entropy, entropy_details, actor_outputs, _ = self.actor.evaluate_actions(mb_actor_inputs, mb_actions, h=None, max_std=self.max_std, mask_on=mask_on)
                
                #  计算一些不参与梯度的监控指标
                with torch.no_grad():
                    active_sum = mb_active_masks.sum()
                    # [新增] 统计采样样本 (包含重复更新累加)
                    ppo_samples_total += mb_active_masks.numel()
                    ppo_valid_samples_total += active_sum

                # [核心修改] A3C 损失: - log_prob * advantage
                surrogate_loss = - (log_probs * mb_advantage)
                actor_loss = (surrogate_loss * mb_active_masks).sum() / (active_sum + mask_eps)
                
                # [修改] 分项 Entropy Loss 计算
                e_cont = entropy_details['cont'] if entropy_details['cont'] is not None else torch.tensor(0., device=self.device)
                e_cat = entropy_details['cat'] if entropy_details['cat'] is not None else torch.tensor(0., device=self.device)
                e_bern = entropy_details['bern'] if entropy_details['bern'] is not None else torch.tensor(0., device=self.device)

                loss_ent_cont = (e_cont * mb_active_masks).sum() / (active_sum + mask_eps)
                loss_ent_cat = (e_cat * mb_active_masks).sum() / (active_sum + mask_eps)
                
                # [新增] 使用 fire_mask 过滤 Bernoulli 熵，只统计允许开火时段的熵
                if 'fire_mask' in actor_outputs and actor_outputs['fire_mask'] is not None:
                    bern_fire_mask = actor_outputs['fire_mask']  # shape: (batch, bern_dim)
                    # 对齐到 (batch, 1)
                    if bern_fire_mask.dim() > 1:
                        bern_fire_mask = bern_fire_mask[:, 0:1]  # 取第一个 Bernoulli 维度
                    fireable_sum = (mb_active_masks * bern_fire_mask).sum()
                    loss_ent_bern = (e_bern * mb_active_masks * bern_fire_mask).sum() / (fireable_sum + mask_eps)
                else:
                    # 回退到原有逻辑（兼容旧版本）
                    loss_ent_bern = (e_bern * mb_active_masks).sum() / (active_sum + mask_eps)

                # =====================================================================
                # [修改] 使用动态 k_cat 计算 Actor Loss
                # =====================================================================
                k_cont = self.k_entropy.get('cont', 0.0)
                k_cat = self.k_entropy.get('cat', 0.0)
                k_bern = self.k_entropy.get('bern', 0.0)
                
                # # 当前的 k_cat 是 e^(log_k_cat)
                # curr_k_cat = torch.exp(self.log_k_cat)
                # --- 关键修改：跟踪目标熵, 熵过小，加熵。熵过大，减熵 ---
                # 比平方误差更加平缓的损失函数候选：
                # 1. pseudo-huber loss:     torch.sqrt(1 + torch.square(target - current)) - 1
                # 2. log-cosh loss:         torch.log(torch.cosh(target_entropy_cat_tensor - loss_ent_cat))

                # 裁剪的范围，绝不能太小
                theoretical_ent_cat_max = 0.0
                for dim in self.actor.action_dims['cat']:
                    theoretical_ent_cat_max += np.log(float(dim))
                diff_cat0 = theoretical_ent_cat_max - target_entropy_cat_tensor
                k_nonlinear_max_cat = torch.sqrt(1+diff_cat0**2)/(diff_cat0+1e-8)

                theoretical_ent_bern_max = -np.log(0.5)*self.actor.action_dims['bern']
                diff_bern0 = theoretical_ent_bern_max - target_entropy_bern_tensor
                k_nonlinear_max_bern = torch.sqrt(1+diff_bern0**2)/(diff_bern0+1e-8)

                k_nonlinear_cat = min(max(k_nonlinear, 0.0), k_nonlinear_max_cat)
                k_nonlinear_bern = min(max(k_nonlinear, 0.0), k_nonlinear_max_bern)

                # 1. Categorical 约束项
                cat_constraint_term = k_cat * (
                    - loss_ent_cat # + 
                    # torch.where(loss_ent_cat > target_entropy_cat_tensor, 
                    # min(max(k_nonlinear_cat, 0.0), 1.0)/(2 * diff_cat0) * \
                    #     torch.square(loss_ent_cat - target_entropy_cat_tensor)
                    # , 0.0)
                )
                # # 2. Bernoulli 约束项 (动态系数：熵越高推力越弱，理论最大时归零)
                # # f(H) = -H + H^2/(2*H_max), 导数 f'(H) = -1 + H/H_max
                # # H=0 时全力推，H=H_max 时导数为0
                # bern_constraint_term = k_bern * (
                #     - loss_ent_bern
                #     + torch.where(loss_ent_bern > target_entropy_bern_tensor,
                #     min(max(k_nonlinear_bern, 0.0), 1.0)/(2 * diff_bern0) * \
                #         torch.square(loss_ent_bern - target_entropy_bern_tensor)
                #     , 0.0)
                # )
                # 3. 组合最终 Actor Loss
                # [修改] 移除 bern_constraint_term（熵约束），由 mini-batch 内的 FRR 替代
                actor_loss = actor_loss + cat_constraint_term - (k_cont * loss_ent_cont)
                
                # 原有非目标熵正则项
                # actor_loss = actor_loss - (k_cont * loss_ent_cont + k_cat * loss_ent_cat + k_bern * loss_ent_bern)

                # # =====================================================================
                # # [修改/重构] Bernoulli 开火头的专属正则化、稀疏惩罚与冷却约束
                # # =====================================================================
                if actor_outputs['bern'] is not None:
                    bern_logits = actor_outputs['bern']
                    bern_probs = torch.sigmoid(bern_logits)
                    
                    # 1. 基础 Logit 越界惩罚 (保持 Logit 在可激活区间)
                    over = F.relu(torch.abs(bern_logits) - bern_max_logits) # 4.0  20步至少打一发，那么开火概率至少也是0.05，对应-3的logits
                    # 只惩罚允许开火的位置
                    if 'fire_mask' in actor_outputs and actor_outputs['fire_mask'] is not None:
                        fire_mask = actor_outputs['fire_mask']  # shape: (batch, bern_dim)
                    else:
                        fire_mask = (bern_logits > -1e6).float()
                    # 只对允许开火且存活的样本求平均，避免被 non-fireable 样本稀释
                    fireable_sum = (mb_active_masks * bern_fire_mask).sum()
                    logit_loss = ((over ** 2) * fire_mask * mb_active_masks).sum() / (fireable_sum + mask_eps)
                    actor_loss = actor_loss + alpha_logit_reg * logit_loss
                    
                    # [新增] 记录未被mask的bern触发概率极值
                    valid_fire_mask = (fire_mask * mb_active_masks).bool()
                    if valid_fire_mask.any():
                        valid_probs = bern_probs[valid_fire_mask]
                        max_fire_prob_list.append(valid_probs.max().item())
                        min_fire_prob_list.append(valid_probs.min().item())


                    # [新增] 3. 单阈值均值约束 (FRR) —— 替代 bern 熵正则项
                    # 约束允许开火样本的平均概率不超过 p_target，用 pseudo-Huber loss
                    # 保护弹药节奏：防止中远距离手痒乱射导致后期无弹可用
                    fire_mask_1d = fire_mask[:, 0] if fire_mask.dim() > 1 else fire_mask.squeeze(-1)
                    active_1d = mb_active_masks.squeeze(-1)
                    fireable_frr = (fire_mask_1d > 0.5) & (active_1d > 0.5)
                    if fireable_frr.sum() > 0:
                        p_mean_fireable = bern_probs[:, 0][fireable_frr].mean()
                        p_target = 0.1  # 目标平均开火概率 20%，对应 6 枚 / 30 个有效开火步
                        delta = F.relu(p_mean_fireable - p_target)
                        # pseudo-Huber loss: sqrt(1 + delta^2) - 1，平缓惩罚，对异常值不爆炸
                        frr_loss = torch.sqrt(1.0 + delta * delta) - 1.0
                        # 复用原熵系数 k_bern 作为强度（建议 0.001~0.005）
                        actor_loss = actor_loss + k_bern * frr_loss

                # Critic Loss
                # Critic 使用 critic_inputs
                v_pred = self.critic(mb_critic_inputs)
                #  reduction='none' 使得我们可以应用 mask
                critic_loss_per_sample = F.mse_loss(v_pred, mb_td_target, reduction='none')
                
                critic_loss = critic_loss_per_sample.mean()
                
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                # if alpha_loss.requires_grad:    # [新增]
                #     self.k_cat_optim.zero_grad()
                if actor_frozen:
                    self.reset_optimizer(reset_type='actor')

                if not actor_frozen:
                    actor_loss.backward()
                critic_loss.backward()
                # if alpha_loss.requires_grad:    # [新增] 反向传播温度损失
                #     alpha_loss.backward()

                pre_clip_actor_grad.append(model_grad_norm(self.actor))
                pre_clip_critic_grad.append(model_grad_norm(self.critic)) 
                
                # [新增] 计算范数比 (Pre-clip)
                grad_norm_ratio_list.append(pre_clip_actor_grad[-1] / (pre_clip_critic_grad[-1] + 1e-8))

                nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.actor_max_grad)
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.critic_max_grad)
                if not actor_frozen:
                    self.actor_optimizer.step()
                self.critic_optimizer.step()
                # if alpha_loss.requires_grad:    # [新增] 步进自适应熵系数
                #     self.k_cat_optim.step()

                # Logging
                actor_grad_list.append(model_grad_norm(self.actor))
                critic_grad_list.append(model_grad_norm(self.critic))            
                actor_loss_list.append(actor_loss.item())
                critic_loss_list.append(critic_loss.item())
                
                # 记录总 Entropy (未加权)
                entropy_total = (entropy * mb_active_masks).sum() / (active_sum + mask_eps)
                entropy_list.append(entropy_total.item()) 
                
                #  [修改] 记录分项 Entropy (现在是 Tensor，需要 .mean().item())
                if entropy_details['cont'] is not None:
                    entropy_cont_list.append(entropy_details['cont'].mean().item())
                if entropy_details['cat'] is not None:
                    entropy_cat_list.append(entropy_details['cat'].mean().item())
                if entropy_details['bern'] is not None:
                    # [修改] 使用 fire_mask 过滤，只记录 fireable 样本的平均熵
                    if 'fire_mask' in actor_outputs and actor_outputs['fire_mask'] is not None:
                        bern_fire_mask = actor_outputs['fire_mask']  # shape: (batch, bern_dim)
                        if bern_fire_mask.dim() > 1:
                            bern_fire_mask = bern_fire_mask[:, 0:1]  # 对齐到 (batch, 1)
                        fireable_sum = (mb_active_masks * bern_fire_mask).sum()
                        bern_entropy_filtered = (entropy_details['bern'] * mb_active_masks * bern_fire_mask).sum() / (fireable_sum + mask_eps)
                        entropy_bern_list.append(bern_entropy_filtered.item())
                    else:
                        entropy_bern_list.append(entropy_details['bern'].mean().item())

        self.actor_loss = np.mean(actor_loss_list)
        self.actor_grad = np.mean(actor_grad_list)
        self.critic_loss = np.mean(critic_loss_list)
        self.critic_grad = np.mean(critic_grad_list)
        self.entropy_mean = np.mean(entropy_list)
        self.pre_clip_critic_grad = np.mean(pre_clip_critic_grad)
        self.pre_clip_actor_grad = np.mean(pre_clip_actor_grad)
        
        #  记录 active 的 advantage 均值
        active_sum_total = active_masks.sum().item()
        if active_sum_total > 0:
            self.advantage = (advantage.abs() * active_masks).sum().item() / active_sum_total
        else:
            self.advantage = 0
        
        #  汇总新指标
        self.approx_kl = np.mean(kl_list)
        #  计算分项 Entropy 均值
        self.entropy_cont = np.mean(entropy_cont_list) if len(entropy_cont_list) > 0 else 0
        self.entropy_cat = np.mean(entropy_cat_list) if len(entropy_cat_list) > 0 else 0
        self.entropy_bern = np.mean(entropy_bern_list) if len(entropy_bern_list) > 0 else 0
        
        # [新增] 赋值有效样本监控项
        self.PPO_samples = ppo_samples_total
        self.PPO_valid_samples = ppo_valid_samples_total
        
        # [新增] 汇总未被mask的bern触发概率极值
        self.max_fire_prob = np.mean(max_fire_prob_list) if len(max_fire_prob_list) > 0 else 0
        self.min_fire_prob = np.mean(min_fire_prob_list) if len(min_fire_prob_list) > 0 else 0


        #  计算 Explained Variance
        # y_true: td_target, y_pred: v_pred_old (更新前的值) 或 v_pred (更新后的值，通常用更新前比较多，或者直接对比)
        # 这里使用 numpy 计算以防 tensor 维度广播问题
        #  explained_var 最好也只看 active 的，但为了简单起见，这里先保持原样或简单过滤
        mask_bool = active_masks.squeeze(-1).bool().cpu().numpy()
        y_true = td_target.flatten().cpu().numpy()[mask_bool]
        y_pred = v_pred_old.flatten().cpu().numpy()[mask_bool] # 比较更新前的 Value 网络预测能力
        
        # if len(y_true) > 1:
        #     var_y = np.var(y_true)
        #     if var_y == 0:
        #         self.explained_var = np.nan
        #     else:
        #         self.explained_var = 1 - np.var(y_true - y_pred) / var_y
        # else:
        #     self.explained_var = 0

        if len(y_true) > 1:
            var_y = np.var(y_true)
            self.td_error_var = np.var(y_true - y_pred) # [新增] TD error 方差
            if var_y < 1e-8:
                self.explained_var = 0.0
            else:
                self.explained_var = 1 - self.td_error_var / var_y
        else:
            self.td_error_var = 0.0
            self.explained_var = 0.0

        check_weights_bias_nan(self.actor, "actor", "update后")
        check_weights_bias_nan(self.critic, "critic", "update后")

    # =========================================================================
    #  [New Method] Bernoulli 开火头保护性有监督训练 (防止机动策略被bern崩溃拖累)
    # =========================================================================
    def fire_prob_protection(self, transition_dict, protect_epochs=4, protect_mini_batch=256, mask_on=0):
        """
        Bern头概率范围保护器。当开火概率整体崩溃（全高或全低）时，以有监督方式
        强行拉回bern头分布，同时切断backbone和其它动作头的梯度，保护机动策略不被拖垮。

        必要条件1 (比值护栏): max_fire_prob / min_fire_prob >= 10，说明分布仍有分化空间，
                               不需要干预，直接跳过。
        必要条件2 (触发case):
          case1: max_fire_prob < 0.05  → 整体开火概率崩到极低，以0.5为监督信号，拉高熵。
          case2: min_fire_prob > 0.1   → 整体开火概率过高，以1e-3为监督信号，压低概率。

        Args:
            transition_dict : 与update()相同格式的经验字典。
            protect_epochs  : 保护性训练的epoch数。
            protect_mini_batch: 每个mini-batch的大小。
            mask_on         : 传给net forward的mask开关，与update保持一致。
        """
        # ── 必要条件1：比值护栏 ──────────────────────────────────────────────────
        ratio = self.max_fire_prob / (self.min_fire_prob + 1e-12)
        if ratio >= 10.0:
            return  # 分布仍有足够分化，不需要干预

        # ── 必要条件2：判断触发case ──────────────────────────────────────────────
        if self.max_fire_prob < 0.05:
            # case1: 概率塌缩到接近0 → 用0.5拉高熵
            target_prob = 0.5
        elif self.min_fire_prob > 0.1:
            # case2: 概率整体过高 → 用1e-3压低
            target_prob = 1e-3
        else:
            return  # 不满足任何触发条件

        # ── 数据准备（复用update的转换逻辑）────────────────────────────────────
        def to_tensor(x, dtype):
            if isinstance(x, np.ndarray):
                return torch.tensor(x, dtype=dtype).to(self.device)
            else:
                return torch.tensor(np.array(x), dtype=dtype).to(self.device)

        if 'obs' in transition_dict:
            actor_inputs = to_tensor(transition_dict['obs'], torch.float)
        else:
            actor_inputs = to_tensor(transition_dict['states'], torch.float)

        if 'active_masks' in transition_dict:
            active_masks = to_tensor(transition_dict['active_masks'], torch.float).view(-1, 1)
        else:
            active_masks = torch.ones(actor_inputs.size(0), 1, device=self.device)

        num_samples = actor_inputs.size(0)
        mb_size = min(protect_mini_batch, num_samples)

        # ── 冻结除bern头以外的所有actor模块 ───────────────────────────────────
        net = self.actor.net  # PolicyNetHybrid 实例

        def set_requires_grad(module_or_param, flag):
            if isinstance(module_or_param, nn.Module):
                for p in module_or_param.parameters():
                    p.requires_grad_(flag)
            else:
                module_or_param.requires_grad_(flag)

        # 逐模块冻结（backbone + 其它动作头）
        set_requires_grad(net.net, False)  # 共享backbone
        if hasattr(net, 'fc_mu'):
            set_requires_grad(net.fc_mu, False)
        if hasattr(net, 'log_std_cont'):
            set_requires_grad(net.log_std_cont, False)
        if hasattr(net, 'fc_cat'):
            set_requires_grad(net.fc_cat, False)
        # bern头保持可训练
        if hasattr(net, 'fc_bern'):
            set_requires_grad(net.fc_bern, True)

        # ── 监督训练循环 ──────────────────────────────────────────────────────
        target_tensor = torch.tensor(target_prob, device=self.device)

        for _ in range(protect_epochs):
            perm = torch.randperm(num_samples, device=self.device)
            for start in range(0, num_samples, mb_size):
                end = min(start + mb_size, num_samples)
                batch_idx = perm[start:end]

                mb_states = actor_inputs[batch_idx]
                mb_active = active_masks[batch_idx]  # (mb, 1)
                active_sum = mb_active.sum()

                actor_out = self.actor.net(mb_states, mask_on=mask_on)

                if actor_out['bern'] is None:
                    break

                bern_logits = actor_out['bern'].clamp(min=-1e8)
                bern_probs = torch.sigmoid(bern_logits)  # (mb, bern_dim)

                # 监督目标：将每个位置的概率拉向target_prob
                # 使用BCE loss，target广播到与bern_probs相同形状
                target_full = target_tensor.expand_as(bern_probs)
                bern_loss_per_sample = F.binary_cross_entropy(
                    bern_probs, target_full, reduction='none'
                ).sum(dim=-1, keepdim=True)  # (mb, 1)

                mask_eps_loc = 1e-5
                bern_loss = (bern_loss_per_sample * mb_active).sum() / (active_sum + mask_eps_loc)

                self.actor_optimizer.zero_grad()
                bern_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.actor_max_grad)
                self.actor_optimizer.step()

        # ── 恢复所有actor模块的梯度反向传播 ──────────────────────────────────
        set_requires_grad(net.net, True)
        if hasattr(net, 'fc_mu'):
            set_requires_grad(net.fc_mu, True)
        if hasattr(net, 'log_std_cont'):
            set_requires_grad(net.log_std_cont, True)
        if hasattr(net, 'fc_cat'):
            set_requires_grad(net.fc_cat, True)
        if hasattr(net, 'fc_bern'):
            set_requires_grad(net.fc_bern, True)

        return

    # =========================================================================
    #  [New Method] 将平铺的 transition_dict 重排为 (num_seqs, seq_len, ...) 形状
    # =========================================================================
    def reshape_for_rnn(self, transition_dict, seq_len):
        """
        将按回合顺序排列的平铺 transition_dict 切分为固定长度序列，供 GRU 网络使用。

        切分规则（与 HybridBuffer_rnn 保持一致）：
          - 以 dones == 1 作为回合边界，逐回合处理。
          - 每个回合内从后往前倒序切块，每块长度为 seq_len；头部余数丢弃。
          - 回合步数 < seq_len 的整个回合丢弃。

        Args:
            transition_dict: dict，键包含
                'states'      : (N, S)
                'next_states' : (N, S)   保留，用于现场计算 V(s')
                'rewards'     : (N,) 或 (N,1)
                'dones'       : (N,) 或 (N,1)
                'actions'     : dict of arrays, 每个值形状 (N, ...)
                可选: 'obs' (N, O), 'active_masks' (N,) 或 (N,1)
            seq_len: int，每条序列的长度

        Returns:
            dict，各字段形状为 (num_seqs, seq_len, ...) 的 numpy 数组；
            'actions' 仍为 dict，每个值形状 (num_seqs, seq_len, ...)。
            若无任何有效序列则抛出 RuntimeError。
        """
        def _to_np(x):
            if isinstance(x, torch.Tensor):
                return x.cpu().numpy()
            return np.array(x)

        states      = _to_np(transition_dict['states'])       # (N, S)
        next_states = _to_np(transition_dict['next_states'])  # (N, S)
        rewards     = _to_np(transition_dict['rewards']).reshape(-1)   # (N,)
        dones       = _to_np(transition_dict['dones']).reshape(-1)     # (N,)

        has_obs          = 'obs' in transition_dict
        has_active_masks = 'active_masks' in transition_dict
        obs          = _to_np(transition_dict['obs'])          if has_obs          else None
        active_masks = _to_np(transition_dict['active_masks']).reshape(-1) if has_active_masks else None

        actions_np = {}
        for k, v in transition_dict['actions'].items():
            actions_np[k] = _to_np(v)  # (N, ...)

        N = dones.shape[0]
        S = states.shape[-1]

        # --- 识别回合边界，倒序切块 ---
        done_indices = list(np.where(dones == 1)[0])
        ep_ends = done_indices[:]
        if len(ep_ends) == 0 or ep_ends[-1] != N - 1:
            ep_ends.append(N - 1)

        valid_seqs = []  # list of (ep_start, block_start) — both absolute indices
        curr_start = 0
        for ep_end in ep_ends:
            ep_len = ep_end - curr_start + 1
            if ep_len < seq_len:
                curr_start = ep_end + 1
                continue
            # 从 ep_end 向前每 seq_len 步切一块
            for block_end in range(ep_end, curr_start + seq_len - 2, -seq_len):
                block_start = block_end - seq_len + 1
                valid_seqs.append((block_start, block_end))
            curr_start = ep_end + 1

        if not valid_seqs:
            raise RuntimeError(
                f"reshape_for_rnn: 没有找到长度 >= {seq_len} 的回合，无法构建序列批次。"
            )

        num_seqs = len(valid_seqs)

        # --- 预分配输出容器 ---
        out = {
            'states':      np.zeros((num_seqs, seq_len, S),           dtype=np.float32),
            'next_states': np.zeros((num_seqs, seq_len, S),           dtype=np.float32),
            'rewards':     np.zeros((num_seqs, seq_len),               dtype=np.float32),
            'dones':       np.zeros((num_seqs, seq_len),               dtype=np.float32),
            'actions':     {},
        }
        if has_obs:
            O = obs.shape[-1]
            out['obs'] = np.zeros((num_seqs, seq_len, O), dtype=np.float32)
        if has_active_masks:
            out['active_masks'] = np.zeros((num_seqs, seq_len), dtype=np.float32)

        for k, v in actions_np.items():
            act_shape = v.shape[1:]  # e.g. (cont_dim,) or (n_cat_heads,) or (bern_dim,)
            dtype = np.int64 if k == 'cat' else np.float32
            out['actions'][k] = np.zeros((num_seqs, seq_len) + act_shape, dtype=dtype)

        # --- 填充 ---
        for i, (s, e) in enumerate(valid_seqs):
            out['states'][i]      = states[s:e+1]
            out['next_states'][i] = next_states[s:e+1]
            out['rewards'][i]     = rewards[s:e+1]
            out['dones'][i]       = dones[s:e+1]
            if has_obs:
                out['obs'][i]          = obs[s:e+1]
            if has_active_masks:
                out['active_masks'][i] = active_masks[s:e+1]
            for k in actions_np:
                out['actions'][k][i]   = actions_np[k][s:e+1]

        return out

    # =========================================================================
    #  [New Helper] 提取出的 MSE 计算逻辑 (供 mixed_update 和 BC_update 复用)
    # =========================================================================
    "离散空间的动作损失函数不对，暂不使用该函数"
    def _compute_mse_loss_with_f(self, actor_input_batch, actions_batch, returns_batch, 
                                 critic_s_batch, # 用于计算 V
                                 max_weight=100.0, use_F=True):
        """
        计算基于 F 函数 (Advantage > 0) 门控的 MSE Loss。
        """
        # 1. 计算 F 函数 (优势加权门控)
        with torch.no_grad():
            # Critic 计算 V 值
            # 注意：如果 actor_input 和 critic_input 不一样 (如 obs vs state)，这里需要传入正确的 tensor
            # 在本类中，通常传入的是用于 Actor 的输入，如果 Actor 用 Obs，Critic 用 State，
            # 外部调用时需确保 critic_s_batch 是正确的全局 State 或与 Actor 输入一致(视具体配置)
            # 这里为了通用性，假设 critic_s_batch 是正确的 Critic 输入
            v_pred = self.critic(critic_s_batch)
            
            # 计算优势
            # 确保 returns_batch 是 (Batch, 1)
            adv = (returns_batch - v_pred) / (torch.sqrt(self.c_sq) + 1e-8)

            if use_F:
                # F 函数：只有 Adv > 0 的样本权重为 1，否则为极小值
                F_mask = torch.where(adv > 0, torch.ones_like(adv), torch.full_like(adv, 1e-6))
                F_mask = torch.clamp(F_mask, max=max_weight)
            else:
                F_mask = torch.ones_like(adv)

        # 2. 获取网络原始输出
        outputs = self.actor.net(actor_input_batch)
        
        # 初始化 Loss Sum (Batch, 1)
        mse_loss_sum = torch.zeros_like(adv)

        # --- Continuous MSE ---
        if 'cont' in self.actor.action_dims and self.actor.action_dims['cont'] > 0:
            mu_current, _ = outputs['cont']
            u_expert = actions_batch['cont']
            mse_loss_sum += F.mse_loss(mu_current, u_expert, reduction='none').sum(dim=-1, keepdim=True)

        # --- Categorical MSE ---
        if 'cat' in self.actor.action_dims and sum(self.actor.action_dims['cat']) > 0:
            cat_probs_current = outputs['cat']
            expert_cat = actions_batch['cat'].long()
            for i, probs in enumerate(cat_probs_current):
                target_one_hot = F.one_hot(expert_cat[:, i], num_classes=probs.size(-1)).float()
                mse_loss_sum += F.mse_loss(probs, target_one_hot, reduction='none').sum(dim=-1, keepdim=True)

        # --- Bernoulli MSE ---
        if 'bern' in self.actor.action_dims and self.actor.action_dims['bern'] > 0:
            bern_logits = outputs['bern']
            bern_probs = torch.sigmoid(bern_logits)
            target_bern = actions_batch['bern']
            mse_loss_sum += F.mse_loss(bern_probs, target_bern, reduction='none').sum(dim=-1, keepdim=True)

        # 3. 应用 FMask 并求平均
        # F_mask: [Batch, 1], mse_loss_sum: [Batch, 1]
        if use_F:
            final_loss = torch.mean(F_mask * mse_loss_sum)
        else:
            final_loss = torch.mean(mse_loss_sum)
        
        return final_loss, F_mask

    
    # =========================================================================
    #  [New Method] BC_update (Critic 同 MARWIL, Actor 用 MSE+F)
    # =========================================================================
    def BC_update(self, il_transition_dict, batch_size=64, c_v=1.0, shuffled=1, max_weight=100.0):
        """
        行为克隆更新 (Behavior Cloning with F-function Constraint)。
        Critic: 使用 MARWIL 风格的回归更新 (拟合 R)。
        Actor: 使用 MSE 回归，但仅在 Advantage > 0 时通过 F 函数生效。
        """
        # 1. 数据准备
        # 可能的局部观测
        if 'obs' in il_transition_dict and len(il_transition_dict['obs']) > 0:
            obs_all = torch.tensor(np.array(il_transition_dict['obs']), dtype=torch.float).to(self.device)
            use_obs = True
        else:
            use_obs = False
            
        # 冻结分布参数，只训练均值/Logits
        if hasattr(self.actor.net, 'log_std_cont'):
            self.actor.net.log_std_cont.requires_grad = False

        states_all = torch.tensor(np.array(il_transition_dict['states']), dtype=torch.float).to(self.device)
        returns_all = torch.tensor(np.array(il_transition_dict['returns']), dtype=torch.float).view(-1, 1).to(self.device)
        
        # 处理 Actions
        raw_actions = il_transition_dict['actions']
        actions_all = {}
        if isinstance(raw_actions, list):
            keys = raw_actions[0].keys()
            temp_dict = {}
            for k in keys:
                temp_dict[k] = np.stack([d[k] for d in raw_actions], axis=0)
            raw_actions = temp_dict
        if isinstance(raw_actions, dict):
            for k, v in raw_actions.items():
                if k == 'cat':
                    actions_all[k] = torch.tensor(v, dtype=torch.long).to(self.device)
                else:
                    actions_all[k] = torch.tensor(v, dtype=torch.float).to(self.device)

        # 2. 索引准备
        total_size = states_all.size(0)
        indices = np.arange(total_size)
        if shuffled:
            np.random.shuffle(indices)

        total_actor_loss = 0
        total_critic_loss = 0
        total_valid_samples = 0
        batch_count = 0

        # 初始化 c_sq 用于 Advantage 归一化 (如果不存在)
        if not hasattr(self, 'c_sq'): 
            self.c_sq = torch.tensor(1.0, device=self.device)

        # 3. Mini-batch 循环
        for start in range(0, total_size, batch_size):
            end = min(start + batch_size, total_size)
            batch_indices = indices[start:end]
            
            s_batch = states_all[batch_indices] 
            r_batch = returns_all[batch_indices]
            
            if use_obs:
                actor_input_batch = obs_all[batch_indices]
            else:
                actor_input_batch = s_batch
            
            # Critic input (通常 state)
            critic_input_batch = s_batch

            actions_batch = {}
            for k, v in actions_all.items():
                actions_batch[k] = v[batch_indices]

            # --- A. Actor Loss (MSE + F) ---
            # 使用 helper 函数计算
            actor_loss, F_mask = self._compute_mse_loss_with_f(
                actor_input_batch, actions_batch, r_batch, critic_input_batch, max_weight, use_F=0
            )

            # --- B. Critic Loss (同 MARWIL) ---
            v_pred = self.critic(critic_input_batch)
            critic_loss = F.mse_loss(v_pred, r_batch) * c_v
            
            # --- C. Optimize ---
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            
            actor_loss.backward()
            critic_loss.backward()
            
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.actor_max_grad)
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.critic_max_grad)
            
            self.actor_optimizer.step()
            self.critic_optimizer.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_valid_samples += F_mask.sum().item()
            batch_count += 1

        avg_actor_loss = total_actor_loss / batch_count if batch_count > 0 else 0
        avg_critic_loss = total_critic_loss / batch_count if batch_count > 0 else 0
        
        return avg_actor_loss, avg_critic_loss, total_valid_samples
    
    
    
    # --- 修改后的 MARWIL_update， 注意原先是0 ---
    def MARWIL_update(self, il_transition_dict, beta=1.0, batch_size=64, alpha=1.0, c_v=1.0, shuffled=1, label_smoothing=0.3, max_weight=100.0,
                      tau=0.8, action_heads_mask=None, no_bern=None, no_cat=None):
        """
        MARWIL 离线更新函数
        输入 actions 结构支持: [{'cat': array([v]), 'bern': array([v])}, ...]
        tau: 非对称损失权重 (Expectile Regression). tau=0.5 为 MSE; tau>0.5 (如0.9) 倾向于高估 Value (拟合好样本)
        action_heads_mask: dict, 例如 {'cont': True, 'cat': True, 'bern': False}
                           指定哪些动作头参与模仿学习 Loss 计算。
                           默认不训练 bern 头，保持与旧版 no_bern=1 一致。
                           为兼容旧代码，仍保留 no_bern/no_cat，但它们会被映射为 mask。
        """
        # 1. 数据准备
        if 'obs' in il_transition_dict and len(il_transition_dict['obs']) > 0:
            obs_all = torch.tensor(np.array(il_transition_dict['obs']), dtype=torch.float).to(self.device)
            use_obs = True
        else:
            use_obs = False
            
        # 预训练阶段通常不训练探索 std
        if hasattr(self.actor.net, 'log_std_cont'):
            self.actor.net.log_std_cont.requires_grad = False

        # 1. 提取全量数据并转为 Tensor
        states_all = torch.tensor(np.array(il_transition_dict['states']), dtype=torch.float).to(self.device)
        returns_all = torch.tensor(np.array(il_transition_dict['returns']), dtype=torch.float).view(-1, 1).to(self.device)
        
        # 统一处理 Actions：List of Dicts -> Dict of Tensors
        raw_actions = il_transition_dict['actions']
        actions_all = {}
        
        # 1. 如果是列表 (List of Dicts)，先堆叠成 Dict of Numpy Arrays
        if isinstance(raw_actions, list):
            keys = raw_actions[0].keys()
            temp_dict = {}
            for k in keys:
                # np.stack 会把 [array([1]), array([2])] 变成 array([[1], [2]]) -> (N, 1)
                temp_dict[k] = np.stack([d[k] for d in raw_actions], axis=0)
            raw_actions = temp_dict # 现在变成了 Dict of Arrays

        # 2. Dict of Arrays -> Dict of Tensors
        if isinstance(raw_actions, dict):
            for k, v in raw_actions.items():
                if k == 'cat':
                    actions_all[k] = torch.tensor(v, dtype=torch.long).to(self.device)
                else:
                    actions_all[k] = torch.tensor(v, dtype=torch.float).to(self.device)
        # ============================================================

        # 2. 准备 Batch 索引
        total_size = states_all.size(0)
        indices = np.arange(total_size)
        if shuffled:
            np.random.shuffle(indices)

        total_actor_loss = 0
        total_critic_loss = 0
        total_c = 0
        batch_count = 0

        # [新增] 权重与 advantage 监控累加器
        total_weight_mean = 0.0
        total_weight_max = 0.0
        total_weight_min = 0.0
        total_clip_frac = 0.0
        total_adv_std = 0.0
        total_adv_p95 = 0.0
        total_adv_max = 0.0
        total_adv_mean = 0.0

        # 3. Mini-batch 循环
        for start in range(0, total_size, batch_size):
            end = min(start + batch_size, total_size)
            batch_indices = indices[start:end]
            
            s_batch = states_all[batch_indices] 
            r_batch = returns_all[batch_indices]
            
            if use_obs:
                actor_input_batch = obs_all[batch_indices]
            else:
                actor_input_batch = s_batch 
            
            # 动作字典切片
            actions_batch = {}
            for k, v in actions_all.items():
                actions_batch[k] = v[batch_indices]

            # A. Advantage & Weights
            with torch.no_grad():
                values = self.critic(s_batch)
                residual = r_batch - values
                
                if not hasattr(self, 'c_sq'): 
                    self.c_sq = torch.tensor(1.0, device=self.device)
                
                batch_mse = (residual ** 2).mean().item()
                self.c_sq = self.c_sq + 1e-8 * (batch_mse - self.c_sq)
                c = torch.sqrt(self.c_sq)
                
                advantage = residual / (c + 1e-8)
                raw_weights = torch.exp(beta * advantage)
                weights = torch.clamp(raw_weights, max=max_weight)

                # [新增] 记录本 batch 的权重统计
                total_weight_mean += weights.mean().item()
                total_weight_max += weights.max().item()
                total_weight_min += weights.min().item()
                total_clip_frac += (weights >= max_weight - 1e-6).float().mean().item()

                # [新增] 记录本 batch advantage 分布统计
                adv = advantage.detach()
                total_adv_std += adv.std().item()
                total_adv_p95 += torch.quantile(adv, 0.95).item()
                total_adv_max += adv.max().item()
                total_adv_mean += adv.mean().item()

            # 解析动作头mask；兼容旧版 no_bern/no_cat
            if action_heads_mask is None:
                action_heads_mask = {'cont': True, 'cat': True, 'bern': False}
                if no_bern is not None:
                    action_heads_mask['bern'] = not no_bern
                if no_cat is not None:
                    action_heads_mask['cat'] = not no_cat
            
            # B. Actor Loss
            raw_il_loss = self.actor.compute_il_loss(
                actor_input_batch,
                actions_batch,
                label_smoothing,
                action_heads_mask=action_heads_mask,
                good_samples=1,
                pre_training=1,
            )
            actor_loss = torch.mean(alpha * weights * raw_il_loss)

            # C. Critic Loss
            v_pred = self.critic(s_batch)
            
            # 原有
            critic_loss = F.mse_loss(v_pred, r_batch) * c_v
            


            # D. Optimize
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.actor_max_grad)
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.critic_max_grad)
            self.actor_optimizer.step()
            self.critic_optimizer.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_c += c.item()
            batch_count += 1

        avg_weight_mean = total_weight_mean / batch_count if batch_count > 0 else 0
        avg_weight_max = total_weight_max / batch_count if batch_count > 0 else 0
        avg_weight_min = total_weight_min / batch_count if batch_count > 0 else 0
        avg_clip_frac = total_clip_frac / batch_count if batch_count > 0 else 0
        avg_adv_std = total_adv_std / batch_count if batch_count > 0 else 0
        avg_adv_p95 = total_adv_p95 / batch_count if batch_count > 0 else 0
        avg_adv_max = total_adv_max / batch_count if batch_count > 0 else 0
        avg_adv_mean = total_adv_mean / batch_count if batch_count > 0 else 0

        avg_actor_loss = total_actor_loss / batch_count if batch_count > 0 else 0
        avg_critic_loss = total_critic_loss / batch_count if batch_count > 0 else 0
        avg_c = total_c / batch_count if batch_count > 0 else 0

        # ============================================================
        # [新增] 监控：在固定的全量 batch 上独立统计每个动作头的 NLL 与熵
        # 全程 no_grad，不参与反传，因此不会干扰现有的网络更新。
        # ============================================================
        if use_obs:
            monitor_input_all = obs_all
        else:
            monitor_input_all = states_all
        with torch.no_grad():
            values_all = self.critic(states_all)
            residual_all = returns_all - values_all
            advantage_all = residual_all / (torch.sqrt(self.c_sq) + 1e-8)
        monitor_metrics = self.actor.compute_marwil_monitor(monitor_input_all, actions_all, advantages=advantage_all)
        # 缓存到 agent 属性，便于训练脚本拉取写入 logger
        self.marwil_nll_cont = monitor_metrics['nll_cont']
        self.marwil_nll_cat = monitor_metrics['nll_cat']
        self.marwil_nll_bern = monitor_metrics['nll_bern']
        self.marwil_entropy_cont = monitor_metrics['entropy_cont']
        self.marwil_entropy_cat = monitor_metrics['entropy_cat']
        self.marwil_entropy_bern = monitor_metrics['entropy_bern']
        self.marwil_accuracy_cont = monitor_metrics['accuracy_cont']
        self.marwil_accuracy_cat = monitor_metrics['accuracy_cat']
        self.marwil_accuracy_bern = monitor_metrics['accuracy_bern']
        self.marwil_weight_mean = avg_weight_mean
        self.marwil_weight_max = avg_weight_max
        self.marwil_weight_min = avg_weight_min
        self.marwil_weight_clip_frac = avg_clip_frac
        self.marwil_adv_std = avg_adv_std
        self.marwil_adv_p95 = avg_adv_p95
        self.marwil_adv_max = avg_adv_max
        self.marwil_adv_mean = avg_adv_mean
        self.marwil_adv_positive_frac = monitor_metrics['adv_positive_frac']

        return avg_actor_loss, avg_critic_loss, avg_c
    
    def Stir(self, transition_dict, target_entropies, max_steps=50, lr=0.05):
        """
        策略搅拌：深拷贝网络并使用纯熵正则项更新，将策略分布推平到目标熵，
        同时尽可能保持原有动作的相对优先级排序不变。
        
        Args:
            transition_dict: 同 update 方法的输入，用于提供计算图所需的 states (和 mask)
            target_entropies: dict, 例如 {'cont': 2.0, 'cat': 1.5, 'bern': 0.6}
            max_steps: 最大允许的搅拌迭代次数
            lr: SGD 学习率 (不宜过大，否则容易导致离散动作排序翻转)
            
        Returns:
            stirred_state_dict: 搅拌后的独立网络参数字典，与原计算图解耦
        """
        # 1. 深度拷贝网络，彻底解耦计算图
        stirred_net = copy.deepcopy(self.actor)
        stirred_net.net.train() # 确保开启梯度计算

        # 2. 冻结特征提取层，仅放开动作头和 std 参数
        # 这是为了最大程度保护原始动作提取出的特征表达，防止由于大跨度更新彻底破坏策略逻辑
        for name, param in stirred_net.net.named_parameters():
            if 'fc_' in name or 'log_std' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # 3. 必须使用无动量的 SGD！
        # 如果使用 Adam，动量积攒会导致 Logits 大幅超调，直接破坏原有的动作优先级排序
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, stirred_net.net.parameters()), lr=lr)

        # 4. 解析输入状态 (复用 update 的接口逻辑)
        if isinstance(transition_dict['states'], np.ndarray):
            states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        else:
            states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        

        # 5. 执行搅拌循环
        for step in range(max_steps):
            outputs = stirred_net.net(states)
            
            loss = 0.0
            current_entropies = {}

            # --- Cont ---
            if 'cont' in self.actor.action_dims and self.actor.action_dims['cont'] > 0:
                mu, std = outputs['cont']
                dist = SquashedNormal(mu, std)
                # 平均到每个样本上
                ent_cont = dist.entropy().mean()
                current_entropies['cont'] = ent_cont.item()
                # 如果当前熵低于目标值，则加上负熵作为 Loss (最小化负熵 = 最大化熵)
                if ent_cont < target_entropies.get('cont', -float('inf')):
                    loss -= ent_cont 

            # --- Cat ---
            if 'cat' in self.actor.action_dims and sum(self.actor.action_dims['cat']) > 0:
                cat_probs_list = outputs['cat']
                ent_cat = 0.0
                for probs in cat_probs_list:
                    dist = Categorical(probs=probs)
                    ent_cat += dist.entropy().mean()
                current_entropies['cat'] = ent_cat.item()
                if ent_cat < target_entropies.get('cat', -float('inf')):
                    loss -= ent_cat

            # --- Bern ---
            # 跳过 Bernoulli 部分的搅拌，避免 mask_on 相关的复杂性

            # 6. 如果没有需要优化的项，直接退出
            if isinstance(loss, float):
                break

            # 7. 反向传播与搅拌更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 8. 更新后检查退出条件：所有指定的动作头都达到了目标熵则停止
            # 注意：必须在 step() 之后检查，避免更新后过冲
            all_met = True
            for key, target in target_entropies.items():
                if current_entropies.get(key, float('inf')) < target:
                    all_met = False
                    break
            if all_met:
                break

        # 8. 计算搅拌后的熵值
        with torch.no_grad():
            # 使用输入的状态计算熵值
            if isinstance(transition_dict['states'], np.ndarray):
                states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
            else:
                states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
            
            outputs = stirred_net.net(states)
            
            # 计算cat熵
            cat_entropy = 0.0
            if 'cat' in self.actor.action_dims and sum(self.actor.action_dims['cat']) > 0:
                cat_probs_list = outputs['cat']
                for probs in cat_probs_list:
                    dist = Categorical(probs=probs)
                    cat_entropy += dist.entropy().mean()
                cat_entropy = cat_entropy.item()
            
            # Bern 熵不计算（搅拌时已跳过）
            bern_entropy = 0.0
            
            # 计算cont熵
            cont_entropy = 0.0
            if 'cont' in self.actor.action_dims and self.actor.action_dims['cont'] > 0:
                mu, std = outputs['cont']
                dist = SquashedNormal(mu, std)
                cont_entropy = dist.entropy().mean().item()
        
        # 9. 转换为 CPU 张量并 clone，彻底切断与本次优化过程及 GPU 的羁绊
        # stirred_net 本身是 HybridActorWrapper 的深拷贝，直接取其 state_dict 即可与外部保存/加载保持一致
        stirred_state_dict = {k: v.cpu().clone() for k, v in stirred_net.state_dict().items()}
        
        # 返回搅拌后的状态字典和熵值信息
        entropy_info = {
            'cat_entropy': cat_entropy,
            'bern_entropy': bern_entropy,
            'cont_entropy': cont_entropy
        }
        
        return stirred_state_dict, entropy_info

    def ADPC_update(self, il_transition_dict, beta=1.0, batch_size=4096, alpha=1.0, c_v=1.0,
                    shuffled=1, chosen_quantile=0.2, no_bern=True, dark_side=1, actor_only=1, epochs=4, 
                    target_entropy_cat=None, ppo_grad_val=None):
        """
        Adversarial Demonstration Policy Correction (ADPC) 更新。
        仅筛选 advantage 最好或者最差的 分位数样本，
        以 label_smoothing=0.99 构造 one-cold 反向标签，对其做反向标签交叉熵训练，
        权重截断在 [0, 1] 防止放大。
        """
        # 用 alpha 调整学习率而非损失权重
        current_lr = self.actor_optimizer.param_groups[0]['lr']
        self.actor_optimizer.param_groups[0]['lr'] = current_lr * alpha

        if 'obs' in il_transition_dict and len(il_transition_dict['obs']) > 0:
            obs_all = torch.tensor(np.array(il_transition_dict['obs']), dtype=torch.float).to(self.device)
            use_obs = True
        else:
            use_obs = False

        states_all  = torch.tensor(np.array(il_transition_dict['states']),  dtype=torch.float).to(self.device)
        returns_all = torch.tensor(np.array(il_transition_dict['returns']),  dtype=torch.float).view(-1, 1).to(self.device)

        raw_actions = il_transition_dict['actions']
        actions_all = {}
        if isinstance(raw_actions, list):
            keys = raw_actions[0].keys()
            temp_dict = {}
            for k in keys:
                temp_dict[k] = np.stack([d[k] for d in raw_actions], axis=0)
            raw_actions = temp_dict
        if isinstance(raw_actions, dict):
            for k, v in raw_actions.items():
                if k == 'cat':
                    actions_all[k] = torch.tensor(v, dtype=torch.long).to(self.device)
                else:
                    actions_all[k] = torch.tensor(v, dtype=torch.float).to(self.device)

        # --- 计算全量 advantage，筛选 最好或最差的 分位数 ---
        with torch.no_grad():
            values_all = self.critic(states_all)
            residual_all = returns_all - values_all
            # if not hasattr(self, 'c_sq'):
            self.c_sq = torch.tensor(1.0, device=self.device)
            c = torch.sqrt(self.c_sq)
            advantage_all = (residual_all / (c + 1e-8)).squeeze(-1)  # (N,)
            # 优势函数归一化（标准化）
            advantage_all = (advantage_all - advantage_all.mean()) / (advantage_all.std() + 1e-8)

            if dark_side:
                # 取最差的 chosen_quantile 样本，且归一化优势度 < 0
                threshold = torch.quantile(advantage_all, chosen_quantile) # 从低到高排位
                selected_mask = (advantage_all <= threshold) & (advantage_all < 0)
            else:
                # 取最好的 chosen_quantile 样本，且归一化优势度 > 0
                threshold = torch.quantile(advantage_all, 1.0 - chosen_quantile) # 从低到高排位
                selected_mask = (advantage_all >= threshold) & (advantage_all > 0)
            chosen_indices = selected_mask.nonzero(as_tuple=False).squeeze(-1)

        if chosen_indices.numel() == 0:
            return 0.0, 0.0

        # 筛选后的子集
        bad_states   = states_all[chosen_indices]
        bad_adv      = advantage_all[chosen_indices]
        bad_actions  = {k: v[chosen_indices] for k, v in actions_all.items()}
        if use_obs:
            bad_obs = obs_all[chosen_indices]
        else:
            bad_obs = bad_states

        # =====================================================================
        # [新增 1]：在开始 mini-batch 更新前，统一提取当前网络对这些样本的 log_probs 作为“旧策略锚点”
        # =====================================================================
        with torch.no_grad():
            anchor_log_probs, _, _, _, _ = self.actor.evaluate_actions(
                bad_obs, bad_actions, max_std=self.max_std, mask_on=0
            )
        
        total_actor_loss  = 0.0
        total_critic_loss = 0.0
        batch_count = 0

        sub_size = chosen_indices.size(0)
        sub_indices = np.arange(sub_size)

        for _ in range(epochs):
            # 目标熵检查：若当前 cat 熵已超出上限，直接终止所有后续 epoch
            if target_entropy_cat is not None:
                with torch.no_grad():
                    _, _, entropy_details_now, _, _ = self.actor.evaluate_actions(
                        bad_obs, bad_actions, max_std=self.max_std, mask_on=0
                    )
                    current_cat_entropy = entropy_details_now['cat'].mean().item()
                if current_cat_entropy > target_entropy_cat:
                    break

            if shuffled:
                np.random.shuffle(sub_indices)
            for start in range(0, sub_size, batch_size):
                end = min(start + batch_size, sub_size)
                bidx = sub_indices[start:end]

                s_batch   = bad_states[bidx]
                adv_batch = bad_adv[bidx]

                if use_obs:
                    actor_input_batch = bad_obs[bidx]
                else:
                    actor_input_batch = s_batch

                a_batch = {k: v[bidx] for k, v in bad_actions.items()}

                # 取出当前 batch 对应的旧策略锚点
                mb_anchor_log_probs = anchor_log_probs[bidx]

                with torch.no_grad():
                    # 权重 = advantage^2，与 beta 无关，恒非负，截断到 [0, 1]（不允许超过1放大）
                    # 非负权重确保：好/差样本的交叉熵误差恒为正向梯度，
                    # 避免负权重与互补交叉熵负负得正形成"接近差动作"的错误更新方向
                    raw_weights = torch.pow(adv_batch, 2) # .unsqueeze(-1) # ？？？之前为啥会有unsqueeze？？？
                    weights = torch.clamp(raw_weights, max=1.0)

                    # =====================================================================
                    # [新增 2]：计算当前网络的新 log_probs，并构造布尔掩码 (Mask)
                    # =====================================================================
                    new_log_probs, _, _, _, _ = self.actor.evaluate_actions(
                        actor_input_batch, a_batch, max_std=self.max_std, mask_on=0
                    )
                    ratio = torch.exp(new_log_probs - mb_anchor_log_probs)

                    # 如果新旧策略比超出 PPO 的容忍范围，mask 对应位置为 0.0，否则为 1.0
                    clip_mask = ((ratio >= 1.0 - self.eps) & (ratio <= 1.0 + self.eps)).float().squeeze(-1) # shape: (Batch, )

                # dark_side=1: 反向标签(0.99) + good_samples=0
                # dark_side=0: 正向模仿(0.01) + good_samples=1
                ls = 0.99 if dark_side else 0.01
                gs = 0 if dark_side else 1
                raw_il_loss = self.actor.compute_il_loss(
                    actor_input_batch,
                    a_batch,
                    label_smoothing=ls,
                    no_bern=no_bern,
                    good_samples=gs,
                    pre_training=1, # 0 原本只是负向交叉熵，但效果还不如构造one-cold分布
                ) # shape: (Batch, )
                
                v_pred = self.critic(s_batch)
                r_batch = returns_all[chosen_indices[bidx]]

                # =====================================================================
                # [新增 3]：将 clip_mask 乘入最终的 Loss。
                # 一旦触发截断，clip_mask=0，整个式子值为0，梯度在这一步被物理抹杀。
                # =====================================================================
                actor_loss = torch.mean(weights * raw_il_loss * clip_mask)
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                # 动态梯度裁剪：若传入了 ppo_grad_val，则上限使用它，否则使用默认的 self.actor_max_grad
                max_g = ppo_grad_val if ppo_grad_val is not None else self.actor_max_grad
                nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=max_g)
                self.actor_optimizer.step()

                
                
                critic_loss = F.mse_loss(v_pred, r_batch) * c_v

                if not actor_only:
                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.critic_max_grad)
                    self.critic_optimizer.step()

                total_actor_loss  += actor_loss.item()
                total_critic_loss += critic_loss.item()
                batch_count += 1

        avg_actor_loss  = total_actor_loss  / batch_count if batch_count > 0 else 0.0
        avg_critic_loss = total_critic_loss / batch_count if batch_count > 0 else 0.0

        # 还原学习率
        self.actor_optimizer.param_groups[0]['lr'] = current_lr

        return avg_actor_loss, avg_critic_loss