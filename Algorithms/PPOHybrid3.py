'''
修改进行中：

改自回归级联结构

准备把PPOHybrid2 的active_mask处理复制过来
'''

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, Bernoulli

import os, sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from Algorithms.Utils import model_grad_norm, check_weights_bias_nan, compute_advantage, SquashedNormal
from Algorithms.MLP_heads import ValueNet

# =============================================================================
# 1. 神经网络定义 (修复：处理可能为空的动作头，增强鲁棒性)
# =============================================================================
class CascadePolicyNet(nn.Module): # 警告，这个结构没有泛化性，必须针对场景单独设计
    def __init__(self, state_dim, hidden_dims, action_dims_dict):
        super().__init__()
        self.action_dims = action_dims_dict
        
        # --- Level 0: 基础特征提取 (Maneuver/Fly) ---
        # 构建隐藏层序列（使用传入的 hidden_dims 列表）
        layers = []
        prev_size = state_dim
        for layer_size in hidden_dims:
            layers.append(nn.Linear(prev_size, layer_size))
            layers.append(nn.ReLU())
            prev_size = layer_size
        # prev_size 为最后一层隐藏单元数，作为特征维度
        self.l0_feat_dim = prev_size
        self.l0_net = nn.Sequential(*layers)
        
        # 假设 Level 0 是必须存在的 cat 动作 (Fly)
        # action_dims_dict['l0_cat'] 应为列表 [14]
        self.l0_dim = action_dims_dict['l0_cat'][0] 
        self.fc_l0_logits = nn.Linear(self.l0_feat_dim, self.l0_dim)

        # --- Level 1: 接收 (状态特征 + Level 0 动作的 One-Hot) ---
        # 输入维度 = Level0 特征 + Level 0 动作维度(14)
        self.l1_input_dim = self.l0_feat_dim + self.l0_dim

        # 同样使用 hidden_dims 构建 l1_net 的隐藏层
        layers = []
        prev_size = self.l1_input_dim
        for layer_size in hidden_dims:
            layers.append(nn.Linear(prev_size, layer_size))
            layers.append(nn.ReLU())
            prev_size = layer_size
        self.l1_feat_dim = prev_size
        self.l1_net = nn.Sequential(*layers)
        
        # Level 1 的动作头 (Fire/Bern)
        # 允许 l1_cat 为空，只保留 l1_bern
        self.has_l1_cat = 'l1_cat' in action_dims_dict and sum(action_dims_dict['l1_cat']) > 0
        if self.has_l1_cat:
            self.fc_l1_cat = nn.Linear(self.l1_feat_dim, sum(action_dims_dict['l1_cat']))
            
        self.has_l1_bern = 'l1_bern' in action_dims_dict and action_dims_dict['l1_bern'] > 0
        if self.has_l1_bern:
            self.fc_l1_bern = nn.Linear(self.l1_feat_dim, action_dims_dict['l1_bern'])

    def forward(self, state, l0_action=None):
        # 1. Level 0 计算
        feat0 = self.l0_net(state)
        l0_logits = self.fc_l0_logits(feat0)
        
        outputs = {'l0_logits': l0_logits, 'l1_logits': None, 'l1_bern': None}

        # 2. Level 1 计算 (需要 l0_action 作为条件)
        # 如果是推理模式且 l0_action 为 None，则只返回 l0_logits 供外部采样，
        # 外部采样后再调用一次 forward 传入 l0_action 获取 l1 结果。
        
        if l0_action is not None:
            # --- 依赖注入 (Autoregressive Input) ---
            # l0_action: (Batch, ) or (Batch, 1) -> 转 One-Hot
            if l0_action.dim() > 1: l0_action = l0_action.squeeze(-1)
            l0_one_hot = F.one_hot(l0_action.long(), num_classes=self.l0_dim).float()
            
            # 拼接特征
            l1_input = torch.cat([feat0, l0_one_hot], dim=-1)
            feat1 = self.l1_net(l1_input)
            
            if self.has_l1_cat:
                outputs['l1_logits'] = self.fc_l1_cat(feat1)
            if self.has_l1_bern:
                outputs['l1_bern'] = self.fc_l1_bern(feat1) # 这里输出 Logits
            
        return outputs

# =============================================================================
# 2. Actor 适配器 (Wrapper) - 修复返回值接口
# =============================================================================
class CascadeActorWrapper(nn.Module): # 警告，这个结构没有泛化性，必须针对场景单独设计
    def __init__(self, policy_net, action_dims_dict, device='cpu'):
        super().__init__()
        self.net = policy_net
        self.action_dims = action_dims_dict # 必须保存
        self.device = device

    def get_action(self, state, explore=True, **kwargs):
        """
        自回归推理: State -> L0 Logits -> Sample L0 -> Net(State, L0) -> L1 Logits -> Sample L1
        """
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)

        # --- Step 1: 决策 Level 0 (Fly) ---
        out0 = self.net(state, l0_action=None) 
        l0_logits = out0['l0_logits']
        
        dist0 = Categorical(logits=l0_logits)
        if explore:
            l0_idx = dist0.sample()
        else:
            l0_idx = torch.argmax(l0_logits, dim=-1)
        
        # --- Step 2: 决策 Level 1 (Fire) ---
        # 将采样到的 l0_idx 传回网络
        out1 = self.net(state, l0_action=l0_idx)
        
        actions_exec = {'l0_cat': l0_idx.cpu().numpy()} # 保持维度 (Batch,)
        # 用于日志记录的概率分布检查
        actions_dist_check = {'l0_cat': F.softmax(l0_logits, dim=-1).detach().cpu().numpy()} 

        # 处理 L1 Bern (Fire)
        if out1['l1_bern'] is not None:
            l1_bern_logits = out1['l1_bern']
            dist1_bern = Bernoulli(logits=l1_bern_logits)
            if explore:
                l1_bern_val = dist1_bern.sample()
            else:
                l1_bern_val = (l1_bern_logits > 0).float()
            
            # [CRITICAL FIX] 关键修改：
            # Bernoulli.sample() 返回 (Batch, 1)，而 Categorical.sample() 返回 (Batch,)。
            # 为了在后续 Buffer 存储时保持维度一致（防止变成 Batch, 1, 1），
            # 同时也为了适配 seq_len 等未来扩展，我们需要把最后一个维度(特征维)去掉，
            # 让它变成 (Batch,) 或 (Batch, Seq)，与 cat 动作保持一致。
            if l1_bern_val.shape[-1] == 1:
               l1_bern_val = l1_bern_val.squeeze(-1)
               
            actions_exec['l1_bern'] = l1_bern_val.cpu().numpy() # (Batch, 1)
            actions_dist_check['l1_bern'] = torch.sigmoid(l1_bern_logits).detach().cpu().numpy()

        # 处理 L1 Cat (如果有)
        if out1['l1_logits'] is not None:
            l1_probs = F.softmax(out1['l1_logits'], dim=-1)
            dist1_cat = Categorical(probs=l1_probs)
            l1_idx = dist1_cat.sample() if explore else torch.argmax(l1_probs, dim=-1)
            actions_exec['l1_cat'] = l1_idx.cpu().numpy()

        # 这里的 raw 和 exec 在离散/伯努利下是一样的
        return actions_exec, actions_exec, None, actions_dist_check

    def evaluate_actions(self, states, actions_raw, **kwargs):
        """
        训练模式：Teacher Forcing。一次性计算所有 LogProbs。
        actions_raw 必须包含: 'l0_cat', 'l1_bern' 等键
        """
        l0_truth = actions_raw['l0_cat'] # (Batch, ) or (Batch, 1)
        if l0_truth.dim() > 1: l0_truth = l0_truth.squeeze(-1)

        # Forward 一次搞定
        outputs = self.net(states, l0_action=l0_truth)
        
        log_probs = 0
        entropy = 0
        entropy_details = {'l0': None, 'l1': None, 'cont': None, 'cat': None, 'bern': None}
        
        # --- L0 Cat (Fly) ---
        dist0 = Categorical(logits=outputs['l0_logits'])
        log_probs += dist0.log_prob(l0_truth).unsqueeze(-1)
        entropy += dist0.entropy().unsqueeze(-1)
        entropy_details['cat'] = dist0.entropy().mean().item() # 记录用

        # --- L1 Bern (Fire) ---
        if outputs['l1_bern'] is not None and 'l1_bern' in actions_raw:
            l1_bern_truth = actions_raw['l1_bern'] # (Batch, 1)
            dist1_bern = Bernoulli(logits=outputs['l1_bern'])
            # Bernoulli log_prob 维度是 (Batch, Dim)，需要 sum 吗？这里 fire dim=1，无需 sum
            lp_bern = dist1_bern.log_prob(l1_bern_truth)
            if lp_bern.dim() > 1: lp_bern = lp_bern.sum(dim=-1, keepdim=True)
            
            log_probs += lp_bern
            entropy += dist1_bern.entropy().sum(dim=-1, keepdim=True)
            entropy_details['bern'] = dist1_bern.entropy().mean().item()

        return log_probs, entropy, entropy_details, None

    def compute_il_loss(self, states, expert_actions, label_smoothing=0.1):
        """
        MARWIL/BC Loss 计算
        Expert Actions Keys: 'l0_cat', 'l1_bern'
        """
        # 取出专家数据的 L0 动作作为条件
        l0_expert = expert_actions['l0_cat']
        if l0_expert.dim() > 1: l0_expert = l0_expert.squeeze(-1)
        
        # Teacher Forcing Forward
        outputs = self.net(states, l0_action=l0_expert)
        
        total_loss = torch.zeros(states.size(0), device=self.device)
        
        # 1. L0 Cat Loss (CrossEntropy)
        l0_logits = outputs['l0_logits']
        # 标准 CE: -log_p[target]
        ce_loss_l0 = F.cross_entropy(l0_logits, l0_expert.long(), reduction='none', label_smoothing=label_smoothing)
        total_loss += ce_loss_l0
        
        # 2. L1 Bern Loss (BCE)
        if outputs['l1_bern'] is not None:
            l1_logits = outputs['l1_bern']
            l1_target = expert_actions['l1_bern']
            
            # Label Smoothing for BCE
            target_smooth = l1_target * (1.0 - label_smoothing) + 0.5 * label_smoothing
            bce_loss = F.binary_cross_entropy_with_logits(l1_logits, target_smooth, reduction='none')
            total_loss += bce_loss.sum(dim=-1)
            
        return total_loss
    
    
# =============================================================================
# 3. PPO 算法类 (精简版)
# =============================================================================

class PPOHybrid:
    def __init__(self, actor, critic, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device, 
                 k_entropy=0.01, critic_max_grad=2, actor_max_grad=2, max_std=0.3):
        
        self.actor = actor # 这是一个 HybridActorWrapper 实例
        self.critic = critic
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
        
        # [新增] 额外的监控指标
        self.approx_kl = 0        # 近似 KL 散度 (判断策略变化幅度)
        self.clip_frac = 0        # 裁剪触发比例 (判断 eps 或 lr 是否合适)
        self.explained_var = 0    # 解释方差 (判断 Critic 拟合程度)
        # [新增] 分项 Entropy 监控
        self.entropy_cat = 0
        self.entropy_bern = 0
        self.entropy_cont = 0

    def set_learning_rate(self, actor_lr=None, critic_lr=None):
        if actor_lr is not None:
            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] = actor_lr
        if critic_lr is not None:
            for param_group in self.critic_optimizer.param_groups:
                param_group['lr'] = critic_lr  

    def take_action(self, state, h0=None, explore=True, max_std=None):
        # 委托给 Actor Wrapper
        max_s = max_std if max_std is not None else self.max_std
        # 注意：这里返回了 hidden_state (虽然是 None)，保持接口一致性
        # actions_exec, actions_raw, _ = self.actor.get_action(state, h=h0, explore=explore, max_std=max_s)
        # return actions_exec, actions_raw
        
        # [修改] 现在接收四个返回值
        actions_exec, actions_raw, h_state, actions_dist_check = self.actor.get_action(state, h=h0, explore=explore, max_std=max_s)
        # [修改] 保持原有的返回两个字典的接口，或者根据需要返回 diagnostic output
        return actions_exec, actions_raw, h_state, actions_dist_check

    def update(self, transition_dict, adv_normed=False, clip_vf=False, clip_range=0.2, shuffled=1):

        # RL 更新阶段：解冻 std 参数
        if hasattr(self.actor.net, 'log_std_param'):
            self.actor.net.log_std_param.requires_grad = True

        # [修改] 6. 智能数据转换：如果已经是 np.ndarray (来自 HybridReplayBuffer)，直接转 Tensor
        # 否则 (来自 list append)，先转 np 再转 Tensor
        def to_tensor(x, dtype):
            if isinstance(x, np.ndarray):
                return torch.tensor(x, dtype=dtype).to(self.device)
            else:
                return torch.tensor(np.array(x), dtype=dtype).to(self.device)

        states = to_tensor(transition_dict['states'], torch.float)
        next_states = to_tensor(transition_dict['next_states'], torch.float)
        dones = to_tensor(transition_dict['dones'], torch.float).view(-1, 1)
        rewards = to_tensor(transition_dict['rewards'], torch.float).view(-1, 1)

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
            
            # 以下为公共部分
            # 如果没有预计算，则现场计算 (注意：如果是并行数据直接展平进来的，这里计算会有偏差)
            with torch.no_grad():
                # 修改：Critic 使用全局 next_states 计算 Target
                td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
                # 修改：Critic 使用全局 states 计算当前 Value
                td_delta = td_target - self.critic(critic_inputs)
                advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu(), dones.cpu()).to(self.device)
                
        # 3. 计算旧策略的 log_probs (使用 Wrapper)
        with torch.no_grad():
            # 修改：Actor 使用 actor_inputs (可能是 obs)
            old_log_probs, _, _, _ = self.actor.evaluate_actions(actor_inputs, actions_on_device, h=None, max_std=self.max_std)
            # 修改：Critic 使用 critic_inputs (全局 states)
            v_pred_old = self.critic(critic_inputs)
            
        # --- [2. 优势归一化] ---
        if adv_normed:
            adv_mean = advantage.mean()
            adv_std = advantage.std(unbiased=False)
            advantage = (advantage - adv_mean) / (adv_std + 1e-8)
        else:
            # 推荐: 即使不归一化，也建议减去均值 (Centering)
            # 这有助于降低方差，且不改变梯度的方向
            adv_mean = advantage.mean()
            advantage = advantage - adv_mean

        # --- [3. Shuffle 逻辑: 在此处打乱所有相关 Tensor] ---
        if shuffled:
            # 生成随机索引
            # 修改：使用 actor_inputs 的大小作为基准 (通常和 states 一样长)
            idx = torch.randperm(actor_inputs.size(0), device=self.device)
            
            # 打乱基础数据
            # 修改：同时打乱 actor_inputs 和 critic_inputs
            # 如果它们指向同一个对象 (states)，也不会出错，只是多做了一次索引操作
            if 'obs' in transition_dict and len(transition_dict['obs']) > 0:
                actor_inputs = actor_inputs[idx] # 打乱 obs
                critic_inputs = critic_inputs[idx] # 打乱 states
            else:
                # 只有 states 的情况
                states = states[idx]
                actor_inputs = states
                critic_inputs = states

            td_target = td_target[idx]
            advantage = advantage[idx]
            old_log_probs = old_log_probs[idx]
            v_pred_old = v_pred_old[idx]
            
            # 打乱字典形式的动作
            for key in actions_on_device:
                actions_on_device[key] = actions_on_device[key][idx]
            
            # 如果有其他 tensor 需要 shuffle (如 truncs) 也要在这里处理
            # todo truncs的处理逻辑 当前env里面没有做，所以算法先不做
            

        # 4. PPO Update Loop
        actor_loss_list, critic_loss_list, entropy_list, ratio_list = [], [], [], []
        actor_grad_list, critic_grad_list = [], []
        pre_clip_actor_grad, pre_clip_critic_grad = [], []

        # [新增] 监控列表
        kl_list = []
        clip_frac_list = []
        # [新增] 分项 Entropy 列表
        entropy_cat_list = []
        entropy_bern_list = []
        entropy_cont_list = []
        
        for _ in range(self.epochs):
            # 计算当前策略的 log_probs 和 entropy (使用 Wrapper)
            # [修改] 接收 entropy_details
            log_probs, entropy, entropy_details ,_ = self.actor.evaluate_actions(actor_inputs, actions_on_device, h=None, max_std=self.max_std)
            
            # [修改] 计算 log_ratio 用于更精准的 KL 计算
            log_ratio = log_probs - old_log_probs
            ratio = torch.exp(log_ratio)
            # [新增] 计算 Approximate KL Divergence (http://joschu.net/blog/kl-approx.html)
            with torch.no_grad():
                # old_approx_kl = (-log_ratio).mean()
                approx_kl = ((ratio - 1) - log_ratio).mean()
                kl_list.append(approx_kl.item())
                
                # [新增] 计算 Clip Fraction (有多少样本触发了裁剪)
                clip_fracs = ((ratio - 1.0).abs() > self.eps).float().mean()
                clip_frac_list.append(clip_fracs.item())
            
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = -torch.min(surr1, surr2).mean() - self.k_entropy * entropy.mean()

            # Critic Loss
            # 修改：Critic 使用 critic_inputs
            v_pred = self.critic(critic_inputs)
            if clip_vf:
                v_pred_clipped = torch.clamp(v_pred, v_pred_old - clip_range, v_pred_old + clip_range)
                vf_loss1 = (v_pred - td_target).pow(2)
                vf_loss2 = (v_pred_clipped - td_target).pow(2)
                critic_loss = torch.max(vf_loss1, vf_loss2).mean()
            else:
                critic_loss = F.mse_loss(v_pred, td_target)
            
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

            # Logging
            actor_grad_list.append(model_grad_norm(self.actor))
            critic_grad_list.append(model_grad_norm(self.critic))            
            actor_loss_list.append(actor_loss.item())
            critic_loss_list.append(critic_loss.item())
            entropy_list.append(entropy.mean().item())
            ratio_list.append(ratio.mean().item())
            
            # [新增] 记录分项 Entropy
            if entropy_details['cont'] is not None:
                entropy_cont_list.append(entropy_details['cont'])
            if entropy_details['cat'] is not None:
                entropy_cat_list.append(entropy_details['cat'])
            if entropy_details['bern'] is not None:
                entropy_bern_list.append(entropy_details['bern'])

        self.actor_loss = np.mean(actor_loss_list)
        self.actor_grad = np.mean(actor_grad_list)
        self.critic_loss = np.mean(critic_loss_list)
        self.critic_grad = np.mean(critic_grad_list)
        self.entropy_mean = np.mean(entropy_list)
        self.ratio_mean = np.mean(ratio_list)
        self.pre_clip_critic_grad = np.mean(pre_clip_critic_grad)
        self.pre_clip_actor_grad = np.mean(pre_clip_actor_grad)
        self.advantage = advantage.abs().mean().item()
        
        # [新增] 汇总新指标
        self.approx_kl = np.mean(kl_list)
        self.clip_frac = np.mean(clip_frac_list)
        # [新增] 计算分项 Entropy 均值
        self.entropy_cont = np.mean(entropy_cont_list) if len(entropy_cont_list) > 0 else 0
        self.entropy_cat = np.mean(entropy_cat_list) if len(entropy_cat_list) > 0 else 0
        self.entropy_bern = np.mean(entropy_bern_list) if len(entropy_bern_list) > 0 else 0
        
        # [新增] 计算 Explained Variance
        # y_true: td_target, y_pred: v_pred_old (更新前的值) 或 v_pred (更新后的值，通常用更新前比较多，或者直接对比)
        # 这里使用 numpy 计算以防 tensor 维度广播问题
        y_true = td_target.flatten().cpu().numpy()
        y_pred = v_pred_old.flatten().cpu().numpy() # 比较更新前的 Value 网络预测能力
        var_y = np.var(y_true)
        if var_y == 0:
            self.explained_var = np.nan
        else:
            self.explained_var = 1 - np.var(y_true - y_pred) / var_y

        check_weights_bias_nan(self.actor, "actor", "update后")
        check_weights_bias_nan(self.critic, "critic", "update后")


    # --- 新增功能 2: MARWIL Update ---
    def MARWIL_update(self, il_transition_dict, beta=1.0, batch_size=128, alpha=1.0, c_v=1.0, shuffled=1, label_smoothing=0.1, max_weight=100.0):
        """
        MARWIL 离线更新函数 (混合动作空间支持版 + Mini-batch)
        """
        # 可能的局部观测
        if 'obs' in il_transition_dict and len(il_transition_dict['obs']) > 0:
            obs_all = torch.tensor(np.array(il_transition_dict['obs']), dtype=torch.float).to(self.device)
            use_obs = True
        else:
            use_obs = False
            
        # --- [1. MARWIL模式：冻结连续动作 std 参数] ---
        if hasattr(self.actor.net, 'log_std_param'):
            self.actor.net.log_std_param.requires_grad = False

        # 1. 提取全量数据并转为 Tensor
        states_all = torch.tensor(np.array(il_transition_dict['states']), dtype=torch.float).to(self.device)
        returns_all = torch.tensor(np.array(il_transition_dict['returns']), dtype=torch.float).view(-1, 1).to(self.device)
        
        # 处理动作字典 (转 Tensor)
        actions_all = {}
        for k, v in il_transition_dict['actions'].items():
            if k == 'cat':  # cont和bern分布都使用float
                actions_all[k] = torch.tensor(np.array(v), dtype=torch.long).to(self.device)
            else:
                actions_all[k] = torch.tensor(np.array(v), dtype=torch.float).to(self.device)

        # 2. 准备 Batch 索引
        total_size = states_all.size(0)
        indices = np.arange(total_size)
        if shuffled:
            np.random.shuffle(indices)

        total_actor_loss = 0
        total_critic_loss = 0
        total_c = 0
        batch_count = 0

        # 3. Mini-batch 循环
        for start in range(0, total_size, batch_size):
            end = min(start + batch_size, total_size)
            batch_indices = indices[start:end]
            
            # 切片取 Batch 数据
            s_batch = states_all[batch_indices] # Critic 总是用全局状态
            
            r_batch = returns_all[batch_indices]
            if use_obs:
                actor_input_batch = obs_all[batch_indices] # Actor 用局部
            else:
                actor_input_batch = s_batch # Actor 用全局
            
            # 动作字典切片
            actions_batch = {}
            for k, v in actions_all.items():
                actions_batch[k] = v[batch_indices]

            # ----------------------------------------------------
            # A. 计算优势 (Advantage) 和 权重 (Weights)
            # ----------------------------------------------------
            with torch.no_grad():
                values = self.critic(s_batch)
                residual = r_batch - values
                
                # 动态更新 c^2 (Moving Average)
                if not hasattr(self, 'c_sq'): 
                    self.c_sq = torch.tensor(1.0, device=self.device)
                
                batch_mse = (residual ** 2).mean().item()
                self.c_sq = self.c_sq + 1e-8 * (batch_mse - self.c_sq)
                c = torch.sqrt(self.c_sq)
                
                # 归一化优势
                advantage = residual / (c + 1e-8)
                
                # 计算权重并截断
                raw_weights = torch.exp(beta * advantage)
                weights = torch.clamp(raw_weights, max=max_weight)

            # ----------------------------------------------------
            # B. 计算 Actor Loss (委托给 Wrapper)
            # ----------------------------------------------------
            # compute_il_loss 返回的是 (Batch, ) 的 loss，没有 reduce
            raw_il_loss = self.actor.compute_il_loss(actor_input_batch, actions_batch, label_smoothing)
            
            # 加权平均: mean( alpha * w * loss )
            actor_loss = torch.mean(alpha * weights * raw_il_loss)

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
            total_c += c.item()
            batch_count += 1

        # 返回平均 Loss
        avg_actor_loss = total_actor_loss / batch_count if batch_count > 0 else 0
        avg_critic_loss = total_critic_loss / batch_count if batch_count > 0 else 0
        avg_c = total_c / batch_count if batch_count > 0 else 0
        
        return avg_actor_loss, avg_critic_loss, avg_c