'''
使用简单熵正则项

GRU这么使用不太合理，使用的时候记得seq_len=1，use_gru=0

通道注意力微调，先升维再用
'''
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, Bernoulli

import os, sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from Algorithms.Utils import model_grad_norm, check_weights_bias_nan, compute_advantage, SquashedNormal
# from Algorithms.MLP_heads import ValueNet

# =============================================================================
# 1. 神经网络定义 (保持不变，只负责 forward 计算)
# =============================================================================
from Algorithms.SharedLayers import ChannelAttention # 调用已定义的通道注意力

# ========
# 策略网络
# ========
class PolicyNetHybrid(torch.nn.Module):
    def __init__(self, state_dim, hidden_dims, 
                 action_dims_dict, init_std=0.5, 
                 reduction_ratio=16, num_layers=1, 
                 use_channel_attention=True, use_gru=True):
        super(PolicyNetHybrid, self).__init__()
        self.action_dims = action_dims_dict
        self.use_channel_attention = use_channel_attention
        self.use_gru = use_gru # 记录开关状态

        # 嵌入层：先扩展到 hidden_dims[0]
        self.embed = nn.Linear(state_dim, hidden_dims[0])
        prev_size = hidden_dims[0]

        # 1. 通道注意力 (hidden_dims[0] -> hidden_dims[0])
        if self.use_channel_attention:
            self.attention = ChannelAttention(prev_size, reduction_ratio)

        # 2. MLP 主干
        layers = []
        for h in hidden_dims:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            prev_size = h
        self.backbone = nn.Sequential(*layers)

        # 只有开启时才定义 GRU
        if self.use_gru:
            # 3. GRU 层 (提取时序特征, 128 -> 128)
            # 使用 batch_first=True 适配 (Batch, SeqLen, Dim)
            # 修改点：传入 num_layers
            self.gru = nn.GRU(prev_size, prev_size, num_layers=num_layers, batch_first=True)

            # 正交初始化 GRU 权重（weight_hh 使用正交，weight_ih 使用 xavier，bias 清零）
            for name, param in self.gru.named_parameters():
                if 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'bias' in name:
                    nn.init.constant_(param.data, 0.0)

        # 4. 动作头 (Heads)
        if 'cont' in self.action_dims and self.action_dims['cont'] > 0:
            self.fc_mu = nn.Linear(prev_size, self.action_dims['cont'])
            self.log_std_cont = nn.Parameter(torch.log(torch.ones(self.action_dims['cont']) * init_std))

        if 'cat' in self.action_dims and sum(self.action_dims['cat']) > 0:
            self.cat_dims = self.action_dims['cat']
            self.fc_cat = nn.Linear(prev_size, sum(self.cat_dims))

        if 'bern' in self.action_dims and self.action_dims['bern'] > 0:
            self.fc_bern = nn.Linear(prev_size, self.action_dims['bern'])
            # 伯努利头特殊初始化：使初始开火概率偏低
            nn.init.constant_(self.fc_bern.bias, -2.0)
            torch.nn.init.normal_(self.fc_bern.weight, mean=0.0, std=1e-3)

    def forward(self, x, h_in=None, max_std=1.0, action_masks=None):
        # 适配维度：如果输入是 2D (B, D)，增加 SeqLen 维度变为 3D (B, 1, D)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        B, S, D = x.shape

        # Step 1: Channel Attention (作用于每一帧)
        x_reshaped = x.reshape(B * S, D)
        x_embed = self.embed(x_reshaped)

        # Step 1: Channel Attention (作用于每一帧)
        if self.use_channel_attention:
            x_att, _ = self.attention(x_embed)
        else:
            x_att = x_embed

        # Step 2: MLP Backbone
        features = self.backbone(x_att).view(B, S, -1)
        
        if self.use_gru:
            # Step 3: GRU 自动递归 (h_in 为序列起始状态)
            # gru_out 形状: (B, S, 128)
            gru_out, h_out = self.gru(features, h_in)
        else:
            # “假 GRU”流程：直接透传特征
            gru_out = features
            h_out = h_in
        
        outputs = {'cont': None, 'cat': None, 'bern': None}

        # --- Heads 分发 ---
        if 'cont' in self.action_dims and self.action_dims['cont'] > 0:
            mu = self.fc_mu(gru_out)
            std = torch.exp(self.log_std_cont).expand_as(mu).clamp(max=max_std)
            outputs['cont'] = (mu, std)

        if 'cat' in self.action_dims and sum(self.action_dims['cat']) > 0:
            logits_all = self.fc_cat(gru_out)
            logits_list = torch.split(logits_all, self.cat_dims, dim=-1)
            outputs['cat'] = [F.softmax(l, dim=-1) for l in logits_list]

        if 'bern' in self.action_dims and self.action_dims['bern'] > 0:
            bern_logits = self.fc_bern(gru_out)
            if action_masks is not None and 'bern' in action_masks:
                # 训练时 mask 需匹配 3D 维度 (B, S, 1)
                m = action_masks['bern']
                if m.dim() == 2: m = m.unsqueeze(1) 
                bern_logits = bern_logits.masked_fill(m == 0, -1e9)
            outputs['bern'] = bern_logits

        return outputs, h_out

# =======
# 价值网络
# =======

class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dims, 
                 reduction_ratio=16, num_layers=1, 
                 use_channel_attention=True, use_gru=True):
        super(ValueNet, self).__init__()
        self.use_channel_attention = use_channel_attention
        self.use_gru = use_gru

        # 嵌入层
        self.embed = nn.Linear(state_dim, hidden_dims[0])
        prev_size = hidden_dims[0]

        if self.use_channel_attention:
            self.attention = ChannelAttention(prev_size, reduction_ratio)

        layers = []
        for h in hidden_dims:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            prev_size = h
        self.backbone = nn.Sequential(*layers)
        
        # 只有开启时才定义 GRU
        if self.use_gru:
            self.gru = nn.GRU(prev_size, prev_size, num_layers=num_layers, batch_first=True)
            # 对 GRU 权重做正交初始化（weight_hh 正交，weight_ih xavier，bias 清零）
            for name, param in self.gru.named_parameters():
                if 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'bias' in name:
                    nn.init.constant_(param.data, 0.0)

        self.fc_out = nn.Linear(prev_size, 1)

    def forward(self, x, h_in=None):
        # --- 增强的维度适配逻辑 [修改部分] ---
        # 1. 如果是 1D (Dim,) -> 变为 (1, 1, Dim)
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        # 2. 如果是 2D (Batch, Dim) -> 变为 (Batch, 1, Dim)
        elif x.dim() == 2:
            x = x.unsqueeze(1)

        # 此时 x 统一为 3D: (Batch, SeqLen, Dim)
        B, S, D = x.shape
        
        x_reshaped = x.reshape(B * S, D)
        x_embed = self.embed(x_reshaped)

        if self.use_channel_attention:
            x_att, _ = self.attention(x_embed)
        else:
            x_att = x_embed

        features = self.backbone(x_att).view(B, S, -1)
        
        if self.use_gru:
            gru_out, h_out = self.gru(features, h_in)
        else:
            gru_out = features
            h_out = h_in
            
        values = self.fc_out(gru_out) # (B, S, 1)
        
        return values, h_out

# =============================================================================
# 2. Actor 适配器 (Wrapper) - 显式支持隐藏状态
# =============================================================================

class HybridActorWrapper(nn.Module):
    """
    统一接口适配器。
    将具体的 PolicyNetHybrid 封装起来，对外提供标准的 get_action 和 evaluate_actions 接口。
    显式支持 GRU 隐藏状态的传递。
    """
    def __init__(self, policy_net, action_dims_dict, action_bounds=None, device='cpu'):
        super(HybridActorWrapper, self).__init__()
        self.net = policy_net
        self.action_dims = action_dims_dict
        self.device = device
        
        # 处理 Action Bounds (保持原样)
        if 'cont' in self.action_dims and self.action_dims['cont'] > 0:
            if action_bounds is None:
                raise ValueError("Continuous action space requires action_bounds")
            self.register_buffer('action_bounds', torch.tensor(action_bounds, dtype=torch.float, device=device))
            self.register_buffer('amin', self.action_bounds[:, 0])
            self.register_buffer('amax', self.action_bounds[:, 1])
            self.register_buffer('action_span', self.amax - self.amin)

    def _scale_action_to_exec(self, a_norm):
        return self.amin + (a_norm + 1.0) * 0.5 * self.action_span

    def get_action(self, state, h=None, explore=True, max_std=None, check_obs=None, bern_threshold=0.5):
        """
        推理接口。
        Args:
            state: numpy array or tensor (1, Dim) or (Batch, Dim)
            h: hidden state (num_layers, Batch, HiddenDim)
            explore: bool or dict
            check_obs: 用于生成动作 mask 的原始观测字典
        Returns:
            actions_exec: dict (numpy), 用于环境执行
            actions_raw: dict (numpy/tensor), 用于存入 buffer
            next_h: GRU 输出的下一时刻隐藏状态
            actions_dist_check: 诊断输出 (概率分布等)
        """
        # 增强的 Batch 检测逻辑 (保持原样)
        is_batch = False
        if not isinstance(state, torch.Tensor):
            if isinstance(state, np.ndarray) and state.ndim > 1:
                is_batch = True
                state = torch.tensor(state, dtype=torch.float).to(self.device)
            else:
                state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        else:
            if state.dim() > 1:
                is_batch = True
        
        # 处理 explore 参数 (保持原样)
        if isinstance(explore, bool):
            explore_opts = {'cont': explore, 'cat': explore, 'bern': explore}
        elif isinstance(explore, dict):
            # 如果传入字典，使用字典的值，对缺失的键默认为 True
            explore_opts = {
                'cont': explore.get('cont', True),
                'cat': explore.get('cat', True),
                'bern': explore.get('bern', True)
            }
        else:
            # 对于其他意外的输入类型，默认全部探索
            explore_opts = {'cont': True, 'cat': True, 'bern': True}

        # 解析 check_obs 并构建 Action Mask (保持原样)
        action_masks = None
        can_fire = True
        if (check_obs is not None) and isinstance(check_obs, dict):
            can_fire = True
            ata_hor = np.arccos(check_obs["target_information"][0])
            ata = check_obs["target_information"][4]
            ata_condition = (ata <= 60 * np.pi / 180 and ata_hor <= 20 * np.pi / 180)
            # [新增] ata_hor 是第一个漂亮结果后新增的mask项
            can_fire = can_fire and ata_condition

            # 2. Target Locked == 1
            locked = check_obs["target_locked"]
            locked_condition = (locked == 1)
            can_fire = can_fire and locked_condition

            # 3. Ammo > 0 (ego_main 最后一个元素是 ammo)
            ammo = check_obs["ego_main"][6]
            ammo_condition = (ammo > 0)
            can_fire = can_fire and ammo_condition

            # 4. 超远距离尾追不打（使用 AA_hor 判断尾追）
            distance = check_obs["target_information"][3]
            AA_hor = check_obs["target_information"][6]
            if (distance > 30e3) and (abs(AA_hor) < np.pi/6):
                can_fire = False

            # 5. 30km 外12s内禁止重复发射第二枚 或 mid-term 有在飞导弹
            # weapon 计时单位兼容原逻辑
            if (distance > 30e3 and check_obs["weapon"] * 120 < 12) or check_obs.get("missile_in_mid_term", False):
                can_fire = False

            # 构建 Tensor Mask: (Batch_Size, Bern_Dim) -> (1, 1)
            # 1.0 表示允许 (保留 Logits)，0.0 表示禁止 (Logits -> -inf)
            mask_val = 1.0 if can_fire else 0.0
            
            # 适配 state 的 batch size
            batch_size = state.size(0)
            mask_tensor = torch.full((batch_size, 1), mask_val, device=self.device, dtype=torch.float)
            
            action_masks = {'bern': mask_tensor}

        # 调用网络，显式传递和接收隐藏状态 [修改]
        # 注意：网络内部会将 2D 输入转为 3D 进行单步递归
        actor_outputs, next_h = self.net(state, h_in=h, max_std=max_std, action_masks=action_masks)
        
        actions_exec = {}
        actions_raw = {}
        actions_dist_check = {} 

        # --- Cont ---
        # if actor_outputs['cont'] is not None:
        if 'cont' in self.action_dims and self.action_dims['cont'] > 0:
            mu, std = actor_outputs['cont']
            # mu, std 形状为 (B, 1, Dim) -> squeeze(1) 还原为采样维度
            mu = mu.squeeze(1)
            std = std.squeeze(1)
            dist = SquashedNormal(mu, std)
            if explore_opts['cont']:
                a_norm, u = dist.sample()
            else:
                u = mu
                a_norm = torch.tanh(u)
            
            a_exec = self._scale_action_to_exec(a_norm)
            
            #  根据是否 Batch 返回不同形状
            if is_batch:
                actions_exec['cont'] = a_exec.cpu().detach().numpy() # (Batch, Dim)
                actions_raw['cont'] = u.cpu().detach().numpy()
                actions_dist_check['cont'] = u.cpu().detach().numpy()
            else:
                actions_exec['cont'] = a_exec[0].cpu().detach().numpy().flatten()
                actions_raw['cont'] = u[0].cpu().detach().numpy().flatten()
                actions_dist_check['cont'] = u[0].cpu().detach().numpy().flatten()

        # --- Cat ---
        # if actor_outputs['cat'] is not None:
        if 'cat' in self.action_dims and sum(self.action_dims['cat']) > 0:
            cat_probs_list = actor_outputs['cat']
            cat_exec_list = []      # 用于 actions_exec
            cat_indices_raw_list = [] # 用于 actions_raw
            cat_probs_check_list = [] #  记录 Cat 概率
            
            for probs in cat_probs_list:
                probs = probs.squeeze(1)
                dist = Categorical(probs=probs)
                idx = dist.sample() if explore_opts['cat'] else torch.argmax(dist.probs, dim=-1)
                
                if is_batch:
                    cat_exec_list.append(idx.cpu().detach().numpy()) # (Batch, )
                    cat_indices_raw_list.append(idx.cpu().detach().numpy())
                    cat_probs_check_list.append(probs.cpu().detach().numpy())
                else:
                    cat_exec_list.append(idx.item())
                    cat_indices_raw_list.append(idx.item())
                    cat_probs_check_list.append(probs[0].cpu().detach().numpy().copy())
            
            # 这里的 actions_exec['cat'] 现在变成了一个包含索引的 numpy 数组
            if is_batch:
                actions_exec['cat'] = np.stack(cat_exec_list, axis=-1) # (Batch, N_Heads)
                actions_raw['cat'] = np.stack(cat_indices_raw_list, axis=-1)
            else:
                actions_exec['cat'] = np.array(cat_exec_list) 
                actions_raw['cat'] = np.array(cat_indices_raw_list)
            
            #  将所有 Cat 概率分布以列表形式存入诊断输出
            actions_dist_check['cat'] = cat_probs_check_list

        # --- Bern ---
        # if actor_outputs['bern'] is not None:
        if 'bern' in self.action_dims and self.action_dims['bern'] > 0:
            bern_logits = actor_outputs['bern']
            bern_logits = bern_logits.squeeze(1)
            dist = Bernoulli(logits=bern_logits)
            bern_action = dist.sample() if explore_opts['bern'] else (dist.probs > bern_threshold).float()
            
            if is_batch:
                actions_exec['bern'] = bern_action.cpu().detach().numpy() # (Batch, Dim)
                actions_raw['bern'] = actions_exec['bern']
                actions_dist_check['bern'] = dist.probs.cpu().detach().numpy()
            else:
                actions_exec['bern'] = bern_action[0].cpu().detach().numpy().flatten()
                actions_raw['bern'] = actions_exec['bern']
                actions_dist_check['bern'] = dist.probs[0].cpu().detach().numpy().flatten()

        return actions_exec, actions_raw, next_h, actions_dist_check 

    def evaluate_actions(self, states, actions_raw, h=None, max_std=None):
        """
        训练接口。计算整个序列块的 log_probs 和 entropy。
        Args:
            states: tensor (Batch, SeqLen, Dim)
            actions_raw: dict of tensors (Batch, SeqLen, ActionDim)
            h: 序列起始隐藏状态 (num_layers, Batch, HiddenDim)
        Returns:
            log_probs, entropy, entropy_details, actor_outputs, next_h
        """
        # 调用网络 [修改]
        actor_outputs, next_h = self.net(states, h_in=h, max_std=max_std)
        
        # 初始化 log_probs 和 entropy，适配 3D (B, S, 1) 或 2D (B, 1)
        if states.dim() == 3:
            B, S, _ = states.shape
            log_probs = torch.zeros(B, S, 1, device=self.device)
            entropy = torch.zeros(B, S, 1, device=self.device)
        else:
            log_probs = torch.zeros(states.size(0), 1, device=self.device)
            entropy = torch.zeros(states.size(0), 1, device=self.device)
        
        entropy_details = {'cont': None, 'cat': None, 'bern': None}

        # --- Cont ---
        if 'cont' in self.action_dims and self.action_dims['cont'] > 0:
            mu, std = actor_outputs['cont']
            dist = SquashedNormal(mu, std)
            u = actions_raw['cont']
            # log_prob 返回 (B, S, Dim), 求和得到 (B, S, 1)
            log_probs += dist.log_prob(0, u).sum(-1, keepdim=True)
            e_cont = dist.entropy().unsqueeze(-1)
            entropy += e_cont
            entropy_details['cont'] = e_cont

        # --- Cat ---
        if 'cat' in self.action_dims and sum(self.action_dims['cat']) > 0:
            cat_probs_list = actor_outputs['cat']
            cat_action = actions_raw['cat'].long()
            
            #  临时列表用于计算 cat 总熵
            e_cat_sum = torch.zeros_like(entropy)
            
            for i, probs in enumerate(cat_probs_list):
                # act_i 为 (B, S, 1), probs 为 (B, S, HeadDim)
                act_i = cat_action[..., i].unsqueeze(-1) # ...表示前面所有维度
                log_probs += torch.log(probs.gather(-1, act_i) + 1e-8)
                dist = Categorical(probs=probs)
                
                # 累加每个离散头的熵
                e_head = dist.entropy().unsqueeze(-1)
                entropy += e_head
                e_cat_sum += e_head
            
            #  记录 cat entropy
            entropy_details['cat'] = e_cat_sum

        # --- Bern ---
        if 'bern' in self.action_dims and self.action_dims['bern'] > 0:
            bern_logits = actor_outputs['bern']
            dist = Bernoulli(logits=bern_logits)
            bern_action = actions_raw['bern']
            log_probs += dist.log_prob(bern_action).sum(-1, keepdim=True)
            
            #  单独记录 bern entropy
            e_bern = dist.entropy().sum(-1, keepdim=True)
            entropy += e_bern
            entropy_details['bern'] = e_bern # [修改] 保持 Tensor 用于 Loss 计算

        return log_probs, entropy, entropy_details, actor_outputs, next_h
    
# =============================================================================
# 3. PPO 算法类 (精简版)
# =============================================================================

class PPOHybrid:
    def __init__(self, actor, critic, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device, 
                 k_entropy={'cont':0.01, 'cat':0.01, 'bern':0.05}, critic_max_grad=2, actor_max_grad=2, max_std=0.3):
        
        self.actor = actor # 这是一个 HybridActorWrapper 实例
        self.critic = critic
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device
        
        # 解析 k_entropy，支持字典输入
        if isinstance(k_entropy, dict):
            self.k_entropy = k_entropy
        else:
            self.k_entropy = {'cont': k_entropy, 'cat': k_entropy, 'bern': k_entropy}
            
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

    def set_learning_rate(self, actor_lr=None, critic_lr=None):
        if actor_lr is not None:
            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] = actor_lr
        if critic_lr is not None:
            for param_group in self.critic_optimizer.param_groups:
                param_group['lr'] = critic_lr  

    def take_action(self, state, h0=None, explore=True, max_std=None, check_obs=None):
        # 委托给 Actor Wrapper
        max_s = max_std if max_std is not None else self.max_std
        # 注意：这里返回了 hidden_state (虽然是 None)，保持接口一致性
        # actions_exec, actions_raw, _ = self.actor.get_action(state, h=h0, explore=explore, max_std=max_s)
        # return actions_exec, actions_raw
        
        # 显式返回 next_h
        actions_exec, actions_raw, h_state, actions_dist_check = self.actor.get_action(
            state, h=h0, explore=explore, max_std=max_s, check_obs=check_obs
        )
        #  保持原有的返回两个字典的接口，或者根据需要返回 diagnostic output
        return actions_exec, actions_raw, h_state, actions_dist_check

    def update(self, transition_dict, adv_normed=False, clip_vf=False, clip_range=0.2, shuffled=1, mini_batch_size=None, alpha_logit_reg=0.05):
        # 0. 确定是否使用递归模式
        use_rnn = 'init_h_actor' in transition_dict
        
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

        # 1. 基础数据准备 (若是 RNN 模式，此处数据已由 Buffer 整理为 3D: [Batch, SeqLen, Dim])
        states = to_tensor(transition_dict['states'], torch.float)
        # 统一提取 next_values 并根据模式处理维度
        raw_next_vals = to_tensor(transition_dict['next_values'], torch.float)
        
        # 维度适配：RNN模式下 Buffer 已经准备好了 3D (B, S, 128)，MLP模式下是展平后的 2D (T*N, 128)
        if use_rnn:
            dones = to_tensor(transition_dict['dones'], torch.float).unsqueeze(-1)
            rewards = to_tensor(transition_dict['rewards'], torch.float).unsqueeze(-1)
            mb_next_values = raw_next_vals.unsqueeze(-1)
            init_h_actor = to_tensor(transition_dict['init_h_actor'], torch.float).transpose(0, 1).contiguous()
            init_h_critic = to_tensor(transition_dict['init_h_critic'], torch.float).transpose(0, 1).contiguous()
            if 'active_masks' in transition_dict:
                active_masks = to_tensor(transition_dict['active_masks'], torch.float).unsqueeze(-1)
            else:
                # 创建与 dones 形状一致的全 1 张量
                active_masks = torch.ones_like(dones)
        else:
            dones = to_tensor(transition_dict['dones'], torch.float).view(-1, 1)
            rewards = to_tensor(transition_dict['rewards'], torch.float).view(-1, 1)
            mb_next_values = raw_next_vals.view(-1, 1)
            init_h_actor = None
            init_h_critic = None
            if 'active_masks' in transition_dict:
                active_masks = to_tensor(transition_dict['active_masks'], torch.float).view(-1, 1)
            else:
                # 创建与 dones 形状一致的全 1 张量
                active_masks = torch.ones_like(dones)

        # 处理 obs (如果存在)
        if 'obs' in transition_dict:
            actor_inputs = to_tensor(transition_dict['obs'], torch.float)
        else:
            actor_inputs = states
        critic_inputs = states
        
        # 动作处理 (转为 3D Tensor)
        actions_from_buffer = transition_dict['actions']
        actions_on_device = {}
        for key, val in actions_from_buffer.items():
            dtype = torch.long if key == 'cat' else torch.float
            actions_on_device[key] = to_tensor(val, dtype)

        # 2. 获取 Advantage (优先使用 Buffer 算好的)
        if 'advantages' in transition_dict and 'td_targets' in transition_dict:
            if use_rnn:
                advantage = to_tensor(transition_dict['advantages'], torch.float).unsqueeze(-1)
                td_target = to_tensor(transition_dict['td_targets'], torch.float).unsqueeze(-1)
            else:
                advantage = to_tensor(transition_dict['advantages'], torch.float).view(-1, 1)
                td_target = to_tensor(transition_dict['td_targets'], torch.float).view(-1, 1)
        else:
            # 现场计算 GAE (不推荐用于并行展平后的数据)
            # #  处理 truncs
            # if 'truncs' in transition_dict:
            #     # 适配维度：RNN模式 [B, S, 1]，MLP模式 [T*N, 1]
            #     truncs = to_tensor(transition_dict['truncs'], torch.float).view(-1, 1)
            # else:
            #     truncs = None # 或者 torch.zeros_like(dones)

            # 以下为公共部分
            # 如果没有预计算，则现场计算 (注意：如果是并行数据直接展平进来的，这里计算会有偏差)
            with torch.no_grad():
                # 计算当前 V
                if use_rnn:
                    v_res_curr, _ = self.critic(critic_inputs, h_in=init_h_critic)
                else:
                    v_res_curr = self.critic(critic_inputs)
                
                # 兼容 Tuple 和 Tensor 返回值
                if isinstance(v_res_curr, tuple):
                    v_curr = v_res_curr[0]
                else:
                    v_curr = v_res_curr
                
                # td_target的计算不应考虑truncs。仅当dones=1时，next_value才为0。
                # truncs的影响由compute_advantage函数内部处理。
                td_target = rewards + self.gamma * mb_next_values * (1 - dones)  # 均为 (B, S, 1) 或 (T*N, 1)
                td_delta = td_target - v_curr
                
                # 准备进入 compute_advantage，如果是 RNN，需转置时间维到第一维
                if use_rnn:
                    # 转置时间维至第1维以适配 compute_advantage
                    # (B, S, 1) -> (S, B, 1)
                    delta_gae = td_delta.transpose(0, 1)
                    dones_gae = dones.transpose(0, 1)
                    truncs_gae = None
                    if 'truncs' in transition_dict:
                        # 补齐 truncs 序列化掩码
                        t_raw = to_tensor(transition_dict['truncs'], torch.float)
                        truncs_gae = t_raw.unsqueeze(-1).transpose(0, 1)

                    adv_res = compute_advantage(self.gamma, self.lmbda, delta_gae.cpu(), dones_gae.cpu(), truncs_gae.cpu() if truncs_gae is not None else None)
                    advantage = adv_res.transpose(0, 1).to(self.device)
                else:
                    truncs = None
                    if 'truncs' in transition_dict:
                        truncs = to_tensor(transition_dict['truncs'], torch.float).view(-1, 1)
                    advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu(), dones.cpu(), truncs.cpu() if truncs is not None else None).to(self.device)
                
                # # 最终更新为 GAE 目标
                # td_target = advantage + v_curr
                
                # [回归老版逻辑] 丢弃 Buffer 算好的 Lambda-Target，在更新时重新计算 1-step TD Target
                # 这确保了与老版 MLP 逻辑在数学上的绝对对齐
                mb_td_target = rewards + self.gamma * mb_next_values * (1.0 - dones)
                
        # 3. 计算旧策略的 log_probs (使用 Wrapper)
        with torch.no_grad():
            # Actor 使用 actor_inputs (可能是 obs)
            # [修改] 接收 5 个返回值
            old_log_probs, _, _, _ ,_ = self.actor.evaluate_actions(actor_inputs, actions_on_device, h=init_h_actor, max_std=self.max_std)
            if use_rnn : 
                v_res_old, _ = self.critic(critic_inputs, h_in=init_h_critic)
            else:
                v_res_old = self.critic(critic_inputs)
            # Critic 使用 critic_inputs (全局 states)
            v_pred_old = v_res_old[0] if isinstance(v_res_old, tuple) else v_res_old

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

        # 4. PPO Update Loop
        actor_loss_list, critic_loss_list, entropy_list, ratio_list = [], [], [], []
        actor_grad_list, critic_grad_list = [], []
        pre_clip_actor_grad, pre_clip_critic_grad = [], []

        #  监控列表
        kl_list = []
        clip_frac_list = []
        #  分项 Entropy 列表
        entropy_cat_list = []
        entropy_bern_list = []
        entropy_cont_list = []
        
        #  防止除零的小数
        mask_eps = 1e-5
        
        num_samples = actor_inputs.size(0)
        if mini_batch_size is None:
            mini_batch_size = num_samples

        # # 如果 transition_dict 带有序列维度 (seq_len)，则确保 mini_batch_size 为 seq_len 的整数倍（向上取整）
        # if use_rnn and mini_batch_size is not None:
        #     seq_len = np.array(transition_dict['dones']).shape[-1]
        #     if seq_len > 1:
        #         units = int(math.ceil(mini_batch_size / float(seq_len)))
        #         mini_batch_size = min(units, num_samples)


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
                mb_old_log_probs = old_log_probs[batch_idx]
                mb_active_masks = active_masks[batch_idx]
                v_pred_old_batch = v_pred_old[batch_idx]
                
                # 切片隐藏状态 (取对应 Batch 的起始 H) [新增]
                mb_h_a = init_h_actor[:, batch_idx, :] if use_rnn else None
                mb_h_c = init_h_critic[:, batch_idx, :] if use_rnn else None
                
                # 切片 Actions (Dict 结构)
                mb_actions = {}
                for k, v in actions_on_device.items():
                    mb_actions[k] = v[batch_idx]

                # 计算当前策略的 log_probs 和 entropy (使用 Wrapper)
                #  接收 entropy_details 和 actor_outputs
                log_probs, entropy, entropy_details, actor_outputs, _ = self.actor.evaluate_actions(mb_actor_inputs, mb_actions, h=mb_h_a, max_std=self.max_std)
                
                #  计算 log_ratio 用于更精准的 KL 计算
                log_ratio = log_probs - mb_old_log_probs
                ratio = torch.exp(log_ratio)
                
                #  计算 Approximate KL Divergence
                with torch.no_grad():
                    #  KL 计算也最好应用 mask，但为了监控方便，这里先保持全局均值或应用mask均值
                    active_sum = mb_active_masks.sum()
                    approx_kl = (((ratio - 1) - log_ratio) * mb_active_masks).sum() / (active_sum + mask_eps)
                    kl_list.append(approx_kl.item())
                    
                    #  计算 Clip Fraction (有多少样本触发了裁剪)
                    clip_fracs = (((ratio - 1.0).abs() > self.eps).float() * mb_active_masks).sum() / (active_sum + mask_eps)
                    clip_frac_list.append(clip_fracs.item())
                
                surr1 = ratio * mb_advantage
                surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * mb_advantage
                
                #  Actor Loss 使用 mask 加权
                surrogate_loss = -torch.min(surr1, surr2)
                # active_sum 已经在上面计算过
                
                actor_loss = (surrogate_loss * mb_active_masks).sum() / (active_sum + mask_eps)
                
                # [修改] 分项 Entropy Loss 计算
                e_cont = entropy_details['cont'] if entropy_details['cont'] is not None else torch.tensor(0., device=self.device)
                e_cat = entropy_details['cat'] if entropy_details['cat'] is not None else torch.tensor(0., device=self.device)
                e_bern = entropy_details['bern'] if entropy_details['bern'] is not None else torch.tensor(0., device=self.device)

                loss_ent_cont = (e_cont * mb_active_masks).sum() / (active_sum + mask_eps)
                loss_ent_cat = (e_cat * mb_active_masks).sum() / (active_sum + mask_eps)
                loss_ent_bern = (e_bern * mb_active_masks).sum() / (active_sum + mask_eps)

                k_cont = self.k_entropy.get('cont', 0.0)
                k_cat = self.k_entropy.get('cat', 0.0)
                k_bern = self.k_entropy.get('bern', 0.0)

                actor_loss = actor_loss - (k_cont * loss_ent_cont + k_cat * loss_ent_cat + k_bern * loss_ent_bern)

                # [新增] Logit Regularization (防止发射概率锁死在 1.0 或 0.0)
                if actor_outputs['bern'] is not None:
                    bern_logits = actor_outputs['bern']
                    # 仅对绝对值超出阈值的部分施加平缓增加的惩罚：
                    # over = max(|logit| - 4, 0)，惩罚 (over)^2 的均值
                    over = F.relu(torch.abs(bern_logits) - 4.0)
                    logit_loss = (over ** 2 * mb_active_masks).sum() / (mb_active_masks.sum() + mask_eps)
                    actor_loss = actor_loss + alpha_logit_reg * logit_loss

                # Critic Loss
                # Critic 使用 critic_inputs
                if use_rnn:
                    # 使用当前 mini-batch 的状态和对应的初始隐藏状态进行递归推理
                    v_res_batch, _ = self.critic(mb_critic_inputs, h_in=mb_h_c)
                    v_pred = v_res_batch[0] if isinstance(v_res_batch, tuple) else v_res_batch
                else:
                    # MLP 模式：仅输入当前 mini-batch 的状态
                    v_res_batch = self.critic(mb_critic_inputs)
                    v_pred = v_res_batch[0] if isinstance(v_res_batch, tuple) else v_res_batch
                
                # --- 价值损失 (Critic Loss) 计算 ---
                # v_pred_old_batch 是在循环开始处通过 v_pred_old[batch_idx] 得到的，这没问题
                if clip_vf:
                    v_pred_clipped = torch.clamp(v_pred, v_pred_old_batch - clip_range, v_pred_old_batch + clip_range)
                    vf_loss1 = (v_pred - mb_td_target).pow(2)
                    vf_loss2 = (v_pred_clipped - mb_td_target).pow(2)
                    
                    '''hubber loss 防止均方误差带来的梯度过大'''
                    # # 改为:
                    # vf_loss1 = F.smooth_l1_loss(v_pred, mb_td_target, reduction='none', beta=1.0)
                    # vf_loss2 = F.smooth_l1_loss(v_pred_clipped, mb_td_target, reduction='none', beta=1.0)
                    
                    critic_loss_per_sample = torch.max(vf_loss1, vf_loss2)
                else:
                    #  reduction='none' 使得我们可以应用 mask
                    critic_loss_per_sample = F.mse_loss(v_pred, mb_td_target, reduction='none')
                    
                    '''huber loss 防止均方误差带来的梯度过大'''
                    # # 修改为 Huber Loss (或者 Smooth L1 Loss)
                    # # beta=1.0 表示误差小于1时用平方，大于1时用线性
                    # critic_loss_per_sample = F.smooth_l1_loss(v_pred, mb_td_target, reduction='none', beta=1.0) 
                
                #  Critic Loss 使用 mask 加权
                critic_loss = (critic_loss_per_sample * mb_active_masks).sum() / (active_sum + mask_eps)
                
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
                
                # 记录总 Entropy (未加权)
                entropy_total = (entropy * mb_active_masks).sum() / (active_sum + mask_eps)
                entropy_list.append(entropy_total.item()) 
                ratio_list.append(ratio.mean().item()) 
                
                #  [修改] 记录分项 Entropy (现在是 Tensor，需要 .mean().item())
                if entropy_details['cont'] is not None:
                    entropy_cont_list.append(entropy_details['cont'].mean().item())
                if entropy_details['cat'] is not None:
                    entropy_cat_list.append(entropy_details['cat'].mean().item())
                if entropy_details['bern'] is not None:
                    entropy_bern_list.append(entropy_details['bern'].mean().item())

        self.actor_loss = np.mean(actor_loss_list)
        self.actor_grad = np.mean(actor_grad_list)
        self.critic_loss = np.mean(critic_loss_list)
        self.critic_grad = np.mean(critic_grad_list)
        self.entropy_mean = np.mean(entropy_list)
        self.ratio_mean = np.mean(ratio_list)
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
        self.clip_frac = np.mean(clip_frac_list)
        #  计算分项 Entropy 均值
        self.entropy_cont = np.mean(entropy_cont_list) if len(entropy_cont_list) > 0 else 0
        self.entropy_cat = np.mean(entropy_cat_list) if len(entropy_cat_list) > 0 else 0
        self.entropy_bern = np.mean(entropy_bern_list) if len(entropy_bern_list) > 0 else 0
        
        #  计算 Explained Variance
        # y_true: td_target, y_pred: v_pred_old (更新前的值) 或 v_pred (更新后的值，通常用更新前比较多，或者直接对比)
        # 这里使用 numpy 计算以防 tensor 维度广播问题
        #  explained_var 最好也只看 active 的，但为了简单起见，这里先保持原样或简单过滤
        mask_bool = active_masks.flatten().bool().cpu().numpy()
        # mask_bool = active_masks.squeeze(-1).bool().cpu().numpy()
        y_true = td_target.flatten().cpu().numpy()[mask_bool]
        y_pred = v_pred_old.flatten().cpu().numpy()[mask_bool] # 比较更新前的 Value 网络预测能力
        
        if len(y_true) > 1:
            var_y = np.var(y_true)
            if var_y < 1e-8:
                self.explained_var = 0.0
            else:
                self.explained_var = 1 - np.var(y_true - y_pred) / var_y
        else:
            self.explained_var = 0.0

        check_weights_bias_nan(self.actor, "actor", "update后")
        check_weights_bias_nan(self.critic, "critic", "update后")
