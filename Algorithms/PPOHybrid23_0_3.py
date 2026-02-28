'''
使用简单熵正则项的备份

分动作头接收entropy_loss

现在给伯努利分布加入actor内mask

多头注意力改为独立网络，用于调节密集奖励权重
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
# from Algorithms.MLP_heads import ValueNet
from Algorithms.SharedLayers import ChannelAttention
import torch.nn.functional as F

# class RewardWeightAttentionNet(nn.Module):
#     def __init__(self, obs_dim, num_dense_rewards, num_heads=4):
#         super(RewardWeightAttentionNet, self).__init__()
#         self.num_dense = num_dense_rewards
        
#         # 直接使用 MHA 处理 (Batch, R-1, Obs_Dim)
#         # 这里把密集奖励的数量 (R-1) 视为序列长度
#         self.mha = nn.MultiheadAttention(
#             embed_dim=obs_dim, 
#             num_heads=num_heads, 
#             batch_first=True
#         )
        
#         # 最后的线性层将观测维度压缩为 1，从而为每个密集奖励生成一个权重
#         self.fc_weight = nn.Linear(obs_dim, 1)
        
#         # 权重初始化：使初始权重接近均匀分布，避免训练初期某些奖励项直接消失
#         nn.init.orthogonal_(self.fc_weight.weight, gain=0.01)
#         nn.init.constant_(self.fc_weight.bias, 0)

#     def forward(self, x):
#         """
#         Input x: (Batch, num_dense, obs_dim)
#         Output: (Batch, num_dense)
#         """
#         # 1. 计算自注意力：让不同奖励项之间通过观测信息进行“协商”
#         # attn_output: (Batch, num_dense, obs_dim)
#         attn_output, _ = self.mha(x, x, x)
        
#         # 2. 映射到标量权重
#         # logits: (Batch, num_dense, 1) -> (Batch, num_dense)
#         logits = self.fc_weight(attn_output).squeeze(-1)
        
#         # 3. 归一化：确保 sum(weights) = 1
#         weights = F.softmax(logits, dim=-1)
#         return weights

class RewardWeightAttentionNet(nn.Module):
    def __init__(self, obs_dim, num_dense_rewards, embed_dim=128, num_heads=4):
        super().__init__()
        self.num_dense = num_dense_rewards
        
        # 1. 将 12 个奖励数值投影到高维 Query 空间
        # 输入: (Batch, 12, 1) -> 输出: (Batch, 12, embed_dim)
        self.query_proj = nn.Linear(1, embed_dim)
        
        # 2. 将状态投影到 Key/Value 空间
        # 输入: (Batch, obs_dim) -> 输出: (Batch, 1, embed_dim)
        self.kv_proj = nn.Linear(obs_dim, embed_dim)
        
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        
        # 3. 最终将混合了状态信息的奖励特征映射回标量权重
        self.fc_weight = nn.Linear(embed_dim, 1)
        
        # RL 惯例：使用较小的增益初始化，保持初始权重接近均匀
        nn.init.orthogonal_(self.fc_weight.weight, gain=0.01)

    def forward(self, state, dense_rewards_raw):
        """
        state: (Batch, obs_dim)
        dense_rewards_raw: (Batch, 12)
        """
        batch_size = state.size(0)
        
        # 构造 Query: (Batch, 12, embed_dim)
        # 将原始奖励数值作为 Query 的原始特征
        q_input = dense_rewards_raw.unsqueeze(-1) # (Batch, 12, 1)
        queries = self.query_proj(q_input) 
        
        # 构造 Key/Value: (Batch, 1, embed_dim)
        # 将状态作为背景信息供奖励查询
        kv_input = state.unsqueeze(1) # (Batch, 1, obs_dim)
        keys = values = self.kv_proj(kv_input)
        
        # 注意力计算 (Cross-Attention)
        # 因为 L_k=1, attn_output 实际上是 queries 对 keys 的线性响应
        # 此时梯度可以顺畅地流回 q_input (dense_rewards_raw)
        attn_output, _ = self.mha(queries, keys, values)
        
        # --- 核心改进：引入残差连接 ---
        # 即使 MHA 的输出因为 Lk=1 而趋同，加上 queries 之后，
        # 12 个槽位的特征就带上了原始奖励的差异性。
        combined_output = attn_output # + queries
        
        # 映射到 Logits 并 Softmax
        # logits: (Batch, 12)
        logits = self.fc_weight(combined_output).squeeze(-1)
        weights = F.softmax(logits, dim=-1)
        
        return weights
     
# =============================================================================
# 1. 策略网络 (集成 Embedding, Attention 与正交初始化)
# =============================================================================
class PolicyNetHybrid(torch.nn.Module):
    def __init__(self, state_dim, hidden_dims, action_dims_dict, init_std=0.5, 
                 num_heads=4, use_attention=True):
        super(PolicyNetHybrid, self).__init__()
        self.action_dims = action_dims_dict
        # self.use_attention = use_attention
        
        # 1. 嵌入层：将状态映射到第一个隐藏层维度
        self.embed = nn.Linear(state_dim, hidden_dims[0])
        
        # # 2. 空间多头注意力层 (替换原来的 ChannelAttention)
        # if self.use_attention:
        #     self.attention = SpatialMultiHeadAttention(hidden_dims[0], num_heads=num_heads)

        # 3. 共享主干网络 (MLP)
        layers = []
        prev_size = hidden_dims[0]
        for layer_size in hidden_dims:
            layers.append(nn.Linear(prev_size, layer_size))
            layers.append(nn.ReLU())
            prev_size = layer_size
        self.backbone = nn.Sequential(*layers)

        # 4. 动作头定义
        # 连续动作头 (Continuous)
        if 'cont' in self.action_dims and self.action_dims['cont'] > 0:
            cont_dim = self.action_dims['cont']
            self.fc_mu = nn.Linear(prev_size, cont_dim)
            self.log_std_cont = nn.Parameter(torch.log(torch.ones(cont_dim) * init_std))

        # 离散动作头 (Categorical)
        if 'cat' in self.action_dims and sum(self.action_dims['cat']) > 0:
            self.cat_dims = self.action_dims['cat']
            self.fc_cat = nn.Linear(prev_size, sum(self.cat_dims))

        # 伯努利动作头 (Bernoulli)
        if 'bern' in self.action_dims and self.action_dims['bern'] > 0:
            self.fc_bern = nn.Linear(prev_size, self.action_dims['bern'])

        # --- 5. 权重初始化 (强化学习推荐方案) ---
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用正交初始化
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                # 防御性判断：只有当 bias 存在时才初始化为 0
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # 针对动作头做微调初始化
        if hasattr(self, 'fc_mu'):
            nn.init.orthogonal_(self.fc_mu.weight, gain=0.01) # 初始 mu 接近 0
        
        if hasattr(self, 'fc_bern'):
            # 关键：手动设置负偏置，使初始开火概率较低 (约 0.12)
            nn.init.constant_(self.fc_bern.bias, -2.0)
            nn.init.normal_(self.fc_bern.weight, std=0.01)
        
        # 离散动作 (fc_cat)：不需要特殊技巧
        # 保持上面的 orthogonal_(gain=sqrt(2)) 即可。
        # 如果你希望初始分布更“均匀”一点，可以用 gain=1.0，但 sqrt(2) 也是完全没问题的。

    def forward(self, x, min_std=1e-6, max_std=1.0, action_masks=None):
        # Step 1: Embedding
        x_f = self.embed(x)
        
        # # Step 2: Attention (判断逻辑)
        # if self.use_attention:
        #     x_f = self.attention(x_f) # MHA 处理
        
        # Step 3: MLP Backbone
        shared_features = self.backbone(x_f)
        
        outputs = {'cont': None, 'cat': None, 'bern': None}

        # --- Continuous ---
        if 'cont' in self.action_dims and self.action_dims['cont'] > 0:
            mu = self.fc_mu(shared_features)
            # 计算 std
            std = torch.exp(self.log_std_cont)
            std = torch.clamp(std, min=min_std, max=max_std)
            # 扩展维度以匹配 batch
            if mu.dim() > 1:
                std = std.unsqueeze(0).expand_as(mu)
            outputs['cont'] = (mu, std)

        # --- Categorical ---
        if 'cat' in self.action_dims and sum(self.action_dims['cat']) > 0:
            cat_logits_all = self.fc_cat(shared_features)
            
            # 1. 切分 Logits
            cat_logits_list = torch.split(cat_logits_all, self.cat_dims, dim=-1)
            
            # 2. 获取温度 (Temp = exp(log_temp))
            # temp_cat 形状: (num_heads, )
            temps = 1.0  # use scalar temp (or replace with tensor if per-head temps are needed)
            
            # 3. 应用温度缩放 (Logits / Temp) 并 Softmax
            # 较高的 Temp -> Logits 数值变小 -> Softmax 后分布趋向均匀 (熵增大)
            # 较低的 Temp -> Logits 数值差距拉大 -> Softmax 后分布趋向 One-hot (熵减小)
            final_probs_list = []
            for i, logits in enumerate(cat_logits_list):
                # 对应的温度: temps[i]
                # scaled_logits = logits / (temps[i] + 1e-8)
                scaled_logits = logits / (temps + 1e-8)
                final_probs_list.append(F.softmax(scaled_logits, dim=-1))
            
            outputs['cat'] = final_probs_list

        # --- Bernoulli (核心修改区域) ---
        if 'bern' in self.action_dims and self.action_dims['bern'] > 0:
            bern_logits = self.fc_bern(shared_features)
            # Action Masking 逻辑
            if action_masks is not None and 'bern' in action_masks:
                mask = action_masks['bern']
                # 确保 mask 和 logits 维度匹配 (Batch, Dim)
                # mask == 0 代表禁止开火，设为极小值
                bern_logits = bern_logits.masked_fill(mask == 0, -1e9)

            # use scalar temp (no tensor ops on plain number)
            temps = 1.0 # torch.exp(self.log_temp_bern)
            scaled_bern_logits = bern_logits / (temps + 1e-8)
            outputs['bern'] = scaled_bern_logits
            
        return outputs

# =============================================================================
# 2. 价值网络 (非递归版本，结构与 PolicyNet 对称)
# =============================================================================
class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dims, num_heads=4, use_attention=True):
        super(ValueNet, self).__init__()
        # self.use_attention = use_attention
        
        self.embed = nn.Linear(state_dim, hidden_dims[0])
        
        # if self.use_attention:
        #     self.attention = SpatialMultiHeadAttention(hidden_dims[0], num_heads=num_heads)
        
        layers = []
        prev_size = hidden_dims[0]
        for h in hidden_dims:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            prev_size = h
        self.backbone = nn.Sequential(*layers)
        
        # 最终输出一个标量 Value
        self.fc_out = nn.Linear(prev_size, 1)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Value 输出层通常使用较小的增益
        nn.init.orthogonal_(self.fc_out.weight, gain=1.0)

    def forward(self, x):
        x_f = self.embed(x)
        # if self.use_attention:
        #     x_f = self.attention(x_f)
        
        features = self.backbone(x_f)
        return self.fc_out(features)

# =============================================================================
# 2. Actor 适配器 (Wrapper) - 核心重构点
# =============================================================================

class HybridActorWrapper(nn.Module):
    """
    统一接口适配器。
    将具体的 PolicyNetHybrid 封装起来，对外提供标准的 get_action 和 evaluate_actions 接口。
    未来如果引入 GRU，只需修改这个 Wrapper 或替换为 RecurrentActorWrapper，PPO 算法本身无需修改。
    """
    def __init__(self, policy_net, action_dims_dict, action_bounds=None, device='cpu'):
        super(HybridActorWrapper, self).__init__()
        self.net = policy_net
        self.action_dims = action_dims_dict
        self.device = device
        
        # 处理 Action Bounds
        if 'cont' in self.action_dims and self.action_dims['cont'] > 0:
            # 如果有连续动作，必须提供 action_bounds
            if action_bounds is None:
                raise ValueError("Continuous action space requires action_bounds")
            self.register_buffer('action_bounds', torch.tensor(action_bounds, dtype=torch.float, device=device))
            self.register_buffer('amin', self.action_bounds[:, 0])
            self.register_buffer('amax', self.action_bounds[:, 1])
            self.register_buffer('action_span', self.amax - self.amin)

    def _scale_action_to_exec(self, a_norm):
        return self.amin + (a_norm + 1.0) * 0.5 * self.action_span

    # [修改] 增加 check_obs 参数，默认为 None
    def get_action(self, state, h=None, explore=True, max_std=None, check_obs=None, bern_threshold=0.5):
        """
        推理接口。
        Args:
            state: numpy array or tensor
            h: hidden state (预留接口，目前未使用)
            explore: bool or dict. If bool, applies to all action types.
                     If dict, e.g., {'cont': True, 'cat': False, 'bern': True}, controls exploration for each type.
        Returns:
            actions_exec: dict (numpy), 用于环境执行
            actions_raw: dict (numpy/tensor), 用于存入 buffer
            next_h: hidden state (预留接口)
            
        注意： 仅在推理时传入check_obs, 训练时禁止传入!!!
        1、目前 get_action 中的 mask 生成只处理单个 check_obs（推理时），
            并把同一 mask 广播到整个 batch；如果要对 batch 内每个样本分别判断需扩展生成逻辑。
        2、evaluate_actions（训练/计算 log_prob）默认未把 action_masks 传给 net ,
            若希望训练时也应用 mask，需要在 evaluate_actions 调用 net 时传入 action_masks。
        """
        #  增强的 Batch 检测逻辑
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
        
        # [修改] 处理 explore 参数，使其支持字典
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

        # =====================================================================
        # [新增] 解析 check_obs 并构建 Action Mask
        # =====================================================================
        action_masks = None
        can_fire = True
        # 当且仅当传入了单个 dict 类型的 check_obs 时启用 mask
        if (check_obs is not None) and isinstance(check_obs, dict):  # and (not explore_opts['bern']):
            # 默认允许开火，下面按规则逐项收敛（保留注释）
            can_fire = True
            # 如果是Batch训练模式，通常check_obs会增加维度，这里只在推理的时候启用

            # 1. ATA <= 60度 (0.5236 rad)
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
        # =====================================================================

        # [修改] 调用网络时传入 action_masks
        actor_outputs = self.net(state, max_std=max_std, action_masks=action_masks)
        
        # # [原有] 调用网络
        # actor_outputs = self.net(state, max_std=max_std)  # 如果需要gru，改动这一行
        
        actions_exec = {}
        actions_raw = {}
        actions_dist_check = {} #  诊断输出

        # --- Cont ---
        if actor_outputs['cont'] is not None:
            mu, std = actor_outputs['cont']
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
        if actor_outputs['cat'] is not None:
            cat_probs_list = actor_outputs['cat']
            cat_exec_list = []      # 用于 actions_exec
            cat_indices_raw_list = [] # 用于 actions_raw
            cat_probs_check_list = [] #  记录 Cat 概率
            
            for probs in cat_probs_list:
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
        if actor_outputs['bern'] is not None:
            bern_logits = actor_outputs['bern']
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

        return actions_exec, actions_raw, None, actions_dist_check # None for hidden state

    def evaluate_actions(self, states, actions_raw, h=None, max_std=None):
        """
        训练接口。计算 log_probs 和 entropy。
        Args:
            states: tensor (B, D)
            actions_raw: dict of tensors
        Returns:
            log_probs: tensor (B, 1)
            entropy: tensor (B, 1)
            next_h: None
            actor_outputs: dict (raw outputs from net) [新增]
        """
        actor_outputs = self.net(states, max_std=max_std)
        log_probs = torch.zeros(states.size(0), 1).to(self.device)
        entropy = torch.zeros(states.size(0), 1).to(self.device)
        
        #  用于记录分项 Entropy 的字典
        entropy_details = {'cont': None, 'cat': None, 'bern': None}

        # --- Cont ---
        if 'cont' in self.action_dims and self.action_dims['cont'] > 0:
            mu, std = actor_outputs['cont']
            dist = SquashedNormal(mu, std)
            u = actions_raw['cont']
            log_probs += dist.log_prob(0, u).sum(-1, keepdim=True)
            entropy += dist.entropy().unsqueeze(-1) # 近似熵
            
            #  单独记录 cont entropy
            e_cont = dist.entropy().unsqueeze(-1)
            entropy += e_cont
            entropy_details['cont'] = e_cont # [修改] 保持 Tensor 用于 Loss 计算

        # --- Cat ---
        if 'cat' in self.action_dims and sum(self.action_dims['cat']) > 0:
            cat_probs_list = actor_outputs['cat']
            cat_action = actions_raw['cat'].long()
            
            #  临时列表用于计算 cat 总熵
            e_cat_sum = torch.zeros_like(entropy)
            
            for i, probs in enumerate(cat_probs_list):
                act_i = cat_action[:, i].unsqueeze(-1)
                log_probs += torch.log(probs.gather(1, act_i) + 1e-8)
                dist = Categorical(probs=probs)
                
                # 累加每个离散头的熵
                e_head = dist.entropy().unsqueeze(-1)
                entropy += e_head
                e_cat_sum += e_head
            
            #  记录 cat entropy
            entropy_details['cat'] = e_cat_sum # [修改] 保持 Tensor 用于 Loss 计算

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

        # [修改] 返回 actor_outputs 以便外部访问 logits
        return log_probs, entropy, entropy_details, actor_outputs, None
    
    def compute_il_loss(self, states, expert_actions, label_smoothing=0.1):
        """
        计算模仿学习 Loss (MARWIL / BC)。
        
        Args:
            states: (Batch, State_Dim)
            expert_actions: 字典, 包含 {'cont': u, 'cat': index, 'bern': float}
                            注意：对于连续动作，这里通常假设传入的是 pre-tanh 的 u，
                            或者你需要在外部处理好。
            label_smoothing: 标签平滑系数
            
        Returns:
            total_loss_per_sample: (Batch, ) 每个样本的 Loss 总和 (未平均)
        """
        actor_outputs = self.net(states) # 获取 raw output (mu/std, logits)
        
        # 初始化一个全 0 的 loss tensor，形状 (Batch, )
        total_loss_per_sample = torch.zeros(states.size(0), device=self.device)

        # --- 1. 连续动作 (Continuous) ---
        # 依据提供的 PPOContinuous 代码，MARWIL 使用 log_prob(u)
        if 'cont' in self.action_dims and self.action_dims['cont'] > 0:
            mu, std = actor_outputs['cont']
            dist = SquashedNormal(mu, std)
            u_expert = expert_actions['cont'] # 假设传入的是 pre-tanh value
            
            # 计算 log_prob，维度求和保持 (Batch, 1) -> squeeze 为 (Batch, )
            # Loss = - log_prob
            cont_loss = -dist.log_prob(0, u_expert).sum(dim=-1)
            total_loss_per_sample += cont_loss

        # --- 2. 离散/多离散动作 (Categorical) ---
        # 依据提供的 Multi-Discrete 代码，使用 CrossEntropy
        if 'cat' in self.action_dims and sum(self.action_dims['cat']) > 0:
            cat_logits_list = actor_outputs['cat'] # 注意：这里 net forward 返回的是 softmax 后的 probs 还是 logits? 
            # 修正：你的 PolicyNetHybrid forward 返回的是 [F.softmax(logits)...]
            # 为了数值稳定性，建议 PolicyNetHybrid 改为返回 logits，或者在这里取 log
            
            # 假设 expert_actions['cat'] 是 (Batch, Num_Heads)
            expert_cat = expert_actions['cat'].long()
            
            for i, probs in enumerate(cat_logits_list):
                # probs: (Batch, N_Class)
                expert_idx = expert_cat[:, i] # (Batch, )
                
                log_probs = torch.log(probs + 1e-10)
                
                if label_smoothing > 0:
                    # Label Smoothing 逻辑
                    n_classes = probs.size(1)
                    one_hot = torch.zeros_like(probs).scatter_(1, expert_idx.unsqueeze(1), 1.0)
                    smooth_target = one_hot * (1.0 - label_smoothing) + (label_smoothing / n_classes)
                    # CrossEntropy: - sum(target * log_p)
                    ce_loss = -torch.sum(smooth_target * log_probs, dim=1)
                else:
                    # 标准 CE: - log_p[target]
                    # gather 需要 index 维度为 (Batch, 1)
                    ce_loss = -log_probs.gather(1, expert_idx.unsqueeze(1)).squeeze(1)
                
                total_loss_per_sample += ce_loss

        # --- 3. 伯努利动作 (Bernoulli) ---
        # -- 1）简单加权BCE --
        # 依据提供的 Bernoulli 代码，使用 BCE
        # if 'bern' in self.action_dims and self.action_dims['bern'] > 0:
        #     bern_logits = actor_outputs['bern']
        #     probs = torch.sigmoid(bern_logits) # 或者是 net 直接输出 logits，这里转 prob
        #     target = expert_actions['bern'] # (Batch, Dim)
        #     # Label Smoothing
        #     if label_smoothing > 0:
        #         target = target * (1.0 - label_smoothing) + 0.5 * label_smoothing
        #     # BCE: - [y*log(p) + (1-y)*log(1-p)]
        #     # 加上 clamp 防止 log(0)
        #     probs = torch.clamp(probs, 1e-10, 1.0 - 1e-10)
        #     # [关键修改] 引入正样本权重 (pos_weight)
        #     # 假设发射只占 1/50，那么 pos_weight 可以设为 10.0 到 50.0 之间
        #     # 这意味着每一条“发射”指令产生的 Loss 会被放大 N 倍
        #     pos_weight = 3  # 建议从 10.0 开始尝试，如果还不敢打就加到 20.0-50.0
        #     bce_loss = -(pos_weight * target * torch.log(probs) + (1.0 - target) * torch.log(1.0 - probs))
        #     # 对动作维度求和 (Batch, Dim) -> (Batch, )
        #     total_loss_per_sample += bce_loss.sum(dim=-1)
        
        # -- 2）Focal Loss --
        if 'bern' in self.action_dims and self.action_dims['bern'] > 0:
            bern_logits = actor_outputs['bern']
            probs = torch.sigmoid(bern_logits)
            probs = torch.clamp(probs, 1e-10, 1.0 - 1e-10)
            target = expert_actions['bern'] # (Batch, 1)
            # Label Smoothing
            if label_smoothing > 0:
                target = target * (1.0 - label_smoothing) + 0.5 * label_smoothing
            # === 方案2：Focal Loss (针对敏感度问题) ===
            # alpha: 平衡因子，类似于 pos_weight 的作用，但范围是 0-1
            # gamma: 聚焦因子，通常设为 2.0。值越大，越忽视简单背景，越关注难分类的发射瞬间
            
            # 建议参数组合：
            # alpha = 0.75 (意味着正样本本身权重是 0.75，负样本是 0.25，自带 3:1 的加权)
            # gamma = 2.0 (标准设置)
            
            alpha = 0.75
            gamma = 2.0
            
            # Focal Loss 公式
            # 对于正样本 (target=1): -alpha * (1-p)^gamma * log(p)
            # 对于负样本 (target=0): -(1-alpha) * p^gamma * log(1-p)
            
            loss_pos = -alpha * torch.pow(1.0 - probs, gamma) * torch.log(probs) * target
            loss_neg = -(1 - alpha) * torch.pow(probs, gamma) * torch.log(1.0 - probs) * (1.0 - target)
            
            bce_loss = loss_pos + loss_neg
            
            total_loss_per_sample += bce_loss.sum(dim=-1)
        
        return total_loss_per_sample
# =============================================================================
# 3. PPO 算法类 (精简版)
# =============================================================================

class PPOHybrid:
    def __init__(self, actor, critic, reward_att_net, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device, 
                 obs_dim=None, num_dense_rewards=None, # 新增参数
                 k_entropy={'cont':0.01, 'cat':0.01, 'bern':0.05}, k_att_entropy = 0, critic_max_grad=2, actor_max_grad=2, max_std=0.3, **kwargs):
        
        self.actor = actor # 这是一个 HybridActorWrapper 实例
        self.critic = critic
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.reward_att_net = reward_att_net
        # 新增：注意力奖励网络
        # reward_att_net = RewardWeightAttentionNet(obs_dim, num_dense_rewards).to(device)
        self.att_optimizer = torch.optim.Adam(self.reward_att_net.parameters(), lr=actor_lr*10) # lr=actor_lr
        
        self.num_dense = num_dense_rewards
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
        
        self.k_att_entropy = k_att_entropy
        
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

        # Att net metrics
        self.att_loss = 0
        self.att_entropy = 0
        self.att_grad = 0
        self.pre_clip_att_grad = 0

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
        
        # [修改] 透传 check_obs
        actions_exec, actions_raw, h_state, actions_dist_check = self.actor.get_action(
            state, h=h0, explore=explore, max_std=max_s, check_obs=check_obs
        )
        #  保持原有的返回两个字典的接口，或者根据需要返回 diagnostic output
        return actions_exec, actions_raw, h_state, actions_dist_check

    def update(self, transition_dict, adv_normed=False, clip_vf=False, clip_range=0.2, shuffled=1, mini_batch_size=None, alpha_logit_reg=0.05, **kwargs):

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
            
        # 【修复：IndexError 保护】空经验池保护
        if len(transition_dict['dones']) == 0:
            return
        
        states = to_tensor(transition_dict['states'], torch.float)
        next_states = to_tensor(transition_dict['next_states'], torch.float)
        dones = to_tensor(transition_dict['dones'], torch.float).view(-1, 1)

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
        
        # 原始奖励数据：(Batch, num_rewards)
        # 最后一列是结果奖励，前面是密集奖励
        raw_rewards_all = to_tensor(transition_dict['rewards'], torch.float)
        dense_rewards_raw = raw_rewards_all[:, :-1]  # (Batch, R-1)
        sparse_reward = raw_rewards_all[:, -1:]      # (Batch, 1)
        # 2. 奖励重塑 (Reward Shaping via Attention)
        
        
        # 【关键点】带梯度计算权重 w
        
        with torch.no_grad():
            # 构造输入: (Batch, R-1, Obs_Dim)
            # initial_att_input = states.unsqueeze(1).repeat(1, self.num_dense, 1)
            # initial_weights = self.reward_att_net(initial_att_input) # weights: (Batch, R-1)
            
            initial_weights = self.reward_att_net(states, dense_rewards_raw)
        
            # 聚合奖励 r_grad (用于回传给 AM 网络)
            fixed_rewards = torch.sum(dense_rewards_raw * initial_weights, dim=-1, keepdim=True) * self.num_dense + sparse_reward
        
        # 【关键点】创建奖励复制品 (用于计算 GAE，不带梯度)
        combined_rewards_fixed = fixed_rewards.detach()
        rewards = combined_rewards_fixed

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
            # Actor 使用 actor_inputs (可能是 obs)
            # [修改] 接收 5 个返回值
            old_log_probs, _, _, _ ,_ = self.actor.evaluate_actions(actor_inputs, actions_on_device, h=None, max_std=self.max_std)
            # Critic 使用 critic_inputs (全局 states)
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

        # 4. PPO Update Loop
        actor_loss_list, critic_loss_list, entropy_list, ratio_list = [], [], [], []
        actor_grad_list, critic_grad_list = [], []
        pre_clip_actor_grad, pre_clip_critic_grad = [], []
 
        # Attention net monitoring lists
        att_loss_list = []
        att_entropy_list = []
        att_grad_list = []
        pre_clip_att_grad = []
        
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
                # v_pred_old_batch = v_pred_old[batch_idx] # 如果需要用到旧 Value
                mb_dense_rewards = dense_rewards_raw[batch_idx]
                mb_sparse_reward = sparse_reward[batch_idx]
                
                # 切片 Actions (Dict 结构)
                mb_actions = {}
                for k, v in actions_on_device.items():
                    mb_actions[k] = v[batch_idx]

                # 计算当前策略的 log_probs 和 entropy (使用 Wrapper)
                #  接收 entropy_details 和 actor_outputs
                log_probs, entropy, entropy_details, actor_outputs, _ = self.actor.evaluate_actions(mb_actor_inputs, mb_actions, h=None, max_std=self.max_std)
                
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
                    logit_loss = (over ** 2).mean()
                    actor_loss = actor_loss + alpha_logit_reg * logit_loss

                # Critic Loss
                # Critic 使用 critic_inputs
                v_pred = self.critic(mb_critic_inputs)
                if clip_vf:
                    v_pred_old_batch = v_pred_old[batch_idx]
                    v_pred_clipped = torch.clamp(v_pred, v_pred_old_batch - clip_range, v_pred_old_batch + clip_range)
                    vf_loss1 = (v_pred - mb_td_target).pow(2)
                    vf_loss2 = (v_pred_clipped - mb_td_target).pow(2)
                    critic_loss_per_sample = torch.max(vf_loss1, vf_loss2)
                else:
                    #  reduction='none' 使得我们可以应用 mask
                    critic_loss_per_sample = F.mse_loss(v_pred, mb_td_target, reduction='none')
                
                #  Critic Loss 使用 mask 加权
                critic_loss = (critic_loss_per_sample * mb_active_masks).sum() / (active_sum + mask_eps)
                
                # --- Attention Network Loss (AM 更新的核心) ---
                # 重新建立 AM 网络与优势的梯度联系
                # 【核心修复】在循环内部实时计算权重以建立梯度路径，防止计算图重复使用报错
                # mb_att_input = mb_actor_inputs.unsqueeze(1).repeat(1, self.num_dense, 1)
                # mb_weights = self.reward_att_net(mb_att_input) # 这里产生带梯度的最新权重
                mb_weights = self.reward_att_net(mb_actor_inputs, mb_dense_rewards)
                
                # mb_reward_grad 是关于 mb_weights 的函数，梯度可以传回 reward_att_net
                # 注意：这里使用当前 Batch 的最新权重重新聚合奖励
                mb_reward_grad = torch.sum(mb_dense_rewards * mb_weights, dim=-1, keepdim=True)
                # 逻辑：如果优势 mb_advantage 为正，AM 应该调高对应的权重来进一步增大聚合奖励
                # 我们使用 mb_advantage 作为系数，因为它代表了动作的“正确性”
                # 注意：mb_advantage 必须 detach，确保 AM 网络只通过改变权重来优化，而不去干扰 Critic
                att_loss = -(mb_reward_grad * mb_advantage.detach() * mb_active_masks).sum() / (active_sum + mask_eps)
                
                # AM 网络的熵正则项：鼓励权重分布不要太集中
                # 【修复】直接使用当前 Batch 的 mb_weights，原先的 [idx] 会导致索引越界
                att_entropy = -(mb_weights * torch.log(mb_weights + 1e-8)).sum(-1).mean()
                
                k_att_entropy = self.k_att_entropy
                # 再加上你设置的 AM 熵正则
                att_loss = att_loss - k_att_entropy * att_entropy
                
                # # # 调试
                # att_loss *= 100.0

                # 记录当前 att_loss / att_entropy （值可在 step 前记录）
                att_loss_list.append(att_loss.item())
                att_entropy_list.append(att_entropy.item())
                 
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                self.att_optimizer.zero_grad()
                
                actor_loss.backward()
                critic_loss.backward()
                att_loss.backward()

                pre_clip_actor_grad.append(model_grad_norm(self.actor))
                pre_clip_critic_grad.append(model_grad_norm(self.critic))
                # 记录 reward_att_net 在裁剪前的梯度范数
                pre_clip_att_grad.append(model_grad_norm(self.reward_att_net))
                nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.actor_max_grad)
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.critic_max_grad)
                nn.utils.clip_grad_norm_(self.reward_att_net.parameters(), self.actor_max_grad)

                self.actor_optimizer.step()
                self.critic_optimizer.step()
                self.att_optimizer.step()
 
                # 记录 reward_att_net 更新后的梯度范数
                att_grad_list.append(model_grad_norm(self.reward_att_net))
 
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

        self.mb_weights = mb_weights
        
        self.actor_loss = np.mean(actor_loss_list)
        self.actor_grad = np.mean(actor_grad_list)
        self.critic_loss = np.mean(critic_loss_list)
        self.critic_grad = np.mean(critic_grad_list)
        self.entropy_mean = np.mean(entropy_list)
        self.ratio_mean = np.mean(ratio_list)
        self.pre_clip_critic_grad = np.mean(pre_clip_critic_grad)
        self.pre_clip_actor_grad = np.mean(pre_clip_actor_grad)
        # 记录 att 指标
        self.att_loss = np.mean(att_loss_list) if len(att_loss_list) > 0 else 0
        self.att_entropy = np.mean(att_entropy_list) if len(att_entropy_list) > 0 else 0
        self.att_grad = np.mean(att_grad_list) if len(att_grad_list) > 0 else 0
        self.pre_clip_att_grad = np.mean(pre_clip_att_grad) if len(pre_clip_att_grad) > 0 else 0
        
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
            if var_y < 1e-8:
                self.explained_var = 0.0
            else:
                self.explained_var = 1 - np.var(y_true - y_pred) / var_y
        else:
            self.explained_var = 0.0

        check_weights_bias_nan(self.actor, "actor", "update后")
        check_weights_bias_nan(self.critic, "critic", "update后")
