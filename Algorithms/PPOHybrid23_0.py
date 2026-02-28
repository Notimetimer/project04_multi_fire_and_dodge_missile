'''
PPO-DPC 消融实验
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
# 1. 神经网络定义 (保持不变，只负责 forward 计算)
# =============================================================================

class PolicyNetHybrid(torch.nn.Module):
    """
    支持混合动作空间的策略网络 (纯 MLP)。
    引入了可学习的温度参数来控制离散和伯努利动作的熵。
    """
    def __init__(self, state_dim, hidden_dims, action_dims_dict, init_std=0.5):
        super(PolicyNetHybrid, self).__init__()
        self.action_dims = action_dims_dict
        
        # 共享主干网络
        layers = []
        prev_size = state_dim
        for layer_size in hidden_dims:
            layers.append(nn.Linear(prev_size, layer_size))
            layers.append(nn.ReLU())
            prev_size = layer_size
        self.net = nn.Sequential(*layers)

        # 1. 连续动作头 (Continuous)
        # 参数: log_std (控制高斯分布宽度)
        if 'cont' in self.action_dims and self.action_dims['cont'] > 0:
            cont_dim = self.action_dims['cont']
            self.fc_mu = nn.Linear(prev_size, cont_dim)
            # 这里的 log_std 依然是状态无关的，对应 PPO 的标准做法
            self.log_std_cont = nn.Parameter(torch.log(torch.ones(cont_dim) * init_std))

        # 2. 离散动作头 (Categorical)
        # 参数: log_temp_cat (控制 Softmax 温度)
        if 'cat' in self.action_dims and sum(self.action_dims['cat']) > 0:
            self.cat_dims = self.action_dims['cat']  # list, e.g., [4, 10]
            total_cat_dim = sum(self.cat_dims)
            self.fc_cat = nn.Linear(prev_size, total_cat_dim)
            
            # 为每一个独立的离散头 (Head) 创建一个温度参数
            # 比如有 [4, 10] 两个头，我们就需要 2 个温度参数
            # 初始化为 0 (即 temp=1.0)，保持原网络特性，让网络自己学去增大熵
            # self.log_temp_cat = nn.Parameter(torch.zeros(len(self.cat_dims))) 

        # 3. 伯努利动作头 (Bernoulli)
        # 参数: log_temp_bern (控制 Sigmoid 陡峭度)
        if 'bern' in self.action_dims and self.action_dims['bern'] > 0:
            bern_dim = self.action_dims['bern']
            self.fc_bern = nn.Linear(prev_size, bern_dim)
            # 初始化 bias 为 -2，使初始开火概率较低（sigmoid(-2) ≈ 0.12）
            nn.init.constant_(self.fc_bern.bias, -2.0)
            
            # 为每一个伯努利动作维度创建一个温度参数
            # 初始化为 0 (即 temp=1.0)
            # self.log_temp_bern = nn.Parameter(torch.zeros(bern_dim))
    
    # [修改] 增加 action_masks 参数, [新增] 增加 temp 参数
    def forward(self, x, min_std=1e-6, max_std=1.0, action_masks=None, temp=1.0):
        shared_features = self.net(x)
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
            # temps = 1.0  # [修改] 使用传入的 temp
            
            # 3. 应用温度缩放 (Logits / Temp) 并 Softmax
            # 较高的 Temp -> Logits 数值变小 -> Softmax 后分布趋向均匀 (熵增大)
            # 较低的 Temp -> Logits 数值差距拉大 -> Softmax 后分布趋向 One-hot (熵减小)
            final_probs_list = []
            for i, logits in enumerate(cat_logits_list):
                # 对应的温度: temps[i]
                # scaled_logits = logits / (temps[i] + 1e-8)
                # 使用传入的 temp 进行缩放, 防止除0
                scaled_logits = logits / (temp + 1e-8)
                final_probs_list.append(F.softmax(scaled_logits, dim=-1))
            
            outputs['cat'] = final_probs_list

        # --- Bernoulli (核心修改区域) ---
        if 'bern' in self.action_dims and self.action_dims['bern'] > 0:
            bern_logits = self.fc_bern(shared_features)
            # [新增] Action Masking 逻辑
            # 如果提供了 mask，将 mask 为 0 (False) 的位置的 logit 设为 -1e9
            if action_masks is not None and 'bern' in action_masks:
                mask = action_masks['bern']
                # 确保 mask 和 logits 维度匹配 (Batch, Dim)
                # mask == 0 代表禁止开火，设为极小值
                bern_logits = bern_logits.masked_fill(mask == 0, -1e9)

            # [修改] 使用传入的 temp
            # temps = 1.0 
            scaled_bern_logits = bern_logits / (temp + 1e-8)
            outputs['bern'] = scaled_bern_logits
            
        return outputs

# =============================================================================
# 2. Actor 适配器 (Wrapper) - 核心重构点
# =============================================================================

class HybridActorWrapper(nn.Module):
    """
    统一接口适配器。
    将具体的 PolicyNetHybrid 封装起来，对外提供标准的 get action 和 evaluate actions 接口。
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

    # [修改] 增加 check_obs 参数，默认为 None， [新增] 增加 temp 参数
    def get_action(self, state, h=None, explore=True, max_std=None, check_obs=None, bern_threshold=0.5, temp=1.0):
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
        1、目前 get action 中的 mask 生成只处理单个 check_obs（推理时），
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
        # 当且仅当传入了单个 dict 类型的 check_obs 时启用 mask, 不受explore影响
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

        # [修改] 调用网络时传入 action_masks 和 temp
        actor_outputs = self.net(state, max_std=max_std, action_masks=action_masks, temp=temp)
        
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
            # entropy += dist.entropy().unsqueeze(-1) # 近似熵
            
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
    
    def compute_il_loss(self, states, expert_actions, label_smoothing=0.1, no_bern=False):
        """
        计算模仿学习 Loss (MARWIL / BC)。
        
        Args:
            states: (Batch, State_Dim)
            expert_actions: 字典, 包含 {'cont': u, 'cat': index, 'bern': float}
                            注意：对于连续动作，这里通常假设传入的是 pre-tanh 的 u，
                            或者你需要在外部处理好。
            label_smoothing: 标签平滑系数
            
        Returns:
            total_loss_per_sample: (Batch, ) 每个样本的 Loss 总和，未加权
        """
        '''
        会增加复杂度的可选改进：模仿学习的时候alpha 传入向量，从而区分密集和稀疏动作的学习强度（密集应该高一些）
        '''
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
                    '''
                    log_probs.gather()
                    从所有动作的概率分布 log_probs 中，精准地抽取出“实际执行了的那个动作” expert_idx 对应的概率值。
                    - 1 (第一个参数)：表示在第 1 维（列维度）进行选取。
                    - expert_idx.unsqueeze(1)：将原来形状为(Batch,)的索引变成(Batch, 1)。
                     这是因为 gather 要求索引的维度必须和原张量一致。
                    - .squeeze(1)：取完值后，形状还是(Batch, 1)用 squeeze 把那个多余的维度删掉，
                    变成平铺的 (Batch,)，方便后续算 Loss。
                    '''
                
                total_loss_per_sample += ce_loss

        # --- 3. 伯努利动作 (Bernoulli) ---
        # -- Focal Loss --
        if not no_bern:
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
        
        # [修改] 解析 k_entropy，支持字典输入
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


    # 重置优化器
    def reset_optimizer(self):
        """
        清除 optimizer 的动量等 state，但保留当前学习率/其他 param_group 设置。
        通过重建 optimizer（保留 param_group 的 lr/betas/eps/weight_decay）来达到清空 state 的目的。
        """
        def _recreate_from(old_optim, params):
            pg = old_optim.param_groups[0]
            lr = pg.get('lr', 1e-3)
            betas = pg.get('betas', (0.9, 0.999))
            eps = pg.get('eps', 1e-8)
            weight_decay = pg.get('weight_decay', 0.0)
            return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        # 重新创建 optim（以清除内部 state），但使用当前的 lr/betas/eps/weight_decay
        self.actor_optimizer = _recreate_from(self.actor_optimizer, self.actor.parameters())
        self.critic_optimizer = _recreate_from(self.critic_optimizer, self.critic.parameters())

        # 清除可能残留的梯度
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
    
    
    def take_action(self, state, h0=None, explore=True, max_std=None, check_obs=None, temperature=1.0):
        # 委托给 Actor Wrapper
        max_s = max_std if max_std is not None else self.max_std
        
        # [修改] 透传 check_obs
        actions_exec, actions_raw, h_state, actions_dist_check = self.actor.get_action(
            state, h=h0, explore=explore, max_std=max_s, check_obs=check_obs, temp=temperature
        )
        #  保持原有的返回两个字典的接口，或者根据需要返回 diagnostic output
        return actions_exec, actions_raw, h_state, actions_dist_check

    def update(self, transition_dict, adv_normed=False, clip_vf=False, clip_range=0.2, shuffled=1, mini_batch_size=None, alpha_logit_reg=0.05):

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

        #  监控列表
        kl_list = []
        clip_frac_list = []
        #  分项 Entropy 列表
        entropy_cat_list = []
        entropy_bern_list = []
        entropy_cont_list = []
        
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
                mb_old_log_probs = old_log_probs[batch_idx]
                mb_active_masks = active_masks[batch_idx]
                # v_pred_old_batch = v_pred_old[batch_idx] # 如果需要用到旧 Value
                
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
                
                # [新增] 统计 PPO 样本
                ppo_samples_total += mb_active_masks.numel()
                ppo_valid_samples_total += mb_active_masks.sum().item()

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
        
        # [新增] 赋值有效样本监控项
        self.PPO_samples = ppo_samples_total
        self.PPO_valid_samples = ppo_valid_samples_total
        
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

    # =========================================================================
    #  [New Helper] 提取出的 MSE 计算逻辑 (供 mixed_update 和 BC_update 复用)
    # =========================================================================
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
    
    
    
    # --- 修改后的 MARWIL_update ---
    def MARWIL_update(self, il_transition_dict, beta=1.0, batch_size=64, alpha=1.0, c_v=1.0, shuffled=1, label_smoothing=0.1, max_weight=100.0,
                      tau=0.8):
        """
        MARWIL 离线更新函数
        输入 actions 结构支持: [{'cat': array([v]), 'bern': array([v])}, ...]
        tau: 非对称损失权重 (Expectile Regression). tau=0.5 为 MSE; tau>0.5 (如0.9) 倾向于高估 Value (拟合好样本)
        """
        # 可能的局部观测
        if 'obs' in il_transition_dict and len(il_transition_dict['obs']) > 0:
            obs_all = torch.tensor(np.array(il_transition_dict['obs']), dtype=torch.float).to(self.device)
            use_obs = True
        else:
            use_obs = False
            
        # 冻结分布参数，只训练均值/Logits
        if hasattr(self.actor.net, 'log_std_cont'):
            self.actor.net.log_std_cont.requires_grad = False
        # if hasattr(self.actor.net, 'log_temp_cat'):
        #     self.actor.net.log_temp_cat.requires_grad = False
        # if hasattr(self.actor.net, 'log_temp_bern'):
        #     self.actor.net.log_temp_bern.requires_grad = False

        # 1. 提取全量数据并转为 Tensor
        states_all = torch.tensor(np.array(il_transition_dict['states']), dtype=torch.float).to(self.device)
        returns_all = torch.tensor(np.array(il_transition_dict['returns']), dtype=torch.float).view(-1, 1).to(self.device)
        
        # ============================================================
        # [修改] 统一处理 Actions：List of Dicts -> Dict of Tensors
        # ============================================================
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

            # B. Actor Loss
            # compute_il_loss 不需要修改，因为它接收的是已经处理好的 Dict of Tensors
            raw_il_loss = self.actor.compute_il_loss(actor_input_batch, actions_batch, label_smoothing)
            actor_loss = torch.mean(alpha * weights * raw_il_loss)

            # C. Critic Loss
            v_pred = self.critic(s_batch)
            
            # 原有
            critic_loss = F.mse_loss(v_pred, r_batch) * c_v
            
            # # # [修改] 使用非对称损失 (Expectile Regression)
            # # # 如果回报高于预估值（好样本），给予极大的权重（tau > 0.5）
            # # # 如果回报低于预估值（差样本），给予较小的权重
            # diff = r_batch - v_pred
            # tau_t = torch.tensor(tau, device=self.device)
            # weight = torch.where(diff > 0, tau_t, 1.0 - tau_t) # 条件，True取值，False取值
            # critic_loss = (weight * (diff**2)).mean() * c_v

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

        avg_actor_loss = total_actor_loss / batch_count if batch_count > 0 else 0
        avg_critic_loss = total_critic_loss / batch_count if batch_count > 0 else 0
        avg_c = total_c / batch_count if batch_count > 0 else 0
        
        return avg_actor_loss, avg_critic_loss, avg_c
    
    # --- 新增功能 3: 混合更新 (PPO + MARWIL) ---
    # --- 修改后的 mixed_update ---
    # --- 修复后的 mixed_update (包含完整监控项) ---
    def mixed_update(self, transition_dict, il_transition_dict, init_il_transition_dict=None, eta=0.1,  # [新增] 专家数据与约束系数
                     # RL 参数
                     adv_normed=False, clip_vf=False, clip_range=0.2, 
                     # IL 参数
                     beta=1.0, il_batch_size=None, alpha=1.0, c_v=1.0, label_smoothing=0.1, max_weight=100.0, il_epochs=1,
                     sil_only_maneuver=True,
                     # 公共参数
                     shuffled=1, mini_batch_size=None, alpha_logit_reg=0.05):
        
        # =====================================================================
        # Phase 1: PPO 更新 (复用现有方法)
        # =====================================================================
        # 这一步会执行 self.epochs 次 PPO 更新，并修改 self.actor_loss 等类属性
        self.update(transition_dict, adv_normed=adv_normed, clip_vf=clip_vf, 
                    clip_range=clip_range, shuffled=shuffled, 
                    mini_batch_size=mini_batch_size, alpha_logit_reg=alpha_logit_reg)

        # --- [Step A] 暂存 PPO 统计指标 & 计算权重 ---
        # 我们需要知道 PPO 到底更新了多少个 Batch，用于后续和 IL 做加权平均
        rl_total_size = len(transition_dict['states'])
        # 假设 self.batch_size 是 PPO 的 mini_batch_size
        # 如果 self.update 内部逻辑不同，这里可能需要调整计算方式
        
        # ppo_num_batches = max(1, int(rl_total_size / mini_batch_size)) * self.epochs
        mb = mini_batch_size if mini_batch_size is not None else rl_total_size
        ppo_num_batches = max(1, (rl_total_size + mb - 1) // mb) * self.epochs
        
        ppo_stats = {
            'actor_loss': self.actor_loss,
            'critic_loss': self.critic_loss,
            'actor_grad': self.actor_grad,
            'critic_grad': self.critic_grad,
            'entropy': self.entropy_mean,
            'ratio': self.ratio_mean,
            'pre_clip_actor': self.pre_clip_actor_grad,
            'pre_clip_critic': self.pre_clip_critic_grad
        }

        # =====================================================================
        # Phase 2: IL 更新 (自模仿 + 专家 MSE)
        # =====================================================================
        # 辅助函数：转 Tensor
        def to_tensor(x, dtype):
            if isinstance(x, np.ndarray):
                return torch.tensor(x, dtype=dtype).to(self.device)
            return torch.tensor(np.array(x), dtype=dtype).to(self.device)

        # --- [Step B] 准备 IL 数据 ---
        # 1. Self-Imitation Data
        # 1. 自模仿数据 (保持原样)
        il_states_all = to_tensor(il_transition_dict['states'], torch.float)
        il_returns_all = to_tensor(il_transition_dict['returns'], torch.float).view(-1, 1)
        use_il_obs = False
        if 'obs' in il_transition_dict and len(il_transition_dict['obs']) > 0:
            il_obs_all = to_tensor(il_transition_dict['obs'], torch.float)
            use_il_obs = True
        il_actions_raw = il_transition_dict['actions']
        il_actions_all = {}
        
        if isinstance(il_actions_raw, list):
            if len(il_actions_raw) > 0:
                keys = il_actions_raw[0].keys()
                temp_dict = {}
                for k in keys:
                    temp_dict[k] = np.stack([d[k] for d in il_actions_raw], axis=0)
                il_actions_raw = temp_dict
            else:
                il_actions_raw = {}
        if isinstance(il_actions_raw, dict):
            for k, v in il_actions_raw.items():
                if k == 'cat':
                    il_actions_all[k] = to_tensor(v, torch.long)
                else:
                    il_actions_all[k] = to_tensor(v, torch.float)
        il_total_size = il_states_all.size(0)
        
        if init_il_transition_dict is not None:
            # 2. 他模仿数据
            init_il_states_all = to_tensor(init_il_transition_dict['states'], torch.float)
            use_init_il_obs = False
            if 'obs' in init_il_transition_dict and len(init_il_transition_dict['obs']) > 0:
                init_il_obs_all = to_tensor(init_il_transition_dict['obs'], torch.float)
                use_init_il_obs = True
            init_il_actions_raw = init_il_transition_dict['actions']
            init_il_actions_all = {}  
            if isinstance(init_il_actions_raw, list):
                if len(init_il_actions_raw) > 0:
                    keys = init_il_actions_raw[0].keys()
                    temp_dict = {}
                    for k in keys:
                        temp_dict[k] = np.stack([d[k] for d in init_il_actions_raw], axis=0)
                    init_il_actions_raw = temp_dict
                else:
                    init_il_actions_raw = {}
            if isinstance(init_il_actions_raw, dict):
                for k, v in init_il_actions_raw.items():
                    if k == 'cat':
                        init_il_actions_all[k] = to_tensor(v, torch.long)
                    else:
                        init_il_actions_all[k] = to_tensor(v, torch.float)
            init_il_returns_all = to_tensor(init_il_transition_dict['returns'], torch.float).view(-1, 1)
            init_il_total_size = init_il_states_all.size(0)
        
        # --- [Step C] 执行 IL 更新 (通常执行 1 次，或者根据 batch_size 循环) ---
        # 这里为了简单且符合"PPO为主，IL为辅"的逻辑，我们执行一次 IL Update
        # 如果 IL 数据量很大，这里可以改为循环
        
        # [修复] 完整初始化所有监控列表
        actor_loss_list, critic_loss_list = [], [],
        actor_grad_list, critic_grad_list = [], []
        pre_clip_actor_grad, pre_clip_critic_grad = [], []
        
        il_actor_loss_list, il_critic_loss_list = [], [] 
        
        il_samples_total = 0
        il_valid_samples_total = 0

        for _ in range(il_epochs):
            actor_loss_init_il = torch.tensor(0.0, device=self.device) # 显式初始化
            # 使用全量(或最大batch) IL 数据，不切分 Mini-Batch
            # 这里也可以稍微 Shuffle 一下 IL 数据，或者直接随机采样 batch_size
            curr_il_batch_size = min(il_batch_size, il_total_size) if il_batch_size is not None else il_total_size
            
            # 简单起见，如果要求全batch，则直接取全部，或者随机取一个大 Batch
            il_indices = np.random.randint(0, il_total_size, curr_il_batch_size)
            
            il_s_batch = il_states_all[il_indices] 
            il_r_batch = il_returns_all[il_indices]
            if use_il_obs:
                il_actor_input_batch = il_obs_all[il_indices]
            else:
                il_actor_input_batch = il_s_batch

            il_actions_batch = {}
            for k, v in il_actions_all.items():
                il_actions_batch[k] = v[il_indices]
            
            if init_il_transition_dict is not None:
                init_il_idx = np.random.randint(0, init_il_total_size, init_il_total_size)
                init_il_returns_batch = init_il_returns_all[init_il_idx]
                if use_init_il_obs:
                    init_il_actor_input_batch = init_il_obs_all[init_il_idx]
                else:
                    init_il_actor_input_batch = init_il_states_all[init_il_idx]
                init_il_actions_batch = {}
                for k, v in init_il_actions_all.items():
                    init_il_actions_batch[k] = v[init_il_idx]
                
            with torch.no_grad():
                il_values = self.critic(il_s_batch)
                residual = il_r_batch - il_values
                
                if not hasattr(self, 'c_sq'): self.c_sq = torch.tensor(1.0, device=self.device)
                batch_mse = (residual ** 2).mean().item()
                # self.c_sq = self.c_sq + 1e-8 * (batch_mse - self.c_sq)
                self.c_sq = torch.tensor(1.0, device=self.device)  # 不要再让缩放系数浮动了
                c = torch.sqrt(self.c_sq)
                
                il_adv = residual / (c + 1e-8)
                # 对负 advantage 强烈抑制权重：adv>0 正常放行，否则乘以 1e-6
                
                '''
                exp可能导致梯度爆炸，对此有以下解决方法（最新的梯度爆炸不是这个原因，代码仅做候补）
                # 1 数值归一化
                il_adv = (il_adv - il_adv.mean()) / (il_adv.std() + 1e-8)
                il_raw_weights = torch.exp(beta * il_adv)
                # 2 减去最大值 (Max-Subtraction Trick)
                il_adv_max = torch.max(il_adv)
                il_raw_weights = torch.exp(beta * (il_adv - il_adv_max))
                # 3 权重截断
                il_raw_weights = torch.exp(beta * il_adv)
                il_raw_weights = torch.clamp(il_raw_weights, max=100.0) # 限制单样本权重最大100倍
                '''
                
                il_raw_weights = beta * il_adv  # torch.exp(beta * il_adv)
                F_word = torch.where(il_adv > 0, torch.ones_like(il_adv), torch.full_like(il_adv, 1e-6))
                il_weights = torch.clamp(il_raw_weights * F_word, min=0, max=max_weight)

                # [新增] 统计 IL 样本强度
                il_samples_total += curr_il_batch_size
                il_valid_samples_total += F_word.sum().item()

            # compute_il_loss 接口不变
            raw_il_loss = self.actor.compute_il_loss(il_actor_input_batch, il_actions_batch, label_smoothing, no_bern=sil_only_maneuver)
            actor_loss_il = torch.mean(il_weights * raw_il_loss)
            
            # --- [new] 对初始轨迹行为克隆 ---
            if init_il_transition_dict is not None:
                # 1. 计算初始轨迹的 F 函数 (优势加权)
                with torch.no_grad():
                    # [修改] 按照你的要求，Critic 直接接收 init_il_actor_input_batch
                    v_init = self.critic(init_il_actor_input_batch)
                    adv_init = (init_il_returns_batch - v_init) / (torch.sqrt(self.c_sq) + 1e-8)
                    
                    # F 函数约束：只有比当前表现好的专家数据才学
                    F_init = torch.where(adv_init > 0, torch.ones_like(adv_init), torch.full_like(adv_init, 1e-6))
                    F_init = torch.clamp(F_init, max=max_weight)

                # 2. 获取网络当前的原始输出 (保留对 net 的直接调用)
                init_outputs = self.actor.net(init_il_actor_input_batch)
                
                # 【核心修正】：初始化为与 adv_init 形状相同的零张量 [Batch, 1]
                # 这样可以确保后续的 += 每一项都是 [Batch, 1] 的累加
                mse_loss_sum = torch.zeros_like(adv_init)
                # mse_loss_sum = torch.tensor(0.0, device=self.device)

                # --- 连续动作 MSE ---
                if 'cont' in self.actor.action_dims and self.actor.action_dims['cont'] > 0:
                    mu_current, _ = init_outputs['cont']
                    u_expert = init_il_actions_batch['cont']
                    mse_loss_sum += F.mse_loss(mu_current, u_expert, reduction='none').sum(dim=-1, keepdim=True)

                # --- 离散动作 MSE ---
                if 'cat' in self.actor.action_dims and sum(self.actor.action_dims['cat']) > 0:
                    cat_probs_current = init_outputs['cat']
                    expert_cat = init_il_actions_batch['cat'].long()
                    for i, probs in enumerate(cat_probs_current):
                        target_one_hot = F.one_hot(expert_cat[:, i], num_classes=probs.size(-1)).float()
                        mse_loss_sum += F.mse_loss(probs, target_one_hot, reduction='none').sum(dim=-1, keepdim=True)

                # --- 伯努利动作 MSE ---
                if 'bern' in self.actor.action_dims and self.actor.action_dims['bern'] > 0:
                    bern_logits = init_outputs['bern']
                    bern_probs = torch.sigmoid(bern_logits)
                    target_bern = init_il_actions_batch['bern']
                    mse_loss_sum += F.mse_loss(bern_probs, target_bern, reduction='none').sum(dim=-1, keepdim=True)

                # 3. 应用 F 函数约束并求平均
                actor_loss_init_mse = torch.mean(F_init * mse_loss_sum)
                actor_loss_init_il = eta * actor_loss_init_mse

            
            il_values_grad = self.critic(il_s_batch)
            critic_loss_il = F.mse_loss(il_values_grad, il_r_batch) * c_v
            
            il_actor_loss_list.append(actor_loss_il.item())
            il_critic_loss_list.append(critic_loss_il.item())

            # 4. 联合反向传播

            il_actor_loss = alpha * (actor_loss_il + actor_loss_init_il)
            il_critic_loss = alpha * critic_loss_il

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            
            il_actor_loss.backward()
            il_critic_loss.backward()
            
            # [修复] 记录裁剪前梯度
            pre_clip_actor_grad.append(model_grad_norm(self.actor))
            pre_clip_critic_grad.append(model_grad_norm(self.critic))
            
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.actor_max_grad)
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.critic_max_grad)
            
            self.actor_optimizer.step()
            self.critic_optimizer.step()

            # [修复] 记录步进后梯度和其他监控项
            actor_grad_list.append(model_grad_norm(self.actor))
            critic_grad_list.append(model_grad_norm(self.critic))            
            actor_loss_list.append(il_actor_loss.item()) # 注意：记录的是混合 Loss
            critic_loss_list.append(il_critic_loss.item())
        

        # =====================================================================
        # Phase 3: 统计指标融合 (加权平均)
        # =====================================================================
        
        # 定义权重：PPO 占 epoch * batches 份，IL 占 1 份 (或者你希望 1:1，可以设 il_weight = ppo_num_batches)
        # 这里按照“梯度更新次数”进行加权是比较科学的
        def weighted_avg(ppo_val, il_val, 
                         ppo_num_batches=ppo_num_batches, il_epochs=il_epochs):
            total_updates = ppo_num_batches + il_epochs
            return (ppo_val * ppo_num_batches + il_val * il_epochs) / total_updates

        # 更新类属性用于 Log
        self.actor_loss = weighted_avg(ppo_stats['actor_loss'], np.mean(actor_loss_list))
        self.critic_loss = weighted_avg(ppo_stats['critic_loss'], np.mean(critic_loss_list))
        
        self.actor_grad = weighted_avg(ppo_stats['actor_grad'], np.mean(actor_grad_list))
        self.critic_grad = weighted_avg(ppo_stats['critic_grad'], np.mean(critic_grad_list))
        
        self.pre_clip_actor_grad = weighted_avg(ppo_stats['pre_clip_actor'], np.mean(pre_clip_actor_grad))
        self.pre_clip_critic_grad = weighted_avg(ppo_stats['pre_clip_critic'], np.mean(pre_clip_critic_grad))
        
        # 熵和Ratio只有PPO有，IL没有，所以乘以系数缩放或者直接保持PPO的
        # 为了避免 Log 里的值看起来突然变小，这里建议直接使用 PPO 的值，因为 IL 不产生 entropy/ratio 概念
        self.entropy_mean = ppo_stats['entropy']
        self.ratio_mean = ppo_stats['ratio']
        
        # 更新样本计数
        self.IL_samples = il_samples_total
        self.IL_valid_samples = il_valid_samples_total

        
        check_weights_bias_nan(self.actor, "actor", "mixed_update后")
        check_weights_bias_nan(self.critic, "critic", "mixed_update后")