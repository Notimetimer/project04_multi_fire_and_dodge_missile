import os
import sys
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, Bernoulli

# 假设 GruMlp 在此路径，如无此文件可替换为标准 nn.GRU 实现
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from Algorithms.SharedLayers import GruMlp
except ImportError:
    # 如果找不到文件，提供一个简单的 Mock 或报错提示
    print("Warning: Algorithm.SharedLayers.GruMlp not found. Please ensure environment is set up.")

# --- 辅助函数 ---

def model_grad_norm(model):
    total_sq = 0.0
    found = False
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach().cpu()
            total_sq += float(g.norm(2).item()) ** 2
            found = True
    return float(total_sq ** 0.5) if found else float('nan')

def check_weights_bias_nan(model, model_name="model", place=None):
    for name, param in model.named_parameters():
        if ("weight" in name) or ("bias" in name):
            if param is None:
                continue
            if torch.isnan(param).any():
                loc = f" at {place}" if place else ""
                raise ValueError(f"NaN detected in {model_name} parameter '{name}'{loc}")

def compute_advantage(gamma, lmbda, td_delta, dones):
    td_delta = td_delta.detach().cpu().numpy()
    dones = dones.detach().cpu().numpy()
    advantage_list = []
    advantage = 0.0
    
    for delta, done in zip(td_delta[::-1], dones[::-1]):
        advantage = delta + gamma * lmbda * advantage * (1 - done)
        advantage_list.append(advantage)
        
    advantage_list.reverse()
    return torch.tensor(np.array(advantage_list), dtype=torch.float)

# --- 独立缩放函数 (External) ---

def scale_action_to_exec(a_norm, action_bounds):
    """
    将标准化的动作 a_norm (在 [-1,1] 范围内) 缩放到环境执行区间。
    独立于 PPO 类之外。
    参数:
        a_norm: numpy array, 归一化动作
        action_bounds: list or numpy array, shape (action_dim, 2), [[min, max], ...]
    """
    if action_bounds is None:
        return a_norm
    
    bounds = np.array(action_bounds)
    amin = bounds[:, 0]
    amax = bounds[:, 1]
    action_span = amax - amin
    
    return amin + (a_norm + 1.0) * 0.5 * action_span

# --- GRU Backbone & Distributions ---

class SharedGruBackbone(nn.Module):
    """共享的 GRU 特征提取器"""
    def __init__(self, state_dim, gru_hidden_size=128, gru_num_layers=1, output_dim=128, batch_first=True):
        super(SharedGruBackbone, self).__init__()
        # 使用 GruMlp，如果环境没有该类，需替换为 nn.GRU + nn.Linear
        self.feature_extractor = GruMlp(
            input_dim=state_dim,
            gru_hidden_size=gru_hidden_size,
            gru_num_layers=gru_num_layers,
            output_dim=output_dim,
            batch_first=batch_first
        )

    def forward(self, x, h_0=None):
        if x.dim() == 2:
            x = x.unsqueeze(1) # (B, D) -> (B, 1, D)
        features_per_step, h_n = self.feature_extractor(x, h_0)
        return features_per_step, h_n

class SquashedNormal:
    def __init__(self, mu, std, eps=1e-6):
        self.mu = mu
        if not torch.is_tensor(std):
            std = torch.as_tensor(std, device=mu.device, dtype=mu.dtype)
        self.std = torch.clamp(std, min=float(eps))
        self.normal = Normal(mu, self.std)
        self.eps = eps

    def sample(self):
        u = self.normal.rsample()
        a = torch.tanh(u)
        return a, u

    def log_prob(self, a, u):
        log_prob_u = self.normal.log_prob(u)
        # tanh 修正项 (Jacobian) 可选，通常为了稳定性直接基于 u 计算概率分布
        # PPO中如果使用 u 作为 action_raw 进行更新，这里返回 u 的 log_prob 即可
        return log_prob_u 

    def entropy(self):
        return self.normal.entropy().sum(-1)

# --- Actor & Critic Networks ---

class PolicyNetHybridGRU(nn.Module):
    """
    支持混合动作空间 + GRU 的策略网络。
    """
    def __init__(self, state_dim, gru_hidden_size, gru_num_layers, hidden_dims, action_dims_dict, init_std=0.5, batch_first=True):
        super(PolicyNetHybridGRU, self).__init__()
        self.action_dims = action_dims_dict
        
        # 1. 独立的 GRU Backbone
        # hidden_dims[0] 作为 backbone 的输出维度，也作为后续 MLP 的输入维度
        middle_dim = hidden_dims[0]
        self.backbone = SharedGruBackbone(state_dim, gru_hidden_size, gru_num_layers, middle_dim, batch_first=batch_first)

        # 2. 共享 MLP 层 (在 GRU 之后)
        layers = []
        prev_size = middle_dim
        # 如果 hidden_dims 有多层，继续构建
        for layer_size in hidden_dims[1:]:
            layers.append(nn.Linear(prev_size, layer_size))
            layers.append(nn.ReLU())
            prev_size = layer_size
        self.net = nn.Sequential(*layers)

        # 3. 动态创建输出头
        # 连续动作头
        if self.action_dims.get('cont', 0) > 0:
            cont_dim = self.action_dims['cont']
            self.fc_mu = nn.Linear(prev_size, cont_dim)
            self.log_std_param = nn.Parameter(torch.log(torch.ones(cont_dim, dtype=torch.float) * init_std))

        # 分类/离散动作头
        if self.action_dims.get('cat', 0) > 0:
            cat_dim = self.action_dims['cat']
            self.fc_cat = nn.Linear(prev_size, cat_dim)

        # 伯努利动作头
        if self.action_dims.get('bern', 0) > 0:
            bern_dim = self.action_dims['bern']
            self.fc_bern = nn.Linear(prev_size, bern_dim)

    def forward(self, x, h_0=None, min_std=1e-6, max_std=0.4):
        # Backbone Forward
        features_per_step, h_n = self.backbone(x, h_0)
        # 取序列最后一步特征用于 MLP
        features = features_per_step[:, -1, :]
        
        shared_features = self.net(features)
        
        outputs = {'cont': None, 'cat': None, 'bern': None}

        # 连续
        if self.action_dims.get('cont', 0) > 0:
            mu = self.fc_mu(shared_features)
            std = torch.exp(self.log_std_param)
            std = torch.clamp(std, min=min_std, max=max_std)
            if mu.dim() > 1:
                std = std.unsqueeze(0).expand_as(mu)
            outputs['cont'] = (mu, std)

        # 分类
        if self.action_dims.get('cat', 0) > 0:
            cat_logits = self.fc_cat(shared_features)
            outputs['cat'] = F.softmax(cat_logits, dim=-1)

        # 伯努利
        if self.action_dims.get('bern', 0) > 0:
            bern_logits = self.fc_bern(shared_features)
            outputs['bern'] = bern_logits

        return outputs, h_n

class ValueNetGRU(nn.Module):
    """独立的 GRU Critic 网络"""
    def __init__(self, state_dim, gru_hidden_size, gru_num_layers, hidden_dims, batch_first=True):
        super(ValueNetGRU, self).__init__()
        
        middle_dim = hidden_dims[0]
        self.backbone = SharedGruBackbone(state_dim, gru_hidden_size, gru_num_layers, middle_dim, batch_first=batch_first)

        layers = []
        prev_size = middle_dim
        for layer_size in hidden_dims[1:]:
            layers.append(torch.nn.Linear(prev_size, layer_size))
            layers.append(nn.ReLU())
            prev_size = layer_size
        self.net = nn.Sequential(*layers)
        self.fc_out = torch.nn.Linear(prev_size, 1)

    def forward(self, x, h_0=None):
        features_per_step, h_n = self.backbone(x, h_0)
        features = features_per_step[:, -1, :]
        
        y = self.net(features)
        value = self.fc_out(y)
        return value, h_n

# --- PPOHybrid 算法 ---

class PPOHybrid:
    def __init__(self, state_dim, gru_hidden_size, gru_num_layers, hidden_dim, action_dims_dict, 
                 actor_lr, critic_lr, lmbda, epochs, eps, gamma, device, 
                 k_entropy=0.01, critic_max_grad=2, actor_max_grad=2, max_std=0.3, batch_first=True):
        
        self.action_dims = action_dims_dict
        self.device = device
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.k_entropy = k_entropy
        self.critic_max_grad = critic_max_grad
        self.actor_max_grad = actor_max_grad
        self.max_std = max_std
        
        # GRU 参数
        self.gru_num_layers = gru_num_layers
        self.gru_hidden_size = gru_hidden_size

        # 1. 创建 Actor (含独立 Backbone)
        self.actor = PolicyNetHybridGRU(state_dim, gru_hidden_size, gru_num_layers, hidden_dim, 
                                        action_dims_dict, batch_first=batch_first).to(device)
        
        # 2. 创建 Critic (含独立 Backbone)
        self.critic = ValueNetGRU(state_dim, gru_hidden_size, gru_num_layers, hidden_dim, 
                                  batch_first=batch_first).to(device)
        
        # 3. 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # 4. 隐藏状态初始化
        self.hidden_state_a = None
        self.hidden_state_c = None

    def reset_hidden_state_a(self):
        self.hidden_state_a = self.get_h0_a(batch_size=1)
        
    def reset_hidden_state_c(self):
        self.hidden_state_c = self.get_h0_c(batch_size=1)

    def get_h0_a(self, batch_size=1):
        h0 = torch.zeros(self.gru_num_layers, batch_size, self.gru_hidden_size, dtype=torch.float32, device=self.device)
        return h0.detach().cpu()

    def get_h0_c(self, batch_size=1):
        h0 = torch.zeros(self.gru_num_layers, batch_size, self.gru_hidden_size, dtype=torch.float32, device=self.device)
        return h0.detach().cpu()
    
    def get_current_hidden_state_a(self):
        return self.hidden_state_a

    def get_current_hidden_state_c(self):
        return self.hidden_state_c

    def _maybe_move_h0_to_device(self, h0):
        if h0 is None: return None
        if torch.is_tensor(h0):
            if h0.device != self.device:
                return h0.to(self.device)
            return h0
        return torch.as_tensor(h0, device=self.device)

    def set_learning_rate(self, actor_lr=None, critic_lr=None):
        if actor_lr is not None:
            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] = actor_lr
        if critic_lr is not None:
            for param_group in self.critic_optimizer.param_groups:
                param_group['lr'] = critic_lr  

    def take_action(self, state, h_0_a=None, explore=True, max_std=None):
        '''
        输出为两项：actions_exec(numpy), actions_raw(numpy)
        并返回新的 hidden_state
        注意：Continuous action 不再此处进行缩放，返回 [-1, 1] 之间的值。
        '''
        # state shape: (SeqLen, StateDim) or (StateDim,) -> (1, SeqLen, StateDim)
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        if h_0_a is not None:
            h_0_a = self._maybe_move_h0_to_device(h_0_a)
            
        max_action_std = max_std if max_std is not None else self.max_std

        # 前向传播
        actor_outputs, h_n_a = self.actor(state, h_0_a, max_std=max_action_std)
        self.hidden_state_a = h_n_a.detach().cpu() # 更新内部隐藏状态

        actions_exec = {} 
        actions_raw = {} 

        # --- Cont ---
        if actor_outputs['cont'] is not None:
            mu, std = actor_outputs['cont']
            dist = SquashedNormal(mu, std)
            if explore:
                a_norm, u = dist.sample()
            else:
                u = mu
                a_norm = torch.tanh(u)
            
            # 此处不进行缩放，直接返回 normalized action
            actions_exec['cont'] = a_norm[0].cpu().detach().numpy().flatten()
            actions_raw['cont'] = u[0].cpu().detach().numpy().flatten()

        # --- Cat ---
        if actor_outputs['cat'] is not None:
            cat_probs = actor_outputs['cat']
            dist = Categorical(probs=cat_probs)
            if explore:
                cat_action_idx = dist.sample()
            else:
                cat_action_idx = torch.argmax(dist.probs, dim=-1)
            actions_exec['cat'] = cat_probs.cpu().detach().numpy()[0].copy()
            actions_raw['cat'] = cat_action_idx.cpu().detach().numpy().flatten()

        # --- Bern ---
        if actor_outputs['bern'] is not None:
            bern_logits = actor_outputs['bern']
            dist = Bernoulli(logits=bern_logits)
            if explore:
                bern_action = dist.sample()
            else:
                bern_action = (dist.probs > 0.5).float()

            actions_exec['bern'] = bern_action[0].cpu().detach().numpy().flatten()
            actions_raw['bern'] = actions_exec['bern']
        
        return actions_exec, actions_raw, self.hidden_state_a

    def get_value(self, state, h_0_c):
        state_input = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        if h_0_c is not None:
            h_0_c = self._maybe_move_h0_to_device(h_0_c)
        value, h_n_c = self.critic(state_input, h_0_c)
        return value.squeeze(0).detach().cpu().numpy(), h_n_c.detach().cpu()

    def update(self, transition_dict, adv_normed=False, clip_vf=False, clip_range=0.2, shuffled=0):
        N = len(transition_dict['states'])
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float).view(-1, 1).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']), dtype=torch.float).view(-1, 1).to(self.device)
        next_values = torch.tensor(np.array(transition_dict['next_values']), dtype=torch.float).to(self.device)
        
        # 提取 hidden states
        h0as_stack = torch.stack([transition_dict['h0as'][i] for i in range(N)]).to(self.device)
        h0cs_stack = torch.stack([transition_dict['h0cs'][i] for i in range(N)]).to(self.device)
        # 调整 shape: (N, Layers, 1, H) -> (Layers, N, H)
        h0as_for_actor = h0as_stack.squeeze(2).permute(1, 0, 2).contiguous()
        h0cs_for_critic = h0cs_stack.squeeze(2).permute(1, 0, 2).contiguous()

        # 提取动作
        actions_from_buffer = transition_dict['actions']
        
        # 计算 TD-target 和 Advantage
        td_target = rewards + self.gamma * next_values * (1 - dones)
        td_delta = td_target - self.critic(states, h0cs_for_critic)[0]
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu(), dones.cpu()).to(self.device)

        # Shuffle 处理
        indices = np.arange(N)
        if shuffled:
            np.random.shuffle(indices)
            states = states[indices]
            rewards = rewards[indices]
            dones = dones[indices]
            next_values = next_values[indices]
            advantage = advantage[indices]
            actions_from_buffer = [actions_from_buffer[i] for i in indices]
            
            # 重新整理 Hidden States
            h0as_stack = h0as_stack[indices]
            h0cs_stack = h0cs_stack[indices]
            h0as = h0as_stack.squeeze(2).permute(1, 0, 2).contiguous()
            h0cs = h0cs_stack.squeeze(2).permute(1, 0, 2).contiguous()
        else:
            h0as = h0as_for_actor
            h0cs = h0cs_for_critic

        # 将动作转为 Tensor
        actions_on_device = {}
        all_keys = actions_from_buffer[0].keys()
        for key in all_keys:
            vals = [d[key] for d in actions_from_buffer]
            if key == 'cat':
                actions_on_device[key] = torch.tensor(np.array(vals), dtype=torch.long).to(self.device)
            else:
                actions_on_device[key] = torch.tensor(np.array(vals), dtype=torch.float).to(self.device)

        if adv_normed:
            adv_mean = advantage.detach().mean()
            adv_std = advantage.detach().std(unbiased=False)
            advantage = (advantage - adv_mean) / (adv_std + 1e-8)

        # 这里的 v_pred_old 应该使用 h0cs 计算
        v_pred_old = self.critic(states, h0cs)[0].detach()

        # 计算 Old Log Probs
        with torch.no_grad():
            old_actor_outputs, _ = self.actor(states, h0as, max_std=self.max_std)
            old_log_probs = torch.zeros(states.size(0), 1).to(self.device)

            if self.action_dims.get('cont', 0) > 0:
                mu_old, std_old = old_actor_outputs['cont']
                dist_old = SquashedNormal(mu_old, std_old)
                u_old = actions_on_device['cont'] # 存储的是 raw u
                old_log_probs += dist_old.log_prob(0, u_old).sum(-1, keepdim=True)
            
            if self.action_dims.get('cat', 0) > 0:
                cat_probs_old = old_actor_outputs['cat']
                cat_action_old = actions_on_device['cat']
                old_log_probs += torch.log(cat_probs_old.gather(1, cat_action_old))

            if self.action_dims.get('bern', 0) > 0:
                bern_logits_old = old_actor_outputs['bern']
                dist_old = Bernoulli(logits=bern_logits_old)
                bern_action_old = actions_on_device['bern']
                old_log_probs += dist_old.log_prob(bern_action_old).sum(-1, keepdim=True)

        actor_loss_list, critic_loss_list = [], []
        actor_grad_list, critic_grad_list = [], []

        for _ in range(self.epochs):
            actor_outputs, _ = self.actor(states, h0as, max_std=self.max_std)
            log_probs = torch.zeros(states.size(0), 1).to(self.device)
            entropy = torch.zeros(states.size(0), 1).to(self.device)

            # Cont
            if self.action_dims.get('cont', 0) > 0:
                mu, std = actor_outputs['cont']
                dist = SquashedNormal(mu, std)
                u = actions_on_device['cont']
                log_probs += dist.log_prob(0, u).sum(-1, keepdim=True)
                entropy += dist.entropy().unsqueeze(-1)

            # Cat
            if self.action_dims.get('cat', 0) > 0:
                cat_probs = actor_outputs['cat']
                cat_action = actions_on_device['cat']
                log_probs += torch.log(cat_probs.gather(1, cat_action))
                dist = Categorical(probs=cat_probs)
                entropy += dist.entropy().unsqueeze(-1)

            # Bern
            if self.action_dims.get('bern', 0) > 0:
                bern_logits = actor_outputs['bern']
                dist = Bernoulli(logits=bern_logits)
                bern_action = actions_on_device['bern']
                log_probs += dist.log_prob(bern_action).sum(-1, keepdim=True)
                entropy += dist.entropy().sum(-1, keepdim=True)

            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = -torch.min(surr1, surr2).mean() - self.k_entropy * entropy.mean()

            # Critic Loss
            current_values, _ = self.critic(states, h0cs)
            if clip_vf:
                v_pred_clipped = torch.clamp(current_values, v_pred_old - clip_range, v_pred_old + clip_range)
                vf_loss1 = (current_values - td_target.detach()).pow(2)
                vf_loss2 = (v_pred_clipped - td_target.detach()).pow(2)
                critic_loss = torch.max(vf_loss1, vf_loss2).mean()
            else:
                critic_loss = F.mse_loss(current_values, td_target.detach())

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()

            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.actor_max_grad)
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.critic_max_grad)

            actor_loss_list.append(actor_loss.item())
            critic_loss_list.append(critic_loss.item())
            actor_grad_list.append(model_grad_norm(self.actor))
            critic_grad_list.append(model_grad_norm(self.critic))

            self.actor_optimizer.step()
            self.critic_optimizer.step()

        self.actor_loss = np.mean(actor_loss_list)
        self.critic_loss = np.mean(critic_loss_list)
        self.actor_grad = np.mean(actor_grad_list)
        self.critic_grad = np.mean(critic_grad_list)
        
        check_weights_bias_nan(self.actor, "actor", "update后")
        check_weights_bias_nan(self.critic, "critic", "update后")