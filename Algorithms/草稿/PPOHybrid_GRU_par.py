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

# --- 新版 compute_advantage (支持 Truncated) ---
def compute_advantage(gamma, lmbda, td_delta, dones, truncateds=None): 
    # 确保输入转为 numpy
    td_delta = td_delta.detach().cpu().numpy()
    dones = dones.detach().cpu().numpy() 

    if truncateds is None:
        # --- 旧式/兼容模式：dones = term OR trunc ---
        advantage_list = []
        advantage = 0.0
        
        for delta, done in zip(td_delta[::-1], dones[::-1]):
            mask = 1.0 - done 
            advantage = delta + gamma * lmbda * advantage * mask
            advantage_list.append(advantage)
        
        advantage_list.reverse()
        return torch.tensor(np.array(advantage_list), dtype=torch.float)
    
    else:
        # --- 新式模式：需要 term (dones) 和 trunc (truncateds) ---
        truncateds = truncateds.detach().cpu().numpy()
        terminateds = dones # term
        
        advantage_list = []
        advantage = 0.0
        
        for delta, term, trunc in zip(td_delta[::-1], terminateds[::-1], truncateds[::-1]):
            # 1. GAE 传递项的修正因子: gamma * lambda * (1 - term) * A_{t+1}
            # 如果是 terminated，不传递未来 advantage
            next_advantage_term = gamma * lmbda * advantage * (1.0 - term)
            
            # 2. 预估 A_t
            advantage = delta + next_advantage_term
            
            # 3. 如果是 truncated，当前步的 Advantage 不应向后传播（或者说被重置），
            # 但当前步本身的价值估计是基于 TD error 的。
            # 这里的逻辑通常是：truncated 意味着环境没结束但被强行切断，
            # 此时我们不希望这个“切断”产生的数学上的 Done 影响 value 学习，
            # 但 GAE 计算上，若被截断，序列确实断了。
            # 在某些实现中，truncated 步的 advantage 会保留，但不传给 t-1。
            # 如下写法：advantage = advantage * (1.0 - trunc) 意味着如果本步 truncated，advantage 归零？
            # 修正理解：这里应指 A_{t} 计算出来后，传给 A_{t-1} 的值。
            # 在循环中 advantage 变量其实代表 "next_advantage" (对于 t-1 来说)。
            # 所以这行代码实际是：准备传给上一时刻的 advantage。
            # 如果当前时刻 t 是 truncated，那么 t 对 t-1 没有贡献（因为并非自然结束）。
            advantage = advantage * (1.0 - trunc)
            
            advantage_list.append(advantage)
        
        advantage_list.reverse()
        return torch.tensor(np.array(advantage_list), dtype=torch.float)

# --- 独立缩放函数 ---

def scale_action_to_exec(a_norm, action_bounds):
    """
    将标准化的动作 a_norm (在 [-1,1] 范围内) 缩放到环境执行区间。
    """
    if action_bounds is None:
        return a_norm
    
    bounds = np.array(action_bounds)
    amin = bounds[:, 0]
    amax = bounds[:, 1]
    action_span = amax - amin
    
    return amin + (a_norm + 1.0) * 0.5 * action_span

# --- 网络结构 ---

class SharedGruBackbone(nn.Module):
    def __init__(self, state_dim, gru_hidden_size=128, gru_num_layers=1, output_dim=128, batch_first=True):
        super(SharedGruBackbone, self).__init__()
        self.feature_extractor = GruMlp(
            input_dim=state_dim,
            gru_hidden_size=gru_hidden_size,
            gru_num_layers=gru_num_layers,
            output_dim=output_dim,
            batch_first=batch_first
        )

    def forward(self, x, h_0=None):
        if x.dim() == 2:
            x = x.unsqueeze(1) 
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
        return log_prob_u 

    def entropy(self):
        return self.normal.entropy().sum(-1)

class PolicyNetHybridGRU(nn.Module):
    def __init__(self, state_dim, gru_hidden_size, gru_num_layers, hidden_dims, action_dims_dict, init_std=0.5, batch_first=True):
        super(PolicyNetHybridGRU, self).__init__()
        self.action_dims = action_dims_dict
        middle_dim = hidden_dims[0]
        self.backbone = SharedGruBackbone(state_dim, gru_hidden_size, gru_num_layers, middle_dim, batch_first=batch_first)

        layers = []
        prev_size = middle_dim
        for layer_size in hidden_dims[1:]:
            layers.append(nn.Linear(prev_size, layer_size))
            layers.append(nn.ReLU())
            prev_size = layer_size
        self.net = nn.Sequential(*layers)

        if self.action_dims.get('cont', 0) > 0:
            cont_dim = self.action_dims['cont']
            self.fc_mu = nn.Linear(prev_size, cont_dim)
            self.log_std_param = nn.Parameter(torch.log(torch.ones(cont_dim, dtype=torch.float) * init_std))

        if self.action_dims.get('cat', 0) > 0:
            cat_dim = self.action_dims['cat']
            self.fc_cat = nn.Linear(prev_size, cat_dim)

        if self.action_dims.get('bern', 0) > 0:
            bern_dim = self.action_dims['bern']
            self.fc_bern = nn.Linear(prev_size, bern_dim)

    def forward(self, x, h_0=None, min_std=1e-6, max_std=0.4):
        features_per_step, h_n = self.backbone(x, h_0)
        features = features_per_step[:, -1, :]
        shared_features = self.net(features)
        
        outputs = {'cont': None, 'cat': None, 'bern': None}

        if self.action_dims.get('cont', 0) > 0:
            mu = self.fc_mu(shared_features)
            std = torch.exp(self.log_std_param)
            std = torch.clamp(std, min=min_std, max=max_std)
            if mu.dim() > 1:
                std = std.unsqueeze(0).expand_as(mu)
            outputs['cont'] = (mu, std)

        if self.action_dims.get('cat', 0) > 0:
            cat_logits = self.fc_cat(shared_features)
            outputs['cat'] = F.softmax(cat_logits, dim=-1)

        if self.action_dims.get('bern', 0) > 0:
            bern_logits = self.fc_bern(shared_features)
            outputs['bern'] = bern_logits

        return outputs, h_n

class ValueNetGRU(nn.Module):
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

        self.actor = PolicyNetHybridGRU(state_dim, gru_hidden_size, gru_num_layers, hidden_dim, 
                                        action_dims_dict, batch_first=batch_first).to(device)
        self.critic = ValueNetGRU(state_dim, gru_hidden_size, gru_num_layers, hidden_dim, 
                                  batch_first=batch_first).to(device)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

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
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        if h_0_a is not None:
            h_0_a = self._maybe_move_h0_to_device(h_0_a)
            
        max_action_std = max_std if max_std is not None else self.max_std

        actor_outputs, h_n_a = self.actor(state, h_0_a, max_std=max_action_std)
        self.hidden_state_a = h_n_a.detach().cpu()

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

    def compute_gae_for_buffer(self, transition_dict):
        """
        [新增方法] 专门为单个子环境的 buffer 计算 GAE。
        适配 GRU：使用 buffer 中存储的 h0cs 计算当前 value，使用 buffer 中已收集的 next_values。
        """
        N = len(transition_dict['states'])
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float).view(-1, 1).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']), dtype=torch.float).view(-1, 1).to(self.device)
        
        # 提取 next_values (GRU模式下通常在 rollout 时通过 get_value 获得并存入 buffer)
        # 如果 buffer 里没有 next_values，GRU 模式下很难精确计算（因为没有 h_n_next）
        assert 'next_values' in transition_dict, "GRU PPO requires 'next_values' in transition_dict for GAE."
        next_vals = torch.tensor(np.array(transition_dict['next_values']), dtype=torch.float).view(-1, 1).to(self.device)

        # 提取 h0cs 用于计算 curr_vals
        h0cs_stack = torch.stack([transition_dict['h0cs'][i] for i in range(N)]).to(self.device)
        # 调整 shape: (N, Layers, 1, H) -> (Layers, N, H)
        h0cs_for_critic = h0cs_stack.squeeze(2).permute(1, 0, 2).contiguous()

        use_new_gae = 'truncs' in transition_dict and len(transition_dict['truncs']) > 0

        with torch.no_grad():
            curr_vals, _ = self.critic(states, h0cs_for_critic)

        if use_new_gae:
            # 新逻辑：区分 terminated (dones) 和 truncated (truncs)
            terminateds = dones
            truncateds = torch.tensor(np.array(transition_dict['truncs']), dtype=torch.float).view(-1, 1).to(self.device)
            
            # td_target 计算时，只有 terminated 才阻止 V(s_t+1) 的引导
            td_target = rewards + self.gamma * next_vals * (1 - terminateds)
            td_delta = td_target - curr_vals
            
            # GAE 计算需要同时传入 terminateds 和 truncateds
            advantage = compute_advantage(self.gamma, self.lmbda, td_delta, terminateds, truncateds).to(self.device)
        else:
            # 旧逻辑：dones = terminated or truncated
            td_target = rewards + self.gamma * next_vals * (1 - dones)
            td_delta = td_target - curr_vals
            advantage = compute_advantage(self.gamma, self.lmbda, td_delta, dones).to(self.device)
        
        # 将结果转回 list 并存入 dict
        transition_dict['advantages'] = advantage.cpu().numpy().flatten().tolist()
        transition_dict['td_targets'] = td_target.cpu().numpy().flatten().tolist()
        
        return transition_dict

    def update(self, transition_dict, adv_normed=False, clip_vf=False, clip_range=0.2, shuffled=0):
        N = len(transition_dict['states'])
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float).view(-1, 1).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']), dtype=torch.float).view(-1, 1).to(self.device)
        next_values = torch.tensor(np.array(transition_dict['next_values']), dtype=torch.float).to(self.device)
        
        # 提取 hidden states
        h0as_stack = torch.stack([transition_dict['h0as'][i] for i in range(N)]).to(self.device)
        h0cs_stack = torch.stack([transition_dict['h0cs'][i] for i in range(N)]).to(self.device)
        h0as_for_actor = h0as_stack.squeeze(2).permute(1, 0, 2).contiguous()
        h0cs_for_critic = h0cs_stack.squeeze(2).permute(1, 0, 2).contiguous()

        # 提取动作
        actions_from_buffer = transition_dict['actions']

        # --- GAE 与 Advantage 准备 ---
        use_new_gae = 'truncs' in transition_dict and len(transition_dict['truncs']) > 0
        
        # 1. 尝试使用预计算的 values
        if 'advantages' in transition_dict and 'td_targets' in transition_dict:
            advantage = torch.tensor(np.array(transition_dict['advantages']), dtype=torch.float).view(-1, 1).to(self.device)
            td_target = torch.tensor(np.array(transition_dict['td_targets']), dtype=torch.float).view(-1, 1).to(self.device)
        else:
            # 2. 如果没有预计算，则现场计算
            # 计算 Current Values (需要 h0cs)
            curr_values, _ = self.critic(states, h0cs_for_critic)
            
            if use_new_gae:
                terminateds = dones
                truncateds = torch.tensor(np.array(transition_dict['truncs']), dtype=torch.float).view(-1, 1).to(self.device)
                td_target = rewards + self.gamma * next_values * (1 - terminateds)
                td_delta = td_target - curr_values
                advantage = compute_advantage(self.gamma, self.lmbda, td_delta, terminateds, truncateds).to(self.device)
            else:
                td_target = rewards + self.gamma * next_values * (1 - dones)
                td_delta = td_target - curr_values
                advantage = compute_advantage(self.gamma, self.lmbda, td_delta, dones).to(self.device)

        # Shuffle 处理
        indices = np.arange(N)
        if shuffled:
            np.random.shuffle(indices)
            states = states[indices]
            rewards = rewards[indices]
            dones = dones[indices]
            next_values = next_values[indices] # 仅用于记录或debug
            advantage = advantage[indices]
            td_target = td_target[indices]
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

        # 计算 v_pred_old (使用 h0cs, 如果被 shuffle 过了则是 shuffle 后的 h0cs)
        v_pred_old = self.critic(states, h0cs)[0].detach()

        # 计算 Old Log Probs
        with torch.no_grad():
            old_actor_outputs, _ = self.actor(states, h0as, max_std=self.max_std)
            old_log_probs = torch.zeros(states.size(0), 1).to(self.device)

            if self.action_dims.get('cont', 0) > 0:
                mu_old, std_old = old_actor_outputs['cont']
                dist_old = SquashedNormal(mu_old, std_old)
                u_old = actions_on_device['cont']
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
        entropy_list, ratio_list = [], []

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
            
            # 熵约束
            entropy_factor = entropy.mean() # torch.clamp(entropy.mean(), -20, 7)
            actor_loss = -torch.min(surr1, surr2).mean() - self.k_entropy * entropy_factor

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

            self.actor_optimizer.step()
            self.critic_optimizer.step()
            
            actor_loss_list.append(actor_loss.item())
            critic_loss_list.append(critic_loss.item())
            actor_grad_list.append(model_grad_norm(self.actor))
            critic_grad_list.append(model_grad_norm(self.critic))
            entropy_list.append(entropy_factor.item())
            ratio_list.append(ratio.mean().item())

        self.actor_loss = np.mean(actor_loss_list)
        self.critic_loss = np.mean(critic_loss_list)
        self.actor_grad = np.mean(actor_grad_list)
        self.critic_grad = np.mean(critic_grad_list)
        self.entropy_mean = np.mean(entropy_list)
        self.ratio_mean = np.mean(ratio_list)
        
        check_weights_bias_nan(self.actor, "actor", "update后")
        check_weights_bias_nan(self.critic, "critic", "update后")