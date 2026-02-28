from torch.distributions import Normal
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# 计算并记录 actor / critic 的梯度范数（L2）
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
    """检查模型中名为 weight/bias 的参数是否包含 NaN，发现则抛出异常。
    参数:
      model: torch.nn.Module
      model_name: 用于错误消息中标识模型（如 "actor"/"critic"）
      place: 字符串，调用位置/上下文（如 "update_loop","pretrain_step"），用于更明确的错误报告
    """
    for name, param in model.named_parameters():
        if ("weight" in name) or ("bias" in name):
            if param is None:
                continue
            if torch.isnan(param).any():
                loc = f" at {place}" if place else ""
                raise ValueError(f"NaN detected in {model_name} parameter '{name}'{loc}")


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def compute_advantage(gamma_t, lambda_t, td_delta):
    """
    时间化 Generalized Advantage Estimation (GAE)
    参数:
        gamma_t: torch.Tensor, 每步的时间化折扣 γ^{τ_t}
        lambda_t: torch.Tensor, 每步的时间化 λ^{τ_t}
        td_delta: torch.Tensor, TD 残差
    返回:
        advantage: torch.Tensor, 与 td_delta 形状相同
    """
    # 确保在 CPU 上按序计算（避免显存循环）
    gamma_t = gamma_t.detach().cpu().numpy().flatten()
    lambda_t = lambda_t.detach().cpu().numpy().flatten()
    td_delta = td_delta.detach().cpu().numpy().flatten()

    advantage_list = []
    advantage = 0.0

    # 按时间反向累计 (从最后一步向前)
    for delta, g, l in zip(td_delta[::-1], gamma_t[::-1], lambda_t[::-1]):
        advantage = g * l * advantage + delta
        advantage_list.append(advantage)

    advantage_list.reverse()
    return torch.tensor(np.array(advantage_list), dtype=torch.float)



class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dims):
        super(ValueNet, self).__init__()
        # self.prelu = torch.nn.PReLU()

        layers = []
        prev_size = state_dim
        for layer_size in hidden_dims:
            layers.append(torch.nn.Linear(prev_size, layer_size))
            # layers.append(self.prelu)
            layers.append(nn.ReLU())
            prev_size = layer_size
        self.net = nn.Sequential(*layers)
        self.fc_out = torch.nn.Linear(prev_size, 1)

        # # 添加参数初始化
        # for layer in self.net:
        #     if isinstance(layer, nn.Linear):
        #         torch.nn.init.xavier_normal_(layer.weight, gain=0.01)
        # torch.nn.init.xavier_normal_(self.fc_out.weight, gain=0.01)

    def forward(self, x):
        y = self.net(x)
        return self.fc_out(y)


class PolicyNetDiscrete(torch.nn.Module):
    def __init__(self, state_dim, hidden_dims, action_dim):
        super(PolicyNetDiscrete, self).__init__()
        self.prelu = torch.nn.PReLU()
        layers = []
        prev_size = state_dim
        for layer_size in hidden_dims:
            layers.append(nn.Linear(prev_size, layer_size))
            # layers.append(self.prelu)
            layers.append(nn.ReLU())
            prev_size = layer_size
        self.net = nn.Sequential(*layers)
        self.fc_out = torch.nn.Linear(prev_size, action_dim)

        # # 固定神经网络初始化参数
        # torch.nn.init.xavier_normal_(self.fc_out.weight, gain=0.01)

    def forward(self, x):
        x = self.net(x)
        return F.softmax(self.fc_out(x), dim=1)


class PPO_discrete:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, state_dim, hidden_dims, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device, k_entropy=0.01, actor_max_grad=2, critic_max_grad=2):
        self.actor = PolicyNetDiscrete(state_dim, hidden_dims, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dims).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device
        self.k_entropy = k_entropy
        self.actor_max_grad=actor_max_grad
        self.critic_max_grad=critic_max_grad


    def set_learning_rate(self, actor_lr=None, critic_lr=None):
        """动态设置 actor 和 critic 的学习率"""
        if actor_lr is not None:
            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] = actor_lr
        if critic_lr is not None:
            for param_group in self.critic_optimizer.param_groups:
                param_group['lr'] = critic_lr    

    # take action
    def take_action(self, state, explore=True):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs) # 离散的输出为类别分布
        if explore:
            action = action_dist.sample()
        else:
            action = torch.argmax(probs)
        # 返回动作索引与对应的概率分布（numpy array）
        probs_np = probs.detach().cpu().numpy()[0].copy() # [0]是batch维度
        return probs_np, action.item()

    def update(self, transition_dict, adv_normed=False, clip_vf=False, clip_range=0.2):
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        # actions 必须为 long 用于 gather 索引
        actions = torch.tensor(np.array(transition_dict['actions']), dtype=torch.long).view(-1, 1).to(self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']), dtype=torch.float).view(-1, 1).to(self.device)
        t_as = torch.tensor(np.array(transition_dict['t_as']), dtype=torch.float).view(-1, 1).to(self.device)
        
        log_probs = torch.log(self.actor(states).gather(1, actions))
        # 添加Actor NaN检查
        if torch.isnan(log_probs).any():
            raise ValueError("NaN in Actor outputs")
        # 添加Critic NaN检查
        critic_values = self.critic(states)
        if torch.isnan(critic_values).any():
            raise ValueError("NaN in Critic outputs")

        # 假设 t_as 已经是形状 [batch_size, 1] 的张量
        # 代表每条 transition 的持续时间（例如秒、帧数等）

        # 时间缩放折扣因子
        gamma_t = torch.pow(self.gamma, t_as)
        lambda_t = torch.pow(self.lmbda, t_as)

        # critic 更新目标
        td_target = rewards + gamma_t * self.critic(next_states) * (1 - dones)

        # TD 残差
        td_delta = td_target - self.critic(states)

        # 计算 advantage，传入逐步的 gamma_t 与 lambda_t
        advantage = compute_advantage(gamma_t, lambda_t, td_delta).to(self.device)

        
        # 优势归一化
        if adv_normed:
            adv_mean, adv_std = advantage.detach().mean(), advantage.detach().std(unbiased=False) 
            # advantage = torch.clamp((advantage - adv_mean) / (adv_std + 1e-8) -10.0, 10.0)
            
            # adv_mean, adv_std = advantage.mean(), advantage.std(unbiased=False) 
            advantage = (advantage - adv_mean) / (adv_std + 1e-8)

        # 提前计算一次旧的 value 预测（用于 value clipping）
        v_pred_old = self.critic(states).detach()  # (N,1)

        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        actor_grad_list = []
        actor_loss_list = []
        critic_grad_list = []
        pre_clip_actor_grad = []
        pre_clip_critic_grad = []
        critic_loss_list = []
        entropy_list = []
        ratio_list = []

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
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

            log_probs = torch.log(self.actor(states).gather(1, actions)) # (N,1)
            ratio = torch.exp(log_probs - old_log_probs) # (N,1)
            surr1 = torch.clamp(ratio, -20, 20) * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage

            probs = self.actor(states)
            action_dist = torch.distributions.Categorical(probs)

            entropy_factor = action_dist.entropy().mean() # torch.clamp(dist.entropy().mean(), -20, 70) # -20, 7 e^2

            actor_loss = torch.mean(-torch.min(surr1, surr2)) - self.k_entropy * entropy_factor # 标量

            # 计算 critic_loss：支持可选的 value clipping（PPO 风格）
            if clip_vf:
                v_pred = self.critic(states)                                  # 当前预测 (N,1)
                v_pred_clipped = torch.clamp(v_pred, v_pred_old - clip_range, v_pred_old + clip_range)
                vf_loss1 = (v_pred - td_target.detach()).pow(2)               # (N,1)
                vf_loss2 = (v_pred_clipped - td_target.detach()).pow(2)       # (N,1)
                critic_loss = torch.max(vf_loss1, vf_loss2).mean()
            else:
                # critic_loss = F.mse_loss(self.critic(states), td_target.detach()) # 原有
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
            entropy_list.append(action_dist.entropy().mean().detach().cpu().item())
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
