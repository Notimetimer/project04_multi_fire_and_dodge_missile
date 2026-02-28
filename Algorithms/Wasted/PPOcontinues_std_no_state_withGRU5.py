'''
联合优化器，同时使用actor_loss和critic_loss
'''

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Algorithms.SharedLayers import GruMlp


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


def compute_advantage(gamma, lmbda, td_delta, dones):
    td_delta = td_delta.detach().cpu().numpy()
    dones = dones.detach().cpu().numpy() # [新增] 转为 numpy
    advantage_list = []
    advantage = 0.0
    
    # [修改] 同时遍历 delta 和 done
    for delta, done in zip(td_delta[::-1], dones[::-1]):
        # 如果当前是 done，说明这是序列的最后一步（或者该步之后没有未来），
        # 此时不应该加上一步（时间上的未来）的 advantage。
        # 注意：这里的 advantage 变量存的是“下一步的优势”，所以要乘 (1-done)
        advantage = delta + gamma * lmbda * advantage * (1 - done)
        advantage_list.append(advantage)
        
    advantage_list.reverse()
    return torch.tensor(np.array(advantage_list), dtype=torch.float)


# --- 新的共享 GRU-MLP 架构 ---
class SharedGruBackbone(nn.Module):
    """共享的 GRU 特征提取器"""
    def __init__(self, state_dim, gru_hidden_size=128, gru_num_layers=1, output_dim=35, batch_first=True):
        super(SharedGruBackbone, self).__init__()
        
        # 使用您新版 SharedLayers 中的 GruMlp
        # 我们只使用它的 GRU 部分，所以 output_dim 可以设为 gru_hidden_size
        self.feature_extractor = GruMlp(
            input_dim=state_dim,
            gru_hidden_size=gru_hidden_size,
            gru_num_layers=gru_num_layers,
            output_dim=output_dim,
            batch_first=batch_first
        )


    def forward(self, x, h_0=None):
        # 如果输入维度为2 (B, D)，增加序列维度S=1
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        
        # 输入 x 的形状: (Batch, SeqLen, StateDim), e.g., (B, 10, 35)
        # h_0: (num_layers, B, gru_hidden_size) 或 None
        # GruMlp 输出形状: (B, SeqLen, gru_hidden_size)
        features_per_step, h_n = self.feature_extractor(x, h_0)
        # 返回每个时间步的特征（B, SeqLen, gru_hidden_size）和隐藏状态，由上层决定如何聚合
        return features_per_step, h_n

class SquashedNormal:
    """带 tanh 压缩的高斯分布。

    采样：u ~ N(mu, std)（使用 rsample 支持 reparam），a = tanh(u)
    log_prob：基于 u 的 normal.log_prob(u) 并加上 tanh 的 Jacobian 修正项：-sum log(1 - tanh(u)^2)
    注意：外部需要把动作缩放到环境动作空间（仿射变换）。
    """

    def __init__(self, mu, std, eps=1e-6):
        self.mu = mu
        if not torch.is_tensor(std):
            std = torch.as_tensor(std, device=mu.device, dtype=mu.dtype)
        self.std = torch.clamp(std, min=float(eps))
        self.normal = Normal(mu, std)
        self.eps = eps
        self.mean = mu

    def sample(self):
        # rsample 以支持 reparameterization 重参数化采样, 结果是可导的
        u = self.normal.rsample()
        a = torch.tanh(u)
        return a, u

    def log_prob(self, a, u):
        # 为数值稳定性添加小量
        log_prob_u = self.normal.log_prob(u)
        jacobian = 0  # 保存u的话就不需要该修正项
        return log_prob_u - jacobian  # 返回形状为 (batch_size, action_dim)

    def entropy(self):
        # 近似：使用 base normal 的熵之和（不考虑 tanh 的修正）
        # 这在实践中通常足够，若需精确熵可用采样估计
        ent = self.normal.entropy().sum(-1)
        return ent


class PolicyNetContinuousGRU(nn.Module):
    """新的 Actor 网络，包含共享 backbone 和一个策略头"""

    def __init__(self, state_dim, gru_hidden_size, gru_num_layers=1, middle_dim=35,
                  head_hidden_dims=[128], action_dim=3, init_std=0.5, batch_first=True):
        super(PolicyNetContinuousGRU, self).__init__()

        # 1. 共享的 GRU 特征提取器
        self.backbone = SharedGruBackbone(state_dim, gru_hidden_size, gru_num_layers, middle_dim, 
                                          batch_first=batch_first)



        # 2. Actor 头 (一个独立的MLP)
        # 输入维度是 GRU 的输出维度
        layers = []
        prev_size = middle_dim
        for layer_size in head_hidden_dims:
            layers.append(nn.Linear(prev_size, layer_size))
            layers.append(nn.ReLU())
            prev_size = layer_size
        self.head = nn.Sequential(*layers)
        self.fc_mu = torch.nn.Linear(prev_size, action_dim)

        # std 参数与之前保持一致
        self.log_std_param = nn.Parameter(torch.log(torch.ones(action_dim, dtype=torch.float) * init_std))

    def forward(self, x, h_0=None, min_std=1e-6, max_std=0.4):
                
        # 如果输入维度为2 (B, D)，增加序列维度S=1
        if x.dim() == 2:
            x = x.unsqueeze(1)
        # backbone 返回每个时间步的特征 (B, SeqLen, gru_hidden_size) 和 h_n
        features_per_step, h_n = self.backbone(x, h_0)
        # 在 Actor 里取序列最后一步的特征作为聚合表示 (B, gru_hidden_size)
        features = features_per_step[:, -1, :]

        head_output = self.head(features)
        mu = self.fc_mu(head_output)

        std = torch.exp(self.log_std_param)
        std = torch.clamp(std, min=min_std, max=max_std)

        # 修正 std 的广播逻辑
        if mu.dim() == 2:  # (B, action_dim)
            std = std.unsqueeze(0).expand(mu.size(0), -1)  # 广播到 (B, action_dim)

        return mu, std, h_n


class ValueNetGRU(nn.Module):
    """新的 Critic 网络，与 Actor 共享同一个 backbone 实例"""

    def __init__(self, shared_backbone, middle_dim, head_hidden_dims):
        super(ValueNetGRU, self).__init__()

        # 1. 引用来自 Actor 的共享 backbone
        self.backbone = shared_backbone

        # 2. Critic 头 (一个独立的MLP)
        layers = []
        prev_size = middle_dim
        for layer_size in head_hidden_dims:
            layers.append(torch.nn.Linear(prev_size, layer_size))
            layers.append(nn.ReLU())
            prev_size = layer_size
        self.head = nn.Sequential(*layers)
        self.fc_out = torch.nn.Linear(prev_size, 1)

    def forward(self, x, h_0=None):
        # x 的形状: (B, SeqLen, StateDim)
        # GRU 特征提取 (梯度会从这里流回共享的 backbone)
        features, h_n = self.backbone(x, h_0)
        features = features[:, -1, :] # 不要多余的seq_len维度 (B, gru_hidden_size)

        temp = self.head(features)
        value = self.fc_out(temp)
        return value, h_n


class PPOContinuous:
    def __init__(self, state_dim, gru_hidden_size, gru_num_layers, middle_dim,
                 head_hidden_dims, action_dim, actor_lr, critic_lr, backbone_lr=None,
                 lmbda=None, epochs=10, eps=0.2, gamma=0.99, device='cpu',
                 k_entropy=0.01, critic_max_grad=2, actor_max_grad=2, max_std=0.3, critic_coef=0.5, batch_first=True):

        # 1. 创建 Actor，它内部包含了共享的 backbone
        self.actor = PolicyNetContinuousGRU(state_dim, gru_hidden_size, gru_num_layers, middle_dim,
                                            head_hidden_dims, action_dim, batch_first=batch_first).to(device)

        # 2. 创建 Critic，并把 Actor 的 backbone 传递给它，以实现共享
        self.critic = ValueNetGRU(self.actor.backbone, middle_dim, head_hidden_dims).to(device)

        # 3. 设置统一优化器，但为 backbone/actor_head/critic_head 保留独立 lr（使用 param_groups）
        backbone_params = list(self.actor.backbone.parameters())
        # actor head params: actor 的所有参数中排除 backbone 部分
        actor_head_params = [p for n, p in self.actor.named_parameters() if not n.startswith("backbone.")]
        critic_head_params = list(self.critic.head.parameters()) + list(self.critic.fc_out.parameters())

        if backbone_lr is None:
            backbone_lr = actor_lr

        self.total_optimizer = torch.optim.Adam([
            {"params": backbone_params, "lr": float(backbone_lr)},
            {"params": actor_head_params, "lr": float(actor_lr)},
            {"params": critic_head_params, "lr": float(critic_lr)},
        ])

        # 便捷引用与参数组保留（用于裁剪/检查）
        self._backbone_params = backbone_params
        self._actor_head_params = actor_head_params
        self._critic_head_params = critic_head_params
        self.critic_coef = critic_coef

        # 确保超参数为正确类型，避免被误传入非数值对象
        self.device = device
        self.gamma = float(gamma)
        self.lmbda = float(lmbda) if lmbda is not None else None
        self.epochs = int(epochs)
        self.eps = float(eps)
        self.k_entropy = float(k_entropy)
        self.critic_max_grad = float(critic_max_grad)
        self.actor_max_grad = float(actor_max_grad)
        self.max_std = float(max_std)

        # 隐藏态信息
        self.gru_num_layers = gru_num_layers
        self.gru_hidden_size = gru_hidden_size
        self.hidden_state = None

    def reset_hidden_state(self):
        """重置隐藏状态"""
        self.hidden_state = self.get_h0(batch_size=1)

    def set_learning_rate(self, actor_lr=None, critic_lr=None, backbone_lr=None):
        """动态设置 actor 和 critic 的学习率"""
        # 如果使用合并的 total_optimizer，按 param_groups 顺序更新 lr
        if backbone_lr is not None:
            self.total_optimizer.param_groups[0]['lr'] = float(backbone_lr)
        if actor_lr is not None:
            self.total_optimizer.param_groups[1]['lr'] = float(actor_lr)
        if critic_lr is not None:
            self.total_optimizer.param_groups[2]['lr'] = float(critic_lr)
            

    def _scale_action_to_exec(self, a, action_bounds):
        """把 normalized action a (in [-1,1]) 缩放到环境区间。

        action_bounds: 形状为 (action_dim, 2) 的二维 NumPy 数组，
                       每行对应 amin 和 amax。
        """
        action_bounds = torch.as_tensor(action_bounds, dtype=a.dtype, device=a.device)
        if action_bounds.dim() == 2:
            # 处理二维张量 (action_dim, 2)
            amin = action_bounds[:, 0]
            amax = action_bounds[:, 1]
        elif action_bounds.dim() == 3:
            # 处理三维张量 (batch, action_dim, 2)
            amin = action_bounds[:, :, 0]
            amax = action_bounds[:, :, 1]
        else:
            raise ValueError("action_bounds 的维度必须是 2 或 3")
        
        # a in (-1,1) -> scale to [amin, amax]
        return amin + (a + 1.0) * 0.5 * (amax - amin)

    def _unscale_exec_to_normalized(self, a_exec, action_bounds):
        """把执行动作 a_exec 反向归一化到 [-1,1]。

        action_bounds: 形状为 (action_dim, 2) 的二维 NumPy 数组，
                       每行对应 amin 和 amax。
        """
        action_bounds = torch.as_tensor(action_bounds, dtype=a_exec.dtype, device=a_exec.device)
        if action_bounds.dim() == 2:
            # 处理二维张量 (action_dim, 2)
            amin = action_bounds[:, 0]
            amax = action_bounds[:, 1]
        elif action_bounds.dim() == 3:
            # 处理三维张量 (batch, action_dim, 2)
            amin = action_bounds[:, :, 0]
            amax = action_bounds[:, :, 1]
        else:
            raise ValueError("action_bounds 的维度必须是 2 或 3")
        
        # 防止除以零
        span = (amax - amin)
        span = torch.where(span == 0, torch.tensor(1e-6, device=span.device, dtype=span.dtype), span)
        a = 2.0 * (a_exec - amin) / span - 1.0
        # numerical stability
        return a.clamp(-0.999999, 0.999999)
    
    # 对外接口（保证numpy输入和numpy输出）
    def unscale_exec_to_normalized(self, a_exec, action_bounds):
        """Public wrapper: accepts numpy or torch, returns numpy on CPU."""
        a_exec_t = torch.as_tensor(a_exec, dtype=torch.float, device=self.device)
        action_bounds_t = torch.as_tensor(action_bounds, dtype=torch.float, device=self.device)
        a_norm_t = self._unscale_exec_to_normalized(a_exec_t, action_bounds_t)
        return a_norm_t.cpu().numpy()
    
    # take action
    def take_action(self, state, h_0=None, action_bounds=None, explore=True, max_std=None):
        # === 修改：处理序列输入 ===
        # state 现在的形状是 (SeqLen, StateDim), e.g., (10, 35)
        # 需要增加一个 batch 维度 -> (1, 10, 35)
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        # 如果外部传入了以 CPU 存储的 hidden（或来自 get_current_hidden_state_*），
        # 在送进模型前需要移动到 device
        if h_0 is not None:
            h_0 = self._maybe_move_h0_to_device(h_0)
        # 检查state中是否存在nan
        if torch.isnan(state).any() or torch.isinf(state).any():
            print('state', state)
        # 检查actor参数中是否存在nan
        check_weights_bias_nan(self.actor, "actor", "take action中")
        if max_std is None:
            max_action_std = self.max_std
        else:
            max_action_std = max_std

        mu, std, h_n = self.actor(state, h_0, min_std=1e-6, max_std=max_action_std)

        # 更新隐藏状态
        self.hidden_state = h_n.detach().cpu()

        # 检查mu, std是否含有nan
        if torch.isnan(mu).any() or torch.isnan(std).any() or torch.isinf(mu).any() or torch.isinf(std).any():
            print('mu', mu)
            print('std', std)
            raise ValueError(
                f"NaN/Inf detected in actor outputs: mu_nan={torch.isnan(mu).any().item()}, "
                f"std_nan={torch.isnan(std).any().item()}, mu_inf={torch.isinf(mu).any().item()}, "
                f"std_inf={torch.isinf(std).any().item()}"
            ) 

        dist = SquashedNormal(mu, std)
        if explore:
            a_norm, u = dist.sample()
        else:
            # use mean action: tanh(mu)
            u = mu
            a_norm = torch.tanh(u)

        a_exec = self._scale_action_to_exec(a_norm, action_bounds)
        # 返回动作、未压缩动作 u 和隐藏状态
        return a_exec[0].cpu().detach().numpy().flatten(), u[0].cpu().detach().numpy().flatten(), self.hidden_state

    def get_value(self, state, h0):
        # 仅在采集经验的时候使用，外部传入为一维向量 (StateDim,)
        # 构造形状 (1,1,StateDim) 并传入 critic
        state_input = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        # 确保 h0 在同一 device（如果有）
        if h0 is not None:
            h0 = self._maybe_move_h0_to_device(h0)

        value, h_out = self.critic(state_input, h0)
        # 返回：value 为 numpy (detached on CPU)，h_out 为 detached CPU tensor
        return value.squeeze(0).detach().cpu().numpy(), h_out.detach().cpu()
    
    
    def get_h0(self, batch_size=1):
        """
        获取 GRU 的初始隐藏状态 h0。
        输出：一个 detach 并转移到 CPU 的 tensor。
        """
        h0 = torch.zeros(self.gru_num_layers, batch_size, self.gru_hidden_size, dtype=torch.float32, device=self.device)
        return h0.detach().cpu()

    def get_current_hidden_state(self):
        """
        获取当前隐藏状态。
        如果隐藏状态为 None，则返回初始隐藏状态。
        """
        # if self.hidden_state is None:
        #     return self.get_h0(batch_size=1)  # 默认 batch_size 为 1
        return self.hidden_state

    def _maybe_move_h0_to_device(self, h0):
        """
        如果 h0 是 Tensor，则把它移动到 self.device（in-place 不做，返回新 tensor）。
        如果 h0 为 None 或非 Tensor，直接返回原值。
        目的：外部/存储的 hidden 通常以 CPU tensor 存储，使用前要移动到模型所在 device。
        """
        if h0 is None:
            return None
        if torch.is_tensor(h0):
            # 保证是同一 device（避免出现 input 在 cuda:0 而 hidden 在 cpu 的错误）
            if h0.device != self.device:
                return h0.to(self.device)
            return h0
        # 如果是 numpy 等，先转为 tensor 再移动
        try:
            return torch.as_tensor(h0, device=self.device)
        except Exception:
            return h0

    def update(self, transition_dict, adv_normed=False, clip_vf=False, clip_range=0.2, shuffled=0):
        # states 张量现在的形状是 (Batch, SeqLen, StateDim)
        # 在 update 方法开头添加 shuffle 功能
        indices = np.arange(len(transition_dict['states']))
        if shuffled:
            np.random.shuffle(indices)

        # 按打乱后的顺序重新排列数据
        states = torch.tensor(np.array(transition_dict['states'])[indices], dtype=torch.float).to(self.device)
        u_s = torch.tensor(np.array(transition_dict['actions'])[indices], dtype=torch.float).to(self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards'])[indices], dtype=torch.float).view(-1, 1).to(self.device)
        # next_states = torch.tensor(np.array(transition_dict['next_states'])[indices], dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones'])[indices], dtype=torch.float).view(-1, 1).to(self.device)
        h0s = torch.stack([transition_dict['h0s'][i] for i in indices]).to(self.device)
        # 修正 h0s 的形状，目标是 (num_layers, B, hidden_size)
        
        h0s = h0s.squeeze(2).permute(1, 0, 2).contiguous()  # 内存连续化
        
        # action_bounds = torch.tensor(np.array(transition_dict['action_bounds'])[indices], dtype=torch.float).to(self.device)
        next_values = torch.tensor(np.array(transition_dict['next_values'])[indices], dtype=torch.float).to(self.device)
        
        if 'max_stds' in transition_dict:
            max_stds = torch.tensor(np.array(transition_dict['max_stds'])[indices], dtype=torch.float).view(-1, 1).to(self.device)
        else:
            max_stds = self.max_std

        # 计算 td_target, advantage
        td_target = rewards + self.gamma * next_values * (1 - dones)
        td_delta = td_target - self.critic(states, h0s)[0]
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu(), dones.cpu()).to(self.device)
        
        # 优势归一化
        if adv_normed:
            adv_mean, adv_std = advantage.detach().mean(), advantage.detach().std(unbiased=False) 
            advantage = (advantage - adv_mean) / (adv_std + 1e-8)
            # advantage = torch.clamp((advantage - adv_mean) / (adv_std + 1e-8) -10.0, 10.0)

        # 提前计算一次旧的 value 预测（用于 value clipping）
        v_pred_old = self.critic(states, h0s)[0].detach()  # (N,1)

        # 策略输出（未压缩的 mu,std）
        mu, std, _ = self.actor(states, h0s, min_std=1e-6, max_std=max_stds) # self.max_std
        # 构造 SquashedNormal 并计算 old_log_probs
        dist = SquashedNormal(mu.detach(), std.detach())

        # # 将执行动作反向归一化到 [-1,1]，以便计算 log_prob
        # actions_normalized = self._unscale_exec_to_normalized(actions_exec, action_bounds)
        
        # # 反算 u = atanh(a)
        u_old = u_s
        # old_log_probs = dist.log_prob(0, u_old) # (N,1)

        # 提前在action_dim维度求和
        old_log_probs = dist.log_prob(0, u_old).sum(-1, keepdim=True)    # -> (N,1)

        if torch.isnan(old_log_probs).any():
            raise ValueError("old_log_probs 包含 NaN，检查 action_bounds 或 actions 的合法性")

        actor_grad_list = []
        actor_loss_list = []
        critic_grad_list = []
        pre_clip_actor_grad = []
        pre_clip_critic_grad = []
        critic_loss_list = []
        entropy_list = []
        ratio_list = []

        for _ in range(self.epochs):
            mu, std, _ = self.actor(states, h0s, min_std=1e-6, max_std=self.max_std)
            if torch.isnan(mu).any() or torch.isnan(std).any():
                raise ValueError("NaN in Actor outputs in loop")
            critic_values = self.critic(states, h0s)[0]
            if torch.isnan(critic_values).any():
                raise ValueError("NaN in Critic outputs in loop")

            # 权重/偏置 NaN 检查（在每次前向后、反向前检查参数）
            check_weights_bias_nan(self.actor, "actor", "update循环中")
            check_weights_bias_nan(self.critic, "critic", "update循环中")

            dist = SquashedNormal(mu, std)
            # 计算当前策略对历史执行动作的 log_prob（使用同一个 u_old）
            # log_probs = dist.log_prob(0, u_old) # (N,1)

            # 提前在action_dim维度求和
            log_probs = dist.log_prob(0, u_old).sum(-1, keepdim=True)   # -> (N,1)

            ratio = torch.exp(log_probs - old_log_probs) # (N,1)
            # surr1 = ratio * advantage
            # clamp surr1
            surr1 = torch.clamp(ratio, -20, 20) * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            # 可选：对surr1用一个很大的范围去clamp防止出现一个很负的数
            entropy_factor = dist.entropy().mean()  # torch.clamp(dist.entropy().mean(), -20, 70) # -20, 7 e^2
            actor_loss_reward_term = -torch.min(surr1, surr2).sum(-1).mean()
            actor_loss = actor_loss_reward_term - self.k_entropy * entropy_factor

            # ↑如果求和之和还要保留原先的张量维度，用torch.sum(torch.min(surr1,surr2),dim=-1,keepdim=True)

            # 计算 critic_loss：支持可选的 value clipping（PPO 风格）
            if clip_vf:
                v_pred = self.critic(states, h0s)[0]                                  # 当前预测 (N,1)
                v_pred_clipped = torch.clamp(v_pred, v_pred_old - clip_range, v_pred_old + clip_range)
                vf_loss1 = (v_pred - td_target.detach()).pow(2)               # (N,1)
                vf_loss2 = (v_pred_clipped - td_target.detach()).pow(2)       # (N,1)
                critic_loss = torch.max(vf_loss1, vf_loss2).mean()
            else:
                critic_loss = F.mse_loss(self.critic(states, h0s)[0], td_target.detach())

            # 使用合并优化器：基于总损失反向传播
            self.total_optimizer.zero_grad()
            total_loss = actor_loss + self.critic_coef * critic_loss
            total_loss.backward()
            
            # 裁剪前梯度（基于分组/模块）
            pre_clip_actor_grad.append(model_grad_norm(self.actor))
            pre_clip_critic_grad.append(model_grad_norm(self.critic))

            # 针对不同参数集合分别裁剪梯度（backbone/actor_head 使用 actor_max_grad，critic_head 使用 critic_max_grad）
            if hasattr(self, "_backbone_params") and len(self._backbone_params) > 0:
                nn.utils.clip_grad_norm_(self._backbone_params, max_norm=self.actor_max_grad)
            if hasattr(self, "_actor_head_params") and len(self._actor_head_params) > 0:
                nn.utils.clip_grad_norm_(self._actor_head_params, max_norm=self.actor_max_grad)
            if hasattr(self, "_critic_head_params") and len(self._critic_head_params) > 0:
                nn.utils.clip_grad_norm_(self._critic_head_params, max_norm=self.critic_max_grad)

            # # 保存用于日志/展示的数值（断开计算图并搬到 CPU）
            actor_grad_list.append(model_grad_norm(self.actor))
            actor_loss_list.append(actor_loss.detach().cpu().item())
            critic_grad_list.append(model_grad_norm(self.critic))            
            critic_loss_list.append(critic_loss.detach().cpu().item())
            entropy_list.append(dist.entropy().mean().detach().cpu().item())
            ratio_list.append(ratio.mean().detach().cpu().item())

            self.total_optimizer.step()
        
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



