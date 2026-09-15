import os, sys
from torch.distributions import Normal
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from math import *

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

from _context import *
from Algorithms.rl_utils import moving_average, old_moving_average


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


# --- 广义优势函数计算 ---
# 旧的实现
# def compute_advantage(gamma, lmbda, td_delta, dones):
#     td_delta = td_delta.detach().cpu().numpy()
#     dones = dones.detach().cpu().numpy() # [新增] 转为 numpy
#     advantage_list = []
#     advantage = 0.0
    
#     # [修改] 同时遍历 delta 和 done
#     for delta, done in zip(td_delta[::-1], dones[::-1]):
#         # 如果当前是 done，说明这是序列的最后一步（或者该步之后没有未来），
#         # 此时不应该加上一步（时间上的未来）的 advantage。
#         # 注意：这里的 advantage 变量存的是“下一步的优势”，所以要乘 (1-done)
#         advantage = delta + gamma * lmbda * advantage * (1 - done)
#         advantage_list.append(advantage)
        
#     advantage_list.reverse()
#     return torch.tensor(np.array(advantage_list), dtype=torch.float)

# --- 保持 compute_advantage 函数，但根据传入参数数量切换逻辑 ---
def compute_advantage(gamma, lmbda, td_delta, dones, truncateds=None): # truncateds 默认为 None
    # 确保输入转为 numpy
    # 只要第0维是 BufferSize (时间维度)，后续维度 (如 [Env, Agent, 1]) 会被 Numpy 广播机制自动处理。
    # zip(td_delta[::-1], ...) 会沿着第0维切片，取出的 delta 形状为 [Env, Agent, 1]，
    # 后续的加减乘除都是 element-wise 的，因此各环境/智能体之间计算是独立的（平行的）。
    td_delta = td_delta.detach().cpu().numpy()
    dones = dones.detach().cpu().numpy() # 假设这里的 dones 是 terminateds (term)

    if truncateds is None:
        # --- 旧式/兼容模式：dones = term OR trunc ---
        # 此时，dones 就是 $\text{done}_t$
        advantage_list = []
        advantage = 0.0 # 初始标量0，第一次运算会自动广播为 [Env, Agent, 1]
        
        for delta, done in zip(td_delta[::-1], dones[::-1]):
            advantage = delta + gamma * lmbda * advantage * (1 - done)
            advantage_list.append(advantage)
        
        advantage_list.reverse()
        return torch.tensor(np.array(advantage_list), dtype=torch.float)
    
    else:
        # --- 新式模式：需要 term (dones) 和 trunc (truncateds) ---
        truncateds = truncateds.detach().cpu().numpy()
        terminateds = dones # $\text{term}_t$
        
        advantage_list = []
        advantage = 0.0
        
        for delta, term, trunc in zip(td_delta[::-1], terminateds[::-1], truncateds[::-1]):
            # 1. GAE 传递项的修正因子: $\gamma \lambda (1 - \text{term}_t) A_{t+1}$
            # 这里的乘法是 element-wise 的，所以不同 env/agent 互不干扰
            next_advantage_term = gamma * lmbda * advantage * (1.0 - term)
            
            # 2. 预估 A_t: $A'_t = \delta_t + \text{next\_advantage\_term}$
            advantage = delta + next_advantage_term
            
            # 3. 最终 A_t 屏蔽: $A_t = (1 - \text{trunc}_t) \cdot A'_t$
            advantage = advantage * (1.0 - trunc)
            
            advantage_list.append(advantage)
        
        advantage_list.reverse()
        return torch.tensor(np.array(advantage_list), dtype=torch.float)


# 计算蒙特卡洛回报（MARWIL），没有考虑并行和多智能体的情况（目前只用于模仿学习）
def compute_monte_carlo_returns(gamma, rewards, dones, truncateds=None):
    """
    计算蒙特卡洛回报 (Discounted Returns / Rt)。
    公式: G_t = r_t + gamma * G_{t+1} * (1 - done_t)
    
    参数:
        gamma (float): 折扣因子
        rewards (tensor/array): 奖励列表
        dones (tensor/array): 终止信号 (terminateds)
        truncateds (tensor/array, optional): 截断信号。默认为 None。
    
    返回:
        torch.Tensor: 计算好的 Returns (G_t)
    """
    
    # --- 1. 数据类型处理 (Tensor -> Numpy) ---
    if isinstance(rewards, torch.Tensor):
        rewards = rewards.detach().cpu().numpy()
    if isinstance(dones, torch.Tensor):
        dones = dones.detach().cpu().numpy()
    if truncateds is not None and isinstance(truncateds, torch.Tensor):
        truncateds = truncateds.detach().cpu().numpy()
        
    # --- 2. 核心计算逻辑 ---
    if truncateds is None:
        # --- 旧式/简单模式：dones 代表所有结束情况 ---
        returns_list = []
        G = 0.0
        
        # 逆序遍历
        for r, done in zip(rewards[::-1], dones[::-1]):
            # 如果当前步 done=True，说明这是序列终点，G_next 归零
            G = r + gamma * G * (1.0 - done)
            returns_list.append(G)
            
        returns_list.reverse()
        return torch.tensor(np.array(returns_list), dtype=torch.float)
    
    else:
        # --- 新式模式：区分 Terminated (正常结束) 和 Truncated (截断/超时) ---
        returns_list = []
        G = 0.0
        
        # 逆序遍历
        for r, term, trunc in zip(rewards[::-1], dones[::-1], truncateds[::-1]):
            # 逻辑说明：
            # 1. G 是上一轮循环计算的 G_{t+1}
            # 2. 如果当前时刻 t 是 truncated (截断) 或者 terminated (终止)，
            #    则不能从 t+1 时刻继承价值 (因为 t 和 t+1 不在同一个逻辑轨迹内，或者 t 已经是终点)。
            #    因此乘以 (1-term) 和 (1-trunc) 进行屏蔽。
            
            mask = (1.0 - term) * (1.0 - trunc)
            G = r + gamma * G * mask
            returns_list.append(G)
            
        returns_list.reverse()
        return torch.tensor(np.array(returns_list), dtype=torch.float)

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
        # a: tanh(u)
        # log_prob(u) - sum log(1 - tanh(u)^2)
        # normal.log_prob 返回每个维度的 log_prob，需要 sum
        # 为数值稳定性添加小量
        log_prob_u = self.normal.log_prob(u)
        # jacobian term
        jacobian = 0 # 保存u的话就不需要该修正项
        # jacobian = 2*(np.log(2.0)-u-F.softplus(-2*u))
        # jacobian = torch.log(1 - a.pow(2) + self.eps)
        # sum over action dim, keep dims consistent: return (N, 1)
        # 取消提前求和 # return (log_prob_u - jacobian).sum(-1, keepdim=True)
        return log_prob_u - jacobian  # 返回形状为 (batch_size, action_dim)

    def entropy(self):
        # 近似：使用 base normal 的熵之和（不考虑 tanh 的修正）
        # 这在实践中通常足够，若需精确熵可用采样估计
        ent = self.normal.entropy().sum(-1)
        return ent


# 线性空间下随机概率分布
class LinearDiscretizedDistribution:
    """
    线性空间离散化分布 (高斯核分布)。
    通过计算 mu 与预设 n 个采样点之间的负平方距离，构建 Categorical 分布。
    适用于 [-1, 1] 或 [-pi/2, pi/2] 等不需要处理周期性边界的连续区间。
    """
    def __init__(self, mu, std, low=-1.0, high=1.0, n=31, eps=1e-8):
        """
        Args:
            mu:    (batch_size, 1) 神经网络输出的均值。
            std:   (batch_size, 1) 或 标量。等效标准差。
            low:   区间的下限。
            high:  区间的上限。
            n:     离散采样点的数量。
        """
        self.device = mu.device
        self.n = n
        self.low = low
        self.high = high
        self.eps = eps

        # 1. 均值限制。虽然高斯分布均值可以超限，但为了采样点覆盖，建议限制在 [low, high]
        self.mu = mu 
        
        # 2. 转换 std 为集中度参数 (等效于 1 / (2 * sigma^2))
        self.std = torch.as_tensor(std, device=self.device).clamp(min=1e-3)
        self.tau_inv = 1.0 / (2.0 * self.std.pow(2) + eps)

        # 3. 在指定范围内预设 n 个等距采样点 (support points)
        # shape: (n,)
        self.v_points = torch.linspace(low, high, n, device=self.device)

        # 4. 计算 Logits: - (v - mu)^2 / (2 * sigma^2)
        # 利用广播机制计算 (batch_size, n)
        # (batch_size, 1) - (n,) -> (batch_size, n)
        dist_sq = (self.v_points - self.mu).pow(2)
        self.logits = -dist_sq * self.tau_inv
        
        self.probs = F.softmax(self.logits, dim=-1)
        self.dist = torch.distributions.Categorical(probs=self.probs)

    def sample(self):
        """
        采样：返回采样点的索引。
        若需要返回物理值，请配合 self.v_points[idx] 使用。
        """
        a = self.dist.sample()
        return a, self.logits

    def log_prob(self, a):
        """计算动作索引 a 的对数概率"""
        return self.dist.log_prob(a)

    def entropy(self):
        """返回离散熵"""
        return self.dist.entropy()

    @property
    def mean_idx(self):
        """返回概率最大的采样点索引"""
        return torch.argmax(self.probs, dim=-1)
    
    @property
    def sampled_value(self):
        """
        辅助方法：如果你的动作空间需要物理值输出，可以直接调用此属性。
        """
        idx = self.dist.sample()
        return self.v_points[idx]

# 圆周空间下随机概率分布
class CircularDiscretizedDistribution:
    """
    圆周空间离散化分布 (星形分布)。
    通过输入向量 (x, y) 与预设的 n 个方向向量做点积，构建 Categorical 分布。
    接口对齐 SquashedNormal。
    强制归一化输入向量，使得 std (等效 sigma) 成为控制分布宽度的唯一变量。
    """
    def __init__(self, vec_h, std, n=12, eps=1e-8):
        """
        Args:
            vec_h: (batch_size, 2) 神经网络输出。我们只取其方向。
            std:   (batch_size, 1) 或 标量。等效标准差 (单位: 弧度)。
                   std 越大，分布越扁平；std 越小，分布越尖锐。
            n:     采样精度。
        """
        self.device = vec_h.device
        self.n = n
        self.eps = eps
        
        # 1. 强制归一化神经网络的输出向量，消除模长对熵的影响
        # 此时 vec_h 仅代表“意图方向”
        self.direction = F.normalize(vec_h, p=2, dim=-1, eps=eps)

        # 2. 将输入的 std 转化为集中度参数 kappa (1/sigma^2)
        # 为了数值稳定性，限制 std 的最小值
        self.std = torch.as_tensor(std, device=self.device).clamp(min=1e-3)
        self.kappa = 1.0 / (self.std.pow(2) + eps)

        # 3. 预设 n 个离散方向向量
        angles = torch.linspace(0, 2 * np.pi, n + 1, device=self.device)[:-1]
        self.v_matrix = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)

        # 4. 计算 Logits: kappa * (direction · v)
        # 这里的 1/sigma^2 起到了原本 Boltzmann 温度倒数的作用
        self.logits = torch.matmul(self.direction, self.v_matrix.t()) * self.kappa
        
        self.probs = F.softmax(self.logits, dim=-1)
        self.dist = torch.distributions.Categorical(probs=self.probs)

    def sample(self):
        """
        采样：从离散分布中抽取索引。
        Returns:
            a: 抽取的动作索引 (0 到 n-1)
            logits: 该样本对应的全部 logits (用于兼容接口或辅助计算)
        """
        a = self.dist.sample()
        return a, self.logits

    def log_prob(self, a, logits_unused=None):
        """
        计算动作 a 的对数概率。
        Args:
            a: 动作索引 (batch_size,)
            logits_unused: 仅为了兼容 SquashedNormal 接口
        """
        # 直接使用 Categorical 的 log_prob
        return self.dist.log_prob(a)

    def entropy(self):
        # 注意：这是离散熵。当 n 很大时，它会趋近于连续分布的熵减去 log(n)
        return self.dist.entropy()

    @property
    def mean_idx(self):
        """
        返回概率最大的方向索引 (等效于高斯分布的均值)
        """
        return torch.argmax(self.probs, dim=-1)


class LinearDiscretizedDistribution_NL:
    """
    线性空间离散化分布 (不规则采样版本)。
    直接传入物理角度列表作为支撑点，高斯核计算 logits。
    适用于采样点不均匀分布的线性区间。
    """
    def __init__(self, mu, std, points=[pi/4, pi/8, 0.0, -pi/8, -pi/2], eps=1e-8):
        """
        Args:
            mu:     (batch_size, 1) 神经网络输出的均值 (tanh 后的值，范围 [-1,1])。
            std:    (batch_size, 1) 等效标准差。
            points: 物理角度列表，长度即为档位数。
        """
        self.device = mu.device
        self.points = torch.tensor(points, dtype=torch.float32, device=self.device)
        self.n = len(points)
        self.eps = eps

        self.mu = mu
        self.std = torch.as_tensor(std, device=self.device).clamp(min=1e-3)
        self.tau_inv = 1.0 / (2.0 * self.std.pow(2) + eps)

        # logits: -(v - mu)^2 / (2*sigma^2), shape (batch, n)
        dist_sq = (self.points - self.mu).pow(2)
        self.logits = -dist_sq * self.tau_inv

        self.probs = F.softmax(self.logits, dim=-1)
        self.dist = torch.distributions.Categorical(probs=self.probs)

    def sample(self):
        a = self.dist.sample()
        return a, self.logits

    def log_prob(self, a):
        return self.dist.log_prob(a)

    def entropy(self):
        return self.dist.entropy()

    @property
    def mean_idx(self):
        return torch.argmax(self.probs, dim=-1)


class CircularDiscretizedDistribution_NL:
    """
    圆周空间离散化分布 (不规则采样版本)。
    通过物理真实的采样角度构建星形分布，确保 std 的物理意义在全空间一致。
    """
    def __init__(self, vec_h, std, angles=[0, pi/3, pi/2, pi, -pi/2, -pi/3], eps=1e-8):
        """
        Args:
            vec_h:  (batch_size, 2) 神经网络输出。强制归一化以提取意图方向。
            std:    (batch_size, 1) 或 标量。等效物理标准差 (单位: 弧度)。
            angles: list, ndarray 或 Tensor。物理采样点的角度序列 (单位: 弧度)。
            eps:    数值稳定性小量。
        """
        self.device = vec_h.device
        self.eps = eps
        
        # 1. 处理传入的物理角度
        # 确保 angles 是 Tensor 且在正确的设备上
        self.angles = torch.as_tensor(angles, device=self.device, dtype=torch.float32)
        self.n = self.angles.size(0)
        
        # 2. 强制归一化输入向量，提取意图方向 (mu)
        self.direction = F.normalize(vec_h, p=2, dim=-1, eps=eps)

        # 3. 将物理 std 转化为集中度参数 kappa (1/sigma^2)
        self.std = torch.as_tensor(std, device=self.device).clamp(min=1e-3)
        self.kappa = 1.0 / (self.std.pow(2) + eps)

        # 4. 构建物理真实的基准向量矩阵 (v_matrix)
        # 每一个行向量对应一个物理动作的方向
        self.v_matrix = torch.stack([
            torch.cos(self.angles), 
            torch.sin(self.angles)
        ], dim=-1) # (n, 2)

        # 5. 计算 Logits: kappa * (direction · v)
        # 点积结果即为 cos(delta_theta)，kappa 控制分布的尖锐程度
        self.logits = torch.matmul(self.direction, self.v_matrix.t()) * self.kappa
        
        # 6. 生成离散概率分布
        self.probs = F.softmax(self.logits, dim=-1)
        self.dist = torch.distributions.Categorical(probs=self.probs)

    def sample(self):
        """
        采样：从离散分布中抽取索引。
        Returns:
            a: 抽取的动作索引 (0 到 n-1)
            logits: 该样本对应的全部 logits (用于兼容接口或辅助计算)
        """
        a = self.dist.sample()
        return a, self.logits

    def log_prob(self, a, logits_unused=None):
        """
        计算动作 a 的对数概率。
        Args:
            a: 动作索引 (batch_size,)
            logits_unused: 仅为了兼容 SquashedNormal 接口
        """
        # 直接使用 Categorical 的 log_prob
        return self.dist.log_prob(a)

    def entropy(self):
        # 注意：这是离散熵。当 n 很大时，它会趋近于连续分布的熵减去 log(n)
        return self.dist.entropy()

    @property
    def mean_idx(self):
        """
        返回概率最大的方向索引 (等效于高斯分布的均值)
        """
        return torch.argmax(self.probs, dim=-1)
    

# 圆周空间下随机概率分布
class CircularContDist4OnPolciy:
    def __init__(self, mu_vec, std, eps=1e-8):
        # mu_vec: (B, 2) 单位向量
        self.eps = float(eps)
        self.device = mu_vec.device
        self.mu_vec = mu_vec.view(-1, 2)
        self.std = std.to(self.device).view(-1, 1)

        # kappa from equivalent std
        self.kappa = 1.0 / (self.std.pow(2) + self.eps)

    def log_prob(self, action_vec):
        # action_vec: (B, 2) 经验池中的动作向量
        action_vec = action_vec.view(-1, 2).to(self.device)
        
        # 使用点积代替 cos(theta - mu)
        dot_product = torch.sum(action_vec * self.mu_vec, dim=-1, keepdim=True)
        
        if hasattr(torch.special, 'i0e'):
            # log I0 = log(i0e(kappa)) + kappa
            log_i0 = torch.log(torch.special.i0e(self.kappa).clamp(min=self.eps)) + self.kappa
        else:
            log_i0 = torch.log(torch.special.i0(self.kappa).clamp(min=self.eps))

        log_norm = torch.log(torch.tensor(2.0 * np.pi, device=self.device)) + log_i0
        return (self.kappa * dot_product) - log_norm

    def sample(self):
        # 采样依然基于 Von Mises 标量，随后转为向量存储
        mu_rad = torch.atan2(self.mu_vec[:, 1], self.mu_vec[:, 0]).detach().cpu().numpy()
        kappa_np = self.kappa.detach().cpu().numpy().reshape(-1)
        
        samples = np.array([np.random.vonmises(m, k) for m, k in zip(mu_rad, kappa_np)])
        # 转回向量 (B, 2)
        samples_vec = np.stack([np.cos(samples), np.sin(samples)], axis=1)
        return torch.as_tensor(samples_vec, dtype=torch.float, device=self.device)

    def entropy(self):
        """使用指数缩放贝塞尔函数计算熵，提高数值稳定性"""
        if hasattr(torch.special, 'i1e'):
            # 使用缩放版本: In(k) = Ine(k) * exp(k)
            i0e = torch.special.i0e(self.kappa).clamp(min=self.eps)
            i1e = torch.special.i1e(self.kappa)
            
            # ratio = I1/I0 = I1e/I0e
            ratio = i1e / i0e
            
            # log(I0) = log(i0e) + kappa
            log_i0 = torch.log(i0e) + self.kappa
            
            # H = -kappa * (I1/I0) + log(2π) + log(I0)
            # 带入缩放项后整理得：
            ent = self.kappa * (1.0 - ratio) + torch.log(torch.tensor(2.0 * np.pi, device=self.device)) + torch.log(i0e)
            return ent
        else:
            # 回退到标准版本（易溢出）
            i0 = torch.special.i0(self.kappa).clamp(min=self.eps)
            i1 = torch.special.i1(self.kappa)
            return -self.kappa * (i1 / i0) + torch.log(torch.tensor(2.0 * np.pi, device=self.device)) + torch.log(i0)

    def mean(self):
        # 返回 (batch, 2) 的均值向量
        return self.mu_vec

