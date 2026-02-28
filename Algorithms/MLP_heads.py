import torch
import torch.nn.functional as F
from torch import nn

# MLP critic head
class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()

        layers = []
        prev_size = state_dim
        for layer_size in hidden_dim:
            layers.append(torch.nn.Linear(prev_size, layer_size))
            layers.append(nn.ReLU())
            prev_size = layer_size
        self.net = nn.Sequential(*layers)
        self.fc_out = torch.nn.Linear(prev_size, 1)

    def forward(self, x):
        y = self.net(x)
        return self.fc_out(y)
    
# 连续动作空间 MLP actor head
class PolicyNetContinuous(torch.nn.Module):
    """输出未压缩（pre-squash）的 mu，std 为与状态无关的可训练参数（每个动作维度一小段）。"""
    def __init__(self, state_dim, hidden_dim, action_dim, init_std=0.5):
        super(PolicyNetContinuous, self).__init__()
        layers = []
        prev_size = state_dim
        for layer_size in hidden_dim:
            layers.append(nn.Linear(prev_size, layer_size))
            layers.append(nn.ReLU())
            prev_size = layer_size
        self.net = nn.Sequential(*layers)
        self.fc_mu = torch.nn.Linear(prev_size, action_dim)
        # state-independent std parameter: store as log(std) for numerical stability
        init_std = float(init_std)
        # one parameter per action dim (unconstrained), optimizer will update this
        self.log_std_param = nn.Parameter(torch.log(torch.ones(action_dim, dtype=torch.float) * init_std))
        # 新增状态
        self._std_frozen = False
        self._force_std_to_max = False

    def forward(self, x, min_std=1e-6, max_std=0.4):
        # mu 仍由网络输出；std 由全局参数确定并广播到 batch
        x = self.net(x)
        mu = self.fc_mu(x)
        # 获取正的 std，使用 exp(log_std) 并进行 clamp
        std = torch.exp(self.log_std_param)

        # 如果用户希望强制 std 等于传入的 max_std（用于有监督阶段）
        if self._force_std_to_max:
            # max_std 可以是 scalar 或 tensor；统一扩展为与 std 同形
            if isinstance(max_std, torch.Tensor):
                max_t = max_std.to(std.device).type_as(std)
            else:
                max_t = torch.full_like(std, float(max_std))
            std = max_t
        else:
            # 正常 clamp（保持 backward 可用或冻结后的值）
            min_t = torch.full_like(std, float(min_std))
            if isinstance(max_std, torch.Tensor):
                max_t = max_std.to(std.device).type_as(std)
            else:
                max_t = torch.full_like(std, float(max_std))
            std = torch.clamp(std, min=min_t, max=max_t)

        # broadcast to batch: shape (batch_size, action_dim)
        if mu.dim() == 2:
            pass
        else:
            std = std.expand_as(mu)

        return mu, std

    # ---------- 新增控制接口 ----------
    def set_fixed_std(self, value=None):
        """把 std 设为指定 value 并冻结参数；若 value 为 None（或未传入），则冻结为当前 std 值。"""
        import math
        with torch.no_grad():
            if value is None:
                # 冻结为当前 log_std_param 的值（即当前 std）
                current_log = self.log_std_param.data.clone()
                self.log_std_param.data.copy_(current_log)
            else:
                self.log_std_param.data.fill_(math.log(float(value)))
        self._std_frozen = True
        self.log_std_param.requires_grad = False

    def clear_fixed_std(self):
        """取消固定并允许参数被训练（恢复 requires_grad=True）。"""
        self._std_frozen = False
        self.log_std_param.requires_grad = True

    def force_std_to_max(self, flag=True):
        """如果 True，则 forward 中直接把 std 置为传入的 max_std（无视 log_std_param/clamp）。"""
        self._force_std_to_max = bool(flag)
    
# 1重轮盘赌动作空间 actor head
class PolicyNetDiscrete(torch.nn.Module):
    def __init__(self, state_dim, hidden_dims, action_dim):
        super(PolicyNetDiscrete, self).__init__()
        # self.prelu = torch.nn.PReLU()
        layers = []
        prev_size = state_dim
        for layer_size in hidden_dims:
            layers.append(nn.Linear(prev_size, layer_size))
            layers.append(nn.ReLU())
            prev_size = layer_size
        self.net = nn.Sequential(*layers)
        self.fc_out = torch.nn.Linear(prev_size, action_dim)

        # # 固定神经网络初始化参数
        # torch.nn.init.xavier_normal_(self.fc_out.weight, gain=0.01)

    def forward(self, x, logits=0, temperature=1.0):
        x = self.net(x)
        out_logits = self.fc_out(x)
        
        # 关键修改：应用温度缩放
        # 注意：只在推理概率时缩放，如果直接请求 logits 用于 loss 计算，
        # 通常由外部决定是否包含温度，但为了 PPO 一致性，建议在这里处理
        scaled_logits = out_logits / temperature

        if not logits:
            return F.softmax(scaled_logits, dim=1)
        else:
            return scaled_logits

# 多重轮盘赌动作空间 actor head
class PolicyNetMultiDiscrete(torch.nn.Module):
    def __init__(self, state_dim, hidden_dims, action_dim):
        super(PolicyNetMultiDiscrete, self).__init__()
        layers = []
        prev_size = state_dim
        for layer_size in hidden_dims:
            layers.append(nn.Linear(prev_size, layer_size))
            layers.append(nn.ReLU())
            prev_size = layer_size
        self.net = nn.Sequential(*layers)

        # 强制要求 action_dim 是一个序列
        if not isinstance(action_dim, (list, tuple)):
            raise TypeError("action_dim must be a list or tuple of integers for PolicyNetMultiDiscrete.")
        self.action_dims = [int(a) for a in action_dim]

        # 总是使用多头输出
        self.fc_outs = nn.ModuleList([nn.Linear(prev_size, a) for a in self.action_dims])

        # 合理初始化输出层（可选，但推荐）
        for fc in self.fc_outs:
            torch.nn.init.xavier_normal_(fc.weight, gain=0.01)
            torch.nn.init.zeros_(fc.bias)

    def forward(self, x, logits=False, temperature=1.0):
        """
        输入 x 可以是任意维度，最后一维必须是 state_dim（特征维）。
        - logits (bool): 若为 True，返回 scaled logits；否则返回 softmax 概率。
        - temperature (float): 应用于 logits 的温度参数。
        - 返回值：一个 list，每项是形状为 [..., action_dim_i] 的张量。
        """
        # 保留前导形状，flatten 到 (N, state_dim)
        *lead_shape, feat_dim = x.shape
        x_flat = x.reshape(-1, feat_dim)
        h = self.net(x_flat)

        outs = []
        for fc in self.fc_outs:
            out_logits = fc(h)  # (N, action_dim_i)
            scaled_logits = out_logits / temperature
            
            if logits:
                result = scaled_logits.view(*lead_shape, scaled_logits.shape[-1])
            else:
                probs = F.softmax(scaled_logits, dim=-1)
                result = probs.view(*lead_shape, probs.shape[-1])
            outs.append(result)
        return outs


# 伯努利动作空间 actor head
class PolicyNetBernouli(torch.nn.Module):
    def __init__(self, state_dim, hidden_dims, action_dim):
        super(PolicyNetBernouli, self).__init__()
        self.prelu = torch.nn.PReLU()
        layers = []
        prev_size = state_dim
        for layer_size in hidden_dims:
            layers.append(nn.Linear(prev_size, layer_size))
            # layers.append(self.prelu)
            layers.append(nn.ReLU())
            prev_size = layer_size
        self.net = nn.Sequential(*layers)

        # ---- 新增：对隐藏层做合理初始化（ReLU -> Kaiming/He） ----
        for m in self.net:
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                torch.nn.init.zeros_(m.bias)

        self.fc_out = torch.nn.Linear(prev_size, action_dim)  # 输出 action_dim 个概率值

        # ---- 使输出 logits 接近 0，从而 sigmoid 输出接近 0.5 ----
        torch.nn.init.normal_(self.fc_out.weight, mean=0.0, std=1e-3)
        torch.nn.init.zeros_(self.fc_out.bias)

    def forward(self, x, temperature=1.0):
        x = self.net(x)
        out_logits = self.fc_out(x)
        # 关键修改：应用温度缩放
        # 注意：只在推理概率时缩放，如果直接请求 logits 用于 loss 计算，
        # 通常由外部决定是否包含温度，但为了 PPO 一致性，建议在这里处理
        scaled_logits = out_logits / temperature
        return scaled_logits  # 不再使用 sigmoid 激活函数


# 混合动作空间 Actor
# todo 待测试后合入


# MAPPO集中式 Critic 带id输入
# todo 待测试后合入，也可能分开到别的文件

