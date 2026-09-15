'''
actor内置开火mask
与去串扰手段结合
'''

import numpy as np
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, Bernoulli
import copy
import os, sys
import json
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from Algorithms.Utils import model_grad_norm, check_weights_bias_nan, compute_advantage, SquashedNormal
from Algorithms.MLP_heads import ValueNet

def sigmoid(x):
    return 1/(1+np.exp(-x))

# =============================================================================
#  机动mask 开关配置
#  由同目录下的 mask_config.json 控制垂直/水平多对一映射与 mask 开关。
#  该配置只在 PolicyNetHybrid.__init__ 中读取一次，缓存为实例属性，
#  forward() 直接使用该实例属性控制机动mask，不会重复读盘。
# =============================================================================

_MASK_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mask_config.json')

def load_mask_config(override=None):
    cfg = {
        'ver': 0,
        'hor': 0,
    }
    if override is not None:
        for k in cfg:
            cfg[k] = int(override.get(k, cfg[k]))
        return cfg
    try:
        with open(_MASK_CONFIG_PATH, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        for k in cfg:
            cfg[k] = int(loaded.get(k, 0))
    except Exception as e:
        print(f"[mask_config] 加载 {_MASK_CONFIG_PATH} 失败，使用默认值。错误: {e}")
    return cfg

# [==== 有些实验没来得及标记，打个补丁来推断mask的情况
def _infer_mask_cfg_from_cat_out_dim(cat_out_dim, source_name):
    """
    根据 fc_cat 输出维度（即 net.fc_cat.2.bias 的长度）推断 ver/hor 配置。
    """
    shape_map = {
        12: {'ver': 0, 'hor': 0}, # 5 + 7
        20: {'ver': 1, 'hor': 0}, # 5+5+3 + 7
        24: {'ver': 1, 'hor': 1}, # 5+5+3 + 5+3+3
        16: {'ver': 0, 'hor': 1}, # 5 + 5+3+3
    }
    cfg = shape_map.get(int(cat_out_dim))
    if cfg is None:
        raise ValueError(
            f"无法从 {source_name} 的 fc_cat.2.bias 长度 {cat_out_dim} 推断 ver/hor 配置，"
            f"已知映射: {shape_map}"
        )
    return cfg


def infer_mask_cfg_from_actor_meta(meta_path):
    """
    从 actor.meta.json 的 net.fc_cat.2.bias 形状推断训练时的 ver/hor 配置。
    返回 {'ver': int, 'hor': int}。
    """
    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        bias_shape = meta.get('net.fc_cat.2.bias')
        if bias_shape is None:
            raise KeyError("net.fc_cat.2.bias not found in meta")
        if not isinstance(bias_shape, list) or len(bias_shape) == 0:
            raise ValueError(f"Unexpected bias shape: {bias_shape}")
        cfg = _infer_mask_cfg_from_cat_out_dim(bias_shape[0], meta_path)
        print(f"[mask_cfg] 从 {meta_path} 推断 ver={cfg['ver']}, hor={cfg['hor']} "
              f"(fc_cat.2.bias={bias_shape})")
        return cfg
    except Exception as e:
        print(f"[mask_cfg] 读取 {meta_path} 失败: {e}，使用默认配置 ver=0, hor=0")
        return {'ver': 0, 'hor': 0}


def infer_mask_cfg_from_state_dict(state_dict):
    """
    从 PyTorch state_dict 中的 net.fc_cat.2.bias 形状推断 ver/hor 配置。
    返回 {'ver': int, 'hor': int}。
    """
    bias_key = 'net.fc_cat.2.bias'
    try:
        if bias_key not in state_dict:
            raise KeyError(f"{bias_key} not found in state_dict")
        bias = state_dict[bias_key]
        cat_out_dim = int(bias.shape[0]) if hasattr(bias, 'shape') else int(len(bias))
        cfg = _infer_mask_cfg_from_cat_out_dim(cat_out_dim, "state_dict")
        print(f"[mask_cfg] 从 state_dict 推断 ver={cfg['ver']}, hor={cfg['hor']} "
              f"({bias_key} length={cat_out_dim})")
        return cfg
    except Exception as e:
        print(f"[mask_cfg] 从 state_dict 推断失败: {e}，使用默认配置 ver=0, hor=0")
        return {'ver': 0, 'hor': 0}
# ====]

# =============================================================================
# 0. RND 网络定义
# =============================================================================

class RNDTargetNet(nn.Module):
    """
    RND 目标网络（权重冻结）。
    - 2层全连接，LeakyReLU，防止负权重初始化后神经元死亡。
    - 正交初始化所有线性层。
    - 内置状态运行时归一化（Welford 在线算法）。
    """
    def __init__(self, state_dim, output_dim=128, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)
        for p in self.parameters():
            p.requires_grad = False
        # 运行时状态归一化统计量
        self.register_buffer('obs_mean', torch.zeros(state_dim))
        self.register_buffer('obs_var', torch.ones(state_dim))
        self.register_buffer('obs_count', torch.tensor(1e-4))

    def update_obs_stats(self, obs_batch):
        """Welford 在线算法更新均值和方差"""
        batch_mean = obs_batch.mean(0)
        batch_var = obs_batch.var(0, unbiased=False)
        batch_count = float(obs_batch.size(0))
        total = self.obs_count + batch_count
        delta = batch_mean - self.obs_mean
        new_mean = self.obs_mean + delta * batch_count / total
        M2 = self.obs_var * self.obs_count + batch_var * batch_count + delta ** 2 * self.obs_count * batch_count / total
        self.obs_mean = new_mean
        self.obs_var = torch.clamp(M2 / total, min=1e-8)
        self.obs_count = total

    def normalize(self, obs):
        return (obs - self.obs_mean) / (self.obs_var.sqrt() + 1e-8)

    def forward(self, x):
        return self.net(self.normalize(x))


class RNDPredictionNet(nn.Module):
    """
    RND 预测网络（持续更新）。
    - 3层全连接，ReLU，表达能力大于 Target 以保证充分拟合。
    - 使用 Kaiming Normal 初始化（与 Target 的正交初始化不同，确保两网络初始参数不同）。
    - 直接接受已归一化的状态向量（由 RNDTargetNet.normalize 提供）。
    """
    def __init__(self, state_dim, output_dim=128, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        # Kaiming Normal 初始化，与 Target 的正交初始化区分开
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x_normalized):
        return self.net(x_normalized)


# =============================================================================
# 1. 神经网络定义 (保持不变，只负责 forward 计算)
# =============================================================================

class PolicyNetHybrid(torch.nn.Module):
    """
    支持混合动作空间的策略网络 (纯 MLP)。
    引入了可学习的温度参数来控制离散和伯努利动作的熵。
    """
    def __init__(self, state_dim, hidden_dims, action_dims_dict, init_std=0.5, head_hidden_layer_num=1, Autoregressive=0, mask_cfg=None):
        super(PolicyNetHybrid, self).__init__()
        self.action_dims = action_dims_dict

        # [新增] 机动mask 开关：只在网络初始化时从
        # mask_config.json 读取一次（或被外部显式传入），永久保存为实例属性，forward() 不再重复读取磁盘。
        mask_cfg = load_mask_config(override=mask_cfg)
        self.ver_map = mask_cfg['ver']
        self.ver_mask = mask_cfg['ver']
        self.hor_map = mask_cfg['hor']
        self.hor_mask = mask_cfg['hor']
        # self.Autoregressive = Autoregressive
        # # 确定bern_dim和主干网络输入维度
        # bern_dim = self.action_dims.get('bern', 0)
        
        # # 如果需要Autoregressive，bern头网络直接处理原始state，输出再拼接到state后面
        # if self.Autoregressive == 1 and bern_dim > 0:
        #     # bern头网络：从原始state直接计算bern_logits
        #     layers_bern = []
        #     prev_size_bern = state_dim
        #     for layer_size in hidden_dims:
        #         layers_bern.append(nn.Linear(prev_size_bern, layer_size))
        #         layers_bern.append(nn.ReLU())
        #         prev_size_bern = layer_size
        #     layers_bern.append(nn.Linear(prev_size_bern, int(prev_size_bern/2)))
        #     layers_bern.append(nn.ReLU())
        #     layers_bern.append(nn.Linear(int(prev_size_bern/2), bern_dim))
        #     self.fc_bern = nn.Sequential(*layers_bern)
        #     nn.init.constant_(self.fc_bern[-1].bias, 0)
        #     # 主干网络输入维度增加bern_dim
        #     backbone_input_dim = state_dim + bern_dim
        # else:
        #     backbone_input_dim = state_dim
        backbone_input_dim = state_dim

        # 共享主干网络
        layers = []
        prev_size = backbone_input_dim
        for layer_size in hidden_dims:
            layers.append(nn.Linear(prev_size, layer_size))
            layers.append(nn.ReLU())
            prev_size = layer_size
        self.net = nn.Sequential(*layers)

        # 1. 连续动作头 (Continuous)
        # 参数: log_std (控制高斯分布宽度)
        if 'cont' in self.action_dims and self.action_dims['cont'] > 0:
            cont_dim = self.action_dims['cont']
            self.log_std_cont = nn.Parameter(torch.log(torch.ones(cont_dim) * init_std))

            # # # 原·单层动作头
            # self.fc_mu = nn.Linear(prev_size, cont_dim)
            
            # 现·2层动作头
            layers = []
            # for _ in range(head_hidden_layer_num):
            layers.append(nn.Linear(prev_size, int(prev_size/2)))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(int(prev_size/2), cont_dim))
            self.fc_mu = nn.Sequential(*layers)
            

        # 2. 离散动作头 (Categorical)
        # 参数: log_temp_cat (控制 Softmax 温度)
        if 'cat' not in self.action_dims:
            self.action_dims['cat'] = []
        if 'cat' in self.action_dims and sum(self.action_dims['cat']) > 0:
            self.cat_dims = list(self.action_dims['cat'])  # list, e.g., [4, 10]
            # [去串扰设计] ver_map/hor_map 分别控制是否启用 13/11 多对一映射；
            # 否则直接按外部维度 self.cat_dims (如 [5, 7]) 构建离散头。
            self.cat_dims_internal = list(self.cat_dims)
            if self.ver_map:
                if len(self.cat_dims_internal) > 0 and self.cat_dims_internal[0] == 5:
                    self.cat_dims_internal[0] = 13
            if self.hor_map:
                if len(self.cat_dims_internal) > 1 and self.cat_dims_internal[1] == 7:
                    self.cat_dims_internal[1] = 11
            total_cat_dim = sum(self.cat_dims_internal)
            # 现·2层动作头
            layers = []
            # for _ in range(head_hidden_layer_num):
            layers.append(nn.Linear(prev_size, int(prev_size/2)))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(int(prev_size/2), total_cat_dim))
            self.fc_cat = nn.Sequential(*layers)
            
            # 为每一个独立的离散头 (Head) 创建一个温度参数
            # 比如有 [4, 10] 两个头，我们就需要 2 个温度参数
            # 初始化为 0 (即 temperature=1.0)，保持原网络特性，让网络自己学去增大熵
            # self.log_temp_cat = nn.Parameter(torch.zeros(len(self.cat_dims))) 

        # 3. 伯努利动作头 (Bernoulli)
        # 参数: log_temp_bern (控制 Sigmoid 陡峭度)
        if 'bern' in self.action_dims and self.action_dims['bern'] > 0:
            bern_dim = self.action_dims['bern']
            
            # # 仅在非Autoregressive模式下才创建共享特征层的bern头
            # if self.Autoregressive != 1:
            # # 原·单层输出
            # self.fc_bern = nn.Linear(prev_size, bern_dim)
            # 初始化 bias 为 -2，使初始开火概率较低（sigmoid(-2) ≈ 0.12）
            # # nn.init.constant_(self.fc_bern.bias, -2.0)

            # 现·2层输出
            layers = []
            # for _ in range(head_hidden_layer_num):
            layers.append(nn.Linear(prev_size, int(prev_size/2)))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(int(prev_size/2), bern_dim))
            self.fc_bern = nn.Sequential(*layers)
            # 初始化 bias 为 -2，使初始开火概率较低（sigmoid(-2) ≈ 0.12）
            """
            bern头的logits必须偏置初始化为负数，假设为x, 那么每一步的开火概率为p0=1/(1+e^x)
            t0=2s为一个决策周期，从105km（刚允许开火）到80km（威胁比较大）开火，双方以两马赫平均接近速率接近，历时36s，
            保险起见折个半，在18s内取憋着不开火的概率为p18=0.5，这样就是同时满足开火概率p18=(1-p0)^9=0.5,
            (1-1/(1+e^x))^9=0.5, 解出bern_logits=-2.5
            """
            nn.init.constant_(self.fc_bern[-1].bias, 0.0) # -2.5) # 2.0
            
            # 为每一个伯努利动作维度创建一个温度参数
            # 初始化为 0 (即 temperature=1.0)
            # self.log_temp_bern = nn.Parameter(torch.zeros(bern_dim))
    
    # [修改] 增加 action_masks 参数, [新增] 增加 temperature 参数
    # [新增] ver_mask/hor_mask: 机动mask开关（__init__ 时从 mask_config.json 读取一次，不重复读盘）
    def forward(self, x, min_std=1e-6, max_std=1.0, action_masks=None, temperature=1.0, mask_on=0):
        ver_mask = self.ver_mask
        hor_mask = self.hor_mask
        if isinstance(temperature, dict):
            temp_cat = temperature.get('cat', 1.0)
            temp_bern = temperature.get('bern', 1.0)
        else:
            temp_cat = temperature
            temp_bern = temperature

        # # --- 处理Autoregressive模式 ---
        # if self.Autoregressive == 1 and hasattr(self, 'fc_bern'):
        #     # 1. 先从原始state计算bern_logits（无mask）
        #     bern_logits_direct = self.fc_bern(x)
        #     # 2. 将bern输出拼接到state后面
        #     x_enhanced = torch.cat([x, bern_logits_direct], dim=-1)
        #     # 3. 使用增强后的输入通过主干网络
        #     shared_features = self.net(x_enhanced)
        # else:
        #     shared_features = self.net(x)
        #     bern_logits_direct = None
        shared_features = self.net(x)
        bern_logits_direct = None

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
            
            # 1. 切分 Logits (内部使用 13 维垂直头 + 11 维水平头切分)
            split_dims = getattr(self, 'cat_dims_internal', self.cat_dims)
            cat_logits_list = list(torch.split(cat_logits_all, split_dims, dim=-1))

            # [去串扰与机动 mask]：对 action_ver 处理 (5个动作)
            if len(cat_logits_list) > 0 and cat_logits_list[0].size(-1) == 13:
                xb_cat = x
                if xb_cat.dim() == 1:
                    xb_cat = xb_cat.unsqueeze(0)
                warning_flag_cat = xb_cat[:, 5] > 1e-6
                missile_in_mid_term_cat = xb_cat[:, 3] > 1e-6
                cond_no_warn_mid = (~warning_flag_cat) & missile_in_mid_term_cat
                cond_no_warn_no_mid = (~warning_flag_cat) & (~missile_in_mid_term_cat)

                ver_logits_all = cat_logits_list[0]  # (Batch, 13)
                v_def = ver_logits_all[:, 8:13]      # 防御：0,1,2,3,4

                if ver_mask:
                    v_off = ver_logits_all[:, 0:5]   # 进攻：0,1,2,3,4
                    v_dis = ver_logits_all[:, 5:8]   # 偏置：2,3,4

                    B = ver_logits_all.size(0)
                    v_off_5 = v_off
                    v_dis_5 = torch.full((B, 5), -1e8, dtype=ver_logits_all.dtype, device=ver_logits_all.device)
                    v_dis_5[:, 2] = v_dis[:, 0]
                    v_dis_5[:, 3] = v_dis[:, 1]
                    v_dis_5[:, 4] = v_dis[:, 2]

                    ver_logits_5 = v_def
                    ver_logits_5 = torch.where(cond_no_warn_mid.unsqueeze(1), v_dis_5, ver_logits_5)
                    ver_logits_5 = torch.where(cond_no_warn_no_mid.unsqueeze(1), v_off_5, ver_logits_5)
                    cat_logits_list[0] = ver_logits_5
                else:
                    cat_logits_list[0] = v_def

            # [去串扰与机动 mask]：对 action_hor 处理
            if len(cat_logits_list) > 1 and cat_logits_list[1].size(-1) == 11:
                xb_cat = x
                if xb_cat.dim() == 1:
                    xb_cat = xb_cat.unsqueeze(0)
                warning_flag_cat = xb_cat[:, 5] > 1e-6
                missile_in_mid_term_cat = xb_cat[:, 3] > 1e-6
                cond_no_warn_mid = (~warning_flag_cat) & missile_in_mid_term_cat
                cond_no_warn_no_mid = (~warning_flag_cat) & (~missile_in_mid_term_cat)

                hor_logits = cat_logits_list[1]  # (Batch, 11)

                if hor_mask:
                    # 11维内部头分别对应三个阶段的白名单，对外聚合为7维动作分布
                    # 0:无中导-追击 1:无中导-左3 2:无中导-置尾 3:无中导-右9 4:无中导-占中
                    # 5:有中导-左偏 6:有中导-置尾 7:有中导-右偏
                    # 8:告警-左3   9:告警-置尾  10:告警-右9
                    #                              0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
                    m_warn_allow   = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=torch.bool, device=x.device) # 告警: 8,9,10
                    m_mid_allow    = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0], dtype=torch.bool, device=x.device) # 无告警+有中导: 5,6,7
                    m_no_mid_allow = torch.tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=torch.bool, device=x.device) # 无告警+无中导: 0,1,2,3,4

                    legal_mask = (
                        warning_flag_cat.unsqueeze(1) & m_warn_allow |
                        cond_no_warn_mid.unsqueeze(1) & m_mid_allow |
                        cond_no_warn_no_mid.unsqueeze(1) & m_no_mid_allow
                    )
                    hor_logits = hor_logits.masked_fill(~legal_mask, -1e8)

                # 把 11 维内部头按阶段语义映射到 7 维外部动作分布
                # 同一外部动作可能由多个内部头承担，但同一时刻只有一个阶段激活，
                # 因此用 logsumexp 聚合（可导，对单激活头退化为恒等）
                # 外部动作: 0追击, 1左偏, 2左3, 3置尾, 4右9, 5右偏, 6占中
                action_head_groups = [
                    [0],        # 0 追击
                    [5],        # 1 左偏
                    [1, 8],     # 2 左3
                    [2, 6, 9],  # 3 置尾
                    [3, 10],    # 4 右9
                    [7],        # 5 右偏
                    [4],        # 6 占中
                ]
                # (B, 11) -> (B, 7)
                cat_logits_list[1] = torch.stack([
                    torch.logsumexp(hor_logits[:, heads], dim=1)
                    for heads in action_head_groups
                ], dim=1)

            # 2. 获取温度 (temperature = exp(log_temp))
            # temp_cat 形状: (num_heads, )
            # temperatures = 1.0  # [修改] 使用传入的 temperature
            # >2 强随机，<0.1 强确定性
            
            # 3. 应用温度缩放 (Logits / temperature) 并 Softmax
            # 较高的 temperature -> Logits 数值变小 -> Softmax 后分布趋向均匀 (熵增大)
            # 较低的 temperature -> Logits 数值差距拉大 -> Softmax 后分布趋向 One-hot (熵减小)
            final_probs_list = []
            for i, logits in enumerate(cat_logits_list):
                # 对应的温度: temperatures[i]
                # 使用 temp_cat 进行缩放, 防止除0
                scaled_logits = logits / (temp_cat + 1e-8)
                final_probs_list.append(F.softmax(scaled_logits, dim=-1))
            
            outputs['cat'] = final_probs_list

        # --- Bernoulli (核心修改区域) ---
        if 'bern' in self.action_dims and self.action_dims['bern'] > 0:
            # # 根据是否使用Autoregressive选择bern_logits来源
            # if self.Autoregressive == 1:
            #     bern_logits = bern_logits_direct
            # else:
            #     bern_logits = self.fc_bern(shared_features)
            bern_logits = self.fc_bern(shared_features)

            # Compute can_fire mask from flattened observation x (always applied)
            xb = x
            if xb.dim() == 1:
                xb = xb.unsqueeze(0)

            # Indices (0-based): cos_ata_hor -> x[:,6], ata -> x[:,10], locked -> x[:,2], ammo -> x[:,20], distance_scaled -> x[:,9]
            cos_ata_hor = torch.clamp(xb[:, 6], -0.999999, 0.999999)
            delta_theta = xb[:, 8]
            ata = xb[:, 10]
            # alt = xb[:, 15] * 5e3
            sin_theta = xb[:, 17]
            # locked = xb[:, 2]
            ammo = xb[:, 20]
            dist = xb[:, 9] * 10e3
            AA_hor = xb[:, 12]
            t_since_launch = xb[:, 21] * 120
            missile_in_mid_term = xb[:, 3] > 1e-6

            ammo_cond = (ammo > 0.0)
            # time_const_cond = t_since_launch >= torch.max(dist/(3*340)/2, torch.as_tensor(10.0, device=dist.device, dtype=dist.dtype))
            time_const_cond = t_since_launch >= torch.clamp_min(dist/(3*340)/2, 10.0)
            # 最小开火冷却时间10s，随开火距离增加 # 10  # 冷却时间10s，全程开启
            
            ata_cond = ata < math.pi / 2
            # 全程只施加弹药与冷却mask；角度/距离mask仅在部署阶段由get_action的check_obs控制
            can_fire = ammo_cond & time_const_cond & ata_cond

            # 禁止大离轴发射导弹
            delta_psi_cond = cos_ata_hor >= math.cos(np.radians(45)) # 25
            can_fire = can_fire & delta_psi_cond
            # 禁止俯冲发射导弹
            theta = torch.arcsin(sin_theta)
            elevation = theta + delta_theta
            theta_cond = theta >= elevation - np.radians(15)
            can_fire = can_fire & theta_cond

            # # 禁止中制导下开火
            # can_fire = can_fire & ~missile_in_mid_term
            
            # # 禁止尾追长距离开火
            # far_chase = (AA_hor < math.pi/2) & (dist > 25e3)
            # can_fire = can_fire & ~ far_chase
            
            # if not can_fire:
            #     print("禁止开火")
            # else:
            #     print("  可以开炮  ")
            

            # build mask for bern dims and apply to first bern dimension only
            bern_dim = self.action_dims.get('bern', 0)
            batch_size = shared_features.size(0)
            mask = torch.ones((batch_size, bern_dim), dtype=torch.bool, device=shared_features.device)
            mask[:, 0] = can_fire.to(dtype=torch.bool)

            # If external action_masks provided (e.g., death masks), combine them (AND)
            if action_masks is not None and 'bern' in action_masks:
                ext_mask = action_masks['bern']
                if isinstance(ext_mask, torch.Tensor):
                    if ext_mask.dim() == 1:
                        ext_mask = ext_mask.unsqueeze(1)
                    ext_bool = (ext_mask != 0).to(dtype=torch.bool, device=shared_features.device)
                else:
                    ext_mask = torch.tensor(np.array(ext_mask), device=shared_features.device)
                    if ext_mask.dim() == 1:
                        ext_mask = ext_mask.unsqueeze(1)
                    ext_bool = (ext_mask != 0).to(dtype=torch.bool, device=shared_features.device)

                if ext_bool.size(1) == 1 and bern_dim > 1:
                    ext_bool = ext_bool.expand(-1, bern_dim)

                mask = mask & ext_bool

            # Apply mask: False -> set logits very small
            bern_logits = bern_logits.masked_fill(mask == 0, -1e8)

            # [修改] 使用我们提取的 temp_bern
            # temperatures = 1.0 
            scaled_bern_logits = bern_logits / (temp_bern + 1e-8)
            outputs['bern'] = scaled_bern_logits
            
            # [新增] 返回 fire_mask，用于在 update 中过滤 Bernoulli 熵计算
            outputs['fire_mask'] = mask.float()  # shape: (batch, bern_dim)
            
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

    # [修改] 增加 check_obs 参数，默认为 None， [新增] 增加 temperature 参数
    def get_action(self, state, h=None, explore=True, max_std=None, check_obs=None, bern_threshold=0.5, temperature=1.0, mask_on=0): # 1
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
        2、evaluate_action（训练/计算 log_prob）默认未把 action_masks 传给 net ,
            若希望训练时也应用 mask，需要在 evaluate_action 调用 net 时传入 action_masks。
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
                'bern': explore.get('bern', True),
            }
        else:
            # 对于其他意外的输入类型，默认全部探索
            explore_opts = {'cont': True, 'cat': True, 'bern': True}

        # =====================================================================
        # check_obs 非空时，从 state 直接提取特征计算角度/距离规则 mask（仅部署阶段）
        # [漏网点1 判定逻辑]: 当外部传入 check_obs(如测试中的 deterministic=True 分支)时，
        # 会执行以下更严苛的规则拦截。若不满足，_deploy_can_fire 将置为 False:
        #   1. wait_til_last_missile_ends: 上一枚导弹中制导期间强制禁止开火 (forward 中此限制已被注释，但此处仍有效)
        #   2. locked_cond: 目标必须已被雷达锁定 (locked > 0)
        #   3. dist_cond: 目标距离必须小于 90km
        #   4. ata_hor: 水平离轴角必须 <= 30° (比 forward 中的 45° 更严格)
        # 外部若传入 check_obs=None (如测试中的 deterministic=False 分支)，则直接跳过此段，不施加这些硬限制。
        # =====================================================================
        _deploy_can_fire = True  # 默认不限制
        if (check_obs is not None) and isinstance(check_obs, dict):
            with torch.no_grad():
                xb = state
                # 照搬 forward 中的索引提取（state 此时已是 (batch, D) tensor）
                cos_ata_hor = torch.clamp(xb[:, 6], -0.999999, 0.999999)
                delta_theta = xb[:, 8]
                ata         = xb[:, 10]
                alt         = xb[:, 15] * 5e3
                sin_theta   = xb[:, 17]
                locked      = xb[:, 2]
                ammo        = xb[:, 20]
                dist        = xb[:, 9] * 10e3
                AA_hor      = xb[:, 12]
                t_since_launch = xb[:, 21] * 120
                missile_in_mid_term = xb[:, 3] > 1e-6 # 上一枚导弹还在中制导阶段，转为 bool

                pi = math.pi
                ata_hor      = torch.acos(cos_ata_hor)
                ata_cond     = (ata <= 60.0 * pi / 180.0) & (ata_hor <= 30.0 * pi / 180.0)
                locked_cond  = (locked > 0)
                dist_cond    = (dist < 90e3) # 105e3)
                delta_theta_cond = 1 # (delta_theta < pi * 30.0 / 180.0) # 内置了更严格的
                wait_til_last_missile_ends = not missile_in_mid_term
                can_fire_full = (ata_cond & locked_cond & dist_cond
                                & delta_theta_cond
                                & wait_til_last_missile_ends)
                _deploy_can_fire = can_fire_full.all().item()  # 转成 bool

        # 调用网络（net 内部只施加弹药+冷却 mask）
        actor_outputs = self.net(state, max_std=max_std, temperature=temperature, mask_on=mask_on)
        
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
            # [漏网点1 生效处]: 部署阶段（check_obs非空）施加角度/距离/中制导规则 mask
            # 若 _deploy_can_fire 为 False，直接将 bern_logits 压为 -1e8，使开火概率归零
            if _deploy_can_fire is False:
                bern_logits = torch.full_like(bern_logits, -1e8)
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

    def evaluate_actions(self, states, actions_raw, h=None, max_std=None, mask_on=0):
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
        actor_outputs = self.net(states, max_std=max_std, mask_on=mask_on)
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
        if 'bern' in self.action_dims and self.action_dims['bern'] > 0 and 'bern' in actions_raw and actions_raw['bern'] is not None and actions_raw['bern'].numel() > 0:
            bern_logits = actor_outputs['bern']
            # Replace -inf logits (from masking) with a large negative finite value for numerical stability during training
            bern_logits = bern_logits.clamp(min=-1e8)
            dist = Bernoulli(logits=bern_logits)
            bern_action = actions_raw['bern']
            log_probs += dist.log_prob(bern_action).sum(-1, keepdim=True)
            
            #  单独记录 bern entropy
            e_bern = dist.entropy().sum(-1, keepdim=True)
            entropy += e_bern
            entropy_details['bern'] = e_bern # [修改] 保持 Tensor 用于 Loss 计算

        # [修改] 返回 actor_outputs 以便外部访问 logits
        return log_probs, entropy, entropy_details, actor_outputs, None
    
    def compute_il_loss(self, states, expert_actions, label_smoothing=0.3, action_heads_mask=None, no_bern=None, no_cat=None, mask_on=0, good_samples=1, pre_training=1):
        """
        计算模仿学习 Loss (MARWIL / BC)。
        
        Args:
            states: (Batch, State_Dim)
            expert_actions: 字典, 包含 {'cont': u, 'cat': index, 'bern': float}
                            注意：对于连续动作，这里通常假设传入的是 pre-tanh 的 u，
                            或者你需要在外部处理好。
            label_smoothing: 标签平滑系数
            action_heads_mask: dict, 例如 {'cont': True, 'cat': True, 'bern': True}
                               指定哪些动作头参与模仿学习 Loss 计算。
                               为兼容旧代码，仍保留 no_bern/no_cat，但它们会被映射为 mask。
            
        Returns:
            total_loss_per_sample: (Batch, ) 每个样本的 Loss 总和，未加权
        """
        '''
        会增加复杂度的可选改进：模仿学习的时候alpha 传入向量，从而区分密集和稀疏动作的学习强度（密集应该高一些）
        '''
        # 解析动作头mask；兼容旧版 no_bern/no_cat
        if action_heads_mask is None:
            action_heads_mask = {'cont': True, 'cat': True, 'bern': True}
            if no_bern is not None:
                action_heads_mask['bern'] = not no_bern
            if no_cat is not None:
                action_heads_mask['cat'] = not no_cat
        
        actor_outputs = self.net(states, mask_on=mask_on) # 获取 raw output (mu/std, logits)
        
        # 初始化一个全 0 的 loss tensor，形状 (Batch, )
        total_loss_per_sample = torch.zeros(states.size(0), device=self.device)

        # --- 1. 连续动作 (Continuous) ---
        # 依据提供的 PPOContinuous 代码，MARWIL 使用 log_prob(u)
        if action_heads_mask.get('cont', False) and 'cont' in self.action_dims and self.action_dims['cont'] > 0:
            mu, std = actor_outputs['cont']
            dist = SquashedNormal(mu, std)
            u_expert = expert_actions['cont'] # 假设传入的是 pre-tanh value
            
            # 计算 log_prob，维度求和保持 (Batch, 1) -> squeeze 为 (Batch, )
            # Loss = - log_prob
            if good_samples: # 如果传入的是好样本，减小距离
                cont_loss = -dist.log_prob(0, u_expert).sum(dim=-1)
            else: # 如果传入的是差样本，要么增大距离，要么别动
                pass
                # cont_loss = +dist.log_prob(0, u_expert).sum(dim=-1)
            total_loss_per_sample += cont_loss

        # --- 2. 离散/多离散动作 (Categorical) ---
        # 依据提供的 Multi-Discrete 代码，使用 CrossEntropy
        if action_heads_mask.get('cat', False) and 'cat' in self.action_dims and sum(self.action_dims['cat']) > 0:
            cat_logits_list = actor_outputs['cat'] # 注意：这里 net forward 返回的是 softmax 后的 probs 还是 logits? 
            # 修正：你的 PolicyNetHybrid forward 返回的是 [F.softmax(logits)...]
            # 为了数值稳定性，建议 PolicyNetHybrid 改为返回 logits，或者在这里取 log
            
            # 假设 expert_actions['cat'] 是 (Batch, Num_Heads)
            expert_cat = expert_actions['cat'].long()
            
            for i, probs in enumerate(cat_logits_list):
                # probs: (Batch, N_Class)
                expert_idx = expert_cat[:, i] # (Batch, )
                '''
                    log_probs.gather()
                    从所有动作的概率分布 log_probs 中，精准地抽取出“实际执行了的那个动作” expert_idx 对应的概率值。
                    - 1 (第一个参数)：表示在第 1 维（列维度）进行选取。
                    - expert_idx.unsqueeze(1)：将原来形状为(Batch,)的索引变成(Batch, 1)。
                     这是因为 gather 要求索引的维度必须和原张量一致。
                    - .squeeze(1)：取完值后，形状还是(Batch, 1)用 squeeze 把那个多余的维度删掉，
                    变成平铺的 (Batch,)，方便后续算 Loss。
                '''
                if pre_training:
                    "预训练使用钉子分布和正向信号"
                    log_probs = torch.log(probs + 1e-10)
                    # Label Smoothing 逻辑
                    n_classes = probs.size(1)
                    one_hot = torch.zeros_like(probs).scatter_(1, expert_idx.unsqueeze(1), 1.0)
                    smooth_target = one_hot * (1.0 - label_smoothing) + (1.0 - one_hot) * (label_smoothing / (n_classes - 1 + 1e-8))
                    # CrossEntropy: - sum(target * log_p)
                    ce_loss = -torch.sum(smooth_target * log_probs, dim=1)
                else:
                    "混合在线训练不再使用钉子分布，仅对采样动作对应的动作头操作"
                    # 设定平滑目标 t
                    # label_smoothing = 0.01 -> t = 0.99 (正向监督，拉升该动作概率)
                    # label_smoothing = 0.99 -> t = 0.01 (负向监督，压低该动作概率)
                    # 从经过 Softmax 的 probs 中，精准提取实际执行动作的概率 x
                    # 注意：这里是对 probs 进行 gather，而不是 log_probs
                    act_probs = probs.gather(1, expert_idx.unsqueeze(1)).squeeze(1)
                    # 数值稳定性保护，严防 log(0) 导致 NaN
                    act_probs = torch.clamp(act_probs, min=1e-8, max=1.0 - 1e-8)
                    # 完美的统一损失函数形式： g(x) = -t*ln(x) - (1-t)*ln(1-x)
                    ce_loss = -(1.0 - label_smoothing) * torch.log(act_probs) - (label_smoothing) * torch.log(1.0 - act_probs)
                    
                total_loss_per_sample += ce_loss

        # --- 3. 伯努利动作 (Bernoulli) ---
        # -- Focal Loss --
        if action_heads_mask.get('bern', False) and 'bern' in self.action_dims and self.action_dims['bern'] > 0 and 'bern' in expert_actions:
            bern_logits = actor_outputs['bern']
            # Clamp masked -inf logits to a large negative finite value for stable sigmoid/log calculations
            bern_logits = bern_logits.clamp(min=-1e8)
            probs = torch.sigmoid(bern_logits)
            probs = torch.clamp(probs, 1e-10, 1.0 - 1e-10)
            target = expert_actions['bern'] # (Batch, 1)
            
            "开火头适度动作平滑"

            # 开火头保持硬标签
            max_target = sigmoid(3.0)
            min_target = sigmoid(-3.0)

            # 对比实验，临时使用软标签给开火头
            # max_target = min(1.0-label_smoothing, sigmoid(3.0))
            # min_target = max(label_smoothing, sigmoid(-3.0))
            
            target = torch.clamp(target, min_target, max_target)

            # 交叉熵公式
            # 正向模仿学习，增加样本中的动作概率
            if good_samples:
                loss_pos = - torch.log(probs) * target
                loss_neg = - torch.log(1.0 - probs) * (1.0 - target)
                bce_loss = loss_pos + loss_neg
            # 负向模仿学习 / 互补标签学习，减少样本中的动作概率
            else:
                loss_pos = - torch.log(probs) * (1.0 - target)
                loss_neg = - torch.log(1.0 - probs) * target
                bce_loss = loss_pos + loss_neg
            
            total_loss_per_sample += bce_loss.sum(dim=-1)
        
        return total_loss_per_sample

    @torch.no_grad()
    def compute_marwil_monitor(self, states, expert_actions, mask_on=0, advantages=None):
        """
        [监控用] 在固定 batch 上计算每个动作头独立的 NLL 与策略熵。
        - 全程 no_grad，不影响主网络更新；
        - NLL 不带 label smoothing / focal / weights，是纯粹的负对数似然；
        - 熵是策略分布本身的熵（不依赖 expert action）。
        - advantages: 可选，(N,) 或 (N,1) 的 advantage tensor，用于计算 adv_positive_frac。

        Returns:
            dict: 包含 nll_cont/nll_cat/nll_bern/entropy_cont/entropy_cat/entropy_bern/adv_positive_frac,
                  缺失的动作头或未传入 advantages 时对应键返回 None。
        """
        actor_outputs = self.net(states, mask_on=mask_on)
        metrics = {
            'nll_cont': None, 'nll_cat': None, 'nll_bern': None,
            'entropy_cont': None, 'entropy_cat': None, 'entropy_bern': None,
            'accuracy_cont': None, 'accuracy_cat': None, 'accuracy_bern': None,
            'adv_positive_frac': None,
        }

        # --- Cont ---
        if 'cont' in self.action_dims and self.action_dims['cont'] > 0 and actor_outputs.get('cont') is not None:
            mu, std = actor_outputs['cont']
            dist = SquashedNormal(mu, std)
            u_expert = expert_actions['cont']
            # 纯 NLL：sum over dim, mean over batch
            nll_cont = (-dist.log_prob(0, u_expert).sum(dim=-1)).mean()
            ent_cont_raw = dist.entropy()
            ent_cont = ent_cont_raw.sum(dim=-1).mean() if ent_cont_raw.dim() > 1 else ent_cont_raw.mean()
            # Accuracy: mu的误差模均值
            accuracy_cont = torch.norm(mu - u_expert, dim=-1).mean()
            metrics['nll_cont'] = nll_cont.item()
            metrics['entropy_cont'] = ent_cont.item()
            metrics['accuracy_cont'] = accuracy_cont.item()

        # --- Cat ---
        if 'cat' in self.action_dims and sum(self.action_dims['cat']) > 0 and actor_outputs.get('cat') is not None:
            cat_probs_list = actor_outputs['cat']
            expert_cat = expert_actions['cat'].long()
            nll_cat_sum = torch.zeros(states.size(0), device=self.device)
            ent_cat_sum = torch.zeros(states.size(0), device=self.device)
            correct_cat_sum = torch.zeros(states.size(0), device=self.device)
            for i, probs in enumerate(cat_probs_list):
                expert_idx = expert_cat[:, i]
                log_probs = torch.log(probs + 1e-10)
                nll_cat_sum += -log_probs.gather(1, expert_idx.unsqueeze(1)).squeeze(1)
                dist = Categorical(probs=probs)
                ent_cat_sum += dist.entropy()
                # Accuracy: argmax和示范的action完全一样
                pred_idx = torch.argmax(probs, dim=1)
                correct_cat_sum += (pred_idx == expert_idx).float()
            metrics['nll_cat'] = nll_cat_sum.mean().item()
            metrics['entropy_cat'] = ent_cat_sum.mean().item()
            # 计算所有cat动作的平均准确度
            total_cat_dims = len(cat_probs_list)
            metrics['accuracy_cat'] = (correct_cat_sum / total_cat_dims).mean().item()

        # --- Bern ---
        if 'bern' in self.action_dims and self.action_dims['bern'] > 0 and actor_outputs.get('bern') is not None and 'bern' in expert_actions:
            bern_logits = actor_outputs['bern'].clamp(min=-1e8)
            dist = Bernoulli(logits=bern_logits)
            target = expert_actions['bern']
            nll_bern = (-dist.log_prob(target).sum(dim=-1)).mean()
            ent_bern = dist.entropy().sum(dim=-1).mean()
            # Accuracy: argmax和示范的action完全一样
            pred_bern = (bern_logits > 0).float()
            accuracy_bern = (pred_bern == target).float().mean()
            metrics['nll_bern'] = nll_bern.item()
            metrics['entropy_bern'] = ent_bern.item()
            metrics['accuracy_bern'] = accuracy_bern.item()

        # --- adv_positive_frac ---
        if advantages is not None:
            adv = advantages.detach().view(-1)
            metrics['adv_positive_frac'] = (adv > 0).float().mean().item()

        return metrics
# =============================================================================
# 3. PPO 算法类 (精简版)
# =============================================================================

class PPOHybrid:
    def __init__(self, actor, critic, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device, 
                 k_entropy={'cont':0.01, 'cat':0.005, 'bern':0.05}, critic_max_grad=2, actor_max_grad=2, max_std=0.7, # ):
                 rnd_state_dim=None, rnd_lr=3e-4, rnd_output_dim=128, rnd_hidden_dim=256):
        
        self.actor = actor # 这是一个 HybridActorWrapper 实例
        self.critic = critic
        
        # critic优化器不动了，保持adam
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        "Adam优化器（有动量）"
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        "RMSprop优化器（有二阶矩但是没有一阶动量），初步测试效果更差"
        # self.actor_optimizer = torch.optim.RMSprop(self.actor.parameters(), lr=actor_lr, alpha=0.99, eps=1e-5)
        "AdamW优化器（weight_decay）"
        # self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=actor_lr, weight_decay=1e-4, eps=1e-5)
        "RAdam优化器(为冷启动优化，但非平稳环境不建议用)"
        # self.actor_optimizer = torch.optim.RAdam(self.actor.parameters(), lr=actor_lr, eps=1e-5)
        
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
        
        # [新增] SAC 风格的可学习 Cat 熵系数 (初始值对应原配置)
        init_k_cat = k_entropy.get('cat', k_entropy)
        self.log_k_cat = torch.nn.Parameter(torch.log(torch.tensor(init_k_cat, device=device)))
        # [新增] 为自适应熵系数配备独立的优化器 (学习率通常与 Actor 保持一致或略大)
        self.k_cat_optim = torch.optim.Adam([self.log_k_cat], lr=actor_lr)
        
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
        
        # [新增] 监控指标
        self.td_error_var = 0     # TD error 的分布方差
        self.grad_norm_ratio = 0  # actor 梯度与 critic 梯度的范数比

        # RND 网络（可选）
        if rnd_state_dim is not None:
            self.rnd_target = RNDTargetNet(rnd_state_dim, output_dim=rnd_output_dim, hidden_dim=rnd_hidden_dim).to(device)
            self.rnd_prediction = RNDPredictionNet(rnd_state_dim, output_dim=rnd_output_dim, hidden_dim=rnd_hidden_dim).to(device)
            self.rnd_optimizer = torch.optim.Adam(self.rnd_prediction.parameters(), lr=rnd_lr)
        else:
            self.rnd_target = None
            self.rnd_prediction = None
            self.rnd_optimizer = None

    def RND_calc(self, transition_dict, beta):
        """
        计算 RND 内在奖励并叠加到外在奖励上。

        步骤：
          1. 用当前 batch 更新状态归一化统计量（Welford 在线算法）
          2. 优化预测网络（蒸馏损失 = MSE(pred, target)）
          3. 计算内在奖励 i = ||pred - target||^2 per sample
          4. 归一化内在奖励（减均值除标准差）
          5. reward_aug = reward + beta * i_normalized

        Args:
            transition_dict: 包含 'states', 'rewards' 等键的字典
            beta: 内在奖励缩放倍率

        Returns:
            new_dict: 奖励已被修改的新字典（浅拷贝，rewards 为新数组）
        """
        assert self.rnd_target is not None and self.rnd_prediction is not None and self.rnd_optimizer is not None, \
            "RND 未初始化，请在 PPOHybrid.__init__ 中传入 rnd_state_dim"

        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)

        # 1. 更新状态归一化统计量
        with torch.no_grad():
            self.rnd_target.update_obs_stats(states)
            states_norm = self.rnd_target.normalize(states)

        # 2. 优化预测网络
        pred = self.rnd_prediction(states_norm)
        with torch.no_grad():
            target = self.rnd_target.net(states_norm)
        distill_loss = F.mse_loss(pred, target)
        self.rnd_optimizer.zero_grad()
        distill_loss.backward()
        self.rnd_optimizer.step()

        # 3. 计算内在奖励（每个样本的特征 MSE）
        with torch.no_grad():
            pred_det = self.rnd_prediction(states_norm)
            target_det = self.rnd_target.net(states_norm)
            intrinsic = ((pred_det - target_det) ** 2).mean(dim=-1, keepdim=True)  # (N, 1)

            # 4. 归一化内在奖励
            i_mean = intrinsic.mean()
            i_std = intrinsic.std() + 1e-8
            intrinsic_norm = (intrinsic - i_mean) / i_std

        # 5. 叠加到外在奖励
        rewards = np.array(transition_dict['rewards'], dtype=np.float32).reshape(-1, 1)
        intrinsic_np = intrinsic_norm.cpu().numpy()
        rewards_aug = rewards + beta * intrinsic_np

        new_dict = dict(transition_dict)
        new_dict['rewards'] = rewards_aug
        mse_raw = i_mean.item()  # 归一化前的原始 MSE 均值，用于监控
        return new_dict, mse_raw

    def RDistill(self, transition_dict, beta, k=1, teacher_actor=None, no_bern=1, learn_type="dual_prob"):
        """
        计算 RDistill (Reward Shaping via Imitation Learning) 内在奖励并叠加到外在奖励上。

        learn_type 决定“距离度量”的构造方式：
          - "dual_prob": 使用 teacher 与 student 两个分布的 KL 散度作为距离。
            这种方式将奖励与 student 分布耦合，可能造成“奖励与实际动作分离”，仅仅是修理动作概率的形状。
          - "single_prob": 只使用 teacher 对经验池里【实际执行动作】的负对数似然(NLL)作为距离，
            奖励与经验池中真实动作直接对应，避免奖励-动作分离。

        通用步骤：
          1. 对 cat 和 bern 部分构造 per-sample 距离 D（KL 或 NLL）
          2. 归一化 D 序列（除以标准差，防除0错误）
          3. 计算内在奖励 = beta * (exp(-k * D_normalized) - 0.99)
          4. 叠加到外在奖励
        cat 与 bern 两部分的区分处理由 no_bern 控制（no_bern=1 时跳过 bern）。

        Args:
            transition_dict: 包含 'states', 'rewards', 'actions' 等键的字典
            beta: 内在奖励缩放倍率
            k: 距离的缩放系数，默认为 1
            teacher_actor: 已加载参数的 PolicyNetHybrid 实例（教师策略）
            no_bern: 1 时不计算 bern 部分
            learn_type: "dual_prob"(师生KL) 或 "single_prob"(teacher对真实动作的NLL)

        Returns:
            new_dict: 奖励已被修改的新字典（浅拷贝，rewards 为新数组）
            dist_mean: 归一化前的距离均值，用于监控
        """
        assert learn_type in ("dual_prob", "single_prob"), f"未知 learn_type: {learn_type}"
        assert teacher_actor is not None, "teacher_actor 不能为空，请传入已加载参数的 PolicyNetHybrid 实例或 RuleTeacherWrapper"

        # actor 使用局部观测 obs（若存在），与 update() 中 actor_inputs 的选择保持一致；
        # 这样才能正确重构规则教师所需的 check_obs，并保证师生 KL 使用同一输入
        if 'obs' in transition_dict and len(transition_dict['obs']) > 0:
            states = torch.tensor(np.array(transition_dict['obs']), dtype=torch.float).to(self.device)
        else:
            states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)

        with torch.no_grad():
            # 1. 获取 teacher 策略分布
            #    类型区分：规则教师(RuleTeacherWrapper)通过 predict_distributions 直接给出软 one-hot 分布；
            #    网络教师(HybridActorWrapper)通过 net(states) 前向得到分布。
            if getattr(teacher_actor, 'is_rule_teacher', False):
                teacher_outputs = teacher_actor.predict_distributions(states)
            else:
                teacher_outputs = teacher_actor.net(states)

            # 2. 计算 per-sample 距离度量
            dist_per_sample = torch.zeros(states.size(0), 1).to(self.device)

            if learn_type == "dual_prob":
                # ===== 师生 KL 散度模式 =====
                student_outputs = self.actor.net(states)

                # --- Categorical 部分 ---
                if teacher_outputs['cat'] is not None and student_outputs['cat'] is not None:
                    for teacher_probs, student_probs in zip(teacher_outputs['cat'], student_outputs['cat']):
                        # KL(P||Q) = sum(P * (log(P) - log(Q)))，加小常数防 log(0)
                        teacher_log_probs = torch.log(teacher_probs + 1e-8)
                        student_log_probs = torch.log(student_probs + 1e-8)
                        kl_cat = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1, keepdim=True)
                        dist_per_sample += kl_cat

                # --- Bernoulli 部分 ---（no_bern=1 时跳过）
                if (not no_bern) and teacher_outputs['bern'] is not None and student_outputs['bern'] is not None:
                    teacher_probs = torch.sigmoid(teacher_outputs['bern'])
                    student_probs = torch.sigmoid(student_outputs['bern'])
                    teacher_log_probs = torch.log(teacher_probs + 1e-8)
                    student_log_probs = torch.log(student_probs + 1e-8)
                    teacher_log_probs_inv = torch.log(1 - teacher_probs + 1e-8)
                    student_log_probs_inv = torch.log(1 - student_probs + 1e-8)
                    kl_bern = (teacher_probs * (teacher_log_probs - student_log_probs) +
                               (1 - teacher_probs) * (teacher_log_probs_inv - student_log_probs_inv)).sum(dim=-1, keepdim=True)
                    dist_per_sample += kl_bern

            elif learn_type == "single_prob":
                # ===== single_prob：teacher 对经验池实际动作的负对数似然(NLL) =====
                # 只用 teacher 执行“经验池里真实动作”的概率产生奖励附加项，
                # 奖励与实际动作直接对应，避免奖励-动作分离。
                actions_from_buffer = transition_dict['actions']
                if isinstance(actions_from_buffer, dict):
                    actions_dict = actions_from_buffer
                else:
                    # 兼容旧格式：list of dicts
                    actions_dict = {}
                    for key in actions_from_buffer[0].keys():
                        actions_dict[key] = np.array([d[key] for d in actions_from_buffer])

                # --- Categorical 部分 ---
                if teacher_outputs['cat'] is not None and 'cat' in actions_dict:
                    cat_actions = torch.as_tensor(np.array(actions_dict['cat']), dtype=torch.long, device=self.device)
                    if cat_actions.dim() == 1:
                        cat_actions = cat_actions.unsqueeze(-1)
                    for h, teacher_probs in enumerate(teacher_outputs['cat']):
                        act_h = cat_actions[:, h].unsqueeze(-1)
                        p_taken = teacher_probs.gather(1, act_h)  # (B,1) teacher 对该动作的概率
                        dist_per_sample += -torch.log(p_taken + 1e-8)  # NLL

                # --- Bernoulli 部分 ---（no_bern=1 时跳过）
                if (not no_bern) and teacher_outputs['bern'] is not None and 'bern' in actions_dict:
                    bern_actions = torch.as_tensor(np.array(actions_dict['bern']), dtype=torch.float, device=self.device)
                    if bern_actions.dim() == 1:
                        bern_actions = bern_actions.unsqueeze(-1)
                    teacher_probs = torch.sigmoid(teacher_outputs['bern'])
                    # NLL = -[a*log(p) + (1-a)*log(1-p)]
                    nll_bern = -(bern_actions * torch.log(teacher_probs + 1e-8) +
                                 (1 - bern_actions) * torch.log(1 - teacher_probs + 1e-8)).sum(dim=-1, keepdim=True)
                    dist_per_sample += nll_bern
            else:
                print("错误的奖励修改类型")
                return dict(transition_dict), 0

            # 3. 归一化距离（除以标准差）
            dist_mean = dist_per_sample.mean()
            dist_std = dist_per_sample.std() + 1e-8
            dist_normalized = torch.clamp(dist_per_sample / dist_std, 0.0, 1.0)

            # 4. 计算内在奖励 = beta * (exp(-k * D_normalized) - 0.99)
            intrinsic = beta * (torch.exp(-k * dist_normalized) - 0.99)

            # 5. 叠加到外在奖励
            rewards = np.array(transition_dict['rewards'], dtype=np.float32).reshape(-1, 1)
            intrinsic_np = intrinsic.cpu().numpy()
            rewards_aug = rewards + intrinsic_np

        new_dict = dict(transition_dict)
        new_dict['rewards'] = rewards_aug
        dist_mean_raw = dist_mean.item()  # 归一化前的距离均值，用于监控
        return new_dict, dist_mean_raw

    def ADistill(self, transition_dict, advantage, alpha_distill, teacher_actor=None, AFiltered=0, conf_thres=0.7, bern_included=0):
        """
        ADistill (Advantage Distillation):
        [当前版本] 不再引导 cat 机动动作，仅对 bern 开火动作做规则共识引导。
        只在学生**实际执行了开火**（bern=1）的样本上调节 advantage；不开火的样本完全不管。
        若所有 teacher_actor（规则教师列表）在 bern 头上达成共识：
          - 全认为该开火 → 给 advantage 加正偏移 (+ adv_std * alpha_distill)
          - 全认为不该开火 → 给 advantage 加负偏移 (- adv_std * alpha_distill)
          - 意见不一致 → 不修改
        teacher_actor 支持单教师或教师列表；列表时取所有教师 bern 决策的一致交集。
        AFiltered=1 时双向过滤：正偏移只在 advantage 高于均值时生效，负偏移只在 advantage 低于均值时生效。
        conf_thres / bern_included 原用于 cat/bern 联合引导，现已不再生效，保留参数仅作接口兼容。
        """
        if alpha_distill <= 0 or teacher_actor is None:
            return advantage

        # 输入状态（优先使用 obs，与 update 中 actor_inputs 保持一致）
        if 'obs' in transition_dict and len(transition_dict['obs']) > 0:
            states = torch.tensor(np.array(transition_dict['obs']), dtype=torch.float).to(self.device)
        else:
            states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)

        # active mask
        if 'active_masks' in transition_dict:
            active_masks = torch.tensor(np.array(transition_dict['active_masks']), dtype=torch.float).to(self.device).view(-1, 1)
        else:
            active_masks = torch.ones_like(advantage)

        # 实际 bern 动作
        actions = transition_dict['actions']
        bern_actions = None
        if isinstance(actions, dict):
            if 'bern' in actions and len(actions['bern']) > 0:
                bern_actions = torch.as_tensor(np.array(actions['bern']), dtype=torch.float, device=self.device)
        else:
            if len(actions) > 0 and isinstance(actions[0], dict) and 'bern' in actions[0] and len(actions[0]['bern']) > 0:
                bern_vals = [d['bern'] for d in actions]
                bern_actions = torch.as_tensor(np.array(bern_vals), dtype=torch.float, device=self.device)

        if bern_actions is None:
            return advantage
        if bern_actions.dim() == 1:
            bern_actions = bern_actions.unsqueeze(-1)

        with torch.no_grad():
            # ---- 规则教师在 bern 头上的共识聚合 ----
            def _get_teacher_outputs(t):
                if getattr(t, 'is_rule_teacher', False):
                    return t.predict_distributions(states)
                else:
                    return t.net(states)

            # 统一为列表处理
            if isinstance(teacher_actor, (list, tuple)):
                teacher_list = list(teacher_actor)
            else:
                teacher_list = [teacher_actor]

            if len(teacher_list) == 0:
                return advantage

            # 收集每个教师的 bern 决策：logit > 0 视为开火（与 RuleTeacherWrapper 的 ±3.0 对应）
            teacher_bern_decisions = []
            for t in teacher_list:
                t_out = _get_teacher_outputs(t)
                t_bern_logits = t_out.get('bern')
                if t_bern_logits is None:
                    return advantage
                teacher_bern_decisions.append(t_bern_logits > 0.0)
            # 形状 (num_teachers, B, bern_dim)，bool
            teacher_bern_decisions = torch.stack(teacher_bern_decisions, dim=0)

            # 所有教师一致认为开火 / 一致认为不开火
            all_fire = teacher_bern_decisions.all(dim=0)       # (B, bern_dim)
            all_no_fire = (~teacher_bern_decisions).all(dim=0)
            has_consensus = all_fire | all_no_fire              # 存在共识的样本

            # 只在学生实际执行了开火（bern=1）的样本上做偏移；不开火的样本完全不管
            student_fired = (bern_actions[:, 0] == 1.0)

            # 匹配条件：active=1 且 学生实际开火 且 教师有共识
            match = active_masks.bool().squeeze(-1) & student_fired & has_consensus[:, 0]

            # 符号：全教师认为该开火 → 正偏移（鼓励学生继续开火）
            #       全教师认为不该开火 → 负偏移（惩罚不该开火时开火）
            sign = torch.zeros(advantage.size(0), device=self.device, dtype=torch.float)
            sign[match & all_fire[:, 0]] = 1.0
            sign[match & all_no_fire[:, 0]] = -1.0

            # AFiltered 分方向过滤：
            # - 教师一致认为该开火（正偏移）：只在 advantage 高于均值时强化
            # - 教师一致认为不该开火（负偏移）：只在 advantage 低于均值时压制
            if AFiltered:
                adv_mean = advantage.mean()
                adv_high = (advantage > adv_mean).squeeze(-1)
                adv_low = (advantage < adv_mean).squeeze(-1)
                valid = ((sign == 1.0) & adv_high) | ((sign == -1.0) & adv_low)
                sign = sign * valid.float()

            if sign.abs().sum() > 0:
                adv_std = advantage.std(unbiased=False)
                advantage = advantage + sign.view_as(advantage) * (adv_std * alpha_distill)

        return advantage

    # =========================================================================
    #  BernDistill: 直接对 bern 头做有监督蒸馏（规则教师共识驱动）
    # =========================================================================
    def BernDistill(self, transition_dict, teacher_actor=None,
                    shuffled=1, mini_batch_size=None, epochs=1, alpha=1.0,
                    grad_clip_ratio=0.5):
        """
        对 bern（开火）头做直接有监督蒸馏，梯度只流过 bern 头参数，
        不影响 backbone / cat 头 / cont 头，彻底隔离对机动策略的干扰。

        教师共识逻辑与 ADistill 一致：
        - teacher_actor 支持单教师或列表；
        - 只在所有教师对 bern 达成一致（全开火 or 全不开火）的样本上施加监督；
        - 只在学生实际执行了开火（bern=1）的样本上施加监督；
        - 教师意见不一致 / 学生未开火 的样本不参与 loss。

        监督目标（微摄动）：
        - 不使用 0/1 硬标签，而是在学生当前开火概率基础上微调：
        - 教师共识=开火 → target = clamp(student_p + 0.01, 0, 1)
        - 教师共识=不开火 → target = clamp(student_p - 0.01, 0, 1)
        - 每次蒸馏只提供微小的概率推力，避免对策略造成大的突变。

        梯度安全：
        - backbone (self.actor.net.net) 和非 bern 头全部 detach（不回传梯度）
        - 双道锁：clip_grad_value_ + 自适应 clip_grad_norm_（锚定 PPO actor_grad）
        """
        if teacher_actor is None or alpha == 0.0 or epochs == 0:
            return

        # 辅助函数
        def to_tensor(x, dtype):
            if isinstance(x, np.ndarray):
                return torch.tensor(x, dtype=dtype).to(self.device)
            return torch.tensor(np.array(x), dtype=dtype).to(self.device)

        # 1. 准备输入
        if 'obs' in transition_dict and len(transition_dict['obs']) > 0:
            states = to_tensor(transition_dict['obs'], torch.float)
        else:
            states = to_tensor(transition_dict['states'], torch.float)

        # active mask
        if 'active_masks' in transition_dict:
            active_masks = to_tensor(transition_dict['active_masks'], torch.float).view(-1)
        else:
            active_masks = torch.ones(states.size(0), device=self.device)

        # 实际 bern 动作
        actions = transition_dict['actions']
        bern_actions = None
        if isinstance(actions, dict):
            if 'bern' in actions and len(actions['bern']) > 0:
                bern_actions = to_tensor(actions['bern'], torch.float)
        else:
            if len(actions) > 0 and isinstance(actions[0], dict) and 'bern' in actions[0] and len(actions[0]['bern']) > 0:
                bern_vals = [d['bern'] for d in actions]
                bern_actions = to_tensor(np.array(bern_vals), torch.float)

        if bern_actions is None:
            return
        if bern_actions.dim() == 1:
            bern_actions = bern_actions.unsqueeze(-1)

        num_samples = states.size(0)
        mb = mini_batch_size if mini_batch_size is not None else num_samples

        # 2. 教师共识判定（全量，一次性算好）
        with torch.no_grad():
            def _get_teacher_outputs(t):
                if getattr(t, 'is_rule_teacher', False):
                    return t.predict_distributions(states)
                else:
                    return t.net(states)

            if isinstance(teacher_actor, (list, tuple)):
                teacher_list = list(teacher_actor)
            else:
                teacher_list = [teacher_actor]

            if len(teacher_list) == 0:
                return

            teacher_bern_decisions = []
            for t in teacher_list:
                t_out = _get_teacher_outputs(t)
                t_bern_logits = t_out.get('bern')
                if t_bern_logits is None:
                    return
                teacher_bern_decisions.append(t_bern_logits > 0.0)
            teacher_bern_decisions = torch.stack(teacher_bern_decisions, dim=0)  # (K, B, 1)

            all_fire = teacher_bern_decisions.all(dim=0)[:, 0]        # (B,)
            all_no_fire = (~teacher_bern_decisions).all(dim=0)[:, 0]   # (B,)
            has_consensus = all_fire | all_no_fire

            student_fired = (bern_actions[:, 0] == 1.0)

            # 有效样本：active=1 & 学生开了火 & 教师有共识
            valid_mask = active_masks.bool() & student_fired & has_consensus

            # 构造摄动方向：
            # [注释掉] 教师共识=开火 → 提高开火概率（避免引导过度开火）
            # 教师共识=不开火 → 压低开火概率，目标设为当前概率的 0.8 倍
            nudge_dir = torch.zeros(num_samples, device=self.device)
            # nudge_dir[all_fire] = 1.0
            nudge_dir[all_no_fire] = -1.0

            # [关键] 在循环开始前一次性算好固定的 target_p，后续 epoch 不再变动
            with torch.no_grad():
                baseline_features = self.actor.net.net(states)
                baseline_logits = self.actor.net.fc_bern(baseline_features)[:, 0]  # (B,)
                baseline_p = torch.sigmoid(baseline_logits)

                # 仅当教师一致认为不该开火时，才压低目标概率
                target_p = baseline_p.clone()
                target_p[all_no_fire] = baseline_p[all_no_fire] * 0.8

        # 若无有效样本，跳过
        if not valid_mask.any():
            return

        # 3. 提取只有 bern 头的参数（隔离 backbone / cat / cont）
        bern_params = list(self.actor.net.fc_bern.parameters())
        if len(bern_params) == 0:
            return

        # 4. 蒸馏训练循环
        distil_loss_list = []
        distil_grad_list = []
        indices = np.arange(num_samples)

        for _ in range(int(epochs)):
            if shuffled:
                np.random.shuffle(indices)

            for start in range(0, num_samples, mb):
                end = min(start + mb, num_samples)
                batch_idx = indices[start:end]

                mb_states = states[batch_idx]
                mb_valid = valid_mask[batch_idx]
                mb_target = target_p[batch_idx]

                if not mb_valid.any():
                    continue

                # 前向：backbone 不回传梯度，只让 bern 头出梯度
                with torch.no_grad():
                    shared_features = self.actor.net.net(mb_states)
                # bern 头正常前向（有梯度）
                bern_logits = self.actor.net.fc_bern(shared_features)  # (mb, 1)

                # 只在有效样本上计算 BCE loss（target 是提前固定好的）
                logits_valid = bern_logits[:, 0][mb_valid]
                target_valid = mb_target[mb_valid]

                loss = F.binary_cross_entropy_with_logits(logits_valid, target_valid, reduction='mean')
                final_loss = alpha * loss

                # 只对 bern_params 做反向传播和更新
                self.actor_optimizer.zero_grad()
                final_loss.backward()

                # 安全起见：清零非 bern 参数的梯度（防止 optimizer 对它们做动量更新）
                for p in self.actor.parameters():
                    if not any(p is bp for bp in bern_params):
                        if p.grad is not None:
                            p.grad.zero_()

                # 双道梯度锁
                nn.utils.clip_grad_value_(bern_params, clip_value=0.5)
                adaptive_l2_limit = max(self.actor_grad, 1e-4) * grad_clip_ratio
                nn.utils.clip_grad_norm_(bern_params, max_norm=adaptive_l2_limit)

                self.actor_optimizer.step()

                distil_loss_list.append(final_loss.item())
                distil_grad_list.append(
                    sum(p.grad.norm().item() for p in bern_params if p.grad is not None) / max(len(bern_params), 1)
                )

        # 5. 记录指标
        if len(distil_loss_list) > 0:
            self.bern_distil_loss = np.mean(distil_loss_list)
            self.bern_distil_grad = np.mean(distil_grad_list)
            check_weights_bias_nan(self.actor, "actor", "BernDistill后")
        else:
            self.bern_distil_loss = 0.0
            self.bern_distil_grad = 0.0

    def set_learning_rate(self, actor_lr=None, critic_lr=None):
        if actor_lr is not None:
            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] = actor_lr
        if critic_lr is not None:
            for param_group in self.critic_optimizer.param_groups:
                param_group['lr'] = critic_lr  


    # 重置优化器，这里仍然是adam，目前只有PBT在使用这个方法
    def reset_optimizer(self, reset_type='both'):
        """
        清除 optimizer 的动量等 state，但保留当前学习率/其他 param_group 设置。
        通过重建 optimizer（保留 param_group 的 lr/betas/eps/weight_decay）来达到清空 state 的目的。
        param_groups 里只存配置参数（lr、betas、eps、weight_decay 等），不存动量状态。
        PyTorch 的动量信息（Adam 的 exp_avg/exp_avg_sq、RMSprop 的 square_avg 等）都放在 old_optim.state 这个独立字典里
        """
        def _recreate_from(old_optim, params):
            optim_cls = type(old_optim)
            pg = old_optim.param_groups[0]
            kwargs = {k: v for k, v in pg.items() if k != 'params'}
            return optim_cls(params, **kwargs)

        if reset_type=='both':
            # 重新创建 optim（以清除内部 state），但使用当前的 lr/betas/eps/weight_decay
            self.actor_optimizer = _recreate_from(self.actor_optimizer, self.actor.parameters())
            self.critic_optimizer = _recreate_from(self.critic_optimizer, self.critic.parameters())
            # 清除可能残留的梯度
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
        elif reset_type=='actor':
            self.actor_optimizer = _recreate_from(self.actor_optimizer, self.actor.parameters())
            self.actor_optimizer.zero_grad()
        elif reset_type=='critic':
            self.critic_optimizer = _recreate_from(self.critic_optimizer, self.critic.parameters())
            self.critic_optimizer.zero_grad()
        else:
            pass
    
    def take_action(self, state, h0=None, explore=True, max_std=None, check_obs=None, temperature=1.0, mask_on=0):
        # 委托给 Actor Wrapper
        max_s = max_std if max_std is not None else self.max_std
        
        # [修改] 透传 check_obs
        actions_exec, actions_raw, h_state, actions_dist_check = self.actor.get_action(
            state, h=h0, explore=explore, max_std=max_s, check_obs=check_obs, temperature=temperature, mask_on=mask_on
        )
        #  保持原有的返回两个字典的接口，或者根据需要返回 diagnostic output
        return actions_exec, actions_raw, h_state, actions_dist_check

    def update(self, transition_dict, adv_normed=False, 
                clip_vf=False, clip_range=0.2, shuffled=1, 
                mini_batch_size=None, alpha_logit_reg=0.05,
                v_trace=None, target_p1=0.65, target_p1_b=0.8, 
                k_nonlinear=0.89, mask_on=0, actor_frozen=0, bern_max_logits=4.0, alpha_distill=0, teacher_actor=None,
                AFiltered=0, conf_thres=0.7, bern_included=0): 
                # [新增] target_p1 默认“一超”概率，剩下来的留给“多强”)
                # [修改] 增加 target_p1_b 参数，对应开火控制的“笃定程度”

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
                if val is None or (isinstance(val, (np.ndarray, torch.Tensor, list)) and len(val) == 0):
                    continue
                if key == 'cat':
                    actions_on_device[key] = to_tensor(val, torch.long)
                else:
                    actions_on_device[key] = to_tensor(val, torch.float)
        else:
            # 兼容旧代码 (list of dicts)
            # 旧逻辑：List of Dicts (较慢)
            all_keys = actions_from_buffer[0].keys()
            for key in all_keys:
                vals = [d[key] for d in actions_from_buffer if key in d]
                if len(vals) == 0:
                    continue
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

                
        # # 策略蒸馏 (Advantage-based): 提升与 teacher 一致的 cat 动作的优势度
        # if alpha_distill > 0 and teacher_actor is not None:
        #     advantage = self.ADistill(transition_dict, advantage, alpha_distill, teacher_actor, AFiltered=AFiltered, conf_thres=conf_thres, bern_included=bern_included)
        
        # 3. 计算旧策略的 log_probs (使用 Wrapper)
        with torch.no_grad():
            # Actor 使用 actor_inputs (可能是 obs)
            # [修改] 接收 5 个返回值
            old_log_probs, _, _, _ ,_ = self.actor.evaluate_actions(actor_inputs, actions_on_device, h=None, max_std=self.max_std, mask_on=mask_on)
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

        # =====================================================================
        # [新增] 计算 Cat 分布的动态目标熵 (Target Entropy)
        # =====================================================================
        target_entropy_cat_total = 0.0
        if 'cat' in self.actor.action_dims and sum(self.actor.action_dims['cat']) > 0:
            for n_actions in self.actor.action_dims['cat']:
                if n_actions > 1:
                    # “一超”概率为 target_p1，其余 (n_actions - 1) 个动作平分剩余概率
                    p_rest = (1.0 - target_p1) / (n_actions - 1)
                    # 香农熵: -p1*ln(p1) - (N-1) * p_rest*ln(p_rest)
                    h_dim = - target_p1 * np.log(target_p1 + 1e-8) - (n_actions - 1) * p_rest * np.log(p_rest + 1e-8)
                    target_entropy_cat_total += h_dim
        target_entropy_cat_tensor = torch.tensor(target_entropy_cat_total, device=self.device)
        
        # =====================================================================
        # [新增] 计算 Bernoulli 分布的动态目标熵 (Target Entropy)
        # =====================================================================
        target_entropy_bern_total = 0.0
        if 'bern' in self.actor.action_dims and self.actor.action_dims['bern'] > 0:
            # 伯努利熵公式: -p*ln(p) - (1-p)*ln(1-p)
            p = target_p1_b
            h_bern_dim = - p * np.log(p + 1e-8) - (1 - p) * np.log(1 - p + 1e-8)
            # 伯努利熵按动作维度累加
            target_entropy_bern_total = h_bern_dim * self.actor.action_dims['bern']
        target_entropy_bern_tensor = torch.tensor(target_entropy_bern_total, device=self.device)

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
        grad_norm_ratio_list = [] # [新增] 范数比列表
        max_fire_prob_list = [] # [新增] 记录未被mask的bern最大触发概率
        min_fire_prob_list = [] # [新增] 记录未被mask的bern最小触发概率
        
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
                log_probs, entropy, entropy_details, actor_outputs, _ = self.actor.evaluate_actions(mb_actor_inputs, mb_actions, h=None, max_std=self.max_std, mask_on=mask_on)
                
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
                
                # 借用一点IMPALA的经验，防止ratio过大导致梯度被炸飞
                # 建议设置数值：2.0~5.0
                if v_trace is not None:
                    ratio = torch.clamp(ratio, max=v_trace)

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
                
                # [新增] 使用 fire_mask 过滤 Bernoulli 熵，只统计允许开火时段的熵
                if 'fire_mask' in actor_outputs and actor_outputs['fire_mask'] is not None:
                    bern_fire_mask = actor_outputs['fire_mask']  # shape: (batch, bern_dim)
                    # 对齐到 (batch, 1)
                    if bern_fire_mask.dim() > 1:
                        bern_fire_mask = bern_fire_mask[:, 0:1]  # 取第一个 Bernoulli 维度
                    fireable_sum = (mb_active_masks * bern_fire_mask).sum()
                    loss_ent_bern = (e_bern * mb_active_masks * bern_fire_mask).sum() / (fireable_sum + mask_eps)
                else:
                    # 回退到原有逻辑（兼容旧版本）
                    loss_ent_bern = (e_bern * mb_active_masks).sum() / (active_sum + mask_eps)

                # =====================================================================
                # [修改] 使用动态 k_cat 计算 Actor Loss
                # =====================================================================
                k_cont = self.k_entropy.get('cont', 0.0)
                k_cat = self.k_entropy.get('cat', 0.0)
                k_bern = self.k_entropy.get('bern', 0.0)
                
                # # 当前的 k_cat 是 e^(log_k_cat)
                # curr_k_cat = torch.exp(self.log_k_cat)
                # --- 关键修改：跟踪目标熵, 熵过小，加熵。熵过大，减熵 ---
                # 比平方误差更加平缓的损失函数候选：
                # 1. pseudo-huber loss:     torch.sqrt(1 + torch.square(target - current)) - 1
                # 2. log-cosh loss:         torch.log(torch.cosh(target_entropy_cat_tensor - loss_ent_cat))

                # 裁剪的范围，绝不能太小
                theoretical_ent_cat_max = 0.0
                for dim in self.actor.action_dims['cat']:
                    theoretical_ent_cat_max += np.log(float(dim))
                diff_cat0 = theoretical_ent_cat_max - target_entropy_cat_tensor
                k_nonlinear_max_cat = torch.sqrt(1+diff_cat0**2)/(diff_cat0+1e-8)

                theoretical_ent_bern_max = -np.log(0.5)*self.actor.action_dims['bern']
                diff_bern0 = theoretical_ent_bern_max - target_entropy_bern_tensor
                k_nonlinear_max_bern = torch.sqrt(1+diff_bern0**2)/(diff_bern0+1e-8)

                k_nonlinear_cat = min(max(k_nonlinear, 0.0), k_nonlinear_max_cat)
                k_nonlinear_bern = min(max(k_nonlinear, 0.0), k_nonlinear_max_bern)

                # 1. Categorical 约束项
                cat_constraint_term = k_cat * (
                    - loss_ent_cat # + 
                    # torch.where(loss_ent_cat > target_entropy_cat_tensor, 
                    # min(max(k_nonlinear_cat, 0.0), 1.0)/(2 * diff_cat0) * \
                    #     torch.square(loss_ent_cat - target_entropy_cat_tensor)
                    # , 0.0)
                )
                # # 2. Bernoulli 约束项 (动态系数：熵越高推力越弱，理论最大时归零)
                # # f(H) = -H + H^2/(2*H_max), 导数 f'(H) = -1 + H/H_max
                # # H=0 时全力推，H=H_max 时导数为0
                # bern_constraint_term = k_bern * (
                #     - loss_ent_bern
                #     + torch.where(loss_ent_bern > target_entropy_bern_tensor,
                #     min(max(k_nonlinear_bern, 0.0), 1.0)/(2 * diff_bern0) * \
                #         torch.square(loss_ent_bern - target_entropy_bern_tensor)
                #     , 0.0)
                # )
                # 3. 组合最终 Actor Loss
                # [修改] 移除 bern_constraint_term（熵约束），由 mini-batch 内的 FRR 替代
                actor_loss = actor_loss + cat_constraint_term - (k_cont * loss_ent_cont)
                
                # 原有非目标熵正则项
                # actor_loss = actor_loss - (k_cont * loss_ent_cont + k_cat * loss_ent_cat + k_bern * loss_ent_bern)

                # # =====================================================================
                # # [修改/重构] Bernoulli 开火头的专属正则化、稀疏惩罚与冷却约束
                # # =====================================================================
                if actor_outputs['bern'] is not None:
                    bern_logits = actor_outputs['bern']
                    bern_probs = torch.sigmoid(bern_logits)
                    
                    # 1. 基础 Logit 越界惩罚 (保持 Logit 在可激活区间)
                    over = F.relu(torch.abs(bern_logits) - bern_max_logits) # 4.0  20步至少打一发，那么开火概率至少也是0.05，对应-3的logits
                    # 只惩罚允许开火的位置
                    if 'fire_mask' in actor_outputs and actor_outputs['fire_mask'] is not None:
                        fire_mask = actor_outputs['fire_mask']  # shape: (batch, bern_dim)
                    else:
                        fire_mask = (bern_logits > -1e6).float()
                    # 只对允许开火且存活的样本求平均，避免被 non-fireable 样本稀释
                    fireable_sum = (mb_active_masks * bern_fire_mask).sum()
                    logit_loss = ((over ** 2) * fire_mask * mb_active_masks).sum() / (fireable_sum + mask_eps)
                    actor_loss = actor_loss + alpha_logit_reg * logit_loss
                    
                    # [新增] 记录未被mask的bern触发概率极值
                    valid_fire_mask = (fire_mask * mb_active_masks).bool()
                    if valid_fire_mask.any():
                        valid_probs = bern_probs[valid_fire_mask]
                        max_fire_prob_list.append(valid_probs.max().item())
                        min_fire_prob_list.append(valid_probs.min().item())


                    # [新增] 3. 单阈值均值约束 (FRR) —— 替代 bern 熵正则项
                    # 约束允许开火样本的平均概率不超过 p_target，用 pseudo-Huber loss
                    # 保护弹药节奏：防止中远距离手痒乱射导致后期无弹可用
                    fire_mask_1d = fire_mask[:, 0] if fire_mask.dim() > 1 else fire_mask.squeeze(-1)
                    active_1d = mb_active_masks.squeeze(-1)
                    fireable_frr = (fire_mask_1d > 0.5) & (active_1d > 0.5)
                    if fireable_frr.sum() > 0:
                        p_mean_fireable = bern_probs[:, 0][fireable_frr].mean()
                        p_target = 0.1  # 目标平均开火概率 20%，对应 6 枚 / 30 个有效开火步
                        delta = F.relu(p_mean_fireable - p_target)
                        # pseudo-Huber loss: sqrt(1 + delta^2) - 1，平缓惩罚，对异常值不爆炸
                        frr_loss = torch.sqrt(1.0 + delta * delta) - 1.0
                        # 复用原熵系数 k_bern 作为强度（建议 0.001~0.005）
                        actor_loss = actor_loss + k_bern * frr_loss

                # Critic Loss
                # Critic 使用 critic_inputs
                v_pred = self.critic(mb_critic_inputs)
                # if clip_vf:
                #     v_pred_old_batch = v_pred_old[batch_idx]

                #     # 新的critic clip方法，按标准差倍数缩放限幅
                #     # 1. 动态计算当前批次 TD Target 的标准差，作为价值尺度的基准
                #     with torch.no_grad():
                #         td_target_std = torch.std(mb_td_target).item()
                #     # 2. 自适应计算 clip_range。设定 0.5 倍标准差为窗口，并用 10.0 进行保底
                #     # 防止训练初期或特定批次由于 Target 过于单一导致 std 接近 0 从而锁死更新
                #     adaptive_clip_range = max(td_target_std * 0.5, 10.0)
                #     # 3. 使用动态计算的范围进行截断
                #     v_pred_clipped = torch.clamp(
                #         v_pred, 
                #         v_pred_old_batch - adaptive_clip_range, 
                #         v_pred_old_batch + adaptive_clip_range
                #     )
                #     vf_loss1 = (v_pred - mb_td_target).pow(2)
                #     vf_loss2 = (v_pred_clipped - mb_td_target).pow(2)
                #     critic_loss_per_sample = torch.max(vf_loss1, vf_loss2)

                # else:
                #  reduction='none' 使得我们可以应用 mask
                critic_loss_per_sample = F.mse_loss(v_pred, mb_td_target, reduction='none')
               
                # 1、有序列时间修正的时候， Critic Loss 使用 mask 加权
                # critic_loss = (critic_loss_per_sample * mb_active_masks).sum() / (active_sum + mask_eps)
                # 2、多智能体情况下，critic不传入active_mask
                critic_loss = critic_loss_per_sample.mean()
                
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                # if alpha_loss.requires_grad:    # [新增]
                #     self.k_cat_optim.zero_grad()
                if actor_frozen:
                    self.reset_optimizer(reset_type='actor')

                if not actor_frozen:
                    actor_loss.backward()
                critic_loss.backward()
                # if alpha_loss.requires_grad:    # [新增] 反向传播温度损失
                #     alpha_loss.backward()

                pre_clip_actor_grad.append(model_grad_norm(self.actor))
                pre_clip_critic_grad.append(model_grad_norm(self.critic)) 
                
                # [新增] 计算范数比 (Pre-clip)
                grad_norm_ratio_list.append(pre_clip_actor_grad[-1] / (pre_clip_critic_grad[-1] + 1e-8))

                nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.actor_max_grad)
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.critic_max_grad)
                if not actor_frozen:
                    self.actor_optimizer.step()
                self.critic_optimizer.step()
                # if alpha_loss.requires_grad:    # [新增] 步进自适应熵系数
                #     self.k_cat_optim.step()

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
                    # [修改] 使用 fire_mask 过滤，只记录 fireable 样本的平均熵
                    if 'fire_mask' in actor_outputs and actor_outputs['fire_mask'] is not None:
                        bern_fire_mask = actor_outputs['fire_mask']  # shape: (batch, bern_dim)
                        if bern_fire_mask.dim() > 1:
                            bern_fire_mask = bern_fire_mask[:, 0:1]  # 对齐到 (batch, 1)
                        fireable_sum = (mb_active_masks * bern_fire_mask).sum()
                        bern_entropy_filtered = (entropy_details['bern'] * mb_active_masks * bern_fire_mask).sum() / (fireable_sum + mask_eps)
                        entropy_bern_list.append(bern_entropy_filtered.item())
                    else:
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
        
        # [新增] 汇总监控项
        self.grad_norm_ratio = np.mean(grad_norm_ratio_list) if len(grad_norm_ratio_list) > 0 else 0
        
        # [新增] 赋值有效样本监控项
        self.PPO_samples = ppo_samples_total
        self.PPO_valid_samples = ppo_valid_samples_total
        
        # [新增] 汇总未被mask的bern触发概率极值
        self.max_fire_prob = np.mean(max_fire_prob_list) if len(max_fire_prob_list) > 0 else 0
        self.min_fire_prob = np.mean(min_fire_prob_list) if len(min_fire_prob_list) > 0 else 0


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
            self.td_error_var = np.var(y_true - y_pred) # [新增] TD error 方差
            if var_y < 1e-8:
                self.explained_var = 0.0
            else:
                self.explained_var = 1 - self.td_error_var / var_y
        else:
            self.td_error_var = 0.0
            self.explained_var = 0.0

        check_weights_bias_nan(self.actor, "actor", "update后")
        check_weights_bias_nan(self.critic, "critic", "update后")

    # =========================================================================
    #  [New Method] Bernoulli 开火头保护性有监督训练 (防止机动策略被bern崩溃拖累)
    # =========================================================================
    def fire_prob_protection(self, transition_dict, protect_epochs=4, protect_mini_batch=256, mask_on=0):
        """
        Bern头概率范围保护器。当开火概率整体崩溃（全高或全低）时，以有监督方式
        强行拉回bern头分布，同时切断backbone和其它动作头的梯度，保护机动策略不被拖垮。

        必要条件1 (比值护栏): max_fire_prob / min_fire_prob >= 10，说明分布仍有分化空间，
                               不需要干预，直接跳过。
        必要条件2 (触发case):
          case1: max_fire_prob < 0.05  → 整体开火概率崩到极低，以0.5为监督信号，拉高熵。
          case2: min_fire_prob > 0.1   → 整体开火概率过高，以1e-3为监督信号，压低概率。

        Args:
            transition_dict : 与update()相同格式的经验字典。
            protect_epochs  : 保护性训练的epoch数。
            protect_mini_batch: 每个mini-batch的大小。
            mask_on         : 传给net forward的mask开关，与update保持一致。
        """
        # ── 必要条件1：比值护栏 ──────────────────────────────────────────────────
        ratio = self.max_fire_prob / (self.min_fire_prob + 1e-12)
        if ratio >= 10.0:
            return  # 分布仍有足够分化，不需要干预

        # ── 必要条件2：判断触发case ──────────────────────────────────────────────
        if self.max_fire_prob < 0.05:
            # case1: 概率塌缩到接近0 → 用0.5拉高熵
            target_prob = 0.5
        elif self.min_fire_prob > 0.05:
            # case2: 概率整体过高 → 用1e-3压低
            target_prob = 1e-2
        else:
            return  # 不满足任何触发条件

        # ── 数据准备（复用update的转换逻辑）────────────────────────────────────
        def to_tensor(x, dtype):
            if isinstance(x, np.ndarray):
                return torch.tensor(x, dtype=dtype).to(self.device)
            else:
                return torch.tensor(np.array(x), dtype=dtype).to(self.device)

        if 'obs' in transition_dict:
            actor_inputs = to_tensor(transition_dict['obs'], torch.float)
        else:
            actor_inputs = to_tensor(transition_dict['states'], torch.float)

        if 'active_masks' in transition_dict:
            active_masks = to_tensor(transition_dict['active_masks'], torch.float).view(-1, 1)
        else:
            active_masks = torch.ones(actor_inputs.size(0), 1, device=self.device)

        num_samples = actor_inputs.size(0)
        mb_size = min(protect_mini_batch, num_samples)

        # ── 冻结除bern头以外的所有actor模块 ───────────────────────────────────
        net = self.actor.net  # PolicyNetHybrid 实例

        def set_requires_grad(module_or_param, flag):
            if isinstance(module_or_param, nn.Module):
                for p in module_or_param.parameters():
                    p.requires_grad_(flag)
            else:
                module_or_param.requires_grad_(flag)

        # 逐模块冻结（backbone + 其它动作头）
        set_requires_grad(net.net, False)  # 共享backbone
        if hasattr(net, 'fc_mu'):
            set_requires_grad(net.fc_mu, False)
        if hasattr(net, 'log_std_cont'):
            set_requires_grad(net.log_std_cont, False)
        if hasattr(net, 'fc_cat'):
            set_requires_grad(net.fc_cat, False)
        # bern头保持可训练
        if hasattr(net, 'fc_bern'):
            set_requires_grad(net.fc_bern, True)

        # ── 监督训练循环 ──────────────────────────────────────────────────────
        target_tensor = torch.tensor(target_prob, device=self.device)

        for _ in range(protect_epochs):
            perm = torch.randperm(num_samples, device=self.device)
            for start in range(0, num_samples, mb_size):
                end = min(start + mb_size, num_samples)
                batch_idx = perm[start:end]

                mb_states = actor_inputs[batch_idx]
                mb_active = active_masks[batch_idx]  # (mb, 1)
                active_sum = mb_active.sum()

                actor_out = self.actor.net(mb_states, mask_on=mask_on)

                if actor_out['bern'] is None:
                    break

                bern_logits = actor_out['bern'].clamp(min=-1e8)
                bern_probs = torch.sigmoid(bern_logits)  # (mb, bern_dim)

                # 监督目标：将每个位置的概率拉向target_prob
                # 使用BCE loss，target广播到与bern_probs相同形状
                target_full = target_tensor.expand_as(bern_probs)
                bern_loss_per_sample = F.binary_cross_entropy(
                    bern_probs, target_full, reduction='none'
                ).sum(dim=-1, keepdim=True)  # (mb, 1)

                mask_eps_loc = 1e-5
                bern_loss = (bern_loss_per_sample * mb_active).sum() / (active_sum + mask_eps_loc)

                self.actor_optimizer.zero_grad()
                bern_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.actor_max_grad)
                self.actor_optimizer.step()

        # ── 恢复所有actor模块的梯度反向传播 ──────────────────────────────────
        set_requires_grad(net.net, True)
        if hasattr(net, 'fc_mu'):
            set_requires_grad(net.fc_mu, True)
        if hasattr(net, 'log_std_cont'):
            set_requires_grad(net.log_std_cont, True)
        if hasattr(net, 'fc_cat'):
            set_requires_grad(net.fc_cat, True)
        if hasattr(net, 'fc_bern'):
            set_requires_grad(net.fc_bern, True)

        return

    # =========================================================================
    #  [New Method] 碎片化有监督机动保护 (水平指令头软约束)
    # =========================================================================
    def maneuver_il_protection(self, transition_dict, alpha=1.0, epochs=4, mini_batch=256, mask_on=0):
        """
        根据状态标志对水平机动头施加碎片化有监督约束，冻结backbone和其他动作头。

        触发规则（仅针对 cat[1] 水平头）：
          - missile_in_mid_term==0 且 warning==0：目标为动作索引 0（追踪）
          - warning==1                          ：目标为动作索引 2/3/4（均等概率，标签平滑）

        不满足上述任一条件的样本（如 missile_in_mid_term==1 且 warning==0）直接跳过。

        Args:
            transition_dict : 与 update() 相同格式的经验字典。
            alpha           : 与 ADPC_update 相同，用于缩放学习率。
            epochs          : 有监督训练的 epoch 数。
            mini_batch      : 每个 mini-batch 的大小。
            mask_on         : 传给 net forward 的 mask 开关。
        """
        def to_tensor(x, dtype):
            if isinstance(x, np.ndarray):
                return torch.tensor(x, dtype=dtype).to(self.device)
            return torch.tensor(np.array(x), dtype=dtype).to(self.device)

        # ── 读取 actor 输入 obs ────────────────────────────────────────────────
        if 'obs' in transition_dict and len(transition_dict['obs']) > 0:
            actor_inputs = to_tensor(transition_dict['obs'], torch.float)
        else:
            actor_inputs = to_tensor(transition_dict['states'], torch.float)

        num_samples = actor_inputs.size(0)

        # ── 从 obs 提取标志位 (obs2obs_check 中确认的列索引) ──────────────────
        # col 3: missile_in_mid_term,  col 5: warning,  col 25: threat_distance (scaled)
        missile_in_mid_term = actor_inputs[:, 3]          # (N,)
        warning_flag        = actor_inputs[:, 5]          # (N,)
        threat_distance     = actor_inputs[:, 25] * 10e3  # (N,) 还原为米

        # ── 构造每个样本的有监督目标 (one-hot 软标签，水平头共7类) ────────────
        # 满足条件1: missile_in_mid_term~0 且 warning~0 → 目标索引 0
        # 满足条件2: warning~1                          → 目标索引 2/3/4 均等
        # 其余: 跳过 (mask=0)
        n_h = self.actor.net.cat_dims[1]  # 水平头类别数，应为 7

        cond1 = (missile_in_mid_term < 0.5) & (warning_flag < 0.5)  # 追踪
        cond2 = (warning_flag > 0.5) & (threat_distance < 15e3)       # 防御（近距告警）

        active = (cond1 | cond2).float().unsqueeze(1)  # (N, 1)
        # active = (cond2).float().unsqueeze(1)  # (N, 1)
        if active.sum() < 1:
            return  # 没有可监督的样本，直接跳过

        # 软标签矩阵 (N, n_h)
        soft_labels = torch.zeros(num_samples, n_h, device=self.device)
        soft_labels[cond1, 0] = 1.0                   # 条件1: 只有动作0
        soft_labels[cond2, 2] = 0.25                   # 条件2: 动作2/3/4 按 1:2:1
        soft_labels[cond2, 3] = 0.50
        soft_labels[cond2, 4] = 0.25

        # active_masks（若存在）
        if 'active_masks' in transition_dict:
            env_masks = to_tensor(transition_dict['active_masks'], torch.float).view(-1, 1)
        else:
            env_masks = torch.ones(num_samples, 1, device=self.device)

        if env_masks.sum() < 1:
            return  # 没有可用样本，直接跳过

        # ── 冻结 backbone / 其他动作头，仅放开 fc_cat ────────────────────────
        net = self.actor.net

        def set_requires_grad(module_or_param, flag):
            if isinstance(module_or_param, nn.Module):
                for p in module_or_param.parameters():
                    p.requires_grad_(flag)
            elif isinstance(module_or_param, nn.Parameter):
                module_or_param.requires_grad_(flag)

        set_requires_grad(net.net, False)
        if hasattr(net, 'fc_mu'):
            set_requires_grad(net.fc_mu, False)
        if hasattr(net, 'log_std_cont'):
            set_requires_grad(net.log_std_cont, False)
        if hasattr(net, 'fc_bern'):
            set_requires_grad(net.fc_bern, False)
        # 只留 fc_cat 可训练
        if hasattr(net, 'fc_cat'):
            set_requires_grad(net.fc_cat, True)

        # ── 缩放学习率 ────────────────────────────────────────────────────────
        current_lr = self.actor_optimizer.param_groups[0]['lr']
        self.actor_optimizer.param_groups[0]['lr'] = current_lr * alpha

        mb_size = min(mini_batch, num_samples)
        eps_loc = 1e-5

        for _ in range(epochs):
            idx_perm = torch.randperm(num_samples, device=self.device)
            for start in range(0, num_samples, mb_size):
                mb_idx = idx_perm[start: start + mb_size]

                mb_obs    = actor_inputs[mb_idx]
                mb_labels = soft_labels[mb_idx]           # (mb, n_h)
                mb_active = active[mb_idx] * env_masks[mb_idx]  # (mb, 1)

                active_sum = mb_active.sum()
                if active_sum < eps_loc:
                    continue

                # 前向：只需要 cat 输出
                actor_out = self.actor.net(mb_obs, mask_on=mask_on)
                cat_probs_list = actor_out['cat']  # list of (mb, dim_i)

                # 取水平头 (index 1)
                h_probs = cat_probs_list[1]  # (mb, n_h)

                # KL(软标签 || 模型分布) = sum(p_target * log(p_target / p_model))
                # 等价于交叉熵 - 标签熵；因标签熵为常数，直接用交叉熵梯度方向即可
                # 使用 NLL = -sum(label * log(probs))
                log_probs = torch.log(h_probs.clamp(min=1e-8))  # (mb, n_h)
                nll_per_sample = -(mb_labels * log_probs).sum(dim=-1, keepdim=True)  # (mb, 1)

                cat_h_loss = (nll_per_sample * mb_active).sum() / (active_sum + eps_loc)

                self.actor_optimizer.zero_grad()
                cat_h_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.actor_max_grad)
                self.actor_optimizer.step()

        # ── 恢复所有模块梯度 ─────────────────────────────────────────────────
        set_requires_grad(net.net, True)
        if hasattr(net, 'fc_mu'):
            set_requires_grad(net.fc_mu, True)
        if hasattr(net, 'log_std_cont'):
            set_requires_grad(net.log_std_cont, True)
        if hasattr(net, 'fc_cat'):
            set_requires_grad(net.fc_cat, True)
        if hasattr(net, 'fc_bern'):
            set_requires_grad(net.fc_bern, True)

        # ── 还原学习率 ────────────────────────────────────────────────────────
        self.actor_optimizer.param_groups[0]['lr'] = current_lr

        return

    # =========================================================================
    #  [New Method] 将平铺的 transition_dict 重排为 (num_seqs, seq_len, ...) 形状
    # =========================================================================
    def reshape_for_rnn(self, transition_dict, seq_len):
        """
        将按回合顺序排列的平铺 transition_dict 切分为固定长度序列，供 GRU 网络使用。

        切分规则（与 HybridBuffer_rnn 保持一致）：
          - 以 dones == 1 作为回合边界，逐回合处理。
          - 每个回合内从后往前倒序切块，每块长度为 seq_len；头部余数丢弃。
          - 回合步数 < seq_len 的整个回合丢弃。

        Args:
            transition_dict: dict，键包含
                'states'      : (N, S)
                'next_states' : (N, S)   保留，用于现场计算 V(s')
                'rewards'     : (N,) 或 (N,1)
                'dones'       : (N,) 或 (N,1)
                'actions'     : dict of arrays, 每个值形状 (N, ...)
                可选: 'obs' (N, O), 'active_masks' (N,) 或 (N,1)
            seq_len: int，每条序列的长度

        Returns:
            dict，各字段形状为 (num_seqs, seq_len, ...) 的 numpy 数组；
            'actions' 仍为 dict，每个值形状 (num_seqs, seq_len, ...)。
            若无任何有效序列则抛出 RuntimeError。
        """
        def _to_np(x):
            if isinstance(x, torch.Tensor):
                return x.cpu().numpy()
            return np.array(x)

        states      = _to_np(transition_dict['states'])       # (N, S)
        next_states = _to_np(transition_dict['next_states'])  # (N, S)
        rewards     = _to_np(transition_dict['rewards']).reshape(-1)   # (N,)
        dones       = _to_np(transition_dict['dones']).reshape(-1)     # (N,)

        has_obs          = 'obs' in transition_dict
        has_active_masks = 'active_masks' in transition_dict
        obs          = _to_np(transition_dict['obs'])          if has_obs          else None
        active_masks = _to_np(transition_dict['active_masks']).reshape(-1) if has_active_masks else None

        actions_np = {}
        for k, v in transition_dict['actions'].items():
            actions_np[k] = _to_np(v)  # (N, ...)

        N = dones.shape[0]
        S = states.shape[-1]

        # --- 识别回合边界，倒序切块 ---
        done_indices = list(np.where(dones == 1)[0])
        ep_ends = done_indices[:]
        if len(ep_ends) == 0 or ep_ends[-1] != N - 1:
            ep_ends.append(N - 1)

        valid_seqs = []  # list of (ep_start, block_start) — both absolute indices
        curr_start = 0
        for ep_end in ep_ends:
            ep_len = ep_end - curr_start + 1
            if ep_len < seq_len:
                curr_start = ep_end + 1
                continue
            # 从 ep_end 向前每 seq_len 步切一块
            for block_end in range(ep_end, curr_start + seq_len - 2, -seq_len):
                block_start = block_end - seq_len + 1
                valid_seqs.append((block_start, block_end))
            curr_start = ep_end + 1

        if not valid_seqs:
            raise RuntimeError(
                f"reshape_for_rnn: 没有找到长度 >= {seq_len} 的回合，无法构建序列批次。"
            )

        num_seqs = len(valid_seqs)

        # --- 预分配输出容器 ---
        out = {
            'states':      np.zeros((num_seqs, seq_len, S),           dtype=np.float32),
            'next_states': np.zeros((num_seqs, seq_len, S),           dtype=np.float32),
            'rewards':     np.zeros((num_seqs, seq_len),               dtype=np.float32),
            'dones':       np.zeros((num_seqs, seq_len),               dtype=np.float32),
            'actions':     {},
        }
        if has_obs:
            O = obs.shape[-1]
            out['obs'] = np.zeros((num_seqs, seq_len, O), dtype=np.float32)
        if has_active_masks:
            out['active_masks'] = np.zeros((num_seqs, seq_len), dtype=np.float32)

        for k, v in actions_np.items():
            act_shape = v.shape[1:]  # e.g. (cont_dim,) or (n_cat_heads,) or (bern_dim,)
            dtype = np.int64 if k == 'cat' else np.float32
            out['actions'][k] = np.zeros((num_seqs, seq_len) + act_shape, dtype=dtype)

        # --- 填充 ---
        for i, (s, e) in enumerate(valid_seqs):
            out['states'][i]      = states[s:e+1]
            out['next_states'][i] = next_states[s:e+1]
            out['rewards'][i]     = rewards[s:e+1]
            out['dones'][i]       = dones[s:e+1]
            if has_obs:
                out['obs'][i]          = obs[s:e+1]
            if has_active_masks:
                out['active_masks'][i] = active_masks[s:e+1]
            for k in actions_np:
                out['actions'][k][i]   = actions_np[k][s:e+1]

        return out

    # =========================================================================
    #  [New Helper] 提取出的 MSE 计算逻辑 (供 mixed_update 和 BC_update 复用)
    # =========================================================================
    "离散空间的动作损失函数不对，暂不使用该函数"
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
    
    
    
    # --- 修改后的 MARWIL_update， 注意原先是0 ---
    def MARWIL_update(self, il_transition_dict, beta=1.0, batch_size=64, alpha=1.0, c_v=1.0, shuffled=1, label_smoothing=0.3, max_weight=100.0,
                      tau=0.8, action_heads_mask=None, no_bern=None, no_cat=None, train_critic=True):
        """
        MARWIL 离线更新函数
        输入 actions 结构支持: [{'cat': array([v]), 'bern': array([v])}, ...]
        tau: 非对称损失权重 (Expectile Regression). tau=0.5 为 MSE; tau>0.5 (如0.9) 倾向于高估 Value (拟合好样本)
        action_heads_mask: dict, 例如 {'cont': True, 'cat': True, 'bern': False}
                           指定哪些动作头参与模仿学习 Loss 计算。
                           默认不训练 bern 头，保持与旧版 no_bern=1 一致。
                           为兼容旧代码，仍保留 no_bern/no_cat，但它们会被映射为 mask。
        """
        # 1. 数据准备
        if 'obs' in il_transition_dict and len(il_transition_dict['obs']) > 0:
            obs_all = torch.tensor(np.array(il_transition_dict['obs']), dtype=torch.float).to(self.device)
            use_obs = True
        else:
            use_obs = False
            
        # 预训练阶段通常不训练探索 std
        if hasattr(self.actor.net, 'log_std_cont'):
            self.actor.net.log_std_cont.requires_grad = False

        # 1. 提取全量数据并转为 Tensor
        states_all = torch.tensor(np.array(il_transition_dict['states']), dtype=torch.float).to(self.device)
        returns_all = torch.tensor(np.array(il_transition_dict['returns']), dtype=torch.float).view(-1, 1).to(self.device)
        
        # 统一处理 Actions：List of Dicts -> Dict of Tensors
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

        # [新增] 权重与 advantage 监控累加器
        total_weight_mean = 0.0
        total_weight_max = 0.0
        total_weight_min = 0.0
        total_clip_frac = 0.0
        total_adv_std = 0.0
        total_adv_p95 = 0.0
        total_adv_max = 0.0
        total_adv_mean = 0.0

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

                # [新增] 记录本 batch 的权重统计
                total_weight_mean += weights.mean().item()
                total_weight_max += weights.max().item()
                total_weight_min += weights.min().item()
                total_clip_frac += (weights >= max_weight - 1e-6).float().mean().item()

                # [新增] 记录本 batch advantage 分布统计
                adv = advantage.detach()
                total_adv_std += adv.std().item()
                total_adv_p95 += torch.quantile(adv, 0.95).item()
                total_adv_max += adv.max().item()
                total_adv_mean += adv.mean().item()

            # 解析动作头mask；兼容旧版 no_bern/no_cat
            if action_heads_mask is None:
                action_heads_mask = {'cont': True, 'cat': True, 'bern': False}
                if no_bern is not None:
                    action_heads_mask['bern'] = not no_bern
                if no_cat is not None:
                    action_heads_mask['cat'] = not no_cat
            
            # B. Actor Loss
            raw_il_loss = self.actor.compute_il_loss(
                actor_input_batch,
                actions_batch,
                label_smoothing,
                action_heads_mask=action_heads_mask,
                good_samples=1,
                pre_training=1,
            )
            actor_loss = torch.mean(alpha * weights * raw_il_loss)

            # C. Critic Loss
            v_pred = self.critic(s_batch)
            
            # 原有
            critic_loss = F.mse_loss(v_pred, r_batch) * c_v
            


            # D. Optimize
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.actor_max_grad)
            self.actor_optimizer.step()
            if train_critic:
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.critic_max_grad)
                self.critic_optimizer.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_c += c.item()
            batch_count += 1

        avg_weight_mean = total_weight_mean / batch_count if batch_count > 0 else 0
        avg_weight_max = total_weight_max / batch_count if batch_count > 0 else 0
        avg_weight_min = total_weight_min / batch_count if batch_count > 0 else 0
        avg_clip_frac = total_clip_frac / batch_count if batch_count > 0 else 0
        avg_adv_std = total_adv_std / batch_count if batch_count > 0 else 0
        avg_adv_p95 = total_adv_p95 / batch_count if batch_count > 0 else 0
        avg_adv_max = total_adv_max / batch_count if batch_count > 0 else 0
        avg_adv_mean = total_adv_mean / batch_count if batch_count > 0 else 0

        avg_actor_loss = total_actor_loss / batch_count if batch_count > 0 else 0
        avg_critic_loss = total_critic_loss / batch_count if batch_count > 0 else 0
        avg_c = total_c / batch_count if batch_count > 0 else 0

        # ============================================================
        # [新增] 监控：在固定的全量 batch 上独立统计每个动作头的 NLL 与熵
        # 全程 no_grad，不参与反传，因此不会干扰现有的网络更新。
        # ============================================================
        if use_obs:
            monitor_input_all = obs_all
        else:
            monitor_input_all = states_all
        with torch.no_grad():
            values_all = self.critic(states_all)
            residual_all = returns_all - values_all
            advantage_all = residual_all / (torch.sqrt(self.c_sq) + 1e-8)
        monitor_metrics = self.actor.compute_marwil_monitor(monitor_input_all, actions_all, advantages=advantage_all)
        # 缓存到 agent 属性，便于训练脚本拉取写入 logger
        self.marwil_nll_cont = monitor_metrics['nll_cont']
        self.marwil_nll_cat = monitor_metrics['nll_cat']
        self.marwil_nll_bern = monitor_metrics['nll_bern']
        self.marwil_entropy_cont = monitor_metrics['entropy_cont']
        self.marwil_entropy_cat = monitor_metrics['entropy_cat']
        self.marwil_entropy_bern = monitor_metrics['entropy_bern']
        self.marwil_accuracy_cont = monitor_metrics['accuracy_cont']
        self.marwil_accuracy_cat = monitor_metrics['accuracy_cat']
        self.marwil_accuracy_bern = monitor_metrics['accuracy_bern']
        self.marwil_weight_mean = avg_weight_mean
        self.marwil_weight_max = avg_weight_max
        self.marwil_weight_min = avg_weight_min
        self.marwil_weight_clip_frac = avg_clip_frac
        self.marwil_adv_std = avg_adv_std
        self.marwil_adv_p95 = avg_adv_p95
        self.marwil_adv_max = avg_adv_max
        self.marwil_adv_mean = avg_adv_mean
        self.marwil_adv_positive_frac = monitor_metrics['adv_positive_frac']

        return avg_actor_loss, avg_critic_loss, avg_c
    
    def Stir(self, transition_dict, target_entropies, max_steps=50, lr=0.05):
        """
        策略搅拌：深拷贝网络并使用纯熵正则项更新，将策略分布推平到目标熵，
        同时尽可能保持原有动作的相对优先级排序不变。
        
        Args:
            transition_dict: 同 update 方法的输入，用于提供计算图所需的 states (和 mask)
            target_entropies: dict, 例如 {'cont': 2.0, 'cat': 1.5, 'bern': 0.6}
            max_steps: 最大允许的搅拌迭代次数
            lr: SGD 学习率 (不宜过大，否则容易导致离散动作排序翻转)
            
        Returns:
            stirred_state_dict: 搅拌后的独立网络参数字典，与原计算图解耦
        """
        # 1. 深度拷贝网络，彻底解耦计算图
        stirred_net = copy.deepcopy(self.actor)
        stirred_net.net.train() # 确保开启梯度计算

        # 2. 冻结特征提取层，仅放开动作头和 std 参数
        # 这是为了最大程度保护原始动作提取出的特征表达，防止由于大跨度更新彻底破坏策略逻辑
        for name, param in stirred_net.net.named_parameters():
            if 'fc_' in name or 'log_std' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # 3. 必须使用无动量的 SGD！
        # 如果使用 Adam，动量积攒会导致 Logits 大幅超调，直接破坏原有的动作优先级排序
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, stirred_net.net.parameters()), lr=lr)

        # 4. 解析输入状态 (复用 update 的接口逻辑)
        if isinstance(transition_dict['states'], np.ndarray):
            states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        else:
            states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        

        # 5. 执行搅拌循环
        for step in range(max_steps):
            outputs = stirred_net.net(states)
            
            loss = 0.0
            current_entropies = {}

            # --- Cont ---
            if 'cont' in self.actor.action_dims and self.actor.action_dims['cont'] > 0:
                mu, std = outputs['cont']
                dist = SquashedNormal(mu, std)
                # 平均到每个样本上
                ent_cont = dist.entropy().mean()
                current_entropies['cont'] = ent_cont.item()
                # 如果当前熵低于目标值，则加上负熵作为 Loss (最小化负熵 = 最大化熵)
                if ent_cont < target_entropies.get('cont', -float('inf')):
                    loss -= ent_cont 

            # --- Cat ---
            if 'cat' in self.actor.action_dims and sum(self.actor.action_dims['cat']) > 0:
                cat_probs_list = outputs['cat']
                ent_cat = 0.0
                for probs in cat_probs_list:
                    dist = Categorical(probs=probs)
                    ent_cat += dist.entropy().mean()
                current_entropies['cat'] = ent_cat.item()
                if ent_cat < target_entropies.get('cat', -float('inf')):
                    loss -= ent_cat

            # --- Bern ---
            # 跳过 Bernoulli 部分的搅拌，避免 mask_on 相关的复杂性

            # 6. 如果没有需要优化的项，直接退出
            if isinstance(loss, float):
                break

            # 7. 反向传播与搅拌更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 8. 更新后检查退出条件：所有指定的动作头都达到了目标熵则停止
            # 注意：必须在 step() 之后检查，避免更新后过冲
            all_met = True
            for key, target in target_entropies.items():
                if current_entropies.get(key, float('inf')) < target:
                    all_met = False
                    break
            if all_met:
                break

        # 8. 计算搅拌后的熵值
        with torch.no_grad():
            # 使用输入的状态计算熵值
            if isinstance(transition_dict['states'], np.ndarray):
                states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
            else:
                states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
            
            outputs = stirred_net.net(states)
            
            # 计算cat熵
            cat_entropy = 0.0
            if 'cat' in self.actor.action_dims and sum(self.actor.action_dims['cat']) > 0:
                cat_probs_list = outputs['cat']
                for probs in cat_probs_list:
                    dist = Categorical(probs=probs)
                    cat_entropy += dist.entropy().mean()
                cat_entropy = cat_entropy.item()
            
            # Bern 熵不计算（搅拌时已跳过）
            bern_entropy = 0.0
            
            # 计算cont熵
            cont_entropy = 0.0
            if 'cont' in self.actor.action_dims and self.actor.action_dims['cont'] > 0:
                mu, std = outputs['cont']
                dist = SquashedNormal(mu, std)
                cont_entropy = dist.entropy().mean().item()
        
        # 9. 转换为 CPU 张量并 clone，彻底切断与本次优化过程及 GPU 的羁绊
        # stirred_net 本身是 HybridActorWrapper 的深拷贝，直接取其 state_dict 即可与外部保存/加载保持一致
        stirred_state_dict = {k: v.cpu().clone() for k, v in stirred_net.state_dict().items()}
        
        # 返回搅拌后的状态字典和熵值信息
        entropy_info = {
            'cat_entropy': cat_entropy,
            'bern_entropy': bern_entropy,
            'cont_entropy': cont_entropy
        }
        
        return stirred_state_dict, entropy_info

    def ADPC_update(self, il_transition_dict, beta=1.0, batch_size=4096, alpha=1.0, c_v=1.0,
                    shuffled=1, chosen_quantile=0.2, no_bern=True, dark_side=1, actor_only=1, epochs=4, 
                    target_entropy_cat=None, ppo_grad_val=None):
        """
        Adversarial Demonstration Policy Correction (ADPC) 更新。
        仅筛选 advantage 最好或者最差的 分位数样本，
        以 label_smoothing=0.99 构造 one-cold 反向标签，对其做反向标签交叉熵训练，
        权重截断在 [0, 1] 防止放大。
        """
        # 用 alpha 调整学习率而非损失权重
        current_lr = self.actor_optimizer.param_groups[0]['lr']
        self.actor_optimizer.param_groups[0]['lr'] = current_lr * alpha

        if 'obs' in il_transition_dict and len(il_transition_dict['obs']) > 0:
            obs_all = torch.tensor(np.array(il_transition_dict['obs']), dtype=torch.float).to(self.device)
            use_obs = True
        else:
            use_obs = False

        states_all  = torch.tensor(np.array(il_transition_dict['states']),  dtype=torch.float).to(self.device)
        returns_all = torch.tensor(np.array(il_transition_dict['returns']),  dtype=torch.float).view(-1, 1).to(self.device)

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

        # --- 计算全量 advantage，筛选 最好或最差的 分位数 ---
        with torch.no_grad():
            values_all = self.critic(states_all)
            residual_all = returns_all - values_all
            # if not hasattr(self, 'c_sq'):
            self.c_sq = torch.tensor(1.0, device=self.device)
            c = torch.sqrt(self.c_sq)
            advantage_all = (residual_all / (c + 1e-8)).squeeze(-1)  # (N,)
            # 优势函数归一化（标准化）
            advantage_all = (advantage_all - advantage_all.mean()) / (advantage_all.std() + 1e-8)

            if dark_side:
                # 取最差的 chosen_quantile 样本，且归一化优势度 < 0
                threshold = torch.quantile(advantage_all, chosen_quantile) # 从低到高排位
                selected_mask = (advantage_all <= threshold) & (advantage_all < 0)
            else:
                # 取最好的 chosen_quantile 样本，且归一化优势度 > 0
                threshold = torch.quantile(advantage_all, 1.0 - chosen_quantile) # 从低到高排位
                selected_mask = (advantage_all >= threshold) & (advantage_all > 0)
            chosen_indices = selected_mask.nonzero(as_tuple=False).squeeze(-1)

        if chosen_indices.numel() == 0:
            return 0.0, 0.0

        # 筛选后的子集
        bad_states   = states_all[chosen_indices]
        bad_adv      = advantage_all[chosen_indices]
        bad_actions  = {k: v[chosen_indices] for k, v in actions_all.items()}
        if use_obs:
            bad_obs = obs_all[chosen_indices]
        else:
            bad_obs = bad_states

        # =====================================================================
        # [新增 1]：在开始 mini-batch 更新前，统一提取当前网络对这些样本的 log_probs 作为“旧策略锚点”
        # =====================================================================
        with torch.no_grad():
            anchor_log_probs, _, _, _, _ = self.actor.evaluate_actions(
                bad_obs, bad_actions, max_std=self.max_std, mask_on=0
            )
        
        total_actor_loss  = 0.0
        total_critic_loss = 0.0
        batch_count = 0

        sub_size = chosen_indices.size(0)
        sub_indices = np.arange(sub_size)

        for _ in range(epochs):
            # 目标熵检查：若当前 cat 熵已超出上限，直接终止所有后续 epoch
            if target_entropy_cat is not None:
                with torch.no_grad():
                    _, _, entropy_details_now, _, _ = self.actor.evaluate_actions(
                        bad_obs, bad_actions, max_std=self.max_std, mask_on=0
                    )
                    current_cat_entropy = entropy_details_now['cat'].mean().item()
                if current_cat_entropy > target_entropy_cat:
                    break

            if shuffled:
                np.random.shuffle(sub_indices)
            for start in range(0, sub_size, batch_size):
                end = min(start + batch_size, sub_size)
                bidx = sub_indices[start:end]

                s_batch   = bad_states[bidx]
                adv_batch = bad_adv[bidx]

                if use_obs:
                    actor_input_batch = bad_obs[bidx]
                else:
                    actor_input_batch = s_batch

                a_batch = {k: v[bidx] for k, v in bad_actions.items()}

                # 取出当前 batch 对应的旧策略锚点
                mb_anchor_log_probs = anchor_log_probs[bidx]

                with torch.no_grad():
                    # 权重 = advantage^2，与 beta 无关，恒非负，截断到 [0, 1]（不允许超过1放大）
                    # 非负权重确保：好/差样本的交叉熵误差恒为正向梯度，
                    # 避免负权重与互补交叉熵负负得正形成"接近差动作"的错误更新方向
                    raw_weights = torch.pow(adv_batch, 2) # .unsqueeze(-1) # ？？？之前为啥会有unsqueeze？？？
                    weights = torch.clamp(raw_weights, max=1.0)

                    # =====================================================================
                    # [新增 2]：计算当前网络的新 log_probs，并构造布尔掩码 (Mask)
                    # =====================================================================
                    new_log_probs, _, _, _, _ = self.actor.evaluate_actions(
                        actor_input_batch, a_batch, max_std=self.max_std, mask_on=0
                    )
                    ratio = torch.exp(new_log_probs - mb_anchor_log_probs)

                    # 如果新旧策略比超出 PPO 的容忍范围，mask 对应位置为 0.0，否则为 1.0
                    clip_mask = ((ratio >= 1.0 - self.eps) & (ratio <= 1.0 + self.eps)).float().squeeze(-1) # shape: (Batch, )

                # dark_side=1: 反向标签(0.99) + good_samples=0
                # dark_side=0: 正向模仿(0.01) + good_samples=1
                ls = 0.99 if dark_side else 0.01
                gs = 0 if dark_side else 1
                raw_il_loss = self.actor.compute_il_loss(
                    actor_input_batch,
                    a_batch,
                    label_smoothing=ls,
                    no_bern=no_bern,
                    good_samples=gs,
                    pre_training=1, # 0 原本只是负向交叉熵，但效果还不如构造one-cold分布
                ) # shape: (Batch, )
                
                v_pred = self.critic(s_batch)
                r_batch = returns_all[chosen_indices[bidx]]

                # =====================================================================
                # [新增 3]：将 clip_mask 乘入最终的 Loss。
                # 一旦触发截断，clip_mask=0，整个式子值为0，梯度在这一步被物理抹杀。
                # =====================================================================
                actor_loss = torch.mean(weights * raw_il_loss * clip_mask)
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                # 动态梯度裁剪：若传入了 ppo_grad_val，则上限使用它，否则使用默认的 self.actor_max_grad
                max_g = ppo_grad_val if ppo_grad_val is not None else self.actor_max_grad
                nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=max_g)
                self.actor_optimizer.step()

                
                
                critic_loss = F.mse_loss(v_pred, r_batch) * c_v

                if not actor_only:
                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.critic_max_grad)
                    self.critic_optimizer.step()

                total_actor_loss  += actor_loss.item()
                total_critic_loss += critic_loss.item()
                batch_count += 1

        avg_actor_loss  = total_actor_loss  / batch_count if batch_count > 0 else 0.0
        avg_critic_loss = total_critic_loss / batch_count if batch_count > 0 else 0.0

        # 还原学习率
        self.actor_optimizer.param_groups[0]['lr'] = current_lr

        return avg_actor_loss, avg_critic_loss