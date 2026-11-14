"""AMAACWrapper：AMAAC（Chen et al., IEEE TAES 2025）注意力 actor-critic 的 PPO 对比实现。

论文对照（Scalable Cooperative Decision-Making in Multi-UAV Confrontations:
An Attention-Based Multiagent Actor–Critic Approach, IEEE TAES 2025）：
- §III-B-1 / 式(20)-(22) / 图4：观测重组为实体序列 o_ofe = (self, ally*, enm*)，
  存活/可观测掩码序列 M_ofe 作用于注意力 key 侧；
- 式(23)-(25)：OFE 注意力层——嵌入序列沿特征维均分为 N_ah=2 头，每头一个线性 W_k，
  Q=K=V=W_k(·) 权重共享（tied），掩码缩放点积注意力，拼接各头后残差 + LayerNorm；
  actor-critic 只取序列首元素 e_1（self token）；
- 式(28) / Table III：actor MLP (3 隐层 ×128, ReLU)；
- 式(29)-(32) / 图6 / Table III：集中式 critic——实体编码序列过 N_c=4 头 tied-QKV
  注意力层，MLP (3,128) 出价值；
- Table III：OFE 输出 16（本实现 embed_dim 默认 64，见偏差 2）、注意力层激活 GELU、
  MLP 隐层 ReLU；Table II 的 RL 超参（PPO clip 0.1、lr 5e-4、γ_A=0.995、λ1=0.98 等）
  属训练器配置，由启动器在 run_config.json 声明，不在本网络封装内固定。

动作空间（与 HanWrapper 完全同一套 TIM/TDM 构造，import 共享保证对比公平性）：
- TIM（目标无关机动，11 个）：(vert,hor) ∈ {0,1,2}×{7,8,9} ∪ {(1,10),(1,11)}；
- TDM（目标相关机动，每敌机 6 个）：vert=1 平飞，hor ∈ 0~5；
- 扁平机动动作数 A = N_TIM + N_enm × N_TDM（2V2 时 11+2×6=23），单一 masked softmax；
- 开火 = 每敌机 Bernoulli，手工禁火门控（可观测、40°锥、已锁定、15°仰角、<100km、
  30s 冷却、弹药、敌机存活）作为硬 mask 叠到 logit 上——与 HanWrapper/ActorWithATTN 同口径。

已声明偏差（declared_deviations，合并/写结论前须同步进 run_config.json）：
1. 论文观测元素统一 10 维零填充 + 共享 W_o；本仓库 fixed(28)/enm(11)/ally(9) 维度异构，
   改为类型专属线性嵌入到统一 embed_dim（与 HanWrapper 相同的输入适配），注意力机制本身与式(23)-(25)一致。
2. 论文 OFE 输出 16 维、告警嵌入 8 维；本实现 embed_dim 默认 64（与主方法容量对齐，可调）。
3. 论文 actor 输入仅 e_1^ofe（机动为目标无关连续量，开火由 MHP 模型门控）；本实现为适配
   TIM/TDM 离散混合动作空间，TIM 头读 self token，TDM/开火头逐敌机读其实体 token（同 HanWrapper）。
4. 无导弹告警嵌入 (e_d,a, e_t,a)：本仓库 agent_inputs 无来袭弹告警序列字段；未使用
   RWR 规避掩码（与 HanWrapper 一致），RWR 信息不进入网络。
5. 无 MHP 命中预测模型与阈值开火（论文 §III-A、算法1）：开火为本仓库统一的学习型
   Bernoulli 头 + 规则硬门控，与主方法/HanWrapper 完全相同。
6. 论文 critic 输入全体 UAV 观测序列、输出全体红机价值序列（CTDE）；本仓库训练管线
   critic 按智能体局部观测逐体调用（接口同主方法 critic_wrapper），故 AMAAC 注意力 critic
   作用于"本机观测实体序列"，价值从 self token 读出；且 get_value 接口不传掩码（同主方法），
   critic 注意力不加 alive mask。
7. 论文 actor/critic 输出层 Tanh（对应连续动作区间）；离散 logits 与价值无界，不施加。

接口约定与 Algorithms/ActorWithATTN.py 保持一致（可被 MAPPOHybrid 直接驱动）：
- get_action(agent_inputs, mask_inputs, rule_inputs, temperature=...) →
  {'enm_idx','vert_idx','hor_idx','fire','log_prob_enm','log_prob_act',...} 张量字典；
- evaluate_actions(agent_inputs, mask_inputs, rule_inputs, actions) →
  {'log_prob','log_prob_enm','log_prob_act','entropy','entropy_details',
   'fire_prob_max','fire_prob_min'}，actions 键与采集侧 Last_Actions 一致（vert/hor/fire/enm_idx）；
- evaluate_fire(agent_inputs, enm_idx, hit) → BCE-with-logits（带梯度），
  并暴露 actor.pilot.fire_head 供 MAPPOHybrid.supervised_fire_update 冻结/优化；
- rule_inputs 为 stack_field([take_some_of_rule_inputs(rule_input)]) 之后的 batched dict。
"""

import numpy as np
import torch
from torch import nn
from torch.distributions import Bernoulli, Categorical
import torch.nn.functional as F

# TIM/TDM 动作空间常量直接 import 自 HanWrapper（单一事实来源，保证对比方法动作空间逐位一致；
# 仅只读引用，不修改 HanWrapper）
from Algorithms.HanWrapper import (JPO_TIM_ACTIONS as TIM_ACTIONS,
                                   JPO_TDM_ACTIONS as TDM_ACTIONS,
                                   N_JPO_TIM as N_TIM, N_JPO_TDM as N_TDM)

_MASK_VAL = 1e8  # 掩码填充值（-inf 量级），与 ActorWithATTN/HanWrapper 口径一致

# (vert,hor) -> 组内 idx 的反向查找表（evaluate_actions 回放动作重编码用）
_VERT_DIM, _HOR_DIM_FULL = 3, 12
_TIM_LUT = np.full((_VERT_DIM, _HOR_DIM_FULL), -1, dtype=np.int64)
for _i, (_v, _h) in enumerate(TIM_ACTIONS):
    _TIM_LUT[_v, _h] = _i
_TDM_LUT = np.full((_VERT_DIM, _HOR_DIM_FULL), -1, dtype=np.int64)
for _i, (_v, _h) in enumerate(TDM_ACTIONS):
    _TDM_LUT[_v, _h] = _i


def _mlp3(in_dim, hidden):
    """论文 Table III：MLP (3, hidden)——3 个隐层、ReLU 激活；输出头单独定义。"""
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
    )


class AMAACAttentionLayer(nn.Module):
    """AMAAC 注意力层（式 23-25 / 式 29-31）：

    嵌入序列沿特征维均分为 n_heads 份；每头持有独立线性 W_k，且 Q=K=V=GELU(W_k(x_k))
    （同一投影权重共享，Table III 指定注意力层激活为 GELU）；掩码缩放点积注意力后
    拼接各头，残差 + LayerNorm。key 掩码 M_ofe/M_alive 屏蔽阵亡/无效实体。
    """

    def __init__(self, embed_dim, n_heads):
        super().__init__()
        assert embed_dim % n_heads == 0, 'embed_dim 必须能被 n_heads 整除'
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.w_k = nn.ModuleList(
            [nn.Linear(self.head_dim, self.head_dim) for _ in range(n_heads)])
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x, key_mask=None):
        # x: (B, S, E)；key_mask: (B, S) bool，True=有效（self 槽位恒有效，softmax 不退化）
        outs = []
        for k, w in enumerate(self.w_k):
            x_k = x[..., k * self.head_dim:(k + 1) * self.head_dim]  # div（式 23）
            qkv = F.gelu(w(x_k))                                     # tied Q=K=V
            scores = torch.matmul(qkv, qkv.transpose(-2, -1)) * self.scale  # (B,S,S)
            if key_mask is not None:
                scores = scores.masked_fill(~key_mask.unsqueeze(1), -_MASK_VAL)
            outs.append(torch.matmul(torch.softmax(scores, dim=-1), qkv))
        out = torch.cat(outs, dim=-1)                                # concat（式 25）
        return self.ln(x + out)                                      # 残差 + LayerNorm


class OFEEncoder(nn.Module):
    """观测-实体编码器（图 4，式 20-25）：

    序列顺序按论文 (self, teammates, opponents) = [fixed(1), ally(Na), enm(Ne)]；
    类型专属线性嵌入（declared_deviation 1）→ N_ah 头 AMAAC 注意力层 →
    首元素 e_1（self token）供 actor/critic 读出。
    """

    def __init__(self, fixed_dim, enm_shape, ally_shape, embed_dim=64,
                 n_heads=2, n_layers=1):
        super().__init__()
        self.enm_num = enm_shape[0]
        self.ally_num = ally_shape[0]
        self.embed_dim = embed_dim
        self.fixed_embed = nn.Linear(fixed_dim, embed_dim)
        self.enm_embed = nn.Linear(enm_shape[1], embed_dim)
        self.ally_embed = nn.Linear(ally_shape[1], embed_dim)
        self.layers = nn.ModuleList(
            [AMAACAttentionLayer(embed_dim, n_heads) for _ in range(n_layers)])

    def forward(self, fixed, enm, ally, enm_mask=None, ally_mask=None):
        # fixed: (B,F), enm: (B,Ne,De), ally: (B,Na,Da)；mask: (B,Ne)/(B,Na) bool
        B = fixed.shape[0]
        e_f = self.fixed_embed(fixed.unsqueeze(1))  # (B,1,E)
        e_a = self.ally_embed(ally)                 # (B,Na,E)
        e_e = self.enm_embed(enm)                   # (B,Ne,E)
        seq = torch.cat((e_f, e_a, e_e), dim=1)     # 论文顺序 (self, ally*, enm*)
        N_a, N_e = e_a.shape[1], e_e.shape[1]

        key_mask = None
        if enm_mask is not None or ally_mask is not None:
            key_mask = torch.ones(B, seq.shape[1], dtype=torch.bool, device=seq.device)
            if ally_mask is not None:
                key_mask[:, 1:1 + N_a] = ally_mask
            if enm_mask is not None:
                key_mask[:, 1 + N_a:1 + N_a + N_e] = enm_mask

        for layer in self.layers:
            seq = layer(seq, key_mask)

        e_self = seq[:, 0]                       # (B,E) 首元素读出
        e_ally = seq[:, 1:1 + N_a]               # (B,Na,E)
        e_enm = seq[:, 1 + N_a:1 + N_a + N_e]    # (B,Ne,E)
        return e_self, e_enm, e_ally


class AMAACPilot(nn.Module):
    """动作头束（论文式 28 的 actor MLP (3,128) 适配到 TIM/TDM 离散混合动作）：

    - TIM 头：self token → MLP(3×128) → 11 个 TIM logits；
    - TDM/开火头：逐敌机实体 token → 共享 MLP(3×128) → 6 个 TDM logits + 1 个开火 logit。
    fire_head 独立成层：MAPPOHybrid.supervised_fire_update 只优化 pilot.fire_head。
    """

    def __init__(self, embed_dim, enm_num, mlp_hidden=128):
        super().__init__()
        self.enm_num = enm_num
        self.tim_backbone = _mlp3(embed_dim, mlp_hidden)
        self.tim_head = nn.Linear(mlp_hidden, N_TIM)
        self.opp_backbone = _mlp3(embed_dim, mlp_hidden)   # 逐敌机共享
        self.tdm_head = nn.Linear(mlp_hidden, N_TDM)
        self.fire_head = nn.Linear(mlp_hidden, 1)

    def forward(self, e_self, e_enm):
        tim_logits = self.tim_head(self.tim_backbone(e_self))      # (B, N_TIM)
        opp_feat = self.opp_backbone(e_enm)                        # (B, Ne, H)
        tdm_logits = self.tdm_head(opp_feat)                       # (B, Ne, N_TDM)
        fire_logits = self.fire_head(opp_feat).squeeze(-1)         # (B, Ne)
        return {'tim': tim_logits, 'tdm': tdm_logits, 'fire': fire_logits}


class AMAACActorWrapper(nn.Module):
    """AMAAC 注意力 actor（OFE 编码器 + TIM/TDM 扁平混合动作头），PPO 训练接口。

    与 ActorWithATTN.ActorWrapper 同一对外契约，可被 MAPPOHybrid 直接驱动；
    动作空间与 HanWrapper 逐位一致（TIM 11 + TDM 6×N_enm）。
    """

    def __init__(self, fixed_dim, enm_shape, ally_shape,
                 embed_dim=64, ofe_heads=2, mlp_hidden=128,
                 device='cpu', ppo_with_bern=1, **_unused):
        super().__init__()
        self.fixed_dim = fixed_dim
        self.enm_shape = tuple(enm_shape)
        self.ally_shape = tuple(ally_shape)
        self.enm_num = self.enm_shape[0]
        self.ally_num = self.ally_shape[0]
        self.n_maneuver = N_TIM + self.enm_num * N_TDM
        # fire_included_in_rl=0 时 RL 的 log_prob/熵不含 fire 分支（口径同 ActorWithATTN）
        self.fire_included_in_rl = bool(ppo_with_bern)

        self.ofe = OFEEncoder(fixed_dim, self.enm_shape, self.ally_shape,
                              embed_dim, ofe_heads)
        self.pilot = AMAACPilot(embed_dim, self.enm_num, mlp_hidden)

        # (vert,hor)->组内 idx 反向查找表（非参数、不入 state_dict）
        self.register_buffer('_tim_lut', torch.as_tensor(_TIM_LUT), persistent=False)
        self.register_buffer('_tdm_lut', torch.as_tensor(_TDM_LUT), persistent=False)
        self.to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    # ------------------------------------------------------------------ 输入整理
    def _pack(self, agent_inputs, mask_inputs):
        """接受单样本（无 batch 维）或批量输入，统一成 (B, ...) 张量。"""
        device = self.device
        fixed = torch.as_tensor(np.asarray(agent_inputs['fixed']), dtype=torch.float32, device=device)
        enm = torch.as_tensor(np.asarray(agent_inputs['enm']), dtype=torch.float32, device=device)
        ally = torch.as_tensor(np.asarray(agent_inputs['ally']), dtype=torch.float32, device=device)
        if fixed.ndim == 1:
            fixed = fixed.unsqueeze(0)
        if enm.ndim == 2:
            enm = enm.unsqueeze(0)
        if ally.ndim == 2:
            ally = ally.unsqueeze(0)
        B = fixed.shape[0]

        def _mask(key, n):
            m = None if mask_inputs is None else mask_inputs.get(key)
            if m is None or np.asarray(m).size == 0:
                return torch.ones(B, n, dtype=torch.bool, device=device)
            m = torch.as_tensor(np.asarray(m), dtype=torch.bool, device=device)
            if m.ndim == 1:
                m = m.unsqueeze(0).expand(B, -1)
            return m.reshape(B, n)

        return fixed, enm, ally, _mask('enm', self.enm_num), _mask('ally', self.ally_num)

    # ------------------------------------------------------------------ 手工禁火门控
    # 与 HanWrapper._fire_gate 完全相同的手工禁火规则（一条不少，阈值一致），
    # 保证对比方法间开火约束口径统一；拷贝而非 import 以保持 baseline 间实现独立。
    def _fire_gate(self, rule_inputs, enm_mask, B, device):
        """返回 (B,Ne) bool：True=允许开火。可观测、40°锥、已锁定、仰角<15°、
        距离<100km、30s 冷却、弹药>0、敌机存活。"""
        if not rule_inputs:
            return torch.ones(B, self.enm_num, dtype=torch.bool, device=device) & enm_mask

        def _arr(x, default):
            return np.asarray(x if x is not None else default)

        enm_infos = rule_inputs.get('enm_infos', {}) or {}
        weapon = rule_inputs.get('weapon_infos', {}) or {}
        eo = _arr(enm_infos.get('enm_observable'), np.zeros((B, self.enm_num))).reshape(B, self.enm_num)
        cdp = _arr(enm_infos.get('cos_enm_delta_psi'), np.full((B, self.enm_num), -1.0)).reshape(B, self.enm_num)
        el = _arr(enm_infos.get('enm_locked'), np.zeros((B, self.enm_num))).reshape(B, self.enm_num)
        edt = _arr(enm_infos.get('enm_delta_theta'), np.full((B, self.enm_num), np.pi)).reshape(B, self.enm_num)
        ed = _arr(enm_infos.get('enm_dist'), np.full((B, self.enm_num), 1e9)).reshape(B, self.enm_num)
        t_fly = _arr(weapon.get('missile_t_fly'), np.full(B, 1e9)).reshape(B)
        ammo = _arr(rule_inputs.get('ammo'), np.zeros(B)).reshape(B)

        in_range = (eo > 0) & (cdp > np.cos(np.radians(40))) & (el > 0) \
            & (edt < np.radians(15)) & (ed < 100e3)                       # (B,Ne)
        global_ok = (t_fly >= 30) & (ammo > 0)                            # (B,)
        gate = in_range & global_ok[:, None]
        gate = torch.as_tensor(gate, dtype=torch.bool, device=device) & enm_mask
        return gate

    # ------------------------------------------------------------------ 核心前向
    def _policy_logits(self, agent_inputs, mask_inputs, rule_inputs):
        """OFE 编码 → TIM/TDM 扁平机动 logits（masked）与开火 logits（raw/masked）。

        扁平顺序：[TIM(11)] + [敌机0 的 TDM(6), 敌机1 的 TDM(6), ...]（同 HanWrapper）。
        """
        device = self.device
        fixed, enm, ally, enm_mask, ally_mask = self._pack(agent_inputs, mask_inputs)
        B = fixed.shape[0]

        e_self, e_enm, _ = self.ofe(fixed, enm, ally, enm_mask, ally_mask)
        heads = self.pilot(e_self, e_enm)

        # 机动掩码：TIM 恒有效；敌机 j 的 N_TDM 个 TDM 槽位随敌机存活/有效掩码
        tdm_mask = enm_mask.unsqueeze(-1).expand(
            B, self.enm_num, N_TDM).reshape(B, -1)                        # (B, Ne*N_TDM)
        maneuver_mask = torch.cat(
            (torch.ones(B, N_TIM, dtype=torch.bool, device=device), tdm_mask), dim=-1)

        logits_flat = torch.cat(
            (heads['tim'], heads['tdm'].reshape(B, -1)), dim=-1)          # (B, A)
        logits_flat_masked = logits_flat.masked_fill(~maneuver_mask, -_MASK_VAL)

        # 开火：手工门控作为硬 mask 叠到 logit 上（sigmoid 后≈0）
        fire_gate = self._fire_gate(rule_inputs, enm_mask, B, device)
        fire_logits_masked = heads['fire'].masked_fill(~fire_gate, -_MASK_VAL)

        return {
            'logits_flat_masked': logits_flat_masked,
            'maneuver_mask': maneuver_mask,
            'fire_logits_masked': fire_logits_masked,
            'fire_logits_raw': heads['fire'],
            'fire_gate': fire_gate,
            'enm_mask': enm_mask,
        }

    def _decode_flat(self, flat_idx, fire_probs, enm_mask):
        """扁平索引 → (vert, hor, enm_idx)。

        TIM：目标无关，交互目标取开火概率最高的敌机 j*（与 HanWrapper 同口径）；
        TDM：目标绑定所属敌机组。(vert,hor) 查 TIM/TDM 映射表。
        """
        device = flat_idx.device
        tim_table = torch.as_tensor(TIM_ACTIONS, dtype=torch.long, device=device)
        tdm_table = torch.as_tensor(TDM_ACTIONS, dtype=torch.long, device=device)
        is_tim = flat_idx < N_TIM
        tim_idx = flat_idx.clamp(max=N_TIM - 1)
        r = (flat_idx - N_TIM).clamp(min=0)
        tdm_enm = torch.div(r, N_TDM, rounding_mode='floor')
        tdm_idx = r % N_TDM

        p_all = fire_probs.masked_fill(~enm_mask, 0.0)
        j_star = p_all.argmax(dim=-1)

        enm_idx = torch.where(is_tim, j_star, tdm_enm)
        vert_idx = torch.where(is_tim, tim_table[tim_idx, 0], tdm_table[tdm_idx, 0])
        hor_idx = torch.where(is_tim, tim_table[tim_idx, 1], tdm_table[tdm_idx, 1])
        return enm_idx, vert_idx, hor_idx

    def _encode_flat(self, vert_idx, hor_idx, enm_idx):
        """回放动作 (vert,hor,enm_idx) → 扁平机动索引（evaluate_actions 用）。

        (vert,hor) ∈ TIM 表 → TIM 组内 idx；∈ TDM 表 → N_TIM + enm_idx*N_TDM + 组内 idx。
        两组互斥，映射唯一；越界组合直接报错（说明回放数据不是本动作空间采的）。
        """
        tim_hit = self._tim_lut[vert_idx, hor_idx]
        tdm_hit = self._tdm_lut[vert_idx, hor_idx]
        is_tim = tim_hit >= 0
        if ((~is_tim) & (tdm_hit < 0)).any():
            raise ValueError('回放动作 (vert,hor) 不在 TIM/TDM 动作空间内，无法重编码')
        flat = torch.where(is_tim, tim_hit,
                           N_TIM + enm_idx * N_TDM + tdm_hit.clamp(min=0))
        return flat

    # ------------------------------------------------------------------ 采样接口
    def get_action(self, agent_inputs, mask_inputs, rule_inputs, h=None,
                   temperature=1.0, enm_idx=None):
        """扁平 masked softmax 采样机动 + 目标处 Bernoulli 采样开火（PPO 行为策略）。

        返回键与 ActorWithATTN.get_action 对齐，供 Combat worker / VsBaseline_MA 直接消费；
        log_prob_enm = log π_flat（含 TDM 目标绑定），log_prob_act = log p_fire（可选剔除），
        二者之和为联合动作 log_prob（与 evaluate_actions['log_prob'] 同一语义）。
        enm_idx：外部指派目标槽位（仅部署/测试），限制 TDM 组到该敌机；训练必须为 None。
        """
        with torch.no_grad():
            out = self._policy_logits(agent_inputs, mask_inputs, rule_inputs)
            logits = out['logits_flat_masked']

            if enm_idx is not None:
                enm_idx = torch.as_tensor(enm_idx, dtype=torch.long,
                                          device=self.device).reshape(-1)
                B, A = logits.shape
                restrict = torch.ones(B, A, dtype=torch.bool, device=self.device)
                for j in range(self.enm_num):
                    sl = slice(N_TIM + j * N_TDM, N_TIM + (j + 1) * N_TDM)
                    restrict[:, sl] = (enm_idx == j).unsqueeze(-1)
                logits = logits.masked_fill(~restrict, -_MASK_VAL)

            temp = max(temperature, 1e-3)
            maneuver_dist = Categorical(logits=logits / temp)
            flat_idx = maneuver_dist.sample()
            log_prob_maneuver = maneuver_dist.log_prob(flat_idx)

            fire_probs = torch.sigmoid(out['fire_logits_masked'])
            enm_idx, vert_idx, hor_idx = self._decode_flat(
                flat_idx, fire_probs, out['enm_mask'])

            # 对选中目标的开火分布（gate 已掩码到 logit 上，被禁火时 p≈0）
            fire_logit_sel = out['fire_logits_masked'].gather(
                1, enm_idx.unsqueeze(1)).squeeze(1)
            fire_dist = Bernoulli(logits=fire_logit_sel / temp)
            fire = fire_dist.sample().to(torch.long)
            log_prob_fire = fire_dist.log_prob(fire.to(torch.float32)) \
                if self.fire_included_in_rl else torch.zeros_like(log_prob_maneuver)

        return {
            'enm_idx': enm_idx,
            'vert_idx': vert_idx,
            'hor_idx': hor_idx,
            'fire': fire,
            'fire_mask': out['fire_gate'].to(torch.long),
            'flat_idx': flat_idx,
            'log_prob_enm': log_prob_maneuver,
            'log_prob_act': log_prob_fire,
        }

    # ------------------------------------------------------------------ PPO 评估接口
    def evaluate_actions(self, agent_inputs, mask_inputs, rule_inputs, actions,
                         h=None, max_std=None):
        """重放批量观测/动作，重算新策略的联合 log_prob 与策略熵（供 PPO 新旧策略比）。

        actions 键与采集侧 Last_Actions 一致：vert / hor / fire / enm_idx；
        (vert,hor,enm_idx) 重编码为扁平机动索引后取 log π_flat。
        """
        device = self.device
        out = self._policy_logits(agent_inputs, mask_inputs, rule_inputs)

        vert_idx = torch.as_tensor(np.asarray(actions['vert']), dtype=torch.long, device=device).reshape(-1)
        hor_idx = torch.as_tensor(np.asarray(actions['hor']), dtype=torch.long, device=device).reshape(-1)
        enm_idx = torch.as_tensor(np.asarray(actions['enm_idx']), dtype=torch.long, device=device).reshape(-1)
        fire = torch.as_tensor(np.asarray(actions['fire']), dtype=torch.long, device=device).reshape(-1)

        flat_idx = self._encode_flat(vert_idx, hor_idx, enm_idx)
        maneuver_dist = Categorical(logits=out['logits_flat_masked'])
        log_prob_enm = maneuver_dist.log_prob(flat_idx)
        ent_maneuver = maneuver_dist.entropy()

        # 开火分支：取回放时选中目标的 logit（gate 掩码后），与采样端同一分布语义
        fire_logit_sel = out['fire_logits_masked'].gather(1, enm_idx.unsqueeze(1)).squeeze(1)
        fire_dist = Bernoulli(logits=fire_logit_sel)
        log_prob_fire = fire_dist.log_prob(fire.to(torch.float32)) \
            if self.fire_included_in_rl else torch.zeros_like(log_prob_enm)
        fire_gate_sel = out['fire_gate'].gather(1, enm_idx.unsqueeze(1)).squeeze(1)
        ent_fire = fire_dist.entropy() * fire_gate_sel.to(torch.float32) \
            if self.fire_included_in_rl else torch.zeros_like(ent_maneuver)

        log_prob_act = log_prob_fire
        entropy = ent_maneuver + ent_fire

        # 未被禁火样本的开火概率极值（仅记录，MAPPOHybrid 按 mini_batch 收集）
        with torch.no_grad():
            fire_prob = torch.sigmoid(out['fire_logits_masked'])
            allowed = out['fire_gate']
            if allowed.any():
                fire_prob_max = fire_prob[allowed].max().item()
                fire_prob_min = fire_prob[allowed].min().item()
            else:
                fire_prob_max = float('nan')
                fire_prob_min = float('nan')

        return {
            'log_prob': log_prob_enm + log_prob_act,
            'log_prob_enm': log_prob_enm,
            'log_prob_act': log_prob_act,
            'entropy': entropy,
            'entropy_details': {
                'maneuver': ent_maneuver,
                'fire': ent_fire,
            },
            'fire_prob_max': fire_prob_max,
            'fire_prob_min': fire_prob_min,
        }

    def evaluate_fire(self, agent_inputs, enm_idx, hit,
                      mask_inputs=None, reduction='mean'):
        """有监督开火头：按历史观测 + 当时选中的敌机，重算开火概率并与命中标签取交叉熵。

        返回带梯度的 BCE-with-logits；与 HanWrapper 同口径：对原始（未门控）fire logit
        计算——监督样本取自真实开火时刻，此时门控 delta 恒为 0。
        调用方（MAPPOHybrid.supervised_fire_update）只放行 pilot.fire_head 的梯度。
        """
        device = self.device
        fixed, enm, ally, enm_mask, ally_mask = self._pack(agent_inputs, mask_inputs)
        B = fixed.shape[0]
        enm_idx = torch.as_tensor(np.asarray(enm_idx), dtype=torch.long, device=device).reshape(-1)
        hit = torch.as_tensor(np.asarray(hit), dtype=torch.float32, device=device).reshape(-1)
        if enm_idx.numel() != B or hit.numel() != B:
            raise ValueError('enm_idx/hit 长度必须等于观测 batch')

        _, e_enm, _ = self.ofe(fixed, enm, ally, enm_mask, ally_mask)
        opp_feat = self.pilot.opp_backbone(e_enm)                    # (B,Ne,H)
        feat_sel = opp_feat.gather(
            1, enm_idx.view(-1, 1, 1).expand(-1, 1, opp_feat.shape[-1])).squeeze(1)
        fire_logit = self.pilot.fire_head(feat_sel).squeeze(-1)      # (B,)
        return F.binary_cross_entropy_with_logits(fire_logit, hit, reduction=reduction)

    # ------------------------------------------------------------------ 存档
    def save(self, path='actor.pt'):
        torch.save({
            'ofe': self.ofe.state_dict(),
            'pilot': self.pilot.state_dict(),
        }, path)

    def load(self, path='actor.pt', device=None):
        target = device or self.device
        checkpoint = torch.load(path, map_location=target)
        self.ofe.load_state_dict(checkpoint['ofe'])
        self.pilot.load_state_dict(checkpoint['pilot'])


class AMAACCriticWrapper(nn.Module):
    """AMAAC 集中式注意力 critic（式 29-32 / 图 6）的仓库接口适配。

    实体编码序列过 N_c=4 头 tied-QKV 注意力层，self token 读出后 MLP(3×128) → 标量价值。
    接口与 CriticWrapper.critic_wrapper 一致：get_value(agent_inputs) -> {'value': (B,)}；
    按本仓库管线逐智能体局部观测调用（declared_deviation 6），不接收掩码。
    """

    def __init__(self, fixed_dim, enm_shape, ally_shape,
                 embed_dim=64, num_heads=4, mlp_hidden=128, device='cpu', **_unused):
        super().__init__()
        self.fixed_dim = fixed_dim
        self.enm_shape = tuple(enm_shape)
        self.ally_shape = tuple(ally_shape)
        self.encoder = OFEEncoder(fixed_dim, self.enm_shape, self.ally_shape,
                                  embed_dim, num_heads)
        self.backbone = _mlp3(embed_dim, mlp_hidden)
        self.value_head = nn.Linear(mlp_hidden, 1)
        self.to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    def get_value(self, agent_inputs, h=None):
        device = self.device
        fixed = torch.as_tensor(np.asarray(agent_inputs['fixed']), dtype=torch.float32, device=device)
        enm = torch.as_tensor(np.asarray(agent_inputs['enm']), dtype=torch.float32, device=device)
        ally = torch.as_tensor(np.asarray(agent_inputs['ally']), dtype=torch.float32, device=device)
        if fixed.ndim == 1:
            fixed = fixed.unsqueeze(0)
        if enm.ndim == 2:
            enm = enm.unsqueeze(0)
        if ally.ndim == 2:
            ally = ally.unsqueeze(0)

        e_self, _, _ = self.encoder(fixed, enm, ally)
        value = self.value_head(self.backbone(e_self)).squeeze(-1)   # (B,)
        return {'value': value}

    def save(self, path='critic.pt'):
        torch.save({
            'encoder': self.encoder.state_dict(),
            'backbone': self.backbone.state_dict(),
            'value_head': self.value_head.state_dict(),
        }, path)

    def load(self, path='critic.pt', device=None):
        target = device or self.device
        checkpoint = torch.load(path, map_location=target)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.backbone.load_state_dict(checkpoint['backbone'])
        self.value_head.load_state_dict(checkpoint['value_head'])
