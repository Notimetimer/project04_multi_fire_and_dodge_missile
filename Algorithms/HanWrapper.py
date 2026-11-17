"""HanWrapper：按 Han et al. (2026) JPO 路线实现的异构 Transformer 策略-价值一体网络。

论文对照（Addressing Hybrid Action of BVR Engagement With Joint Policy Optimization, IEEE TAES 2026）：
- 式(9)：fixed(ego)/enm/ally 三类实体各自独立的嵌入矩阵；
- 式(11)-(13)：每层 9 个类型专属 Q/K/V 投影；
- 式(15)：阵亡/无效敌机在注意力 key 侧掩码；
- 式(16)：残差 + LayerNorm；实验部分为 2 层、E=32；
- 式(17)：ego 嵌入经 MLP 出 TIM 策略/Q 头（目标无关机动）；
- 式(18)：每个敌机嵌入经共享 MLP 出 TDM logits、开火 logit、TDM Q；
- 式(21)：TIM ∪ TDM 拼接为扁平 logits，带 mask 的单一 softmax（一次性采样，
  不做"先选敌机再选机动"的自回归分解）；
- 式(19)(20)(26)：开火 = 每敌机 sigmoid 命中概率，训练/测试均 Bernoulli 采样
  （不设开火阈值，与 PPO 测试口径一致）。
- 论文实验超参：2 层注意力 E=32；MLP 头 2 隐层×128；kI=kD=0.8；ξ=1.5；Ω̄1=0.8。

动作空间（与 Algorithms/ActorWithHANATTN.py / AMAACWrapper.py 逐位一致）：
- TIM（目标无关机动，全局共享组，11 个）：(vert,hor) ∈ {0,1,2}×{7,8,9} ∪ {(1,10),(1,11)}；
- TDM（目标相关机动，每个敌机一份，6 个）：vert 固定为平飞 1，hor ∈ 0~5；
- TIM/TDM 组合互不重叠，(vert,hor) ↔ idx 双向唯一；
- 扁平机动动作数 A = N_TIM + N_enm × N_TDM（2V2 时 11 + 2×6 = 23）；
- 开火门控与 ActorWithATTN/PPO 一致：可观测、40°锥、锁定、俯仰<15°、距离<100km、
  30s 冷却、弹药>0、敌机存活。
- 不使用 RWR 规避掩码：与 ActorWithHANATTN 同一理由——TIM 的 hor 索引 7~11 不在
  ActorWithATTN/HanWrapper 的规避放行集合 {2,3,4} 内，照搬会把 TIM 整体禁死。

已声明偏差（declared_deviations）：
1) 论文式(19)(26) 的开火目标恒为 argmax_j πF,j 的 j*，与机动动作解耦；本实现为与
   ActorWithATTN 的 enm_idx 语义对齐，TDM 动作的开火目标绑定其机动目标敌机，TIM 动作取 j*。
（开火门控不再是偏差：与 ActorWithATTN/PPO 完全一致的 8 条限制。）

接口约定与 Algorithms/ActorWithATTN.py 保持一致：
- get_action(agent_inputs, mask_inputs, rule_inputs, temperature=...) 返回
  {'enm_idx','vert_idx','hor_idx','fire',...} 张量字典，供 Combat worker 直接消费；
- rule_inputs 为 stack_field([take_some_of_rule_inputs(rule_input)]) 之后的 batched dict。
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Bernoulli, Categorical
import torch.nn.functional as F

_MASK_VAL = 1e8  # 掩码填充值（-inf 量级），与 ActorWithATTN 口径一致

VERT_DIM = 3
# 对比环境 maneuver_for_1_enm 支持的水平动作索引范围：[0, 12)
HOR_DIM_FULL = 12

# TIM（目标无关机动）：全局共享组，共 11 个 (vert, hor)
TIM_ACTIONS = (
    (0, 7), (0, 8), (0, 9),
    (1, 7), (1, 8), (1, 9),
    (2, 7), (2, 8), (2, 9),
    (1, 10), (1, 11),
)
N_TIM = len(TIM_ACTIONS)

# TDM（目标相关机动）：action_v 固定为平飞 1，hor 0~5，每个敌机一份，共 6 个
TDM_ACTIONS = tuple((1, h) for h in range(6))
N_TDM = len(TDM_ACTIONS)

assert not (set(TIM_ACTIONS) & set(TDM_ACTIONS)), 'TIM/TDM 动作组合必须互斥'
assert len(set(TIM_ACTIONS)) == N_TIM and len(set(TDM_ACTIONS)) == N_TDM, '动作组合必须唯一'

# 兼容旧名（ActorWithHANATTN / AMAACWrapper 只读引用这两组常量，保持单一事实来源）
JPO_TIM_ACTIONS = TIM_ACTIONS
N_JPO_TIM = N_TIM
JPO_TDM_ACTIONS = TDM_ACTIONS
N_JPO_TDM = N_TDM


@dataclass(frozen=True)
class NetworkOutput:
    """与预留抽象接口对齐的输出束：统一动作顺序下的策略 logits / 双 Q / 有效动作掩码。"""
    policy_logits: Any   # (B, A) 扁平机动 logits（已掩码）
    q1: Any              # (B, 2A) 第 1 套 Q（机动 × 开火）
    q2: Any              # (B, 2A) 第 2 套 Q
    valid_actions: Any   # (B, 2A) bool 联合动作掩码


# 异构自注意力层：fixed/enm/ally 三类实体各自持有独立的 Q/K/V 投影（每层 9 个矩阵，式 11-13），
# 拼接成统一序列做 scaled dot-product 自注意力，key 侧按 enm/ally 掩码（式 15），
# 输出按类型切回后做残差 + 类型专属 LayerNorm（式 16）。
class HetAttnLayer(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.scale = embed_dim ** -0.5
        self.w_q_f = nn.Linear(embed_dim, embed_dim)
        self.w_k_f = nn.Linear(embed_dim, embed_dim)
        self.w_v_f = nn.Linear(embed_dim, embed_dim)
        self.w_q_e = nn.Linear(embed_dim, embed_dim)
        self.w_k_e = nn.Linear(embed_dim, embed_dim)
        self.w_v_e = nn.Linear(embed_dim, embed_dim)
        self.w_q_a = nn.Linear(embed_dim, embed_dim)
        self.w_k_a = nn.Linear(embed_dim, embed_dim)
        self.w_v_a = nn.Linear(embed_dim, embed_dim)
        # 式(16)：LayerNorm(e + ζV)，按类型各一份
        self.ln_f = nn.LayerNorm(embed_dim)
        self.ln_e = nn.LayerNorm(embed_dim)
        self.ln_a = nn.LayerNorm(embed_dim)

    def forward(self, e_f, e_e, e_a, enm_mask=None, ally_mask=None):
        # e_f: (B,1,E), e_e: (B,Ne,E), e_a: (B,Na,E)；mask: (B,Ne)/(B,Na) bool
        Q = torch.cat((self.w_q_f(e_f), self.w_q_e(e_e), self.w_q_a(e_a)), dim=1)
        K = torch.cat((self.w_k_f(e_f), self.w_k_e(e_e), self.w_k_a(e_a)), dim=1)
        V = torch.cat((self.w_v_f(e_f), self.w_v_e(e_e), self.w_v_a(e_a)), dim=1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B,s,s)
        B, s, _ = Q.shape
        N_e, N_a = e_e.shape[1], e_a.shape[1]
        if enm_mask is not None or ally_mask is not None:
            key_valid = torch.ones(B, Q.shape[1], dtype=torch.bool, device=Q.device)
            if enm_mask is not None:
                key_valid[:, 1:1 + N_e] = enm_mask
            if ally_mask is not None:
                key_valid[:, 1 + N_e:1 + N_e + N_a] = ally_mask
            scores = scores.masked_fill(~key_valid.unsqueeze(1), -_MASK_VAL)

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)  # (B,s,E)

        # 残差 + LayerNorm（式 16）
        e_f_new = self.ln_f(e_f + out[:, 0:1, :])
        e_e_new = self.ln_e(e_e + out[:, 1:1 + N_e, :])
        e_a_new = self.ln_a(e_a + out[:, 1 + N_e:1 + N_e + N_a, :])
        return e_f_new, e_e_new, e_a_new


def _mlp(in_dim, hidden, out_dim):
    """论文实验配置：MLP 头含两个隐层、每层 hidden 节点（默认 128）。"""
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, out_dim))


class EntityBackbone(nn.Module):
    """式(9)(11)-(16)：三类实体独立嵌入 + n_layers 层异构自注意力。"""

    def __init__(self, fixed_dim, enm_dim, ally_dim, embed_dim=32, n_layers=2):
        super().__init__()
        self.fixed_embed = nn.Linear(fixed_dim, embed_dim)
        self.enm_embed = nn.Linear(enm_dim, embed_dim)
        self.ally_embed = nn.Linear(ally_dim, embed_dim)
        self.attn_layers = nn.ModuleList([HetAttnLayer(embed_dim) for _ in range(n_layers)])

    def forward(self, fixed, enm, ally, enm_mask, ally_mask):
        e_f = self.fixed_embed(fixed.unsqueeze(1))  # (B,1,E)
        e_e = self.enm_embed(enm)                   # (B,Ne,E)
        e_a = self.ally_embed(ally)                 # (B,Na,E)
        for layer in self.attn_layers:
            e_f, e_e, e_a = layer(e_f, e_e, e_a, enm_mask, ally_mask)
        return e_f.squeeze(1), e_e  # (B,E), (B,Ne,E)


class HanWrapper(nn.Module):
    """异构 Transformer 骨干 + 扁平混合动作头 + 双 Q + 独立开火头。

    已声明偏差（declared_deviations）：
    1) 策略与价值各自持有独立骨干、参数完全不共享（标准 SAC 口径）；论文式(35)
       为共享骨干一次反传。
    2) 双 Q 覆盖"机动 × 开火"联合动作（2A 个），开火进入 soft value 与策略损失，
       使 RL 梯度能更新开火头；论文式(19)(28) 把开火排除在 RL 梯度之外。
    3) 开火 BCE 监督只更新独立 fire_head（backbone 侧 detach），与 PPO 的
       supervised_fire_update 同口径；论文式(28) 未指定冻结范围。
    其余结构（式 9-18）与论文一致。
    """
    def __init__(self, fixed_dim, enm_shape, ally_shape,
                 embed_dim=32, n_layers=2, head_hidden=128, device='cpu', **_unused):
        super().__init__()
        self.fixed_dim = fixed_dim
        self.enm_shape = tuple(enm_shape)
        self.ally_shape = tuple(ally_shape)
        self.enm_num = self.enm_shape[0]
        self.ally_num = self.ally_shape[0]
        self.embed_dim = embed_dim
        self.n_maneuver = N_TIM + self.enm_num * N_TDM  # 扁平机动动作数 A

        # 式(9)-(16) 策略骨干：采样、部署、开火监督都只走这一条
        self.actor_backbone = EntityBackbone(fixed_dim, self.enm_shape[1], self.ally_shape[1],
                                             embed_dim, n_layers)
        # critic 独立骨干：与策略骨干不共享参数，避免策略梯度移动 critic 的输入表征
        self.critic_backbone = EntityBackbone(fixed_dim, self.enm_shape[1], self.ally_shape[1],
                                              embed_dim, n_layers)

        # 式(17)：ego 嵌入 → TIM 策略头（目标无关机动）
        self.tim_pi_head = _mlp(embed_dim, head_hidden, N_TIM)
        # 式(18)：敌机嵌入 → 共享 MLP → TDM logits
        self.opp_pi_head = _mlp(embed_dim, head_hidden, N_TDM)
        # 开火头独立一层（与 PPO 的 fire_head 同口径）：监督梯度不回流骨干，RL 梯度经联合 Q 更新
        self.fire_head = nn.Linear(embed_dim, 1)

        # 双 Q 覆盖"机动 × 开火"联合动作，输出顺序与 pi_joint 一致（a·2 + fire）：
        # TIM 头读 ego token，TDM 头逐敌机读其 token。
        self.tim_q1_head = _mlp(embed_dim, head_hidden, 2 * N_TIM)
        self.tim_q2_head = _mlp(embed_dim, head_hidden, 2 * N_TIM)
        self.opp_q1_head = _mlp(embed_dim, head_hidden, 2 * N_TDM)
        self.opp_q2_head = _mlp(embed_dim, head_hidden, 2 * N_TDM)
        self.n_joint = 2 * self.n_maneuver  # 联合动作数 |A_man × {0,1}|

        # (vert,hor) 正向动作表（非参数、不入 state_dict）
        self.register_buffer('tim_table', torch.as_tensor(np.asarray(TIM_ACTIONS), dtype=torch.long),
                             persistent=False)
        self.register_buffer('tdm_table', torch.as_tensor(np.asarray(TDM_ACTIONS), dtype=torch.long),
                             persistent=False)

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
            if m is None:
                return torch.ones(B, n, dtype=torch.bool, device=device)
            m = torch.as_tensor(np.asarray(m), dtype=torch.bool, device=device).reshape(B, n) \
                if np.asarray(m).ndim >= 1 else torch.ones(B, n, dtype=torch.bool, device=device)
            return m

        enm_mask = _mask('enm', self.enm_num)
        ally_mask = _mask('ally', self.ally_num)
        return fixed, enm, ally, enm_mask, ally_mask

    # ------------------------------------------------------------------ 参数分组（actor/critic 独立优化）
    def actor_parameters(self):
        """策略侧参数：策略骨干 + TIM/TDM 机动头 + 开火头（开火头由 RL 与 BCE 共同更新）。"""
        return list(self.actor_backbone.parameters()) + list(self.tim_pi_head.parameters()) \
            + list(self.opp_pi_head.parameters()) + list(self.fire_head.parameters())

    def fire_parameters(self):
        """开火监督专用参数：只有独立 fire_head（BCE 不回流骨干，与 PPO 口径一致）。"""
        return list(self.fire_head.parameters())

    def critic_parameters(self):
        """价值侧参数：critic 骨干 + TIM/TDM 双 Q 头。"""
        return list(self.critic_backbone.parameters()) \
            + list(self.tim_q1_head.parameters()) + list(self.tim_q2_head.parameters()) \
            + list(self.opp_q1_head.parameters()) + list(self.opp_q2_head.parameters())

    # ------------------------------------------------------------------ 手工禁火门控
    def _fire_gate(self, rule_inputs, enm_mask, B, device):
        """返回 (B,N) bool：True=允许开火。规则一条不少：
        可观测、40°锥、已锁定、仰角<15°、距离<100km、30s 冷却、弹药>0、敌机存活。"""
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
            & (edt < np.radians(15)) & (ed < 100e3)                       # (B,N)
        global_ok = (t_fly >= 30) & (ammo > 0)                            # (B,)
        gate = in_range & global_ok[:, None]
        gate = torch.as_tensor(gate, dtype=torch.bool, device=device) & enm_mask
        return gate

    # ------------------------------------------------------------------ 核心前向（可微，SAC 训练与采样共用）
    def _policy_heads(self, e_f, e_e, enm_mask, rule_inputs, B):
        """式(17)(18)：ego → TIM logits；敌机 → TDM logits 与开火 logit。

        扁平顺序：[TIM(N_TIM)] + [敌机0 的 TDM(N_TDM), 敌机1 的 TDM(N_TDM), ...]，
        TIM 槽位恒有效，敌机 j 的 TDM 槽位随该敌机有效掩码一并屏蔽（式 21 的 mask）。
        开火门控作为硬 mask 叠到逐敌机 logit 上；不使用 RWR 规避掩码（见文件头说明）。
        联合分布 pi_joint 以 a·2 + fire 索引：TIM 的开火目标为 j*=argmax πF,j，
        TDM 的开火目标为其绑定的敌机。
        """
        device = self.device
        # 式(17)：ego 嵌入 → TIM logits
        tim_logits = self.tim_pi_head(e_f)                       # (B, N_TIM)
        # 式(18)：敌机嵌入 → TDM logits；开火 logit 由独立 fire_head 产生
        tdm_logits = self.opp_pi_head(e_e)                       # (B, Ne, N_TDM)
        fire_logits_raw = self.fire_head(e_e).squeeze(-1)        # (B, Ne)

        # 机动掩码：TIM 恒有效；敌机 j 的 N_TDM 个 TDM 槽位随敌机存活/有效掩码
        tdm_mask = enm_mask.unsqueeze(-1).expand(
            B, self.enm_num, N_TDM).reshape(B, -1)                    # (B, Ne*N_TDM)
        maneuver_mask = torch.cat(
            (torch.ones(B, N_TIM, dtype=torch.bool, device=device), tdm_mask), dim=-1)  # (B, A)

        logits_flat = torch.cat((tim_logits, tdm_logits.reshape(B, -1)), dim=-1)  # (B, A)
        logits_flat_masked = logits_flat.masked_fill(~maneuver_mask, -_MASK_VAL)
        log_pi = logits_flat_masked - torch.logsumexp(logits_flat_masked, dim=-1, keepdim=True)
        pi = torch.softmax(logits_flat_masked, dim=-1)

        # 开火：手工门控作为硬 mask 叠加到 logit 上（sigmoid 后≈0）
        fire_gate = self._fire_gate(rule_inputs, enm_mask, B, device)
        fire_logits = fire_logits_raw.masked_fill(~fire_gate, -_MASK_VAL)
        p_fire = torch.sigmoid(fire_logits).masked_fill(~enm_mask, 0.0)   # (B, Ne)

        # 开火目标：TIM 取 j*=argmax πF,j；TDM 取该机动绑定的敌机
        j_star = p_fire.argmax(dim=-1)                                   # (B,)
        p_star = p_fire.gather(1, j_star.unsqueeze(1)).squeeze(1)        # (B,)

        # 联合分布 π(a, f) = π_man(a) · π_fire(target(a), f)，索引 a·2 + f
        pi_tim = pi[:, :N_TIM]                                           # (B, N_TIM)
        pi_tim_joint = torch.stack(
            [pi_tim * (1.0 - p_star).unsqueeze(-1),
             pi_tim * p_star.unsqueeze(-1)], dim=-1).reshape(B, N_TIM * 2)
        pi_tdm = pi[:, N_TIM:].reshape(B, self.enm_num, N_TDM)           # (B, Ne, N_TDM)
        pi_tdm_joint = torch.stack(
            [pi_tdm * (1.0 - p_fire).unsqueeze(-1),
             pi_tdm * p_fire.unsqueeze(-1)], dim=-1).reshape(B, -1)      # (B, Ne*N_TDM*2)
        pi_joint = torch.cat((pi_tim_joint, pi_tdm_joint), dim=-1)        # (B, 2A)

        out = {
            'logits_flat_masked': logits_flat_masked,  # (B, A)
            'log_pi': log_pi, 'pi': pi,                # (B, A)
            'pi_joint': pi_joint,                      # (B, 2A) 联合分布（Q/熵用）
            'maneuver_mask': maneuver_mask,            # (B, A)
            'fire_logits': fire_logits,                # (B, Ne) 已掩码
            'fire_logits_raw': fire_logits_raw,        # (B, Ne) 未掩码（供 BCE）
            'fire_prob': p_fire,                       # (B, Ne) 已掩码+死亡置零
            'fire_gate': fire_gate,
            'enm_mask': enm_mask,
        }
        return out

    def _forward_policy(self, agent_inputs, mask_inputs, rule_inputs):
        """策略部分前向：扁平机动 logits、掩码与开火门控；不算 Q 与分组熵。"""
        fixed, enm, ally, enm_mask, ally_mask = self._pack(agent_inputs, mask_inputs)
        e_f, e_e = self.actor_backbone(fixed, enm, ally, enm_mask, ally_mask)
        out = self._policy_heads(e_f, e_e, enm_mask, rule_inputs, fixed.shape[0])
        return out, e_f, e_e

    def forward_actor(self, agent_inputs, mask_inputs, rule_inputs):
        """部署/采样专用：只算策略分布与开火门控，不计算 Q 与分组熵。"""
        out, _, _ = self._forward_policy(agent_inputs, mask_inputs, rule_inputs)
        return out

    def forward_sac(self, agent_inputs, mask_inputs, rule_inputs):
        """返回扁平机动分布、双 Q、分组熵等全部训练所需量（带梯度）。

        扁平机动顺序：[TIM(11)] + [敌机0 的 TDM(6), 敌机1 的 TDM(6), ...]；
        联合（机动×开火）索引 = 机动 idx · 2 + fire，双 Q 宽度为 2A。
        分组：G0=TIM，G1..GN=各敌机 TDM；组质量、组内条件 log_prob、分组熵（式 29-30）。
        """
        fixed, enm, ally, enm_mask, ally_mask = self._pack(agent_inputs, mask_inputs)
        B = fixed.shape[0]
        e_f, e_e = self.actor_backbone(fixed, enm, ally, enm_mask, ally_mask)
        out = self._policy_heads(e_f, e_e, enm_mask, rule_inputs, B)
        pi, log_pi = out['pi'], out['log_pi']
        enm_mask = out['enm_mask']
        eps = 1e-8

        # critic 走独立骨干（参数不共享）：Q 只依赖 critic 自己的表征
        e_f_c, e_e_c = self.critic_backbone(fixed, enm, ally, enm_mask, ally_mask)
        q1 = torch.cat((self.tim_q1_head(e_f_c).reshape(B, N_TIM * 2),
                        self.opp_q1_head(e_e_c).reshape(B, -1)), dim=-1)   # (B, 2A)
        q2 = torch.cat((self.tim_q2_head(e_f_c).reshape(B, N_TIM * 2),
                        self.opp_q2_head(e_e_c).reshape(B, -1)), dim=-1)   # (B, 2A)

        # 分组（联合索引空间 a·2+f）：G0=TIM（N_TIM*2 个联合槽位），G1..GN=各敌机 TDM
        group_slices = [slice(0, N_TIM * 2)] + [
            slice(N_TIM * 2 + j * 2 * N_TDM, N_TIM * 2 + (j + 1) * 2 * N_TDM)
            for j in range(self.enm_num)]
        pr_g = torch.stack([out['pi_joint'][:, sl].sum(-1) for sl in group_slices], dim=-1)  # (B,1+Ne)
        pr_g_safe = pr_g.clamp(min=eps)
        log_pr_g = torch.log(pr_g_safe)
        log_cond = torch.cat(
            [torch.log(out['pi_joint'][:, sl].clamp_min(eps)) - log_pr_g[:, g:g + 1]
             for g, sl in enumerate(group_slices)], dim=-1)                        # (B,2A)

        # H^J 为全部 A 个机动槽位的熵（含 TIM/TDM，不含开火位）
        H_J = -(pi * log_pi).sum(-1)                                              # (B,)
        # H^I：TIM 组内条件熵（式 36）
        pi_tim = pi[:, :N_TIM]
        log_pi_tim = log_pi[:, :N_TIM]
        log_cond_tim = log_pi_tim - log_pr_g[:, 0:1]
        H_I = -(pi_tim * log_cond_tim).sum(-1) / pr_g_safe[:, 0]
        # H^D_j：各敌机 TDM 组内条件熵（式 37）
        pi_tdm = pi[:, N_TIM:].reshape(B, self.enm_num, N_TDM)
        log_pi_tdm = log_pi[:, N_TIM:].reshape(B, self.enm_num, N_TDM)
        pr_tdm = pi_tdm.sum(-1, keepdim=True)
        log_cond_tdm = log_pi_tdm - torch.log(pr_tdm.clamp_min(eps))
        H_Dj = -(pi_tdm * log_cond_tdm).sum(-1) / pr_tdm.squeeze(-1).clamp_min(eps)  # (B,Ne)
        # H^D 对存活敌机取均值（式 37 的 1/Ñ Σ 口径）
        enm_mask_f = enm_mask.to(torch.float32)
        H_D = (H_Dj * enm_mask_f).sum(-1) / enm_mask_f.sum(-1).clamp(min=1.0)

        out.update({
            'q1': q1, 'q2': q2,                        # (B, 2A)
            'log_cond': log_cond,                      # (B, 2A) 组内条件 log prob
            'pr_g': pr_g,                              # (B, 1+Ne)
            'H_J': H_J, 'H_I': H_I, 'H_D': H_D, 'H_Dj': H_Dj,
        })
        return out

    def forward(self, context: Any) -> NetworkOutput:
        """与预留抽象接口对齐：context = (agent_inputs, mask_inputs, rule_inputs)。"""
        agent_inputs, mask_inputs, rule_inputs = context
        out = self.forward_sac(agent_inputs, mask_inputs, rule_inputs)
        # 抽象接口的 valid_actions 与双 Q 同宽（联合动作：每个机动槽位 × 2 个开火位）
        joint_mask = out['maneuver_mask'].repeat_interleave(2, dim=-1)
        return NetworkOutput(out['logits_flat_masked'], out['q1'], out['q2'], joint_mask)

    # ------------------------------------------------------------------ 动作解码
    def get_action(self, agent_inputs, mask_inputs, rule_inputs, h=None,
                   temperature=1.0, deterministic=None):
        """一次扁平采样：TIM ∪ TDM 单一 softmax 采样机动；开火一律 Bernoulli 概率采样（不设阈值）。

        解码：flat_idx < N_TIM 走 TIM（目标无关，交互目标取开火概率最高的 j*）；
        否则走敌机 j 的 TDM 组，enm_idx = (flat_idx - N_TIM) // N_TDM。
        deterministic=None 默认 False：测试通路统一用 temperature=0.3 的 Categorical 采样，
        不再把 temperature<1 推断成 argmax；开火在训练/测试都用 Bernoulli 采样（与 PPO 测试一致）。
        deterministic=True 仅保留给需要严格 argmax 的调试用途。
        """
        if deterministic is None:
            deterministic = False
        with torch.no_grad():
            out = self.forward_actor(agent_inputs, mask_inputs, rule_inputs)
            logits = out['logits_flat_masked']

            if deterministic:
                flat_idx = logits.argmax(dim=-1)  # 扁平分布 argmax
            else:
                flat_idx = Categorical(logits=logits / max(temperature, 1e-3)).sample()

            # 解码扁平索引 → TIM 或（敌机 j 的 TDM）→ 查映射表得 (vert, hor)
            is_tim = flat_idx < N_TIM
            tim_idx = flat_idx.clamp(max=N_TIM - 1)
            r = (flat_idx - N_TIM).clamp(min=0)
            tdm_enm = torch.div(r, N_TDM, rounding_mode='floor')
            tdm_idx = r % N_TDM

            vert_idx = torch.where(is_tim, self.tim_table[tim_idx, 0], self.tdm_table[tdm_idx, 0])
            hor_idx = torch.where(is_tim, self.tim_table[tim_idx, 1], self.tdm_table[tdm_idx, 1])

            # 开火概率（已掩码）：TIM 与目标无关，取开火概率最高的敌机 j* 作为交互目标
            p_all = out['fire_prob'].masked_fill(~out['enm_mask'], 0.0)
            j_star = p_all.argmax(dim=-1)
            enm_idx = torch.where(is_tim, j_star, tdm_enm)

            # 对选中目标取开火概率；若该敌机被屏蔽（理论上不会），退化为 j*
            p_primary = p_all.gather(1, enm_idx.unsqueeze(1)).squeeze(1)
            need_fallback = ~out['enm_mask'].gather(1, enm_idx.unsqueeze(1)).squeeze(1)
            if need_fallback.any():
                enm_idx = torch.where(need_fallback, j_star, enm_idx)
                p_primary = torch.where(need_fallback,
                                        p_all.gather(1, j_star.unsqueeze(1)).squeeze(1),
                                        p_primary)

            # 开火一律按 Bernoulli 采样：开火头由 BCE 监督、概率校准到真实命中率，
            # 用固定阈值（如 0.8）会让部署时几乎不开火（导致测试恒 0 胜、0 击杀）。
            fire = Bernoulli(probs=p_primary.clamp(0.0, 1.0)).sample().to(torch.long)

        return {
            'enm_idx': enm_idx,
            'vert_idx': vert_idx,
            'hor_idx': hor_idx,
            'fire': fire,
            'fire_target': enm_idx,
            'flat_idx': flat_idx,
            'fire_prob_primary': float(p_primary.mean().item()) if p_primary.numel() else float('nan'),
        }

    # ------------------------------------------------------------------ 开火监督（式 28，梯度只到独立 fire_head）
    def evaluate_fire(self, agent_inputs, fire_target_idx, hit,
                      mask_inputs=None, reduction='mean'):
        """Lf：历史开火时刻观测 → 当前网络对该敌机的开火 logit → BCE-with-logits。

        监督样本都取自真实开火时刻（门控当时已通过），故用未掩码 raw logit；
        mask_inputs 缺省视为全部槽位有效（记录中不含 mask）。
        梯度切断：backbone 输出 detach，只有独立 fire_head 收 BCE 梯度
        （与 PPO 的 supervised_fire_update 冻结其余参数同口径）；
        开火头本身仍由 SAC 的联合 Q 策略梯度正常更新。
        """
        fixed, enm, ally, enm_mask, ally_mask = self._pack(agent_inputs, mask_inputs)
        e_f, e_e = self.actor_backbone(fixed, enm, ally, enm_mask, ally_mask)
        fire_logits = self.fire_head(e_e.detach()).squeeze(-1)  # (B, Ne) raw，梯度止于 fire_head

        B = fixed.shape[0]
        idx = torch.as_tensor(np.asarray(fire_target_idx), dtype=torch.long, device=self.device).reshape(B)
        hit = torch.as_tensor(np.asarray(hit), dtype=torch.float32, device=self.device).reshape(B)
        logit_star = fire_logits.gather(1, idx.unsqueeze(1)).squeeze(1)
        return F.binary_cross_entropy_with_logits(logit_star, hit, reduction=reduction)

    # ------------------------------------------------------------------ 存取
    def save(self, path='han_actor.pt'):
        torch.save(self.state_dict(), path)

    def load(self, path='han_actor.pt', device=None):
        self.load_state_dict(torch.load(path, map_location=device or self.device))
