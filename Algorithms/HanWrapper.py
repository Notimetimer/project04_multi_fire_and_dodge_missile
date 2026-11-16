"""HanWrapper：按 Han et al. (2026) JPO 路线实现的异构 Transformer 策略-价值一体网络。

【动作空间变更】TIM 已删除：机动动作与 ActorWithATTN 同步为 vert(3) × hor(7) = 21 个
TDM/敌机，单一 masked softmax；开火头与手工禁火门控保持不变。告警规避同样与
ActorWithATTN._warning 同口径（warning=1 时 hor 只开放三线/置尾/九线的对应子集，
vert 不受限）。下文"TIM/TDM"的原始
描述保留作历史对照，现行动作空间以本文件常量 TDM_ACTIONS / N_TDM 为准。

论文对照（Addressing Hybrid Action of BVR Engagement With Joint Policy Optimization, IEEE TAES 2026）：
- 式(9)：fixed(ego)/enm/ally 三类实体各自独立的嵌入矩阵；
- 式(11)-(13)：每层 9 个类型专属 Q/K/V 投影；
- 式(15)：阵亡/无效敌机在注意力 key 侧掩码；
- 式(16)：残差 + LayerNorm；实验部分为 2 层、E=32；
- 式(17)-(18)：ego 嵌入出 TIM 策略/Q 头；每个敌机嵌入经共享 MLP 出 TDM logits、开火 logit、TDM Q；
- 式(21)：TIM ∪ TDM 拼接为扁平 logits，带 mask 的单一 softmax；
- 式(19)(20)(26)：开火 = 每敌机 sigmoid 命中概率，训练/测试均 Bernoulli 采样
  （不设开火阈值，与 PPO 测试口径一致）。
- 论文实验超参：2 层注意力 E=32；MLP 头 2 隐层×128；kI=kD=0.8；ξ=1.5；Ω̄1=0.8。

已声明偏差（declared_deviations）：
1) 论文式(19)(26) 的开火目标恒为 argmax_j πF,j 的 j*，与机动动作解耦；本实现为与
   ActorWithATTN 的 enm_idx 语义对齐，TDM 动作的开火目标绑定其机动目标敌机，TIM 动作仍取 j*。
（开火门控不再是偏差：已恢复为与 ActorWithATTN/PPO 完全一致的 8 条限制。）

对比用动作空间（idx ↔ (vert, hor) 双向映射表与合法 mask 表均在本文件内部维护）：
- TIM（目标无关机动，全局共享组，11 个）：(vert,hor) ∈ {0,1,2}×{7,8,9} ∪ {(1,10),(1,11)}；
- TDM（目标相关机动，每个敌机一份，6 个）：vert 固定为平飞 1，hor ∈ 0~5；
- TIM/TDM 的组合互不重叠，(vert,hor) ↔ idx 双向唯一；
- 扁平机动动作数 A = N_TIM + N_enm × N_TDM（2V2 时 11 + 2×6 = 23）。
- 开火门控与 ActorWithATTN/PPO 一致：可观测、40°锥、锁定、俯仰<15°、距离<100km、
  30s 冷却、弹药>0、敌机存活。

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
# 与 ActorWithATTN 同步的水平维度：0追 1左crank 2三线 3置尾 4九线 5右crank 6回中心
HOR_DIM = 7

# TDM（目标相关机动）：vert 0~2 × hor 0~6，每个敌机一份，共 21 个
TDM_ACTIONS = tuple((v, h) for v in range(VERT_DIM) for h in range(HOR_DIM))
N_TDM = len(TDM_ACTIONS)

# 告警规避掩码（与 Algorithms/ActorWithATTN.py 的 _EVADE_HOR_MASK / _warning 完全同一口径）：
# warning=1 时按 sin_threat_delta_psi 的符号只放行部分水平动作，vert 三个方向不限制。
# hor 顺序：0 追踪 1 左crank 2 三线 3 置尾 4 九线 5 右crank 6 回中心
HOR_EVADE_ALLOW = {1: (2, 3), 0: (2, 3, 4), -1: (3, 4)}

# 以下 JPO 原始动作空间常量仅供对比算法 Algorithms/AMAACWrapper.py 只读引用
# （TIM 11 + 每敌机 TDM 6）；HanSAC 网络已删除 TIM/TDM 原始动作空间，不再使用它们。
# TIM（目标无关机动）：全局共享组，共 11 个 (vert, hor)
JPO_TIM_ACTIONS = (
    (0, 7), (0, 8), (0, 9),
    (1, 7), (1, 8), (1, 9),
    (2, 7), (2, 8), (2, 9),
    (1, 10), (1, 11),
)
N_JPO_TIM = len(JPO_TIM_ACTIONS)

# TDM（目标相关机动）：action_v 固定为平飞 1，hor 0~5，每个敌机一份，共 6 个
JPO_TDM_ACTIONS = tuple((1, h) for h in range(6))
N_JPO_TDM = len(JPO_TDM_ACTIONS)

assert len(set(TDM_ACTIONS)) == N_TDM, 'TDM 动作组合必须唯一'



@dataclass(frozen=True)
class NetworkOutput:
    """与预留抽象接口对齐的输出束：统一动作顺序下的策略 logits / 双 Q / 有效动作掩码。"""
    policy_logits: Any   # (B, A) 扁平机动 logits（已掩码）
    q1: Any              # (B, A) 第 1 套 Q
    q2: Any              # (B, A) 第 2 套 Q
    valid_actions: Any   # (B, A) bool 机动动作掩码


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
    2) 双 Q 覆盖"机动 × 开火"联合动作（每敌机 2·N_TDM 个），开火进入 soft value 与
       策略损失，使 RL 梯度能更新开火头；论文式(19)(28) 把开火排除在 RL 梯度之外。
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
        self.n_maneuver = self.enm_num * N_TDM  # 扁平机动动作数 A

        # 式(9)-(16) 策略骨干：采样、部署、开火监督都只走这一条
        self.actor_backbone = EntityBackbone(fixed_dim, self.enm_shape[1], self.ally_shape[1],
                                             embed_dim, n_layers)
        # critic 独立骨干：与策略骨干不共享参数，避免策略梯度移动 critic 的输入表征
        self.critic_backbone = EntityBackbone(fixed_dim, self.enm_shape[1], self.ally_shape[1],
                                              embed_dim, n_layers)

        # 式(18)：敌机嵌入 → 共享 MLP → TDM logits；双 Q 覆盖"机动 × 开火"联合动作
        self.opp_pi_head = _mlp(embed_dim, head_hidden, N_TDM)
        # 开火头独立一层（与 PPO 的 fire_head 同口径）：监督梯度不回流骨干，RL 梯度经联合 Q 更新
        self.fire_head = nn.Linear(embed_dim, 1)
        self.opp_q1_head = _mlp(embed_dim, head_hidden, 2 * N_TDM)
        self.opp_q2_head = _mlp(embed_dim, head_hidden, 2 * N_TDM)
        self.n_joint = self.enm_num * 2 * N_TDM  # 联合动作数 |A_man × {0,1}|

        # TDM 的 hor 索引：TDM_ACTIONS 按 v 外层、h 内层排列，故 hor = tdm_idx % HOR_DIM。
        # 告警规避掩码要靠它把 hor 维的允许集合铺到 21 个 TDM 槽位上。
        self.register_buffer('tdm_hor_idx', torch.arange(N_TDM, dtype=torch.long) % HOR_DIM,
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
        """策略侧参数：策略骨干 + 机动头 + 开火头（开火头由 RL 与 BCE 共同更新）。"""
        return list(self.actor_backbone.parameters()) + list(self.opp_pi_head.parameters()) \
            + list(self.fire_head.parameters())

    def fire_parameters(self):
        """开火监督专用参数：只有独立 fire_head（BCE 不回流骨干，与 PPO 口径一致）。"""
        return list(self.fire_head.parameters())

    def critic_parameters(self):
        """价值侧参数：critic 骨干 + 双 Q 头。"""
        return list(self.critic_backbone.parameters()) \
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

    # ------------------------------------------------------------------ 告警规避掩码（与 PPO 同口径）
    def _warning_mask(self, rule_inputs, enm_mask, B, device):
        """返回 (B, Ne, N_TDM) bool：True=该敌机槽位下的这个 TDM 允许被选中。

        与 ActorWithATTN._warning / _EVADE_HOR_MASK 完全同一口径：
        warning=1 时水平动作只允许 {2 三线, 3 置尾, 4 九线} 中与 sin_threat_delta_psi
        符号对应的子集（sign=+1 → {2,3}、sign=0 → {2,3,4}、sign=-1 → {3,4}），
        垂直动作（vert 0/1/2）不限制；warning=0 时不加任何限制。
        """
        allow_hor = np.ones((B, HOR_DIM), dtype=bool)
        rwr = (rule_inputs or {}).get('RWR_infos') or {}
        if rwr:
            warning = np.asarray(rwr.get('warning', np.zeros(B)), dtype=np.float32).reshape(-1)
            sin_dp = np.asarray(rwr.get('sin_threat_delta_psi', np.zeros(B)),
                                dtype=np.float32).reshape(-1)
            # 长度对齐不了（异常输入）时退回"全允许"，保证每行至少有一个可选动作
            if warning.size == B and sin_dp.size == B:
                warn_idx = warning > 0
                sign = np.sign(sin_dp)
                for s, allow in HOR_EVADE_ALLOW.items():
                    sel = warn_idx & (sign == s)
                    if sel.any():
                        allow_hor[sel] = False
                        allow_hor[np.flatnonzero(sel)[:, None], np.array(allow)] = True

        allow_hor_t = torch.as_tensor(allow_hor, dtype=torch.bool, device=device)  # (B, HOR_DIM)
        tdm_allow = allow_hor_t[:, self.tdm_hor_idx]                # (B, N_TDM)，hor 维铺到 TDM
        return enm_mask.unsqueeze(-1) & tdm_allow.unsqueeze(1)      # (B, Ne, N_TDM)

    # ------------------------------------------------------------------ 核心前向（可微，SAC 训练与采样共用）
    def _policy_heads(self, e_e, enm_mask, rule_inputs, B):
        """式(18)：敌机嵌入 → 扁平机动分布、开火 logits/门控、联合（机动×开火）分布。

        机动 mask = 敌机存活 mask × 告警规避 mask（与 ActorWithATTN 的 hor 规避同一口径），
        因此告警时"追踪/巡逻/crank"这些会迎弹的水平动作概率严格为 0。
        """
        device = self.device
        # 式(18)：敌机嵌入 → TDM logits；开火 logit 由独立 fire_head 产生
        tdm_logits = self.opp_pi_head(e_e)                       # (B, Ne, N_TDM)
        fire_logits_raw = self.fire_head(e_e).squeeze(-1)        # (B, Ne)

        # 机动掩码：敌机 j 的 N_TDM 个 TDM 槽位随敌机存活/有效掩码，
        # 并叠加告警规避（warning 时只留三线/置尾/九线里与威胁方位对应的那几个 hor）
        maneuver_mask = self._warning_mask(rule_inputs, enm_mask, B, device).reshape(B, -1)  # (B, A)

        logits_flat = tdm_logits.reshape(B, -1)                  # (B, A)
        logits_flat_masked = logits_flat.masked_fill(~maneuver_mask, -_MASK_VAL)
        log_pi = logits_flat_masked - torch.logsumexp(logits_flat_masked, dim=-1, keepdim=True)
        pi = torch.softmax(logits_flat_masked, dim=-1)

        # 开火：手工门控作为硬 mask 叠加到 logit 上（sigmoid 后≈0）
        fire_gate = self._fire_gate(rule_inputs, enm_mask, B, device)
        fire_logits = fire_logits_raw.masked_fill(~fire_gate, -_MASK_VAL)
        p_fire = torch.sigmoid(fire_logits).masked_fill(~enm_mask, 0.0)   # (B, Ne)

        # 联合分布 π(机动, 开火) = π(机动|敌机组) · π(开火|该敌机)
        # 联合索引 = j·(2·N_TDM) + a·2 + f，与 Q 头的 reshape 顺序一致
        pi_man = pi.reshape(B, self.enm_num, N_TDM)
        pi_joint = torch.stack(
            [pi_man * (1.0 - p_fire).unsqueeze(-1),
             pi_man * p_fire.unsqueeze(-1)], dim=-1).reshape(B, -1)       # (B, 2A)

        out = {
            'logits_flat_masked': logits_flat_masked,  # (B, A)
            'log_pi': log_pi, 'pi': pi,                # (B, A)
            'pi_joint': pi_joint,                      # (B, 2A) 联合分布（Q/熵用）
            'maneuver_mask': maneuver_mask,
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
        out = self._policy_heads(e_e, enm_mask, rule_inputs, fixed.shape[0])
        return out, e_f, e_e

    def forward_actor(self, agent_inputs, mask_inputs, rule_inputs):
        """部署/采样专用：只算策略分布与开火门控，不计算 Q 与分组熵。"""
        out, _, _ = self._forward_policy(agent_inputs, mask_inputs, rule_inputs)
        return out

    def forward_sac(self, agent_inputs, mask_inputs, rule_inputs):
        """返回扁平机动分布、双 Q、per-enm 分组熵等全部训练所需量（带梯度）。

        扁平顺序：[TIM(11)] + [敌机0 的 TDM(6), 敌机1 的 TDM(6), ...]；开火 logits 单独返回。
        （TIM 已删除，实际扁平顺序：[敌机0 的 TDM(21), 敌机1 的 TDM(21), ...]。）
        """
        fixed, enm, ally, enm_mask, ally_mask = self._pack(agent_inputs, mask_inputs)
        B = fixed.shape[0]
        e_f, e_e = self.actor_backbone(fixed, enm, ally, enm_mask, ally_mask)
        out = self._policy_heads(e_e, enm_mask, rule_inputs, B)
        pi_j = out['pi_joint']                      # (B, 2A) 联合（机动×开火）分布
        enm_mask = out['enm_mask']
        eps = 1e-8
        log_pi_j = torch.log(pi_j.clamp_min(eps))   # π=0 的槽位贡献 0·log0=0，无污染

        # critic 走独立骨干（参数不共享）：Q 只依赖 critic 自己的表征
        e_e_c = self.critic_backbone(fixed, enm, ally, enm_mask, ally_mask)[1]
        # 每敌机 2·N_TDM 个联合动作，索引顺序与 pi_joint 一致：j·(2N) + a·2 + f
        q1 = self.opp_q1_head(e_e_c).reshape(B, self.enm_num, N_TDM, 2).reshape(B, -1)
        q2 = self.opp_q2_head(e_e_c).reshape(B, self.enm_num, N_TDM, 2).reshape(B, -1)

        # 分组：G1..GN=各敌机的联合动作组；组质量、组内条件 log prob、分组熵（式 29-30）
        group_slices = [
            slice(j * 2 * N_TDM, (j + 1) * 2 * N_TDM) for j in range(self.enm_num)]
        pr_g = torch.stack([pi_j[:, sl].sum(-1) for sl in group_slices], dim=-1)  # (B,N)
        pr_g_safe = pr_g.clamp(min=eps)
        log_pr_g = torch.log(pr_g_safe)

        log_cond = torch.cat(
            [log_pi_j[:, sl] - log_pr_g[:, g:g + 1] for g, sl in enumerate(group_slices)],
            dim=-1)                                                              # (B,2A)

        # 联合熵 H^J 与组内条件熵 H^D_j（掩码槽位 pi=0，0*有限值=0 不贡献）
        H_J = -(pi_j * log_pi_j).sum(-1)                                         # (B,)
        H_Dj = torch.stack([
            -(pi_j[:, sl] * log_cond[:, sl]).sum(-1) / pr_g_safe[:, j]
            for j, sl in enumerate(group_slices)], dim=-1)                        # (B,Ne)
        # H^D 对存活敌机取均值（式 37 的 1/Ñ Σ 口径）
        enm_mask_f = enm_mask.to(torch.float32)
        H_D = (H_Dj * enm_mask_f).sum(-1) / enm_mask_f.sum(-1).clamp(min=1.0)

        out.update({
            'q1': q1, 'q2': q2,                        # (B, 2A)
            'log_cond': log_cond,                      # (B, 2A) 组内条件 log prob
            'pr_g': pr_g,                              # (B, Ne)
            'H_J': H_J, 'H_D': H_D, 'H_Dj': H_Dj,
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
        """一次扁平采样：TDM 单一 softmax 采样机动；开火一律 Bernoulli 概率采样（不设阈值）。

        解码：flat_idx = enm_idx * N_TDM + tdm_idx，再查 TDM_ACTIONS 得 (vert, hor)。
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
                flat_idx = logits.argmax(dim=-1)  # 联合分布 argmax
            else:
                flat_idx = Categorical(logits=logits / max(temperature, 1e-3)).sample()

            # 解码扁平索引 → 敌机 j 的 TDM → 查映射表得 (vert, hor)
            device = logits.device
            tdm_table = torch.as_tensor(TDM_ACTIONS, dtype=torch.long, device=device)
            enm_idx = torch.div(flat_idx, N_TDM, rounding_mode='floor')
            tdm_idx = flat_idx % N_TDM

            vert_idx = tdm_table[tdm_idx, 0]
            hor_idx = tdm_table[tdm_idx, 1]

            # 开火概率（已掩码）：与选中的 TDM 敌机绑定
            fire_logits = out['fire_logits']  # (B, Ne) 已掩码
            p_all = torch.sigmoid(fire_logits)
            p_all = p_all.masked_fill(~out['enm_mask'], 0.0)
            j_star = p_all.argmax(dim=-1)

            # 机动掩码已屏蔽死敌机，enm_idx 理论上有效；若无效则退化为 j*
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
