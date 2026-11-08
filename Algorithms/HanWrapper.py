"""HanWrapper：按 Han et al. (2026) JPO 路线实现的异构 Transformer 策略-价值一体网络。

论文对照（Addressing Hybrid Action of BVR Engagement With Joint Policy Optimization, IEEE TAES 2026）：
- 式(9)：fixed(ego)/enm/ally 三类实体各自独立的嵌入矩阵；
- 式(11)-(13)：每层 9 个类型专属 Q/K/V 投影；
- 式(15)：阵亡/无效敌机在注意力 key 侧掩码；
- 式(16)：残差 + LayerNorm；实验部分为 2 层、E=32；
- 式(17)-(18)：ego 嵌入出 TIM 策略/Q 头；每个敌机嵌入经共享 MLP 出 TDM logits、开火 logit、TDM Q；
- 式(21)：TIM ∪ TDM 拼接为扁平 logits，带 mask 的单一 softmax（本文件 maneuver 部分）；
- 式(19)(20)(26)：开火 = 每敌机 sigmoid 命中概率，训练时 Bernoulli 采样，测试时 argmax + 阈值 0.8。

与论文的有意偏差（动作空间沿用项目既有设定，不重引入论文原版动作）：
- 项目机动动作为 (vert, hor) 对，见 Envs/Tasks/ChooseStrategyEnv260421_action_spaces.py maneuver_for_1_enm；
  * hor ∈ {2,3,4}（3线/置尾/9线）在 warning 下相对威胁方向 → 按用户规定视为 TIM，vert 固定为平飞(1)；
  * hor∈{0,1,5,6}（追踪/左crank/右crank/回中心）× vert∈{0,1,2} → TDM，每个敌机一份（12 个/敌机）；
  * 扁平机动动作数 A = N_TIM + N_enm*N_TDM = 3 + 2*12 = 27（2V2）。
- 手工禁火门控全部保留为开火头的硬 mask（锁定、40°锥、15°仰角、<100km、30s 冷却、弹药、敌机存活）。

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
# hor 语义（maneuver_for_1_enm）：0 追踪 / 1 左crank / 5 右crank / 2 3线 / 3 置尾 / 4 9线 / 6 回中心
TIM_HOR = (2, 3, 4)        # warning 时相对威胁方向 → 论文 TIM（目标无关机动）
TDM_HOR = (0, 1, 5, 6)     # 相对所选目标 → 论文 TDM（目标相关机动）
TIM_VERT = 1               # TIM 动作垂直通道固定为"平飞(+1°)"
N_TIM = len(TIM_HOR)                       # 3
N_TDM = len(TDM_HOR) * VERT_DIM            # 每敌机 12
FIRE_THRESHOLD = 0.8       # 论文 IV-B：测试时开火阈值

# TDM 扁平索引 k ∈ [0,12) -> (hor, vert)
_TDM_HOR_LIST = [TDM_HOR[k // VERT_DIM] for k in range(N_TDM)]
_TDM_VERT_LIST = [k % VERT_DIM for k in range(N_TDM)]


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
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))


class HanWrapper(nn.Module):
    """异构 Transformer 骨干 + 扁平混合动作头 + 双 Q + 开火头（策略与价值共享骨干，式 35 联合反传）。"""

    def __init__(self, fixed_dim, enm_shape, ally_shape,
                 embed_dim=32, n_layers=2, head_hidden=64, device='cpu', **_unused):
        super().__init__()
        self.fixed_dim = fixed_dim
        self.enm_shape = tuple(enm_shape)
        self.ally_shape = tuple(ally_shape)
        self.enm_num = self.enm_shape[0]
        self.ally_num = self.ally_shape[0]
        self.embed_dim = embed_dim
        self.n_maneuver = N_TIM + self.enm_num * N_TDM  # 扁平机动动作数 A

        # 式(9)：三类实体独立嵌入
        self.fixed_embed = nn.Linear(fixed_dim, embed_dim)
        self.enm_embed = nn.Linear(self.enm_shape[1], embed_dim)
        self.ally_embed = nn.Linear(self.ally_shape[1], embed_dim)
        # 2 层异构自注意力（论文实验设置）
        self.attn_layers = nn.ModuleList([HetAttnLayer(embed_dim) for _ in range(n_layers)])

        # 式(17)：ego 嵌入 → TIM 策略头 + TIM 双 Q
        self.tim_pi_head = _mlp(embed_dim, head_hidden, N_TIM)
        self.tim_q1_head = _mlp(embed_dim, head_hidden, N_TIM)
        self.tim_q2_head = _mlp(embed_dim, head_hidden, N_TIM)
        # 式(18)：敌机嵌入 → 共享 MLP → TDM logits + 开火 logit；TDM 双 Q
        self.opp_pi_head = _mlp(embed_dim, head_hidden, N_TDM + 1)  # 末位为开火 logit
        self.opp_q1_head = _mlp(embed_dim, head_hidden, N_TDM)
        self.opp_q2_head = _mlp(embed_dim, head_hidden, N_TDM)

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

    def _backbone(self, fixed, enm, ally, enm_mask, ally_mask):
        e_f = self.fixed_embed(fixed.unsqueeze(1))  # (B,1,E)
        e_e = self.enm_embed(enm)                   # (B,Ne,E)
        e_a = self.ally_embed(ally)                 # (B,Na,E)
        for layer in self.attn_layers:
            e_f, e_e, e_a = layer(e_f, e_e, e_a, enm_mask, ally_mask)
        return e_f.squeeze(1), e_e  # (B,E), (B,Ne,E)

    # ------------------------------------------------------------------ 手工禁火门控（全部保留为硬 mask）
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
    def forward_sac(self, agent_inputs, mask_inputs, rule_inputs):
        """返回扁平混合动作分布、双 Q、分组条件熵等全部训练所需量（带梯度）。

        扁平顺序：[TIM(3)] + [敌机0 的 TDM(12), 敌机1 的 TDM(12), ...]；开火 logits 单独返回。
        """
        device = self.device
        fixed, enm, ally, enm_mask, ally_mask = self._pack(agent_inputs, mask_inputs)
        B = fixed.shape[0]
        e_f, e_e = self._backbone(fixed, enm, ally, enm_mask, ally_mask)

        # 策略/Q 头
        tim_logits = self.tim_pi_head(e_f)                       # (B,3)
        opp_out = self.opp_pi_head(e_e)                          # (B,N,13)
        tdm_logits = opp_out[..., :N_TDM]                        # (B,N,12)
        fire_logits_raw = opp_out[..., N_TDM]                    # (B,N)
        q1 = torch.cat((self.tim_q1_head(e_f),
                        self.opp_q1_head(e_e).reshape(B, -1)), dim=-1)   # (B,A)
        q2 = torch.cat((self.tim_q2_head(e_f),
                        self.opp_q2_head(e_e).reshape(B, -1)), dim=-1)   # (B,A)

        # 机动掩码：TIM 恒有效；敌机 j 的 12 个 TDM 随敌机存活/有效掩码
        tdm_mask = enm_mask.unsqueeze(-1).expand(B, self.enm_num, N_TDM).reshape(B, -1)
        maneuver_mask = torch.cat(
            (torch.ones(B, N_TIM, dtype=torch.bool, device=device), tdm_mask), dim=-1)  # (B,A)

        logits_flat = torch.cat((tim_logits, tdm_logits.reshape(B, -1)), dim=-1)
        logits_flat_masked = logits_flat.masked_fill(~maneuver_mask, -_MASK_VAL)
        log_pi = logits_flat_masked - torch.logsumexp(logits_flat_masked, dim=-1, keepdim=True)
        pi = torch.softmax(logits_flat_masked, dim=-1)

        # 分组：G0=TIM，G1..GN=各敌机 TDM；组质量、组内条件 log prob、分组熵（式 29-30 所需）
        eps = 1e-8
        group_slices = [slice(0, N_TIM)] + [
            slice(N_TIM + j * N_TDM, N_TIM + (j + 1) * N_TDM) for j in range(self.enm_num)]
        pr_g = torch.stack([pi[:, sl].sum(-1) for sl in group_slices], dim=-1)  # (B,1+N)
        pr_g_safe = pr_g.clamp(min=eps)
        log_pr_g = torch.log(pr_g_safe)

        log_cond = torch.cat(
            [log_pi[:, sl] - log_pr_g[:, g:g + 1] for g, sl in enumerate(group_slices)], dim=-1)  # (B,A)
        with torch.no_grad():
            pass
        # 联合熵 H^J 与组内条件熵 H^I / H^D_j（masked 槽位 pi=0，0*有限值=0 不贡献）
        H_J = -(pi * log_pi).sum(-1)                                          # (B,)
        H_I = -(pi[:, group_slices[0]] * log_cond[:, group_slices[0]]).sum(-1) / pr_g_safe[:, 0]
        H_Dj = torch.stack([
            -(pi[:, sl] * log_cond[:, sl]).sum(-1) / pr_g_safe[:, j + 1]
            for j, sl in enumerate(group_slices[1:])], dim=-1)                # (B,N)
        # H^D 对存活敌机取均值（式 37 的 1/Ñ Σ 口径）
        enm_mask_f = enm_mask.to(torch.float32)
        H_D = (H_Dj * enm_mask_f).sum(-1) / enm_mask_f.sum(-1).clamp(min=1.0)

        # 开火：手工门控作为硬 mask 叠加到 logit 上（sigmoid 后≈0）
        fire_gate = self._fire_gate(rule_inputs, enm_mask, B, device)
        fire_logits = fire_logits_raw.masked_fill(~fire_gate, -_MASK_VAL)

        return {
            'logits_flat_masked': logits_flat_masked,  # (B,A)
            'log_pi': log_pi, 'pi': pi,                # (B,A)
            'q1': q1, 'q2': q2,                        # (B,A)
            'log_cond': log_cond,                      # (B,A) 组内条件 log prob
            'pr_g': pr_g,                              # (B,1+N)
            'H_J': H_J, 'H_I': H_I, 'H_D': H_D, 'H_Dj': H_Dj,
            'maneuver_mask': maneuver_mask,
            'fire_logits': fire_logits,                # (B,N) 已掩码
            'fire_logits_raw': fire_logits_raw,        # (B,N) 未掩码（供 BCE）
            'fire_gate': fire_gate,
            'enm_mask': enm_mask,
        }

    def forward(self, context: Any) -> NetworkOutput:
        """与预留抽象接口对齐：context = (agent_inputs, mask_inputs, rule_inputs)。"""
        agent_inputs, mask_inputs, rule_inputs = context
        out = self.forward_sac(agent_inputs, mask_inputs, rule_inputs)
        return NetworkOutput(out['logits_flat_masked'], out['q1'], out['q2'], out['maneuver_mask'])

    # ------------------------------------------------------------------ 动作解码
    def _primary_target(self, fire_logits, enm_mask):
        """论文式(19)：j* = argmax_j πF,j（已掩码，命中概率最高的可打敌机）。"""
        p = torch.sigmoid(fire_logits)  # (B,N)
        p = p.masked_fill(~enm_mask, 0.0)
        j_star = p.argmax(dim=-1)  # 全无效时退化为 0（此时 p=0，不会真开火）
        return j_star, p

    def get_action(self, agent_inputs, mask_inputs, rule_inputs, h=None,
                   temperature=1.0, deterministic=None, fire_threshold=FIRE_THRESHOLD):
        """一次扁平采样：TIM ∪ TDM 单一 softmax 采样机动；开火独立 Bernoulli（训练）/阈值（测试）。

        deterministic=None 时按 temperature<1 推断（兼容 VsBaseline_MA 测试通路的 temperature=0.3）。
        """
        if deterministic is None:
            deterministic = bool(temperature < 1.0)
        with torch.no_grad():
            out = self.forward_sac(agent_inputs, mask_inputs, rule_inputs)
            logits = out['logits_flat_masked']

            if deterministic:
                flat_idx = logits_flat_argmax = logits.argmax(dim=-1)  # 式(22)：联合分布 argmax
            else:
                flat_idx = Categorical(logits=logits / max(temperature, 1e-3)).sample()

            j_star, fire_p = self._primary_target(out['fire_logits'], out['enm_mask'])
            p_star = fire_p.gather(1, j_star.unsqueeze(1)).squeeze(1)  # (B,)
            if deterministic:
                fire = (p_star > fire_threshold).to(torch.long)        # 式(20)：阈值 0.8
            else:
                fire = Bernoulli(probs=p_star.clamp(0.0, 1.0)).sample().to(torch.long)  # 式(26)

            # 解码扁平索引 → (enm_idx, vert, hor)
            device = logits.device
            is_tim = flat_idx < N_TIM
            r = (flat_idx - N_TIM).clamp(min=0)
            tdm_enm = r // N_TDM
            k = r % N_TDM
            hor_tdm = torch.as_tensor(_TDM_HOR_LIST, device=device)[k]
            vert_tdm = (k % VERT_DIM)
            hor_tim = torch.as_tensor(TIM_HOR, device=device)[flat_idx.clamp(max=N_TIM - 1)]

            enm_idx = torch.where(is_tim, j_star, tdm_enm)
            hor_idx = torch.where(is_tim, hor_tim, hor_tdm)
            vert_idx = torch.where(is_tim, torch.full_like(flat_idx, TIM_VERT), vert_tdm)

        return {
            'enm_idx': enm_idx,
            'vert_idx': vert_idx,
            'hor_idx': hor_idx,
            'fire': fire,
            'fire_target': j_star,
            'flat_idx': flat_idx,
            'fire_prob_max': float(p_star.max().item()) if p_star.numel() else float('nan'),
        }

    # ------------------------------------------------------------------ 开火监督（式 28，梯度穿骨干）
    def evaluate_fire(self, agent_inputs, fire_target_idx, hit,
                      mask_inputs=None, reduction='mean'):
        """Lf：历史开火时刻观测 → 当前网络对该敌机的开火 logit → BCE-with-logits。

        监督样本都取自真实开火时刻（门控当时已通过），故用未掩码 raw logit；
        mask_inputs 缺省视为全部槽位有效（记录中不含 mask）。
        不冻结任何参数：梯度经开火头流穿整个异构骨干（式 35 的 LNet 求和反传）。
        """
        fixed, enm, ally, enm_mask, ally_mask = self._pack(agent_inputs, mask_inputs)
        e_f, e_e = self._backbone(fixed, enm, ally, enm_mask, ally_mask)
        fire_logits = self.opp_pi_head(e_e)[..., N_TDM]  # (B,N) raw

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
