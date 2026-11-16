"""ActorWithHANATTN：Han 异构自注意力 encoder + TIM/TDM 混合动作头的 PPO actor。

论文对照（Addressing Hybrid Action of BVR Engagement With Joint Policy Optimization,
Han et al., IEEE TAES 2026）：
- 式(9)(11)-(16)：fixed(ego) / enm 两类实体各自独立的 Q/K/V 投影（原实现含 ally 一档，
  本次按需求删除 ally），拼接为统一序列做掩码缩放点积自注意力，残差 + LayerNorm；
  论文实验为 2 层注意力，本实现层数由 attn_layers 控制、默认 2。
- 式(17)(18)：ego 嵌入经 MLP 出 TIM logits；每个敌机嵌入经共享 MLP 出 TDM logits
  与开火 logit（式 18 的 MLP_F / MLP_D 共享前置处理器，输出头独立）。
- 式(21)：TIM logits 与各敌机 TDM logits 拼接为一个扁平动作向量，带 mask 的单一
  softmax；本次改造即"一次性带 mask 采样"，不再自回归选敌机 → 选机动。
- 式(19)(20)：开火目标 j* = argmax_j πF,j；本仓库沿用 ActorWithATTN/AMAACWrapper 口径，
  对"选中敌机"（TDM 动作为其绑定的敌机，TIM 动作为 j*）取 Bernoulli 采样，并把
  手工禁火规则作为硬 mask 叠到 logit 上（8 条：可观测、40°锥、锁定、俯仰<15°、
  距离<100km、30s 冷却、弹药>0、敌机存活）。

与 Algorithms/AMAACWrapper.py 的动作空间完全一致（直接 import HanWrapper 的
JPO_TIM_ACTIONS / JPO_TDM_ACTIONS / N_JPO_TIM / N_JPO_TDM，单一事实来源）：
- TIM（目标无关机动，11 个）：(vert,hor) ∈ {0,1,2}×{7,8,9} ∪ {(1,10),(1,11)}；
- TDM（目标相关机动，每敌机 6 个）：vert=1 平飞，hor ∈ 0~5；
- 扁平机动动作数 A = N_TIM + N_enm × N_TDM（2V2 时 11 + 2×6 = 23）。

已声明偏差（declared_deviations，合并/写结论前须同步进 run_config.json）：
1. 交叉注意力（PointerNetAttention / CrossAttentionPooling）整体删除：敌机选取由
   扁平 masked softmax 一并完成，不再有独立的指针网络目标分布。
2. 队友特征不再读取：序列只保留 (ego, enm*)，ally 的嵌入与 Q/K/V/LayerNorm 全删。
3. 论文式(19)(26) 的开火目标恒为 j* 且与机动解耦；本实现为与 ActorWithATTN 的
   enm_idx 语义对齐，TDM 动作的开火目标绑定其机动绑定的敌机，TIM 动作取 j*。
4. 未使用 RWR 规避掩码（与 AMAACWrapper 一致）。原因：TIM 的 hor 索引 7~11 不在
   ActorWithATTN/ HanWrapper 的规避放行集合 {2,3,4} 内，照搬会把 TIM 整体禁死，
   与"TIM/TDM 混合动作空间"不对称，故不做该掩码。
5. 论文用 hybrid SAC 优化（熵调节、双 Q、离策略）；本实现按需求只保留 PPO 侧：
   get_action 采样、evaluate_actions 重算新策略 log_prob/熵，优化器逻辑在
   MAPPOHybrid，本文件不含任何 SAC/双 Q/目标网络逻辑。
6. 无 MHP 命中预测模型与阈值开火（论文 §III-A、算法1）：开火为本仓库统一的学习型
   Bernoulli 头 + 规则硬门控。

接口约定与 Algorithms/ActorWithATTN.py / AMAACWrapper.py 保持一致（可被
MAPPOHybrid 直接驱动）：
- get_action(agent_inputs, mask_inputs, rule_inputs, temperature=..., enm_idx=...) →
  {'enm_idx','vert_idx','hor_idx','fire','fire_mask','flat_idx','log_prob_enm','log_prob_act'}；
  log_prob_enm = log π_flat（含 TDM 目标绑定），log_prob_act = log p_fire，二者之和为
  联合动作 log_prob（与 evaluate_actions['log_prob'] 同一语义）。
- evaluate_actions(agent_inputs, mask_inputs, rule_inputs, actions) →
  {'log_prob','log_prob_enm','log_prob_act','entropy','entropy_details',
   'fire_prob_max','fire_prob_min'}；actions 键与采集侧 Last_Actions 一致
  （vert / hor / fire / enm_idx）。
- evaluate_fire(agent_inputs, enm_idx, hit) → BCE-with-logits（带梯度），并暴露
  actor.pilot.fire_head 供 MAPPOHybrid.supervised_fire_update 冻结/优化。
- rule_inputs 为 stack_field([take_some_of_rule_inputs(rule_input)]) 之后的 batched dict。
"""

import numpy as np
import torch
from torch import nn
from torch.distributions import Bernoulli, Categorical
import torch.nn.functional as F

# TIM/TDM 动作空间常量直接 import 自 HanWrapper（单一事实来源，保证与 AMAACWrapper
# 逐位一致；仅只读引用，不修改 HanWrapper）
from Algorithms.HanWrapper import (JPO_TIM_ACTIONS as TIM_ACTIONS,
                                   JPO_TDM_ACTIONS as TDM_ACTIONS,
                                   N_JPO_TIM as N_TIM, N_JPO_TDM as N_TDM)

_MASK_VAL = 1e8  # 掩码填充值（-inf 量级），与 ActorWithATTN/AMAACWrapper/HanWrapper 口径一致

# (vert,hor) 取值范围由动作表推出，不硬编码 3/12
_VERT_DIM = max(v for v, _ in tuple(TIM_ACTIONS) + tuple(TDM_ACTIONS)) + 1
_HOR_DIM_FULL = max(h for _, h in tuple(TIM_ACTIONS) + tuple(TDM_ACTIONS)) + 1

# (vert,hor) -> 组内 idx 的反向查找表（evaluate_actions 回放动作重编码用）
_TIM_LUT = np.full((_VERT_DIM, _HOR_DIM_FULL), -1, dtype=np.int64)
for _i, (_v, _h) in enumerate(TIM_ACTIONS):
    _TIM_LUT[_v, _h] = _i
_TDM_LUT = np.full((_VERT_DIM, _HOR_DIM_FULL), -1, dtype=np.int64)
for _i, (_v, _h) in enumerate(TDM_ACTIONS):
    _TDM_LUT[_v, _h] = _i


def _mlp(in_dim, hidden):
    """论文 Table：动作头 MLP 含两个隐层、ReLU 激活；输出头单独定义。"""
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
    )


class HeterogeneousSelfAttention(nn.Module):
    """异构自注意力层（式 11-13、15、16），保留原有 HanAttention 的核心结构：

    - fixed(ego) 与 enm 两类实体各自持有独立的 Q/K/V 投影（原实现的 ally 一档已删除，
      队友特征不再参与任何计算）；
    - 沿 seq 维拼接为统一序列做一次 scaled dot-product 自注意力；
    - key 侧按 enm 掩码屏蔽阵亡/无效敌机，ego 槽位恒有效（保证每行 softmax 不退化）；
    - 按类型切回后各做残差 + LayerNorm（式 16），使同类层可堆叠到 2 层。
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.scale = embed_dim ** -0.5
        # ego 类独立投影
        self.w_q_f = nn.Linear(embed_dim, embed_dim)
        self.w_k_f = nn.Linear(embed_dim, embed_dim)
        self.w_v_f = nn.Linear(embed_dim, embed_dim)
        # enm 类独立投影
        self.w_q_e = nn.Linear(embed_dim, embed_dim)
        self.w_k_e = nn.Linear(embed_dim, embed_dim)
        self.w_v_e = nn.Linear(embed_dim, embed_dim)
        # 式(16)：LayerNorm(e + ζV)，按类型各一份
        self.ln_f = nn.LayerNorm(embed_dim)
        self.ln_e = nn.LayerNorm(embed_dim)

    def forward(self, e_f, e_e, enm_mask=None):
        # e_f: (B, 1, D), e_e: (B, Ne, D)；enm_mask: (B, Ne) bool
        Q = torch.cat((self.w_q_f(e_f), self.w_q_e(e_e)), dim=1)   # (B, 1+Ne, D)
        K = torch.cat((self.w_k_f(e_f), self.w_k_e(e_e)), dim=1)
        V = torch.cat((self.w_v_f(e_f), self.w_v_e(e_e)), dim=1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, s, s)
        if enm_mask is not None:
            B, s, _ = Q.shape
            N_e = e_e.shape[1]
            key_valid = torch.ones(B, s, dtype=torch.bool, device=Q.device)
            key_valid[:, 1:1 + N_e] = enm_mask
            scores = scores.masked_fill(~key_valid.unsqueeze(1), -_MASK_VAL)

        out = torch.matmul(torch.softmax(scores, dim=-1), V)        # (B, s, D)
        N_e = e_e.shape[1]
        e_f_new = self.ln_f(e_f + out[:, 0:1, :])                   # (B, 1, D)
        e_e_new = self.ln_e(e_e + out[:, 1:1 + N_e, :])             # (B, Ne, D)
        return e_f_new, e_e_new


class HANEntityEncoder(nn.Module):
    """观测-实体编码器：类型专属线性嵌入 → n_layers 层 HeterogeneousSelfAttention。

    序列顺序 (ego, enm*)，读取 ego token 与各敌机 token；不再构造/读取 ally 序列。
    """

    def __init__(self, fixed_dim, enm_dim, embed_dim=32, attn_layers=2):
        super().__init__()
        self.fixed_embed = nn.Linear(fixed_dim, embed_dim)
        self.enm_embed = nn.Linear(enm_dim, embed_dim)
        self.layers = nn.ModuleList(
            [HeterogeneousSelfAttention(embed_dim) for _ in range(attn_layers)])

    def forward(self, fixed, enm, enm_mask=None):
        # fixed: (B, F), enm: (B, Ne, De)；mask: (B, Ne) bool
        e_f = self.fixed_embed(fixed.unsqueeze(1))   # (B, 1, E)
        e_e = self.enm_embed(enm)                    # (B, Ne, E)
        for layer in self.layers:
            e_f, e_e = layer(e_f, e_e, enm_mask)
        return e_f.squeeze(1), e_e


class HANPilot(nn.Module):
    """异构动作头（式 17-18 适配到 TIM/TDM 离散混合动作）：

    - TIM 头：ego token → MLP(2×hidden) → N_TIM 个 logits（目标无关机动）；
    - TDM / 开火头：逐敌机 token → 共享 MLP(2×hidden) → N_TDM 个 TDM logits
      + 1 个开火 logit（每敌机一组，共享前置处理器、输出头独立）。
    fire_head 独立成层：MAPPOHybrid.supervised_fire_update 只优化 pilot.fire_head。
    """

    def __init__(self, embed_dim, mlp_hidden=128):
        super().__init__()
        self.tim_backbone = _mlp(embed_dim, mlp_hidden)
        self.tim_head = nn.Linear(mlp_hidden, N_TIM)
        self.opp_backbone = _mlp(embed_dim, mlp_hidden)   # 逐敌机共享
        self.tdm_head = nn.Linear(mlp_hidden, N_TDM)
        self.fire_head = nn.Linear(mlp_hidden, 1)

    def forward(self, e_self, e_enm):
        tim_logits = self.tim_head(self.tim_backbone(e_self))      # (B, N_TIM)
        opp_feat = self.opp_backbone(e_enm)                        # (B, Ne, H)
        tdm_logits = self.tdm_head(opp_feat)                       # (B, Ne, N_TDM)
        fire_logits = self.fire_head(opp_feat).squeeze(-1)         # (B, Ne)
        return {'tim': tim_logits, 'tdm': tdm_logits, 'fire': fire_logits}


class ActorWrapper(nn.Module):
    """HAN 注意力 actor（2 层异构自注意力 encoder + TIM/TDM 扁平混合动作头），PPO 接口。

    对外契约与 ActorWithATTN.ActorWrapper / AMAACWrapper.AMAACActorWrapper 一致。
    """

    def __init__(self, fixed_dim, enm_shape, ally_shape,
                 embed_dim=32, attn_layers=2, mlp_hidden=128,
                 device='cpu', ppo_with_bern=1, **_unused):
        super().__init__()
        self.fixed_dim = fixed_dim
        self.enm_shape = tuple(enm_shape)
        # ally_shape 仅为保持构造签名兼容：队友特征不再被读取（见模块 docstring 偏差 2）
        self.ally_shape = tuple(ally_shape)
        self.enm_num = self.enm_shape[0]
        self.ally_num = self.ally_shape[0]
        self.n_maneuver = N_TIM + self.enm_num * N_TDM
        # fire_included_in_rl=0 时 RL 的 log_prob/熵不含 fire 分支（口径同 ActorWithATTN）
        self.fire_included_in_rl = bool(ppo_with_bern)

        self.encoder = HANEntityEncoder(fixed_dim, self.enm_shape[1], embed_dim, attn_layers)
        self.pilot = HANPilot(embed_dim, mlp_hidden)

        # (vert,hor)->组内 idx 反向查找与正向动作表（非参数、不入 state_dict）
        self.register_buffer('_tim_lut', torch.as_tensor(_TIM_LUT), persistent=False)
        self.register_buffer('_tdm_lut', torch.as_tensor(_TDM_LUT), persistent=False)
        self.register_buffer('_tim_table', torch.as_tensor(np.asarray(TIM_ACTIONS), dtype=torch.long),
                             persistent=False)
        self.register_buffer('_tdm_table', torch.as_tensor(np.asarray(TDM_ACTIONS), dtype=torch.long),
                             persistent=False)
        self.to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    # ------------------------------------------------------------------ 输入整理
    def _pack(self, agent_inputs, mask_inputs):
        """接受单样本（无 batch 维）或批量输入，统一成 (B, ...) 张量。

        只读 fixed / enm 与 enm 掩码；agent_inputs['ally'] 与 mask_inputs['ally'] 一律不读。
        """
        device = self.device
        fixed = torch.as_tensor(np.asarray(agent_inputs['fixed']), dtype=torch.float32, device=device)
        enm = torch.as_tensor(np.asarray(agent_inputs['enm']), dtype=torch.float32, device=device)
        if fixed.ndim == 1:
            fixed = fixed.unsqueeze(0)
        if enm.ndim == 2:
            enm = enm.unsqueeze(0)
        B = fixed.shape[0]

        m = None if mask_inputs is None else mask_inputs.get('enm')
        if m is None or np.asarray(m).size == 0:
            enm_mask = torch.ones(B, self.enm_num, dtype=torch.bool, device=device)
        else:
            enm_mask = torch.as_tensor(np.asarray(m), dtype=torch.bool, device=device)
            if enm_mask.ndim == 1:
                enm_mask = enm_mask.unsqueeze(0).expand(B, -1)
            enm_mask = enm_mask.reshape(B, self.enm_num)
        return fixed, enm, enm_mask

    # ------------------------------------------------------------------ 手工禁火门控
    # 与 AMAACWrapper._fire_gate / HanWrapper._fire_gate 完全相同的手工禁火规则
    # （一条不少，阈值一致），拷贝而非 import 以保持同目录 baseline 各自实现独立。
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
        cdp = _arr(enm_infos.get('cos_enm_delta_psi'),
                   np.full((B, self.enm_num), -1.0)).reshape(B, self.enm_num)
        el = _arr(enm_infos.get('enm_locked'), np.zeros((B, self.enm_num))).reshape(B, self.enm_num)
        edt = _arr(enm_infos.get('enm_delta_theta'),
                   np.full((B, self.enm_num), np.pi)).reshape(B, self.enm_num)
        ed = _arr(enm_infos.get('enm_dist'),
                  np.full((B, self.enm_num), 1e9)).reshape(B, self.enm_num)
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
        """encoder 编码 → TIM/TDM 扁平机动 logits（masked）与逐敌机开火 logits（raw/masked）。

        扁平顺序：[TIM(N_TIM)] + [敌机0 的 TDM(N_TDM), 敌机1 的 TDM(N_TDM), ...]；
        TIM 槽位恒有效，敌机 j 的 TDM 槽位随该敌机的有效掩码一并屏蔽（式 21 的 mask）。
        """
        device = self.device
        fixed, enm, enm_mask = self._pack(agent_inputs, mask_inputs)
        B = fixed.shape[0]

        e_self, e_enm = self.encoder(fixed, enm, enm_mask)
        heads = self.pilot(e_self, e_enm)

        tdm_mask = enm_mask.unsqueeze(-1).expand(
            B, self.enm_num, N_TDM).reshape(B, -1)                        # (B, Ne*N_TDM)
        maneuver_mask = torch.cat(
            (torch.ones(B, N_TIM, dtype=torch.bool, device=device), tdm_mask), dim=-1)

        logits_flat = torch.cat(
            (heads['tim'], heads['tdm'].reshape(B, -1)), dim=-1)          # (B, A)
        logits_flat_masked = logits_flat.masked_fill(~maneuver_mask, -_MASK_VAL)

        # 开火：手工门控作为硬 mask 叠到逐敌机 logit 上（sigmoid 后≈0）
        fire_gate = self._fire_gate(rule_inputs, enm_mask, B, device)
        fire_logits_masked = heads['fire'].masked_fill(~fire_gate, -_MASK_VAL)

        return {
            'logits_flat_masked': logits_flat_masked,
            'maneuver_mask': maneuver_mask,
            'fire_logits_masked': fire_logits_masked,
            'fire_gate': fire_gate,
            'enm_mask': enm_mask,
        }

    def _decode_flat(self, flat_idx, fire_probs, enm_mask):
        """一次性采样的扁平索引 → (vert, hor, enm_idx)。

        TIM：目标无关，交互目标取开火概率最高的敌机 j*（与 AMAACWrapper 同口径）；
        TDM：目标绑定所属敌机组。 (vert,hor) 查 TIM/TDM 动作表。
        """
        tim_table, tdm_table = self._tim_table, self._tdm_table
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
        return torch.where(is_tim, tim_hit, N_TIM + enm_idx * N_TDM + tdm_hit.clamp(min=0))

    # ------------------------------------------------------------------ 采样接口
    def get_action(self, agent_inputs, mask_inputs, rule_inputs, h=None,
                   temperature=1.0, enm_idx=None):
        """一次性 masked softmax 采样机动 + 对选中敌机 Bernoulli 采样开火（PPO 行为策略）。

        enm_idx：外部指派目标槽位（仅部署/测试），把 TDM 组限制到该敌机；训练必须为 None。
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
            # 温度只作用于机动分布：开火 logit 不做温度缩放，与 ActorWithATTN 的测试口径一致
            fire_logit_sel = out['fire_logits_masked'].gather(
                1, enm_idx.unsqueeze(1)).squeeze(1)
            fire_dist = Bernoulli(logits=fire_logit_sel)
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

        # 开火分支：取回放时选中目标的 logit（gate 掩码后），与采样端同一分布语义；
        # 与采样端一致不做温度缩放（温度只作用于机动分布，evaluate_actions 本身也不收 temperature）
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

        返回带梯度的 BCE-with-logits；与 AMAACWrapper 同口径：对原始（未门控）fire logit
        计算——监督样本取自真实开火时刻，此时门控 delta 恒为 0。
        调用方（MAPPOHybrid.supervised_fire_update）只放行 pilot.fire_head 的梯度。
        """
        device = self.device
        fixed, enm, enm_mask = self._pack(agent_inputs, mask_inputs)
        B = fixed.shape[0]
        enm_idx = torch.as_tensor(np.asarray(enm_idx), dtype=torch.long, device=device).reshape(-1)
        hit = torch.as_tensor(np.asarray(hit), dtype=torch.float32, device=device).reshape(-1)
        if enm_idx.numel() != B or hit.numel() != B:
            raise ValueError('enm_idx/hit 长度必须等于观测 batch')

        _, e_enm = self.encoder(fixed, enm, enm_mask)
        opp_feat = self.pilot.opp_backbone(e_enm)                    # (B,Ne,H)
        feat_sel = opp_feat.gather(
            1, enm_idx.view(-1, 1, 1).expand(-1, 1, opp_feat.shape[-1])).squeeze(1)
        fire_logit = self.pilot.fire_head(feat_sel).squeeze(-1)      # (B,)
        return F.binary_cross_entropy_with_logits(fire_logit, hit, reduction=reduction)

    # ------------------------------------------------------------------ 存档
    def save(self, path='actor.pt'):
        torch.save({
            'encoder': self.encoder.state_dict(),
            'pilot': self.pilot.state_dict(),
        }, path)

    def load(self, path='actor.pt', device=None):
        target = device or self.device
        checkpoint = torch.load(path, map_location=target)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.pilot.load_state_dict(checkpoint['pilot'])
