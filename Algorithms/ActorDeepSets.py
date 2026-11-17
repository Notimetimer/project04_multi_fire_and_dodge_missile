"""ActorDeepSets.py —— 纯 MLP 的 Set-MLP / Deep Sets 目标选取 baseline（不使用注意力）。

结构动机
--------
主方法 ActorWithATTN 用"异构自注意力 + 交叉注意力指针网络"做目标选取。本 baseline 想回答：
在完全不用 Q-K 点积 / 自注意力 / 两两配对注意力的前提下，纯 MLP + 集合池化能否仍学到
"目标之间的相对重要性"。答案是"部分能"：

- 每个实体经共享 phi MLP 编码为 e_i（Deep Sets 的共享元素编码，Zaheer et al., NIPS 2017，
  是多智能体/集合输入场景最常用的集合聚合结构之一）；
- 掩码 mean/max 池化得到集合上下文 c（对实体数 N 与排列都不敏感，可扩展到任意 N）；
- 共享打分头 s_i = MLP([q, e_i, c])，再对掩码后的分数做 softmax / Categorical；
  其中 q 为本机 fixed 嵌入。因为 s_i 是同一个函数作用于"实体自身的表示 + 全体统计量"，
  网络可以学到：
    * 相对排序（softmax 下的重要性强弱）；
    * 与 q 的条件关系（"相对我而言谁更重要"）；
    * 与集合统计量（最近/最远/均值）的比较关系。
  但无法表达任意两两关系（如"谁离谁更近""谁和谁同向"），那需要 RelationNet 级别的成对输入。
  故能力阶梯为：独立打分 < 本结构的 Set-MLP < RelationNet < 主方法 ATTN；
  本 baseline 恰好卡在中间，用来隔离"注意力/成对交互"的增益。

与 Deep Sets 论文的精确对应（避免"同名但不同构"的误读）：
- 共享 phi + 掩码求和/均值池化 + 上下文投影 rho 对应 Sec 3.1 的 invariant 模型
  rho(Σ φ(x))（Theorem 2）；掩码 mean 是 sum 的归一化写法，max 对应 Sec 3.1 式(4) 的 maxpool 变体；
- 逐实体打分 s_i = MLP([q, e_i, c]) 对应 Sec 3.1 式(4) 的置换等变层
  f(x) = σ(λ I x + γ pool(x) 1)：每个元素的输出同时依赖"自身特征"和"全体池化统计量"；
- 以本机嵌入 q 作为条件对应 Sec 3.1 的"附加元信息 z 的条件化 φ(x|z)"；
- 注意：本结构不是论文 invariant 输出头本身（它只给一个集合级标量，不产生逐实体分数），
  而是用其等变层结构生成逐实体分数，再做 softmax/采样完成选择。

与主方法 ActorWithATTN.py 的对照关系（对比公平性）：
- pilot 直接从 ActorWithATTN 复用（不复制、不改动），机动/开火分支逐层一致；
- 观测 (fixed/enm/ally)、mask、规则输入、动作语义、log_prob/entropy 口径、checkpoint 选择规则一致；
- 敌机仍离散采样（enm_idx / log_prob_enm / ent_enm），队友仍为 softmax 加权聚合，
  与主方法的"指针网络采样敌机 + 交叉注意力聚合队友"在接口与语义上对齐；
- 唯一被替换的是选择器的特征交互方式：自注意力/点积交叉注意力 → 共享 MLP 打分 + 集合池化。

声明偏差（declared_deviations，合并/写结论前须同步进 run_config.json）：
1. selector 由 ATTN 改为 Set-MLP/Deep Sets；实体表示容量与主方法 embed 层同量级
   （enm/ally 额外有一层非线性 phi，属于本 baseline 的算法内部结构，不改变观测/动作/奖励）。
2. enm/ally 的打分头共享参数（置换等变），队友聚合权重由学习型打分头给出，
   与主方法 CrossAttentionPooling 的软聚合语义对齐（但无 query-key 点积）。
3. 不使用 agent_inputs['enm_chosen']：本结构不把"队友已选"作为交互特征；
   接口仍接受该键以兼容回放。
4. 退化处理：全部敌机死亡时固定指向 0 号槽位（最小 id），该行 log_prob_enm=0、ent_enm=0；
   主方法 ActorWithATTN 在该状态下会退化为全掩码均匀分布。此差异只影响无存活敌机的退化步，
   正常战斗的观测/动作/奖励/评测协议不变（规则对手 BasicRules_Homo 同样使用 at=0）。
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Bernoulli, Categorical

# pilot / 掩码常量与主方法共用同一份实现（原件只读，不复制）
from Algorithms.ActorWithATTN import pilot, _MASK_VAL


def _masked_pool(entity_seq, mask):
    """掩码 mean/max 池化：把 (B, N, d) 压成 (B, 2d)。

    全掩码的行（无有效实体）用 0 兜底，避免 -inf 进入后续 MLP。
    """
    if mask is None:
        mask = torch.ones(entity_seq.shape[:2], dtype=torch.bool, device=entity_seq.device)
    valid = mask.unsqueeze(-1)
    weight = valid.to(entity_seq.dtype)
    has_entity = valid.any(dim=1)  # (B, 1)：该行是否存在有效实体
    denom = weight.sum(dim=1).clamp(min=1.0)
    mean = (entity_seq * weight).sum(dim=1) / denom
    masked_max = entity_seq.masked_fill(~valid, torch.finfo(entity_seq.dtype).min)
    peak = masked_max.max(dim=1).values
    # 全掩码行的 peak 是 finfo.min（有限值），isfinite 挡不住，必须按 has_entity 显式置零，
    # 否则 -3.4e38 进入 context 的 Linear 会溢出成 ±inf，反传出现 NaN 并把网络参数污染。
    peak = torch.where(has_entity, peak, torch.zeros_like(peak))
    return torch.cat((mean, peak), dim=-1)


class DeepSetsSelector(nn.Module):
    """Set-MLP 选择器：共享实体编码 + 掩码池化上下文 + 共享 MLP 打分。

    输出契约与 ActorWithATTN.selector 完全一致：
    (selected_info, enm_idx, mini_rule_inputs, log_prob_enm, ent_enm)
    """

    def __init__(self, fixed_dim, enm_shape, ally_shape, embed_dim=64, hidden_dim=64):
        super().__init__()
        self.fixed_dim = fixed_dim
        self.enm_shape = enm_shape
        self.ally_shape = ally_shape
        self.enm_num = enm_shape[0]
        self.ally_num = ally_shape[0]
        self.embed_dim = embed_dim

        # 本机 query 嵌入（与主方法 fixed_embed 同结构）
        self.fixed_embed = nn.Linear(fixed_dim, embed_dim)

        # Deep Sets 的 phi：enm/ally 各自一套共享实体编码（同一序列内参数共享，置换等变）
        self.enm_phi = nn.Sequential(
            nn.Linear(enm_shape[1], hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim), nn.ReLU(),
        )
        self.ally_phi = nn.Sequential(
            nn.Linear(ally_shape[1], hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim), nn.ReLU(),
        )

        # 集合上下文投影：mean||max -> embed_dim
        self.enm_context = nn.Sequential(nn.Linear(2 * embed_dim, embed_dim), nn.ReLU())
        self.ally_context = nn.Sequential(nn.Linear(2 * embed_dim, embed_dim), nn.ReLU())

        # 共享打分头 s_i = MLP([q, e_i, c]) -> 标量；enm / ally 各自一套
        self.enm_score = nn.Sequential(
            nn.Linear(3 * embed_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.ally_score = nn.Sequential(
            nn.Linear(3 * embed_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _masked_scores(score_head, query, entity_seq, context):
        """把共享打分头作用于每个实体：concat(q, e_i, c)，q/c 沿 N 维广播。

        score_head 参数对 N 个实体共享 → 置换等变；集合上下文 c 让分数可依赖全体统计量。
        """
        batch, num = entity_seq.shape[0], entity_seq.shape[1]
        query = query.unsqueeze(1).expand(-1, num, -1)
        context = context.unsqueeze(1).expand(-1, num, -1)
        return score_head(torch.cat((query, entity_seq, context), dim=-1)).squeeze(-1)

    def forward(self, agent_inputs, mask_inputs, rule_inputs,
                enm_idx=None, temperature=1.0):
        fixed = agent_inputs['fixed']   # (B, fixed_dim)
        enm = agent_inputs['enm']       # (B, Ne, enm_dim)
        ally = agent_inputs['ally']     # (B, Na, ally_dim)
        enm_mask = mask_inputs.get('enm')   # (B, Ne) 或 None
        ally_mask = mask_inputs.get('ally') # (B, Na) 或 None

        query = self.fixed_embed(fixed)          # (B, d)
        e_e = self.enm_phi(enm)                  # (B, Ne, d)，共享 phi
        e_a = self.ally_phi(ally)                # (B, Na, d)，共享 phi

        # ── 敌机：集合上下文 + 共享打分 → Categorical 采样 ─────────────────────
        ctx_enm = self.enm_context(_masked_pool(e_e, enm_mask))          # (B, d)
        enm_scores = self._masked_scores(self.enm_score, query, e_e, ctx_enm)
        # 温度缩放后重新铺掩码兜底：与主方法同口径，掩码槽位统一 -_MASK_VAL
        enm_scores = enm_scores / temperature
        if enm_mask is not None:
            enm_scores = enm_scores.masked_fill(~enm_mask, -_MASK_VAL)
        # 全部敌机死亡：无合法目标，固定指向 0 号槽位（最小 id，环境默认填充），
        # 该行 log_prob=0、熵=0；与规则对手 BasicRules_Homo 无存活敌机时 at=0 一致。
        no_valid_enm = None if enm_mask is None else ~enm_mask.any(dim=-1, keepdim=True)
        if no_valid_enm is not None and no_valid_enm.any() and enm_scores.shape[-1] > 0:
            fallback = torch.full_like(enm_scores, -_MASK_VAL)
            fallback[..., 0] = 0.0
            enm_scores = torch.where(no_valid_enm, fallback, enm_scores)

        enm_categorical = Categorical(logits=enm_scores)

        if enm_idx is None:
            enm_idx = enm_categorical.sample()

        log_prob_enm = enm_categorical.log_prob(enm_idx)
        ent_enm = enm_categorical.entropy()

        e_enm = e_e.gather(
            1, enm_idx.reshape(-1, 1, 1).expand(-1, 1, self.embed_dim)).squeeze(1)

        # ── 队友：集合上下文 + 共享打分 → softmax 加权聚合 ────────────────────
        ctx_ally = self.ally_context(_masked_pool(e_a, ally_mask))       # (B, d)
        ally_scores = self._masked_scores(self.ally_score, query, e_a, ctx_ally)
        if ally_mask is not None:
            ally_scores = ally_scores.masked_fill(~ally_mask, -_MASK_VAL)
        ally_weights = torch.softmax(ally_scores, dim=-1)               # (B, Na)
        e_ally = torch.einsum('bn,bnd->bd', ally_weights, e_a)          # (B, d)

        # 与主方法同样的三段拼接：fixed query / 选中敌机 / 聚合队友
        selected_info = torch.cat((query, e_enm, e_ally), dim=-1)        # (B, 3d)

        # ── rule_inputs 按 enm_idx 抽取，组装 mini_rule_inputs（与主方法同口径）──
        mini_rule_inputs = {}
        mini_rule_inputs['ammo'] = rule_inputs.get('ammo')
        mini_rule_inputs['RWR_infos'] = rule_inputs.get('RWR_infos')
        mini_rule_inputs['weapon_infos'] = rule_inputs.get('weapon_infos')
        enm_infos = rule_inputs.get('enm_infos', {})  # dict{key: [B, n_enm]}
        enm_idx_col = enm_idx.unsqueeze(1)            # (B,) -> (B, 1)
        mini_rule_inputs['enm_info'] = {}
        for k, v in enm_infos.items():
            v_t = torch.as_tensor(v, device=enm_idx.device)
            mini_rule_inputs['enm_info'][k] = v_t.gather(1, enm_idx_col).squeeze(1)

        return selected_info, enm_idx, mini_rule_inputs, log_prob_enm, ent_enm


class ActorWrapper(nn.Module):
    """Deep Sets baseline 的 actor 封装：对外契约逐项对齐 ActorWithATTN.ActorWrapper。"""

    def __init__(self, fixed_dim, enm_shape, ally_shape,
                 embed_dim=64, hidden_dims=(256, 256), device='cpu', ppo_with_bern=1):
        super().__init__()
        self.fixed_dim = fixed_dim
        self.fire_included_in_rl = bool(ppo_with_bern)
        self.enm_shape = enm_shape
        self.ally_shape = ally_shape
        self.enm_num = enm_shape[0]
        self.ally_num = ally_shape[0]
        self.input_dim = 3 * embed_dim

        self.selector = DeepSetsSelector(fixed_dim, enm_shape, ally_shape, embed_dim).to(device)
        self.pilot = pilot(self.input_dim, hidden_dims).to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    def _to_tensor(self, value, dtype, shape):
        tensor = torch.as_tensor(value, dtype=dtype, device=self.device)
        if tensor.ndim == len(shape):
            tensor = tensor.unsqueeze(0)
        return tensor

    def pack_inputs(self, agent_inputs, mask_inputs):
        fixed = self._to_tensor(agent_inputs['fixed'], torch.float32, (self.fixed_dim,))
        enm = self._to_tensor(agent_inputs['enm'], torch.float32, self.enm_shape)
        ally = self._to_tensor(agent_inputs['ally'], torch.float32, self.ally_shape)

        enm_mask = np.asarray(mask_inputs.get('enm', []), dtype=bool).reshape(-1)
        if enm_mask.size == 0:
            enm_mask = np.ones(enm.shape[1], dtype=bool)
        enm_mask = torch.as_tensor(enm_mask, dtype=torch.bool, device=self.device)
        if enm_mask.ndim == 1:
            enm_mask = enm_mask.unsqueeze(0).expand(fixed.shape[0], -1)

        ally_mask = np.asarray(mask_inputs.get('ally', []), dtype=bool).reshape(-1)
        if ally_mask.size == 0:
            ally_mask = np.ones(ally.shape[1], dtype=bool)
        ally_mask = torch.as_tensor(ally_mask, dtype=torch.bool, device=self.device)
        if ally_mask.ndim == 1:
            ally_mask = ally_mask.unsqueeze(0).expand(fixed.shape[0], -1)

        return (
            {'fixed': fixed, 'enm': enm, 'ally': ally},
            {'enm': enm_mask, 'ally': ally_mask},
        )

    def get_action(self, agent_inputs, mask_inputs, rule_inputs,
                   h=None, temperature=1.0, enm_idx=None):
        packed_agent, packed_mask = self.pack_inputs(agent_inputs, mask_inputs)
        if enm_idx is not None:
            enm_idx = torch.as_tensor(enm_idx, dtype=torch.long, device=self.device).reshape(-1)
        selected_info, enm_idx, mini_rule_inputs, log_prob_enm, _ = self.selector(
            packed_agent, packed_mask, rule_inputs, enm_idx=enm_idx, temperature=temperature,
        )
        pilot_logits = self.pilot(selected_info, mini_rule_inputs)

        vert_dist = Categorical(logits=pilot_logits['vert'] / temperature)
        hor_dist = Categorical(logits=pilot_logits['hor'] / temperature)
        fire_dist = Bernoulli(logits=pilot_logits['fire'].squeeze(-1))

        vert_idx = vert_dist.sample()
        hor_idx = hor_dist.sample()
        fire = fire_dist.sample().to(torch.long)

        log_prob_act = (
            vert_dist.log_prob(vert_idx)
            + hor_dist.log_prob(hor_idx)
            + (fire_dist.log_prob(fire.to(torch.float32)) if self.fire_included_in_rl else 0.0)
        )

        return {
            'enm_idx': enm_idx,
            'fire': fire,
            'hor_idx': hor_idx,
            'vert_idx': vert_idx,
            'fire_mask': pilot_logits['fire_mask'],
            'log_prob_enm': log_prob_enm,
            'log_prob_act': log_prob_act,
        }

    def evaluate_actions(self, agent_inputs, mask_inputs, rule_inputs,
                         actions, h=None, max_std=None):
        """入参顺序与 ActorWithATTN 一致，MAPPOHybrid.update 按该位置顺序调用。"""
        device = self.device

        fixed = torch.as_tensor(agent_inputs['fixed'], dtype=torch.float32, device=device)
        enm = torch.as_tensor(agent_inputs['enm'], dtype=torch.float32, device=device)
        ally = torch.as_tensor(agent_inputs['ally'], dtype=torch.float32, device=device)
        enm_mask = torch.as_tensor(np.asarray(mask_inputs['enm']), dtype=torch.bool, device=device)
        ally_mask = torch.as_tensor(np.asarray(mask_inputs['ally']), dtype=torch.bool, device=device)

        enm_idx = torch.as_tensor(actions['enm_idx'], dtype=torch.long, device=device)
        vert_idx = torch.as_tensor(actions['vert'], dtype=torch.long, device=device)
        hor_idx = torch.as_tensor(actions['hor'], dtype=torch.long, device=device)
        fire = torch.as_tensor(actions['fire'], dtype=torch.long, device=device)

        selected_info, _, mini_rule_inputs, log_prob_enm, ent_enm = self.selector(
            {'fixed': fixed, 'enm': enm, 'ally': ally},
            {'enm': enm_mask, 'ally': ally_mask},
            rule_inputs, enm_idx=enm_idx,
        )

        pilot_logits = self.pilot(selected_info, mini_rule_inputs)
        vert_dist = Categorical(logits=pilot_logits['vert'])
        hor_dist = Categorical(logits=pilot_logits['hor'])
        fire_dist = Bernoulli(logits=pilot_logits['fire'].squeeze(-1))

        log_prob_vert = vert_dist.log_prob(vert_idx)
        log_prob_hor = hor_dist.log_prob(hor_idx)
        log_prob_fire = (fire_dist.log_prob(fire.to(torch.float32))
                         if self.fire_included_in_rl else torch.zeros_like(log_prob_vert))
        log_prob_act = log_prob_vert + log_prob_hor + log_prob_fire

        fire_mask = pilot_logits['fire_mask'].to(torch.float32)
        ent_vert = vert_dist.entropy()
        ent_hor = hor_dist.entropy()
        ent_fire = (fire_dist.entropy() * fire_mask
                    if self.fire_included_in_rl else torch.zeros_like(ent_vert))
        ent_act = ent_vert + ent_hor + ent_fire

        with torch.no_grad():
            fire_prob = torch.sigmoid(pilot_logits['fire'].squeeze(-1))
            fire_allowed = fire_mask.bool()
            if fire_allowed.any():
                fire_prob_max = fire_prob[fire_allowed].max().item()
                fire_prob_min = fire_prob[fire_allowed].min().item()
            else:
                fire_prob_max = float('nan')
                fire_prob_min = float('nan')

        entropy = ent_enm + ent_act
        log_prob = log_prob_enm + log_prob_act

        return {
            'log_prob': log_prob,
            'log_prob_enm': log_prob_enm,
            'log_prob_act': log_prob_act,
            'entropy': entropy,
            'entropy_details': {
                'enm': ent_enm,
                'vert': ent_vert,
                'hor': ent_hor,
                'fire': ent_fire,
                'act': ent_act,
            },
            'fire_prob_max': fire_prob_max,
            'fire_prob_min': fire_prob_min,
        }

    def evaluate_fire(self, agent_inputs, enm_idx, hit,
                      mask_inputs=None, rule_inputs=None, reduction='mean'):
        """有监督开火头：与 ActorWithATTN.evaluate_fire 同一路径与语义。"""
        device = self.device
        fixed = torch.as_tensor(agent_inputs['fixed'], dtype=torch.float32, device=device)
        enm = torch.as_tensor(agent_inputs['enm'], dtype=torch.float32, device=device)
        ally = torch.as_tensor(agent_inputs['ally'], dtype=torch.float32, device=device)
        if fixed.ndim == 1:
            fixed = fixed.unsqueeze(0)
        if enm.ndim == 2:
            enm = enm.unsqueeze(0)
        if ally.ndim == 2:
            ally = ally.unsqueeze(0)
        batch_size = fixed.shape[0]

        enm_idx = torch.as_tensor(np.asarray(enm_idx), dtype=torch.long, device=device).reshape(-1)
        hit = torch.as_tensor(np.asarray(hit), dtype=torch.float32, device=device).reshape(-1)
        if enm_idx.numel() != batch_size or hit.numel() != batch_size:
            raise ValueError('enm_idx/hit 长度必须等于观测 batch：%d/%d vs %d'
                             % (enm_idx.numel(), hit.numel(), batch_size))

        if mask_inputs is None:
            enm_mask = torch.ones(batch_size, enm.shape[1], dtype=torch.bool, device=device)
            ally_mask = torch.ones(batch_size, ally.shape[1], dtype=torch.bool, device=device)
        else:
            enm_mask = torch.as_tensor(np.asarray(mask_inputs['enm']), dtype=torch.bool, device=device)
            ally_mask = torch.as_tensor(np.asarray(mask_inputs['ally']), dtype=torch.bool, device=device)
            if enm_mask.ndim == 1:
                enm_mask = enm_mask.unsqueeze(0).expand(batch_size, -1)
            if ally_mask.ndim == 1:
                ally_mask = ally_mask.unsqueeze(0).expand(batch_size, -1)

        rule_inputs = {} if rule_inputs is None else rule_inputs

        selected_info, _, mini_rule_inputs, _, _ = self.selector(
            {'fixed': fixed, 'enm': enm, 'ally': ally},
            {'enm': enm_mask, 'ally': ally_mask},
            rule_inputs, enm_idx=enm_idx,
        )

        if rule_inputs:
            fire_logits = self.pilot(selected_info, mini_rule_inputs)['fire'][:, 0]
        else:
            fire_logits = self.pilot.fire_head(self.pilot.backbone(selected_info))[:, 0]

        return F.binary_cross_entropy_with_logits(fire_logits, hit, reduction=reduction)

    def save(self, path='actor.pt'):
        torch.save({
            'selector': self.selector.state_dict(),
            'pilot': self.pilot.state_dict(),
        }, path)

    def load(self, path='actor.pt', device=None):
        target = device or self.device
        checkpoint = torch.load(path, map_location=target)
        self.selector.load_state_dict(checkpoint['selector'])
        self.pilot.load_state_dict(checkpoint['pilot'])
