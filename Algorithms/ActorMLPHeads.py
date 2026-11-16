"""ActorMLPHeads.py —— 方案3 baseline：规则目标选取（敌机采样 / 队友距离优先聚合）。

与主方法 Algorithms/ActorWithATTN.py 的对照关系（对比公平性）：
- pilot 直接从 ActorWithATTN 复用（不复制、不改动），机动/开火分支与主方法逐层一致；
- 观测 (fixed/enm/ally)、mask、规则输入、动作语义、log_prob/entropy 口径、checkpoint 选择规则一致；
- 唯一被替换的是 selector：本 baseline 不训练目标选取。

规则（字段序取自 Envs/BattleHomo.py 的 ENM_KEYS / ALLY_KEYS；enm 20 维，ally 9 维）：
- 敌机：距离最小优先（enm_dist，索引 6），|AA| 最大次优先（cos_enm_AA，索引 9，
  以 1-|cos_enm_AA| 作为极小罚项）；优先级越高 → logits 越大；
  随后 Categorical 采样，保留 enm_idx / log_prob_enm / ent_enm 的对外接口，
  但 logits 不含可训练参数（与主方法的"学习型指针网络打分"相对照）。
- 队友：仅距离最小（ally_dist，索引 4），以 softmax(-ally_dist / temperature) 加权聚合，
  不做离散选取（与主方法的交叉注意力聚合相对照，且同样支持掩码）。

声明偏差（declared_deviations，合并/写结论前须同步进 run_config.json）：
1. selector 由"异构自注意力 + 交叉注意力指针网络"改为"规则打分 + Categorical 采样 /
   规则 softmax 聚合"；实体表示仍由可训练线性嵌入得到（与主方法 selector 的 embed 层同量级），
   pilot 完全复用。因此本 baseline 对比的是目标选取机制，而非表示容量。
2. 方案3 的敌机/队友选取不可训练：规则 logits 是观测常量，log_prob_enm 只用于接口与
   PPO 比率对齐，不产生策略梯度。这是方案3的既定假设，不是实现缺陷。
3. 不使用 agent_inputs['enm_chosen']：规则选取不依赖"队友已选"信息；接口仍接受该键以兼容回放。
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Bernoulli, Categorical

# pilot / 掩码常量与主方法共用同一份实现（原件只读，不复制）
from Algorithms.ActorWithATTN import pilot, _MASK_VAL

# ENM_KEYS / ALLY_KEYS 中的字段索引（顺序见 Envs/BattleHomo.py）
_ENM_DIST_IDX = 6      # enm_dist
_ENM_AA_IDX = 9        # cos_enm_AA
_ALLY_DIST_IDX = 4     # ally_dist
# enm_dist / ally_dist 在环境中已按 50 km 缩放；1e-3 约等于 50 m，
# 只用于"近等距"时让 |AA| 次优先，不影响距离主导关系。
_AA_TIE_BREAK = 1e-3


class ActorMLPHeads(nn.Module):
    """方案3 的规则选择器：可训练实体嵌入 + 不可训练规则打分/聚合。

    输出契约与 ActorWithATTN.selector 完全一致：
    (selected_info, enm_idx, mini_rule_inputs, log_prob_enm, ent_enm)
    """

    def __init__(self, fixed_dim, enm_shape, ally_shape, embed_dim=64):
        super().__init__()
        self.fixed_dim = fixed_dim
        self.enm_shape = enm_shape
        self.ally_shape = ally_shape
        self.enm_num = enm_shape[0]
        self.ally_num = ally_shape[0]
        self.embed_dim = embed_dim

        # 三类输入各自的嵌入层，与主方法 selector 的 embed 层保持同一结构/维度
        self.fixed_embed = nn.Linear(fixed_dim, embed_dim)
        self.enm_embed = nn.Linear(enm_shape[1], embed_dim)
        self.ally_embed = nn.Linear(ally_shape[1], embed_dim)

    def forward(self, agent_inputs, mask_inputs, rule_inputs,
                enm_idx=None, temperature=1.0):
        fixed = agent_inputs['fixed']   # (B, fixed_dim)
        enm = agent_inputs['enm']       # (B, Ne, enm_dim)
        ally = agent_inputs['ally']     # (B, Na, ally_dim)
        enm_mask = mask_inputs.get('enm')   # (B, Ne) 或 None
        ally_mask = mask_inputs.get('ally') # (B, Na) 或 None

        e_f = self.fixed_embed(fixed)       # (B, d)，作为 pilot 的 query_state
        e_e = self.enm_embed(enm)           # (B, Ne, d)
        e_a = self.ally_embed(ally)         # (B, Na, d)

        # ── 敌机：距离最小优先、|AA| 最大次优先（规则常量，无可训练参数）──────────
        enm_distance = enm[..., _ENM_DIST_IDX]
        cos_aa = enm[..., _ENM_AA_IDX]
        priority = enm_distance + _AA_TIE_BREAK * (1.0 - cos_aa.abs())
        enm_logits = -priority / temperature
        if enm_mask is not None:
            enm_logits = enm_logits.masked_fill(~enm_mask, -_MASK_VAL)
        enm_categorical = Categorical(logits=enm_logits)

        if enm_idx is None:
            enm_idx = enm_categorical.sample()

        log_prob_enm = enm_categorical.log_prob(enm_idx)
        ent_enm = enm_categorical.entropy()

        e_enm = e_e.gather(
            1, enm_idx.reshape(-1, 1, 1).expand(-1, 1, self.embed_dim)).squeeze(1)

        # ── 队友：距离最小优先的 softmax 聚合（规则常量权重）──────────────────
        ally_distance = ally[..., _ALLY_DIST_IDX]
        ally_logits = -ally_distance / temperature
        if ally_mask is not None:
            ally_logits = ally_logits.masked_fill(~ally_mask, -_MASK_VAL)
        ally_weights = torch.softmax(ally_logits, dim=-1)          # (B, Na)
        e_ally = torch.einsum('bn,bnd->bd', ally_weights, e_a)     # (B, d)

        # 与主方法同样的三段拼接：fixed / 选中敌机 / 聚合队友
        selected_info = torch.cat((e_f, e_enm, e_ally), dim=-1)    # (B, 3d)

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
    """方案3 的 actor 封装：对外契约逐项对齐 ActorWithATTN.ActorWrapper。"""

    def __init__(self, fixed_dim, enm_shape, ally_shape,
                 embed_dim=64, hidden_dims=(256, 256), device='cpu', ppo_with_bern=1):
        super().__init__()
        self.fixed_dim = fixed_dim
        # fire_included_in_rl 语义与主方法一致：0 时 RL 的 log_prob/熵不含 fire 分支
        self.fire_included_in_rl = bool(ppo_with_bern)
        self.enm_shape = enm_shape
        self.ally_shape = ally_shape
        self.enm_num = enm_shape[0]
        self.ally_num = ally_shape[0]
        self.input_dim = 3 * embed_dim

        self.selector = ActorMLPHeads(fixed_dim, enm_shape, ally_shape, embed_dim).to(device)
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
        # temperature：玻尔兹曼采样温度，仅测试时使用；训练采样走默认值 1
        # enm_idx：外部集中指派的敌机槽位，仅部署/测试使用；训练路径必须为 None
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
        """训练接口：严格用回放观测/掩码/规则与已采样动作重算新策略 log_prob 与熵。

        注意入参顺序与 ActorWithATTN 一致：(agent_inputs, mask_inputs, rule_inputs, actions)，
        MAPPOHybrid.update 按该位置顺序调用。
        """
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


# 兼容只读文件 Algorithms/MAPPOHybrid.py 顶部的 import（该文件按规范不可修改）
ActorMLPWrapper = ActorWrapper
