import os
from math import *
import numpy as np
import torch
from torch import nn
from torch.distributions import Bernoulli, Categorical
import torch.nn.functional as F

HORIZONTAL_ACTIONS = (0, 1, 3, 5)
VERT_DIM = 3                      # 垂直动作维度
HOR_DIM = 7                       # 水平动作维度
_MASK_VAL = 1e8                   # 掩码填充值（-inf 量级，保证 softmax 后概率为 0）

# 规避模式（warning=True）下，仅允许水平动作索引 {2,3,4}，其余维度置为 -1e8
# hor顺序： 前，右前，右，后，左，左前，巡逻
_EVADE_HOR_MASK = torch.tensor([-1, -1, 0, 0, 0, -1, -1], dtype=torch.float32) * _MASK_VAL
# # 安全模式（no_warning_or_enm=True）下，仅允许水平动作索引 {6}，其余维度置为 -1e8
# _SAFE_HOR_MASK = torch.tensor([-1, -1, -1, -1, -1, -1, 0], dtype=torch.float32) * _MASK_VAL


# 单头交叉注意力模块：query 来自自身状态，key/value 来自实体序列。
# 每个实体类型（enm / ally）各持有一份独立实例，参数互不共享。
class PointerNetAttention(nn.Module):
    def __init__(self, query_dim, entity_dim, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.scale = embed_dim ** -0.5
        self.query_proj = nn.Linear(query_dim, embed_dim)
        self.key_proj = nn.Linear(entity_dim, embed_dim)

    def forward(self, query_state, entity_seq, mask=None):
        query = self.query_proj(query_state)         # (B, E)
        keys = self.key_proj(entity_seq)              # (B, N, E)
        scores = torch.einsum('be,bne->bn', query, keys) * self.scale
        if mask is not None:
            scores = scores.masked_fill(~mask, -_MASK_VAL)
        return scores, keys


# 选择器网络：用两个独立的交叉注意力模块分别压缩 enm / ally 的 seq_len 维度，
# 选出一个敌机一个队友，把它们的嵌入与 fixed 拼接后交给 pilot。
class selector(nn.Module):
    def __init__(self, fixed_dim=28, enm_shape=(3, 10), ally_shape=(2, 9), embed_dim=64):
        super().__init__()
        self.fixed_dim = fixed_dim
        self.enm_shape = enm_shape
        self.ally_shape = ally_shape
        self.enm_num = enm_shape[0]
        self.ally_num = ally_shape[0]
        self.embed_dim = embed_dim

        # 敌机与友机各自独立的注意力模块：query/key 投影参数互不共享
        self.enm_attn = PointerNetAttention(fixed_dim, enm_shape[1], embed_dim)
        self.ally_attn = PointerNetAttention(fixed_dim, ally_shape[1], embed_dim)

    def forward(self, agent_inputs, mask_inputs, rule_inputs, enm_idx=None, ally_idx=None, temperature=1.0):
        fixed = agent_inputs['fixed'] # 应该是(batch, fixed_dim)
        enm = agent_inputs['enm'] # 应该是(batch, enm_num, enm_shape[1])
        ally = agent_inputs['ally'] # 应该是(batch, ally_num, ally_shape[1])
        enm_mask = mask_inputs.get('enm') # 应该是(batch, enm_num)
        ally_mask = mask_inputs.get('ally') # 应该是(batch, ally_num)

        # 两个独立的交叉注意力：各自的 query/key 投影产生打分与嵌入
        # type A 变量query
        query_enm = fixed
        # # type B 常数query
        # query_enm = torch.zeros_like(fixed)
        # 聚合敌机特征
        enm_scores, enm_keys = self.enm_attn(query_enm, enm, enm_mask)
        # 队友信息需要本机query
        ally_scores, ally_keys = self.ally_attn(fixed, ally, ally_mask)

        # 敌人选取分布和队友选取分布（temperature 仅测试时调节采样平滑度，默认1不改变分布）
        # 缩放后重新铺掩码兜底：掩码槽位统一用 -_MASK_VAL（有限值），温度缩放不会溢出为 -inf
        enm_scores = enm_scores / temperature
        ally_scores = ally_scores / temperature
        if enm_mask is not None:
            enm_scores = enm_scores.masked_fill(~enm_mask, -_MASK_VAL)
        if ally_mask is not None:
            ally_scores = ally_scores.masked_fill(~ally_mask, -_MASK_VAL)
        enm_dist = Categorical(logits=enm_scores)
        ally_dist = Categorical(logits=ally_scores)

        if enm_idx is None:
            enm_idx = enm_dist.sample()
        if ally_idx is None:
            ally_idx = ally_dist.sample()

        log_prob_enm = enm_dist.log_prob(enm_idx)
        log_prob_ally = ally_dist.log_prob(ally_idx)

        # 两个选择分布的熵（mask 槽位 logits=-inf，p=0 对熵无贡献），供 evaluate 组装 H(enm)+H(ally)
        ent_enm = enm_dist.entropy()
        ent_ally = ally_dist.entropy()

        # enm_keys: (B, enm_num, embed_dim)
        # enm_idx: (B,) -> reshape (-1,1,1) -> (B,1,1) -> expand -> (B, 1, embed_dim)
        # gather(1, ...): 沿 seq_len 维取被选中的实体嵌入 -> (B, 1, embed_dim)
        # squeeze(1) -> e_enm: (B, embed_dim)
        e_enm = enm_keys.gather(1, enm_idx.reshape(-1, 1, 1).expand(-1, 1, self.embed_dim)).squeeze(1)
        e_ally = ally_keys.gather(1, ally_idx.reshape(-1, 1, 1).expand(-1, 1, self.embed_dim)).squeeze(1)

        # selected_info: concat((B, fixed_dim), (B, embed_dim), (B, embed_dim)) -> (B, fixed_dim + 2*embed_dim)
        selected_info = torch.cat((fixed, e_enm, e_ally), dim=-1)
        
        # rule_inputs 处理：按 enm_idx 抽取对应敌机信息，组装 mini_rule_inputs
        # 输入 rule_inputs 维度结构（张量化后）：
        #   {'ammo': [B], 'enm_infos': {key: [B, n_enm]}, 'RWR_infos': {key: [B]}, 'weapon_infos': {key: [B]}}
        # 输出 mini_rule_inputs 中 enm_info 已从 [B, n_enm] 按 enm_idx 抽取为 [B]
        mini_rule_inputs = {}
        mini_rule_inputs['ammo'] = rule_inputs.get('ammo')
        mini_rule_inputs['RWR_infos'] = rule_inputs.get('RWR_infos')
        mini_rule_inputs['weapon_infos'] = rule_inputs.get('weapon_infos')
        enm_infos = rule_inputs.get('enm_infos', {})  # dict{key: [B, n_enm]}，每个 key 是所有敌机的某项属性
        enm_idx_col = enm_idx.unsqueeze(1)  # (B,) -> (B, 1) for gather along dim=1
        mini_rule_inputs['enm_info'] = {}
        for k, v in enm_infos.items():
            v_t = torch.as_tensor(v, device=enm_idx.device)
            mini_rule_inputs['enm_info'][k] = v_t.gather(1, enm_idx_col).squeeze(1)  # (B,)
            # 输出[b, 0] = v_t[b, enm_idx_col[b, 0]] = v_t[b, enm_idx[b]]
        return selected_info, enm_idx, ally_idx, mini_rule_inputs, log_prob_enm, log_prob_ally, ent_enm, ent_ally


# 飞行员网络：接收 selector 选出的信息，输出 vert/hor/fire 三路逻辑值。
# 规则掩码以 -1e8 偏移量的形式
# 汇流到对应 logits 上，由 actorWrapper 负责后续的采样与 log_prob 计算。
class pilot(nn.Module):
    def __init__(self, input_dim, hidden_dims=(256, 256)):
        super().__init__()
        layers = []
        prev = input_dim
        for hidden in hidden_dims:
            layers.extend((nn.Linear(prev, hidden), nn.ReLU()))
            prev = hidden
        self.backbone = nn.Sequential(*layers)

        self.vert_head = nn.Linear(prev, VERT_DIM)
        self.hor_head = nn.Linear(prev, HOR_DIM)
        self.fire_head = nn.Linear(prev, 1)

    @staticmethod
    def _extract_flag(mini_rule_inputs, key, batch_size, device, default=False):
        if isinstance(mini_rule_inputs, list):
            vals = [bool((r or {}).get(key, default)) for r in mini_rule_inputs]
        elif isinstance(mini_rule_inputs, dict):
            vals = [bool((mini_rule_inputs or {}).get(key, default))] * batch_size
        else:
            vals = [bool(default)] * batch_size
        return torch.tensor(vals, dtype=torch.bool, device=device)

    def _warning(self, mini_rule_inputs, batch_size, device):
        # 向量化：按 sin_threat_delta_psi 符号批量生成 hor delta 掩码
        RWR = mini_rule_inputs.get('RWR_infos') or {}
        warning = np.asarray(RWR.get('warning', np.zeros(batch_size, dtype=bool))).reshape(-1).astype(bool)
        sin_dp = np.asarray(RWR.get('sin_threat_delta_psi', np.zeros(batch_size))).reshape(-1)
        delta = np.zeros((batch_size, HOR_DIM), dtype=np.float32)
        if warning.any():
            sign = np.sign(sin_dp)
            # hor顺序：前，右前，右，后，左，左前，巡逻
            for s, m in {1: [-1, -1, 0, 0, -1, -1, -1],
                         0: [-1, -1, 0, 0,  0, -1, -1],
                         -1: [-1, -1, -1, 0, 0, -1, -1]}.items():
                sel = (sign == s) & warning
                if sel.any():
                    delta[sel] = np.array(m, dtype=np.float32) * _MASK_VAL
        return (torch.as_tensor(warning, dtype=torch.bool, device=device),
                torch.as_tensor(delta, dtype=torch.float32, device=device))

    def _enm_in_range(self, mini_rule_inputs, batch_size, device):
        # 向量化：批量判断敌机是否在可攻击区，不可攻击时禁火
        enm = mini_rule_inputs.get('enm_info') or {}
        eo = np.asarray(enm.get('enm_observable', np.zeros(batch_size))).reshape(-1)
        cdp = np.asarray(enm.get('cos_enm_delta_psi', np.zeros(batch_size))).reshape(-1)
        el = np.asarray(enm.get('enm_locked', np.zeros(batch_size))).reshape(-1).astype(bool)
        edt = np.asarray(enm.get('enm_delta_theta', np.zeros(batch_size))).reshape(-1)
        ed = np.asarray(enm.get('enm_dist', np.zeros(batch_size))).reshape(-1)
        in_range = (eo > 0) & (cdp > np.cos(np.radians(40))) & el & (edt < np.radians(15)) & (ed < 100e3)
        out_range = (~in_range)
        delta_fire = ((~in_range).astype(np.float32) * (-_MASK_VAL)).reshape(-1, 1)
        return (torch.as_tensor(out_range, dtype=torch.bool, device=device),
                torch.as_tensor(delta_fire, dtype=torch.float32, device=device))

    def _just_fired(self, mini_rule_inputs, batch_size, device):
        # 向量化：missile_t_fly < 30 视为刚发射，禁火
        weapon = mini_rule_inputs.get('weapon_infos') or {}
        t_fly = np.asarray(weapon.get('missile_t_fly', np.full(batch_size, 1e9))).reshape(-1)
        just_fired = (t_fly < 30)
        delta_fire = (just_fired.astype(np.float32) * (-_MASK_VAL)).reshape(-1, 1)
        return (torch.as_tensor(just_fired, dtype=torch.bool, device=device),
                torch.as_tensor(delta_fire, dtype=torch.float32, device=device))
    
    def _has_ammo(self, mini_rule_inputs, batch_size, device):
        # 向量化：ammo <= 0 时无弹可用，禁火
        ammo_arr = mini_rule_inputs.get('ammo', None)
        if ammo_arr is None:
            no_ammo = np.zeros(batch_size, dtype=bool)
        else:
            no_ammo = (np.asarray(ammo_arr).reshape(-1) <= 0)
        delta_fire = (no_ammo.astype(np.float32) * (-_MASK_VAL)).reshape(-1, 1)
        return (torch.as_tensor(no_ammo, dtype=torch.bool, device=device),
                torch.as_tensor(delta_fire, dtype=torch.float32, device=device))
        
    def forward(self, selected_info, mini_rule_inputs):
        features = self.backbone(selected_info)

        # 主干输出三路原始 logits
        vert_logits = self.vert_head(features)           # (B, vert_num)
        pre_hor_logits = self.hor_head(features)         # (B, HOR_DIM)
        fire_logits = self.fire_head(features)           # (B, 1)

        B = selected_info.shape[0]
        device = selected_info.device

        warning_flag, delta_hor_evade = self._warning(mini_rule_inputs, B, device)
        out_range_flag, delta_fire_out_range = self._enm_in_range(mini_rule_inputs, B, device)
        just_fired_flag, delta_fire_recent = self._just_fired(mini_rule_inputs, B, device)
        no_ammo_flag, delta_fire_noammo = self._has_ammo(mini_rule_inputs, B, device)

        # 第一处汇流：hor_logits = pre_hor_logits + delta_evade
        # delta_evade 由 _warning 按威胁方向动态生成；
        hor_logits = pre_hor_logits + delta_hor_evade

        # 第二处汇流：fire_logits += 多源禁火叠加
        fire_logits = fire_logits + delta_fire_out_range + delta_fire_recent + delta_fire_noammo  # (B, 1)

        # 第三处汇流：打包为 logits 字典，由 actorWrapper 采样并计算 log_prob
        return {
            'hor': hor_logits,
            'vert': vert_logits,
            'fire': fire_logits,
            'fire_mask': 1 - (out_range_flag | just_fired_flag | no_ammo_flag).long(),
        }


# 统筹 selector 与 pilot 的 ActorWrapper：
# 对外接口只有 (agent_inputs, mask_inputs, rule_inputs) → 完整动作与概率。
class ActorWrapper(nn.Module):
    def __init__(self, fixed_dim=28, enm_shape=(2, 10), ally_shape=(1, 9),
                 embed_dim=64, hidden_dims=(256, 256), device='cpu'):
        super().__init__()
        self.fixed_dim = fixed_dim
        self.enm_shape = enm_shape
        self.ally_shape = ally_shape
        self.enm_num = enm_shape[0]
        self.ally_num = ally_shape[0]
        self.input_dim = fixed_dim + 2 * embed_dim

        self.selector = selector(fixed_dim, enm_shape, ally_shape, embed_dim).to(device)
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
            enm_mask = np.ones(self.enm_num, dtype=bool)
        enm_mask = torch.as_tensor(enm_mask, dtype=torch.bool, device=self.device)
        if enm_mask.ndim == 1:
            enm_mask = enm_mask.unsqueeze(0).expand(fixed.shape[0], -1)

        ally_mask = np.asarray(mask_inputs.get('ally', []), dtype=bool).reshape(-1)
        if ally_mask.size == 0:
            ally_mask = np.ones(self.ally_num, dtype=bool)
        ally_mask = torch.as_tensor(ally_mask, dtype=torch.bool, device=self.device)
        if ally_mask.ndim == 1:
            ally_mask = ally_mask.unsqueeze(0).expand(fixed.shape[0], -1)

        return (
            {'fixed': fixed, 'enm': enm, 'ally': ally},
            {'enm': enm_mask, 'ally': ally_mask},
        )

    def get_action(self, agent_inputs, mask_inputs, rule_inputs, h=None, temperature=1.0):
        # temperature：玻尔兹曼采样温度，仅测试时使用；训练采样走默认值1，分布不变
        packed_agent, packed_mask = self.pack_inputs(agent_inputs, mask_inputs)
        selected_info, enm_idx, ally_idx, mini_rule_inputs, log_prob_enm, log_prob_ally, _, _ = self.selector(
            packed_agent, packed_mask, rule_inputs, temperature=temperature,
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
            + fire_dist.log_prob(fire.to(torch.float32))
        )

        return {
            'enm_idx': enm_idx,
            'ally_idx': ally_idx,
            'fire': fire,
            'hor_idx': hor_idx,
            'vert_idx': vert_idx,
            'fire_mask': pilot_logits['fire_mask'],
            'log_prob_enm': log_prob_enm,
            'log_prob_ally': log_prob_ally,
            'log_prob_act': log_prob_act,
        }

    def evaluate_actions(self, agent_inputs, mask_inputs, rule_inputs, actions, h=None, max_std=None):
        # 训练接口：严格传入回放中的批量观测/掩码/规则输入与已采样动作，不做 sample，
        # 只重算"新策略"在这些动作上的 log_prob 与策略熵，供 PPO 新旧策略比使用。
        device = self.device

        # 回放数据自带 batch 维（fixed:[B,*], enm:[B,N,*]），直接转 tensor；
        # 不能走 pack_inputs：其中的 reshape(-1) 是单样本路径，会把 [B,N] 掩码摊平
        fixed = torch.as_tensor(agent_inputs['fixed'], dtype=torch.float32, device=device)
        enm = torch.as_tensor(agent_inputs['enm'], dtype=torch.float32, device=device)
        ally = torch.as_tensor(agent_inputs['ally'], dtype=torch.float32, device=device)
        enm_mask = torch.as_tensor(np.asarray(mask_inputs['enm']), dtype=torch.bool, device=device)
        ally_mask = torch.as_tensor(np.asarray(mask_inputs['ally']), dtype=torch.bool, device=device)
        packed_agent = {'fixed': fixed, 'enm': enm, 'ally': ally}
        packed_mask = {'enm': enm_mask, 'ally': ally_mask}

        # 回放动作键：vert / hor / fire / enm_idx / ally_idx（与采集脚本 last_Actions 一致）
        enm_idx = torch.as_tensor(actions['enm_idx'], dtype=torch.long, device=device)
        ally_idx = torch.as_tensor(actions['ally_idx'], dtype=torch.long, device=device)
        vert_idx = torch.as_tensor(actions['vert'], dtype=torch.long, device=device)
        hor_idx = torch.as_tensor(actions['hor'], dtype=torch.long, device=device)
        fire = torch.as_tensor(actions['fire'], dtype=torch.long, device=device)

        # selector 传入回放中的 enm_idx/ally_idx：只算这两个已选样本的 log_prob 与选定 embedding，
        # ent_enm/ent_ally 是遍历全部槽位的 Categorical 熵 H(enm)、H(ally)
        selected_info, _, _, mini_rule_inputs, log_prob_enm, log_prob_ally, ent_enm, ent_ally = self.selector(
            packed_agent, packed_mask, rule_inputs, enm_idx=enm_idx, ally_idx=ally_idx,
        )

        # pilot 在"给定敌人与队友选取"下重放三路分布（条件策略 act|enm_idx, ally_idx, o）
        pilot_logits = self.pilot(selected_info, mini_rule_inputs)
        vert_dist = Categorical(logits=pilot_logits['vert'])
        hor_dist = Categorical(logits=pilot_logits['hor'])
        fire_dist = Bernoulli(logits=pilot_logits['fire'].squeeze(-1))

        log_prob_vert = vert_dist.log_prob(vert_idx)
        log_prob_hor = hor_dist.log_prob(hor_idx)
        log_prob_fire = fire_dist.log_prob(fire.to(torch.float32))
        log_prob_act = log_prob_vert + log_prob_hor + log_prob_fire

        # H(act|enm_idx, ally_idx, o)：三路动作条件熵之和；
        # fire_mask=0 的样本开火被规则禁死，Bernoulli 退化，不贡献火动作熵
        fire_mask = pilot_logits['fire_mask'].to(torch.float32)
        ent_vert = vert_dist.entropy()
        ent_hor = hor_dist.entropy()
        ent_fire = fire_dist.entropy() * fire_mask
        ent_act = ent_vert + ent_hor + ent_fire

        # 统计未被禁火样本的开火概率极值（仅记录用，无梯度，mini_batch 级标量）
        with torch.no_grad():
            fire_prob = torch.sigmoid(pilot_logits['fire'].squeeze(-1))
            fire_allowed = fire_mask.bool()
            if fire_allowed.any():
                fire_prob_max = fire_prob[fire_allowed].max().item()
                fire_prob_min = fire_prob[fire_allowed].min().item()
            else:
                fire_prob_max = float('nan')
                fire_prob_min = float('nan')

        # 联合策略：H = H(enm) + H(ally) + H(act|enm_idx, ally_idx, o)
        entropy = ent_enm + ent_ally + ent_act
        # 联合 log_prob：PPO 新旧策略比的分子（分母用回放中存的旧 log_prob_*）
        log_prob = log_prob_enm + log_prob_ally + log_prob_act

        return {
            'log_prob': log_prob,
            'log_prob_enm': log_prob_enm,
            'log_prob_ally': log_prob_ally,
            'log_prob_act': log_prob_act,
            'entropy': entropy,
            'entropy_details': {
                'enm': ent_enm,
                'ally': ent_ally,
                'vert': ent_vert,
                'hor': ent_hor,
                'fire': ent_fire,
                'act': ent_act,
            },
            'fire_prob_max': fire_prob_max,
            'fire_prob_min': fire_prob_min,
        }

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
