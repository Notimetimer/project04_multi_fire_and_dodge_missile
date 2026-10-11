import os
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
_EVADE_HOR_MASK = torch.tensor([-1, -1, 0, 0, 0, -1, -1], dtype=torch.float32) * _MASK_VAL
# 安全模式（no_warning_or_enm=True）下，仅允许水平动作索引 {6}，其余维度置为 -1e8
_SAFE_HOR_MASK = torch.tensor([-1, -1, -1, -1, -1, -1, 0], dtype=torch.float32) * _MASK_VAL


# 单头交叉注意力模块：query 来自自身状态，key/value 来自实体序列。
# 每个实体类型（enm / ally）各持有一份独立实例，参数互不共享。
class EntityAttention(nn.Module):
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
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
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
        self.enm_attn = EntityAttention(fixed_dim, enm_shape[1], embed_dim)
        self.ally_attn = EntityAttention(fixed_dim, ally_shape[1], embed_dim)

    def forward(self, agent_inputs, mask_inputs, enm_idx=None, ally_idx=None):
        fixed = agent_inputs['fixed'] # 应该是(batch, fixed_dim)
        enm = agent_inputs['enm'] # 应该是(batch, enm_num, enm_shape[1])
        ally = agent_inputs['ally'] # 应该是(batch, ally_num, ally_shape[1])
        enm_mask = mask_inputs.get('enm') # 应该是(batch, enm_num)
        ally_mask = mask_inputs.get('ally') # 应该是(batch, ally_num)

        # 两个独立的交叉注意力：各自的 query/key 投影产生打分与嵌入
        enm_scores, enm_keys = self.enm_attn(fixed, enm, enm_mask)
        ally_scores, ally_keys = self.ally_attn(fixed, ally, ally_mask)

        # 敌人选取分布和队友选取分布
        enm_dist = Categorical(logits=enm_scores)
        ally_dist = Categorical(logits=ally_scores)

        if enm_idx is None:
            enm_idx = enm_dist.sample()
        if ally_idx is None:
            ally_idx = ally_dist.sample()

        log_prob_enm = enm_dist.log_prob(enm_idx)
        log_prob_ally = ally_dist.log_prob(ally_idx)

        # enm_keys: (B, enm_num, embed_dim)
        # enm_idx: (B,) -> reshape (-1,1,1) -> (B,1,1) -> expand -> (B, 1, embed_dim)
        # gather(1, ...): 沿 seq_len 维取被选中的实体嵌入 -> (B, 1, embed_dim)
        # squeeze(1) -> e_enm: (B, embed_dim)
        e_enm = enm_keys.gather(1, enm_idx.reshape(-1, 1, 1).expand(-1, 1, self.embed_dim)).squeeze(1)
        e_ally = ally_keys.gather(1, ally_idx.reshape(-1, 1, 1).expand(-1, 1, self.embed_dim)).squeeze(1)

        # selected_info: concat((B, fixed_dim), (B, embed_dim), (B, embed_dim)) -> (B, fixed_dim + 2*embed_dim)
        selected_info = torch.cat((fixed, e_enm, e_ally), dim=-1)
        return selected_info, enm_idx, ally_idx, log_prob_enm, log_prob_ally


# 飞行员网络：接收 selector 选出的信息，输出 vert/hor/fire 三路逻辑值。
# 规则掩码（warning / no_warning_or_enm / just_fired）以 -1e8 偏移量的形式
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
    def _extract_flag(rule_inputs, key, batch_size, device, default=False):
        if isinstance(rule_inputs, list):
            vals = [bool((r or {}).get(key, default)) for r in rule_inputs]
        elif isinstance(rule_inputs, dict):
            vals = [bool((rule_inputs or {}).get(key, default))] * batch_size
        else:
            vals = [bool(default)] * batch_size
        return torch.tensor(vals, dtype=torch.bool, device=device)

    def _warning(self, rule_inputs, batch_size, device):
        # 用户自定义“警告”判定条件
        RWR_infos = rule_inputs["RWR_infos"]
        warning = RWR_infos["warning"]
        return warning

    def _no_warning_or_enm(self, rule_inputs, batch_size, device):
        # TODO: 用户自定义“无警告或敌机”判定条件
        return self._extract_flag(rule_inputs, 'no_warning_or_enm', batch_size, device, default=False)

    def _just_fired(self, rule_inputs, batch_size, device):
        # TODO: 用户自定义“刚开火过”判定条件
        # 示例（留空，按需替换）：
        #   return self._extract_flag(rule_inputs, 'just_fired', batch_size, device, default=False)
        return torch.zeros(batch_size, dtype=torch.bool, device=device)

    def forward(self, selected_info, rule_inputs):
        features = self.backbone(selected_info)

        # 主干输出三路原始 logits
        vert_logits = self.vert_head(features)           # (B, vert_num)
        pre_hor_logits = self.hor_head(features)         # (B, HOR_DIM)
        fire_logits = self.fire_head(features)           # (B, 1)

        B = selected_info.shape[0]
        device = selected_info.device

        warning = self._warning(rule_inputs, B, device)
        no_warning_or_enm = self._no_warning_or_enm(rule_inputs, B, device)
        just_fired = self._just_fired(rule_inputs, B, device)

        # 第一处汇流：hor_logits = pre_hor_logits + delta_evade + delta_safe
        # warning 为真时叠加规避掩码，no_warning_or_enm 为真时叠加安全掩码
        delta_evade = warning.float().unsqueeze(-1) * _EVADE_HOR_MASK.to(device)
        delta_safe = no_warning_or_enm.float().unsqueeze(-1) * _SAFE_HOR_MASK.to(device)
        hor_logits = pre_hor_logits + delta_evade + delta_safe

        # 第二处汇流：fire_logits += delta_fire
        # just_fired 为真时将开火逻辑值置为 -1e8（禁火）
        delta_fire = just_fired.float().unsqueeze(-1) * (-_MASK_VAL)   # (B, 1)
        fire_logits = fire_logits + delta_fire

        # 第三处汇流：打包为 logits 字典，由 actorWrapper 采样并计算 log_prob
        return {
            'hor': hor_logits,
            'vert': vert_logits,
            'fire': fire_logits,
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

    def forward(self, agent_inputs, mask_inputs, rule_inputs, actions=None):
        packed_agent, packed_mask = self.pack_inputs(agent_inputs, mask_inputs)
        sel_actions = None if actions is None else {
            'enm_idx': actions['enm_idx'], 'ally_idx': actions['ally_idx']}
        selected_info, enm_idx, ally_idx, log_prob_enm, log_prob_ally = self.selector(
            packed_agent, packed_mask,
            enm_idx=None if sel_actions is None else sel_actions['enm_idx'],
            ally_idx=None if sel_actions is None else sel_actions['ally_idx'],
        )
        # pilot 只输出三路 logits，采样与 log_prob 在 wrapper 中完成
        pilot_logits = self.pilot(selected_info, rule_inputs)

        vert_dist = Categorical(logits=pilot_logits['vert'])
        hor_dist = Categorical(logits=pilot_logits['hor'])
        fire_dist = Bernoulli(logits=pilot_logits['fire'].squeeze(-1))

        if actions is None:
            vert = vert_dist.sample()
            hor = hor_dist.sample()
            fire = fire_dist.sample().to(torch.long)
        else:
            vert = actions['vert']
            hor = actions['hor']
            fire = actions['fire']

        log_prob_act = (
            vert_dist.log_prob(vert)
            + hor_dist.log_prob(hor)
            + fire_dist.log_prob(fire.to(torch.float32))
        )

        return {
            'hor': hor,
            'vert': vert,
            'fire': fire,
            'log_prob_act': log_prob_act,
            'enm_idx': enm_idx,
            'ally_idx': ally_idx,
            'log_prob_enm': log_prob_enm,
            'log_prob_ally': log_prob_ally,
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
