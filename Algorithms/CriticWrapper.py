import numpy as np
import torch
from torch import nn


# 多头交叉注意力模块：query 来自自身状态（fixed），key/value 来自实体序列。
# enm / ally 各持有一份独立实例，参数互不共享；实体特征经注意力聚合后
# 扩散进自身状态表示（允许特征在 entity 之间流向 query）。
class EntityCrossAttention(nn.Module):
    def __init__(self, query_dim, entity_dim, embed_dim, num_heads=4):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, embed_dim)     # 把 fixed 投影到统一嵌入维度
        self.entity_proj = nn.Linear(entity_dim, embed_dim)   # 把实体特征投影到统一嵌入维度
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, query_state, entity_seq):
        query = self.query_proj(query_state).unsqueeze(1)   # (B, fixed_dim) -> (B, 1, E)
        entity = self.entity_proj(entity_seq)               # (B, N, entity_dim) -> (B, N, E)
        # need_weights=True 额外返回注意力权重，即"注意力计算时产生的每实体标量打分"
        agg, weights = self.attn(query, entity, entity, need_weights=True)
        # agg: (B, 1, E) -> (B, E) 是该实体组对 fixed 的聚合特征
        # weights: (B, 1, N) -> (B, N) 是每个实体槽位的标量 logits
        return agg.squeeze(1), weights.squeeze(1)


# 采集器网络：只接收 agent_inputs（丢弃 mask_inputs 与 rule_inputs），
# 用两个独立的多头注意力模块（enm 组 / ally 组）让实体特征扩散，
# 输出固定结构的特征向量 e、enm_logits、ally_logits。
class collector(nn.Module):
    def __init__(self, fixed_dim, enm_shape, ally_shape,
                 embed_dim=64, num_heads=4):
        super().__init__()
        self.fixed_dim = fixed_dim
        self.enm_shape = enm_shape
        self.ally_shape = ally_shape
        self.enm_num = enm_shape[0]
        self.ally_num = ally_shape[0]
        self.embed_dim = embed_dim

        # enm 与 ally 严格使用两个不同的多头注意力模块，输出两组不同特征
        self.enm_attn = EntityCrossAttention(fixed_dim, enm_shape[1], embed_dim, num_heads)
        self.ally_attn = EntityCrossAttention(fixed_dim, ally_shape[1], embed_dim, num_heads)

    def forward(self, agent_inputs):
        fixed = agent_inputs['fixed']   # (B, fixed_dim)
        enm = agent_inputs['enm']       # (B, enm_num, enm_dim)
        ally = agent_inputs['ally']     # (B, ally_num, ally_dim)

        # 两个独立的交叉注意力：fixed 作 query，实体序列作 key/value
        agg_enm, enm_logits = self.enm_attn(fixed, enm)      # (B, E), (B, enm_num)
        agg_ally, ally_logits = self.ally_attn(fixed, ally)  # (B, E), (B, ally_num)

        # e 固定结构：cat((B, fixed_dim), 聚合enm (B, E), 聚合ally (B, E))
        e = torch.cat((fixed, agg_enm, agg_ally), dim=-1)    # (B, fixed_dim + 2*E)
        return e, enm_logits, ally_logits


# 评估器网络：单线性输出头的 MLP，输入固定维度特征向量 e，输出 V(s) 值。
class judge(nn.Module):
    def __init__(self, input_dim, hidden_dims=(256, 256)):
        super().__init__()
        layers = []
        prev = input_dim
        for hidden in hidden_dims:
            layers.extend((nn.Linear(prev, hidden), nn.ReLU()))
            prev = hidden
        self.backbone = nn.Sequential(*layers)
        self.value_head = nn.Linear(prev, 1)

    def forward(self, e):
        features = self.backbone(e)                  # (B, hidden)
        value = self.value_head(features)            # (B, 1)
        return value.squeeze(-1)                     # (B,) 的 V(s)


# 统筹 collector 与 judge 的 CriticWrapper：
# 对外接口为 agent_inputs -> enm_logits / ally_logits / value，并提供 save / load。
class critic_wrapper(nn.Module):
    def __init__(self, fixed_dim, enm_shape, ally_shape,
                 embed_dim=64, num_heads=4, hidden_dims=(256, 256), device='cpu'):
        super().__init__()
        self.fixed_dim = fixed_dim
        self.enm_shape = enm_shape
        self.ally_shape = ally_shape
        self.enm_num = enm_shape[0]
        self.ally_num = ally_shape[0]
        self.input_dim = fixed_dim + 2 * embed_dim

        self.collector = collector(fixed_dim, enm_shape, ally_shape, embed_dim, num_heads).to(device)
        self.judge = judge(self.input_dim, hidden_dims).to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    def _to_tensor(self, value, dtype, shape):
        # 单样本输入自动补 batch 维，回放批量输入保持原样
        tensor = torch.as_tensor(value, dtype=dtype, device=self.device)
        if tensor.ndim == len(shape):
            tensor = tensor.unsqueeze(0)
        return tensor

    def get_value(self, agent_inputs, h=None):
        # critic 只使用 agent_inputs，丢弃 mask_inputs 与 rule_inputs
        fixed = self._to_tensor(agent_inputs['fixed'], torch.float32, (self.fixed_dim,))
        enm = self._to_tensor(agent_inputs['enm'], torch.float32, self.enm_shape)
        ally = self._to_tensor(agent_inputs['ally'], torch.float32, self.ally_shape)

        e, enm_logits, ally_logits = self.collector({'fixed': fixed, 'enm': enm, 'ally': ally})
        value = self.judge(e)

        return {
            'enm_logits': enm_logits,
            'ally_logits': ally_logits,
            'value': value,
        }

    def save(self, path='critic.pt'):
        torch.save({
            'collector': self.collector.state_dict(),
            'judge': self.judge.state_dict(),
        }, path)

    def load(self, path='critic.pt', device=None):
        target = device or self.device
        checkpoint = torch.load(path, map_location=target)
        self.collector.load_state_dict(checkpoint['collector'])
        self.judge.load_state_dict(checkpoint['judge'])
