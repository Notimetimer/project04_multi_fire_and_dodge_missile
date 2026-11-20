"""COMAWrapper：COMA 集中式 critic 版网络（actor 侧完全复用 HanWrapper）。

来源与关系（对比算法规范 §"共享优先，原件不动"）：
- 本文件不复制 HanWrapper.py，直接 import 其 actor 组件（EntityBackbone / _mlp / 动作常量）；
- COMAWrapper 继承 HanWrapper，只替换 critic：HanWrapper 的分散式双 Q 在 __init__ 里置空，
  新增集中式单 Q（COMA_Critic）；actor 骨干、TIM/TDM 头、fire_head、get_action、
  evaluate_fire、save/load 全部沿用父类实现（HanWrapper.py 原文件字节不变）。

集中式 critic 结构（为什么这样设计）：
1) 每个红方 agent 的本地观测 (fixed, enm, ally) 先过共享 EntityBackbone（即 HanWrapper
   式(9)-(16) 的异构注意力编码），得到每机一个 token；
2) 各 agent 的动作 one-hot 嵌入加到 token 上作为 key/value；agent i 的 query 只允许看
   **队友**的 key（自身 key 被屏蔽），因此输出 Q_i(s, u_{-i}, ·) 与自己的动作无关，
   一次前向就给出整行反事实 Q，不需要对每个候选动作重复前向；
3) 共享 MLP 头输出 (B, N, |U|)，|U| = 2A（机动×开火联合动作，与 pi_joint 索引一致，
   联合索引 = flat_idx·2 + fire）；
4) 队伍规模变长：N 来自输入张量，掩码按存活 agent 屏蔽，训练 2V2、评测 4V4/6V6 无需
   重新定义网络维度。

已声明偏差（declared_deviations）：
1) critic 的"状态"是各 agent 本地可观测的集合编码，不是上帝视角真值状态；
2) 用异构注意力 + 队伍级交叉注意力取代 COMA 论文/PyMARL 的全连接 MLP critic；
3) 保留独立 fire_head：开火仍由 BCE 监督，反事实基线里的开火概率 detach（动作空间因此
   是 2A 而不是 A）；
4) 不做 GRU：actor 与 critic 都是单步前馈（由集中 critic 补状态）。

接口：
- get_action / forward_actor / evaluate_fire / save / load：与 HanWrapper 完全一致；
- forward_joint(agent_input, mask_input, rule_input)：批量 (B,N,...) 输入 → 逐 agent 的
  pi_joint / log_pi / 掩码，形状为 (B,N,...)；
- forward_critic(agent_input, mask_input, joint_act, agent_alive)：批量 (B,N,...) →
  (B,N,2A) 的反事实 Q 行（Q_i(s, u_{-i}, a) 对所有 a）；
- forward(context) 不再提供：COMA 训练路径请调用上面两个方法。
"""

import numpy as np
import torch
from torch import nn

from Algorithms.HanWrapper import HanWrapper, EntityBackbone, _mlp

_MASK_NEG = -1e8  # 注意力屏蔽填充值


def _flat_tree(tree, B, N):
    """(B,N,...) 嵌套结构 → (B*N,...)，用于复用 HanWrapper 的逐 agent 前向。"""
    if tree is None:
        return None
    if isinstance(tree, dict):
        return {k: _flat_tree(v, B, N) for k, v in tree.items()}
    arr = np.asarray(tree)
    if arr.ndim <= 1:
        return arr.reshape(B * N)
    return arr.reshape(B * N, *arr.shape[2:])


def _unflat_tree(tree, B, N):
    """把逐 agent 输出 (B*N,...) 还原成 (B,N,...)。"""
    if isinstance(tree, dict):
        return {k: _unflat_tree(v, B, N) for k, v in tree.items()}
    if not torch.is_tensor(tree):
        return tree
    return tree.reshape(B, N, *tree.shape[1:])


def pack_team_arrays(agent_input, mask_input, enm_num, ally_num, device):
    """(B,N,...) 观测/掩码 → 张量五元组；在线 critic 与目标 critic 共用同一套输入整理。"""
    arr = np.asarray(agent_input['fixed'])
    B, N = (1, arr.shape[0]) if arr.ndim == 2 else (arr.shape[0], arr.shape[1])
    fixed = torch.as_tensor(np.asarray(agent_input['fixed'], dtype=np.float32), device=device)
    enm = torch.as_tensor(np.asarray(agent_input['enm'], dtype=np.float32), device=device)
    ally = torch.as_tensor(np.asarray(agent_input['ally'], dtype=np.float32), device=device)
    if fixed.ndim == 2:
        fixed = fixed.unsqueeze(0)
    if enm.ndim == 3:
        enm = enm.unsqueeze(0)
    if ally.ndim == 3:
        ally = ally.unsqueeze(0)

    def _mask(key, n):
        m = None if mask_input is None else mask_input.get(key)
        if m is None:
            return torch.ones(B, N, n, dtype=torch.bool, device=device)
        m = torch.as_tensor(np.asarray(m), dtype=torch.bool, device=device)
        if m.ndim == 2:
            m = m.unsqueeze(0)
        return m.reshape(B, N, n)

    return fixed, enm, ally, _mask('enm', enm_num), _mask('ally', ally_num)


class TeamCrossAttnLayer(nn.Module):
    """队伍级多头交叉注意力：query=本机 token，key/value=各 agent token + 动作嵌入。

    kv_mask 是 (B, N_query, N_key) 的布尔掩码：本机自身 key 恒为 False，
    因此注意力输出不含自身动作信息，Q_i(s, u_{-i}, ·) 一次前向即可得到。
    """

    def __init__(self, embed_dim, n_heads=4):
        super().__init__()
        assert embed_dim % n_heads == 0, 'embed_dim 必须能被 n_heads 整除'
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.w_q = nn.Linear(embed_dim, embed_dim)
        self.w_k = nn.Linear(embed_dim, embed_dim)
        self.w_v = nn.Linear(embed_dim, embed_dim)
        self.w_o = nn.Linear(embed_dim, embed_dim)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, tok, kv, kv_mask):
        B, N, E = tok.shape
        h, d = self.n_heads, self.head_dim
        q = self.w_q(tok).reshape(B, N, h, d).permute(0, 2, 1, 3)   # (B,h,N,d)
        k = self.w_k(kv).reshape(B, N, h, d).permute(0, 2, 1, 3)
        v = self.w_v(kv).reshape(B, N, h, d).permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale   # (B,h,N,N)
        scores = scores.masked_fill(~kv_mask[:, None, :, :], _MASK_NEG)
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(B, N, E)
        # N=1（或队友全灭）时没有有效 key，softmax 会退化成均匀分布，这里显式清零
        has_key = kv_mask.any(-1)                                    # (B,N)
        ctx = ctx * has_key[:, :, None].to(ctx.dtype)
        return self.ln(tok + self.w_o(ctx))


class COMA_Critic(nn.Module):
    """集中式单 Q：Q_i(s, u_{-i}, ·)，输出 (B,N,2A)。

    local_enc 为共享的逐 agent 编码（EntityBackbone，式 9-16 的异构注意力），
    act_emb 把联合动作索引 flat_idx·2+fire 映射为嵌入，q_head 共享。
    """

    def __init__(self, fixed_dim, enm_dim, ally_dim, n_joint,
                 embed_dim=32, n_layers=2, head_hidden=128, n_heads=4):
        super().__init__()
        self.n_joint = int(n_joint)
        self.embed_dim = int(embed_dim)
        self.local_enc = EntityBackbone(fixed_dim, enm_dim, ally_dim, embed_dim, n_layers)
        self.act_emb = nn.Embedding(self.n_joint, embed_dim)
        self.team_layers = nn.ModuleList(
            [TeamCrossAttnLayer(embed_dim, n_heads) for _ in range(n_layers)])
        self.q_head = _mlp(embed_dim, head_hidden, self.n_joint)

    def forward(self, fixed, enm, ally, enm_mask=None, ally_mask=None,
                joint_act=None, agent_alive=None):
        # fixed: (B,N,F)；enm: (B,N,Ne,D)；ally: (B,N,Na,D)；joint_act: (B,N) long
        B, N = fixed.shape[0], fixed.shape[1]
        if enm_mask is None:
            enm_mask = torch.ones(B, N, enm.shape[2], dtype=torch.bool, device=fixed.device)
        if ally_mask is None:
            ally_mask = torch.ones(B, N, ally.shape[2], dtype=torch.bool, device=fixed.device)
        if agent_alive is None:
            agent_alive = torch.ones(B, N, dtype=torch.bool, device=fixed.device)
        if joint_act is None:
            raise ValueError('COMA_Critic 需要联合动作索引 joint_act')

        # 逐 agent 本地编码：把 (B,N) 并进 batch 维，复用 EntityBackbone
        e_f, _ = self.local_enc(
            fixed.reshape(B * N, -1),
            enm.reshape(B * N, enm.shape[2], enm.shape[3]),
            ally.reshape(B * N, ally.shape[2], ally.shape[3]),
            enm_mask.reshape(B * N, -1), ally_mask.reshape(B * N, -1))
        tok = e_f.reshape(B, N, self.embed_dim)                        # (B,N,E)

        act = joint_act.reshape(B, N).clamp(0, self.n_joint - 1)
        kv = tok + self.act_emb(act)                                   # 队友动作进 key/value

        # 每个 query 只允许看队友（自身 key 屏蔽）→ Q_i 与自身动作无关
        eye = torch.eye(N, dtype=torch.bool, device=fixed.device).unsqueeze(0)
        kv_mask = agent_alive[:, :, None] & (~eye)
        for layer in self.team_layers:
            tok = layer(tok, kv, kv_mask)
        return self.q_head(tok)                                        # (B,N,2A)


class COMAWrapper(HanWrapper):
    """COMA 网络壳：actor 复用 HanWrapper，critic 换成集中式单 Q。"""

    def __init__(self, fixed_dim, enm_shape, ally_shape,
                 embed_dim=32, n_layers=2, head_hidden=128, n_heads=4,
                 device='cpu', **_unused):
        super().__init__(fixed_dim, enm_shape, ally_shape,
                         embed_dim=embed_dim, n_layers=n_layers,
                         head_hidden=head_hidden, device=device)
        # 移除父类的分散式双 Q（置 None 会同时从 _modules 注销，state_dict 不再包含它们）
        for _name in ('critic_backbone', 'tim_q1_head', 'tim_q2_head',
                      'opp_q1_head', 'opp_q2_head'):
            setattr(self, _name, None)
        self.critic = COMA_Critic(fixed_dim, self.enm_shape[1], self.ally_shape[1],
                                  self.n_joint, embed_dim, n_layers, head_hidden, n_heads)
        self.critic.to(self.device)

    # ------------------------------------------------------------------ 参数分组
    def critic_parameters(self):
        """价值侧参数：集中式 critic（本地编码 + 动作嵌入 + 队伍注意力 + Q 头）。"""
        return list(self.critic.parameters())

    # ------------------------------------------------------------------ 队伍批量整理
    @staticmethod
    def _team_ndim(agent_input):
        """返回 (B, N)：接受 (N,...) 单步或 (B,N,...) 批量。"""
        arr = np.asarray(agent_input['fixed'])
        if arr.ndim == 2:          # (N,F) → 单步
            return 1, arr.shape[0]
        return arr.shape[0], arr.shape[1]

    def _pack_team(self, agent_input, mask_input):
        """把 (B,N,...) 观测/掩码整理成张量；缺省 mask 视为全部有效。"""
        return pack_team_arrays(agent_input, mask_input, self.enm_num, self.ally_num, self.device)

    # ------------------------------------------------------------------ 集中式 critic 前向
    def forward_critic(self, agent_input, mask_input, joint_act, agent_alive=None):
        """返回反事实 Q 行 (B,N,2A)：Q_i(s, u_{-i}, a) 对所有 a。

        joint_act 为联合动作索引 flat_idx·2+fire（与 pi_joint 索引一致）；
        agent_alive 为 (B,N)/(N,) 的 0/1 或 bool，缺省全存活。
        """
        fixed, enm, ally, enm_mask, ally_mask = self._pack_team(agent_input, mask_input)
        B, N = fixed.shape[0], fixed.shape[1]
        act = torch.as_tensor(np.asarray(joint_act), dtype=torch.long, device=self.device)
        if act.ndim == 1:
            act = act.unsqueeze(0)
        act = act.reshape(B, N)
        if agent_alive is None:
            alive = torch.ones(B, N, dtype=torch.bool, device=self.device)
        else:
            alive = torch.as_tensor(np.asarray(agent_alive), dtype=torch.bool,
                                    device=self.device).reshape(B, N)
        return self.critic(fixed, enm, ally, enm_mask, ally_mask, act, alive)

    # ------------------------------------------------------------------ 逐 agent 策略前向（批量）
    def forward_joint(self, agent_input, mask_input, rule_input):
        """(B,N,...) 输入 → 逐 agent 的 pi / pi_joint / log_pi / 开火门控等，形状 (B,N,...)。

        实现上把 (B,N) 并进 batch 维交给父类 forward_actor，再把输出还原成 (B,N,...)，
        因此策略分布、动作掩码、开火门控与部署路径 get_action 逐位一致。
        """
        B, N = self._team_ndim(agent_input)
        out = self.forward_actor(_flat_tree(agent_input, B, N),
                                 _flat_tree(mask_input, B, N),
                                 _flat_tree(rule_input, B, N))
        return _unflat_tree(out, B, N)

    # ------------------------------------------------------------------ 禁用父类 SAC 接口
    def forward(self, context):
        """COMA 训练路径不走父类的 SAC 前向（其 critic 已被移除）。"""
        raise NotImplementedError(
            'COMAWrapper 使用集中式 critic：请调用 forward_joint / forward_critic')
