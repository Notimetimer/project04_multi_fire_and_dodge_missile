"""HANSACHybrid：按 Han et al. (2026) JPO 路线的离策略混合 SAC 智能体。

论文对照：
- 式(29)-(30)：联合目标 J^J = Σ πJ·Q + αJ·H^J；TIM/TDM 目标在其上加组内条件熵项
  （本实现把 αI·Pr(TIM)·H^I 与 αD·Σ_j Pr(TDM_j)·H^D_j 直接并入软值函数与策略损失，
  与"indicator 加权"形式在期望意义下等价）；
- 式(31)-(33)：双 Q 的 TD 目标，软值含联合熵与分组熵，target 网络软更新（式 39，τ=0.05）；
- 式(34)：策略损失 Lπ；式(35)：LNet = Lf + LQ1 + LQ2 + Lπ 求和后一次反传（不冻结主干）；
- 式(36)-(38)：αI/αD/αJ 三个温度分别自动调节；期望熵 Ḅ^I=kI·log|AI|、Ḅ^D=kD·log|AD|，
  Ḅ^J 按式(48) 由当前组质量自适应合成（Pr(TIM)·Ḅ^I + Σ Pr(TDM_j)·Ḅ^D + H(Γ)）；
- 经验回放：D_BFM（机动转移，容量 1e5）与 D_f（开火命中记录，容量 1e3，批 128）；
- 开火不进 RL 梯度：Lf 只含 BCE（式 28），开火不出现在 policy/Q/熵项中。

数据约定（与 CombatHanSAC_parall_homo.worker_process_han 对齐）：
单步 transition = {
  'obs':  {'agent_input','mask_input','rule_input'},   # 决策时刻观测三元组
  'action': {'flat_idx': int, ...},                    # 扁平机动动作索引
  'reward': float, 'next_obs': {...}, 'done': 0/1, 'active_mask': 0/1
}
开火记录 = {'fire_obs': agent_input, 'enm_idx': int, 'hit_record': 0/1}（环境端回填）。
"""

import copy
import random
from collections import deque
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass

from Algorithms.HanWrapper import HanWrapper, N_TIM, N_TDM


@dataclass(frozen=True)
class TransitionBatch:
    """与预留抽象接口对齐的批结构（本实现内部使用 dict 形式，此结构仅供外部调用方参考）。"""
    context: Any
    action: Any
    reward: Any
    next_context: Any
    terminated: Any
    truncated: Any


@dataclass(frozen=True)
class UpdateResult:
    metrics: Mapping[str, float]


# 与 _context.stack_field 同语义的递归堆叠（Algorithms 包内自带一份，避免依赖训练侧 _context）
def _stack_field(field_list):
    sample = next((v for v in field_list if v is not None), None)
    if sample is None:
        return None
    if isinstance(sample, dict):
        keys = set()
        for item in field_list:
            if item is not None:
                keys.update(item.keys())
        return {k: _stack_field([item.get(k) if item is not None else None
                                 for item in field_list]) for k in keys}
    return np.array(field_list)


def _tree_index(tree, i):
    """递归取 dict-of-arrays 的第 i 个样本（stack_field 的逆操作）。"""
    if isinstance(tree, dict):
        return {k: _tree_index(v, i) for k, v in tree.items()}
    return tree[i]


class ReplayBuffer:
    """D_BFM：机动转移回放池，容量 1e5（论文规模），随机均匀采样。"""

    def __init__(self, capacity=int(1e5)):
        self.capacity = int(capacity)
        self.storage = []
        self.pos = 0

    def push(self, transition):
        if len(self.storage) < self.capacity:
            self.storage.append(transition)
        else:
            self.storage[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def push_stacked(self, stacked_trans):
        """把 worker 堆叠好的 dict-of-arrays 拆成单步转移逐条写入（键名归一化为单数）。"""
        key_map = {'actions': 'action', 'rewards': 'reward',
                   'dones': 'done', 'active_masks': 'active_mask'}
        n = len(stacked_trans['dones'])
        for i in range(n):
            single = _tree_index(stacked_trans, i)
            self.push({key_map.get(k, k): v for k, v in single.items()})

    def sample(self, batch_size):
        return random.sample(self.storage, min(batch_size, len(self.storage)))

    def __len__(self):
        return len(self.storage)


class HANSACHybrid:
    """离策略混合 SAC：双 Q + 目标网络 + 三温度自动调节 + 开火 BCE 监督（式 35 联合反传）。"""

    def __init__(self, networks: HanWrapper,
                 lr=1e-4, gamma=0.9, tau=0.05, batch_size=128,
                 buffer_size=int(1e5),
                 fire_buffer_size=1000, fire_batch_size=128, fire_min_samples=100,
                 kI=0.8, kD=0.8, alpha_init=0.2, max_grad_norm=5.0,
                 device='cpu'):
        self.ac = networks
        self.ac_target = copy.deepcopy(networks)
        self.ac_target.eval()
        for p in self.ac_target.parameters():
            p.requires_grad_(False)

        self.gamma = gamma          # 论文 γ=0.9
        self.tau = tau              # 式(39) τ=0.05
        self.batch_size = batch_size
        self.device = device
        self.max_grad_norm = max_grad_norm

        # LNet = Lf + LQ1 + LQ2 + Lπ 对同一参数集一次反传（式 35），单个优化器
        self.net_optimizer = torch.optim.Adam(self.ac.parameters(), lr=lr)

        # 三个温度的 log 参数（式 36-38 自动调节）
        self.log_alpha_I = torch.tensor(np.log(alpha_init), device=device, requires_grad=True)
        self.log_alpha_D = torch.tensor(np.log(alpha_init), device=device, requires_grad=True)
        self.log_alpha_J = torch.tensor(np.log(alpha_init), device=device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha_I, self.log_alpha_D, self.log_alpha_J], lr=lr)

        # 期望熵：Ḅ^I = kI·log|AI|，Ḅ^D = kD·log|AD|（论文取 kI=kD=0.8）
        self.B_I = float(kI * np.log(N_TIM))
        self.B_D = float(kD * np.log(N_TDM))

        self.buffer = ReplayBuffer(buffer_size)                    # D_BFM
        self.fire_buffer = deque(maxlen=int(fire_buffer_size))     # D_f
        self.fire_batch_size = int(fire_batch_size)
        self.fire_min_samples = int(fire_min_samples)

        self.update_count = 0

    # ------------------------------------------------------------------ 数据写入
    def push_stacked(self, stacked_trans):
        self.buffer.push_stacked(stacked_trans)

    def push_fire_records(self, records):
        self.fire_buffer.extend(records)

    # ------------------------------------------------------------------ 内部量
    def _soft_value_terms(self, out, alpha_I, alpha_D, alpha_J):
        """给定 forward_sac 输出，返回 (ΣπQmin 所需 qmin, 熵加权和)。
        熵加权和 = αJ·H^J + αI·Pr(G0)·H^I + αD·Σ_j Pr(Gj)·H^D_j。"""
        qmin = torch.min(out['q1'], out['q2'])
        ent = alpha_J * out['H_J'] \
            + alpha_I * out['pr_g'][:, 0] * out['H_I'] \
            + alpha_D * (out['pr_g'][:, 1:] * out['H_Dj']).sum(-1)
        return qmin, ent

    # ------------------------------------------------------------------ 主更新
    def update(self, batch: Any = None, n_updates=1) -> UpdateResult:
        """执行 n_updates 次梯度更新；batch 参数仅为抽象接口兼容（数据来自内部回放池）。"""
        metrics_acc = {}
        n_done = 0
        for _ in range(int(n_updates)):
            if len(self.buffer) < self.batch_size:
                break
            m = self._update_once()
            n_done += 1
            for k, v in m.items():
                metrics_acc[k] = metrics_acc.get(k, 0.0) + v
        if n_done == 0:
            return UpdateResult({})
        return UpdateResult({k: v / n_done for k, v in metrics_acc.items()})

    def _update_once(self):
        device = self.device
        ac, ac_t = self.ac, self.ac_target

        # ---------------- 采样 D_BFM ----------------
        trans = self.buffer.sample(self.batch_size)
        obs = _stack_field([t['obs'] for t in trans])
        next_obs = _stack_field([t['next_obs'] for t in trans])
        flat_idx = torch.as_tensor(np.array([t['action']['flat_idx'] for t in trans]),
                                   dtype=torch.long, device=device)
        reward = torch.as_tensor(np.array([t['reward'] for t in trans], dtype=np.float32),
                                 dtype=torch.float32, device=device)
        done = torch.as_tensor(np.array([t['done'] for t in trans], dtype=np.float32),
                               dtype=torch.float32, device=device)
        active = torch.as_tensor(np.array([t['active_mask'] for t in trans], dtype=np.float32),
                                 dtype=torch.float32, device=device)
        act_sum = active.sum().clamp(min=1.0)

        alpha_I = self.log_alpha_I.exp().detach()
        alpha_D = self.log_alpha_D.exp().detach()
        alpha_J = self.log_alpha_J.exp().detach()

        # ---------------- 在线/目标网络前向 ----------------
        out = ac.forward_sac(obs['agent_input'], obs['mask_input'], obs['rule_input'])
        with torch.no_grad():
            out_t = ac_t.forward_sac(next_obs['agent_input'], next_obs['mask_input'],
                                     next_obs['rule_input'])

        # ---------------- 双 Q 的 TD 损失（式 31-33） ----------------
        with torch.no_grad():
            qmin_t, ent_t = self._soft_value_terms(out_t, alpha_I, alpha_D, alpha_J)
            V_t = (out_t['pi'] * qmin_t).sum(-1) + ent_t
            y = reward + self.gamma * (1.0 - done) * V_t

        q1_taken = out['q1'].gather(1, flat_idx.unsqueeze(1)).squeeze(1)
        q2_taken = out['q2'].gather(1, flat_idx.unsqueeze(1)).squeeze(1)
        LQ1 = (((q1_taken - y) ** 2) * active).sum() / act_sum
        LQ2 = (((q2_taken - y) ** 2) * active).sum() / act_sum

        # ---------------- 策略损失（式 34：−J，含联合熵与分组熵） ----------------
        qmin, ent = self._soft_value_terms(out, alpha_I, alpha_D, alpha_J)
        Lpi = (-((out['pi'] * qmin).sum(-1) + ent) * active).sum() / act_sum

        # ---------------- 开火监督 Lf（式 28，不冻结主干） ----------------
        Lf = torch.zeros((), device=device)
        if len(self.fire_buffer) >= self.fire_min_samples:
            fire_batch = random.sample(list(self.fire_buffer),
                                       min(self.fire_batch_size, len(self.fire_buffer)))
            fire_obs = _stack_field([r['fire_obs'] for r in fire_batch])
            fire_idx = np.array([r['enm_idx'] for r in fire_batch])
            fire_hit = np.array([r['hit_record'] for r in fire_batch], dtype=np.float32)
            Lf = self.ac.evaluate_fire(fire_obs, fire_idx, fire_hit)

        # ---------------- LNet 求和，一次反传（式 35） ----------------
        LNet = LQ1 + LQ2 + Lpi + Lf
        self.net_optimizer.zero_grad()
        LNet.backward()
        torch.nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
        self.net_optimizer.step()

        # ---------------- 温度更新（式 36-38，熵全部 detach） ----------------
        pr_g_det = out_t['pr_g']  # 用目标策略的组质量合成期望联合熵（式 48 口径，无梯度）
        H_gamma = -(pr_g_det * torch.log(pr_g_det.clamp(min=1e-8))).sum(-1)   # H(Γ)
        B_J = (pr_g_det[:, 0] * self.B_I + pr_g_det[:, 1:].sum(-1) * self.B_D + H_gamma).detach()

        L_alpha_I = (self.log_alpha_I * ((out['H_I'] - self.B_I) * active).detach().sum() / act_sum)
        L_alpha_D = (self.log_alpha_D * ((out['H_D'] - self.B_D) * active).detach().sum() / act_sum)
        L_alpha_J = (self.log_alpha_J * ((out['H_J'] - B_J) * active).detach().sum() / act_sum)
        alpha_loss = L_alpha_I + L_alpha_D + L_alpha_J
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # ---------------- 目标网络软更新（式 39） ----------------
        with torch.no_grad():
            for p, tp in zip(self.ac.parameters(), self.ac_target.parameters()):
                tp.mul_(1 - self.tau).add_(p, alpha=self.tau)

        self.update_count += 1
        return {
            'loss_q': float((LQ1 + LQ2).detach().cpu()),
            'loss_pi': float(Lpi.detach().cpu()),
            'loss_fire': float(Lf.detach().cpu()) if torch.is_tensor(Lf) else 0.0,
            'H_J': float(out['H_J'].mean().detach().cpu()),
            'H_I': float(out['H_I'].mean().detach().cpu()),
            'H_D': float(out['H_D'].mean().detach().cpu()),
            'alpha_I': float(alpha_I), 'alpha_D': float(alpha_D), 'alpha_J': float(alpha_J),
            'B_J': float(B_J.mean().cpu()),
            'q_mean': float(qmin.mean().detach().cpu()),
            'fire_buffer': float(len(self.fire_buffer)),
            'replay_size': float(len(self.buffer)),
        }

    # ------------------------------------------------------------------ 存取（断点续训）
    def state_dict(self):
        return {
            'ac': self.ac.state_dict(),
            'ac_target': self.ac_target.state_dict(),
            'net_optimizer': self.net_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
            'log_alpha_I': float(self.log_alpha_I.detach().cpu()),
            'log_alpha_D': float(self.log_alpha_D.detach().cpu()),
            'log_alpha_J': float(self.log_alpha_J.detach().cpu()),
            'update_count': self.update_count,
        }

    def load_state_dict(self, state):
        self.ac.load_state_dict(state['ac'])
        self.ac_target.load_state_dict(state['ac_target'])
        self.net_optimizer.load_state_dict(state['net_optimizer'])
        self.alpha_optimizer.load_state_dict(state['alpha_optimizer'])
        self.log_alpha_I.data = torch.tensor(state['log_alpha_I'], device=self.device)
        self.log_alpha_D.data = torch.tensor(state['log_alpha_D'], device=self.device)
        self.log_alpha_J.data = torch.tensor(state['log_alpha_J'], device=self.device)
        self.update_count = state.get('update_count', 0)
