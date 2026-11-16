"""HANSACHybrid：按 Han et al. (2026) JPO 路线的离策略混合 SAC 智能体。

论文对照：
- 式(29)-(30)：联合目标 J^J = Σ πJ·Q + αJ·H^J；TIM/TDM 目标在其上加组内条件熵项
  （本实现把 αI·Pr(TIM)·H^I 与 αD·Σ_j Pr(TDM_j)·H^D_j 直接并入软值函数与策略损失，
  与"indicator 加权"形式在期望意义下等价）；
- 式(31)-(33)：双 Q 的 TD 目标，软值含联合熵与分组熵，target 网络软更新（式 39，τ 由调用方设定）；
- 式(34)：策略损失 Lπ；式(35)：本实现不沿用论文的共享骨干单次联合反传，改为
  actor / critic 各自独立骨干与独立优化器、分步反传（标准 SAC 口径，见类 docstring 偏差声明）；
  （更新节奏：Q 每次更新都反传，π/温度每 actor_update_every 次更新反传一次，
   Lf 每 fire_update_every 次更新反传一次，各自在自己的参数上裁剪梯度。）
- 式(36)-(38)：αI/αD/αJ 三个温度分别自动调节；期望熵 Ḅ^I=kI·log|AI|、Ḅ^D=kD·log|AD|，
  Ḅ^J 按式(48) 由当前组质量自适应合成（Pr(TIM)·Ḅ^I + Σ Pr(TDM_j)·Ḅ^D + H(Γ)）；
- 经验回放：D_BFM（机动转移，容量 1e5）与 D_f（开火命中记录，容量 1e3，批 128）；
- 开火进 RL 梯度（已声明偏差）：双 Q 覆盖"机动 × 开火"联合动作，开火位进入 soft value、
  策略损失与熵项，因此 RL 梯度可以更新独立 fire_head；Lf（式 28）只做 BCE 且 backbone
  已 detach，梯度只落在 fire_head（与 PPO 的 supervised_fire_update 同口径）。

数据约定（与 CombatHanSAC_parall_homo.worker_process_han 对齐）：
单步 transition = {
  'obs':  {'agent_input','mask_input','rule_input'},   # 决策时刻观测三元组
  'action': {'flat_idx': int, ...},                    # 扁平机动动作索引
  'reward': float, 'next_obs': {...}, 'active_mask': 0/1,
  'terminated': 0/1,   # 环境终结（胜负已分）：TD 目标不 bootstrap
  'truncated': 0/1,    # 时限截断：TD 目标照常 bootstrap（time-limit aware）
}
开火记录 = {'fire_obs': agent_input, 'enm_idx': int, 'hit_record': 0/1}（环境端回填）。
末端门控：回放池中末端转移（terminated+truncated）累计不足 min_terminal_samples 时不做梯度更新。
"""

import copy
import random
from collections import deque
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass

from Algorithms.HanWrapper import HanWrapper, N_TDM


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
        self.n_terminated = 0    # 累计入池的环境终结帧数（末端门控用）
        self.n_truncated = 0     # 累计入池的时限截断帧数（末端门控用）

    @property
    def n_ended(self):
        return self.n_terminated + self.n_truncated

    def push(self, transition):
        if len(self.storage) < self.capacity:
            self.storage.append(transition)
        else:
            self.storage[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def push_stacked(self, stacked_trans):
        """把 worker 堆叠好的 dict-of-arrays 拆成单步转移逐条写入（键名归一化为单数）。"""
        key_map = {'actions': 'action', 'rewards': 'reward',
                   'terminateds': 'terminated', 'truncateds': 'truncated',
                   'active_masks': 'active_mask',
                   'next_states': 'next_obs'}
        n = len(stacked_trans['terminateds'])
        for i in range(n):
            single = _tree_index(stacked_trans, i)
            self.n_terminated += int(single['terminateds'])
            self.n_truncated += int(single['truncateds'])
            self.push({key_map.get(k, k): v for k, v in single.items()})

    def sample(self, batch_size):
        return random.sample(self.storage, min(batch_size, len(self.storage)))

    def __len__(self):
        return len(self.storage)


class HANSACHybrid:
    """离策略混合 SAC：双 Q + 目标网络 + 温度自动调节（αD/per-Ñ αJ）+ 开火 BCE 监督。

    已声明偏差：
    1) actor 与 critic 各自持有独立骨干、独立优化器（标准 SAC 口径），取代论文式(35)
       的共享骨干单次反传；
    2) 双 Q 覆盖"机动 × 开火"联合动作，开火进入 policy/Q/熵项（论文把开火排除在 RL 梯度外）；
    3) Lf 只更新独立 fire_head，backbone 侧 detach（与 PPO 的 supervised_fire_update 同口径）。
    （TIM 已删除：αI/H_I 一并移除，实际只保留 αD/per-Ñ αJ 两类温度。）"""

    def __init__(self, networks: HanWrapper,
                 lr=1e-4, alpha_lr=None, gamma=0.9, tau=0.05, batch_size=128,
                 buffer_size=int(1e5),
                 fire_buffer_size=1000, fire_batch_size=128, fire_min_samples=100,
                 min_terminal_samples=32,
                 kI=0.8, kD=0.8, xi=1.5, alpha_init=0.2, alpha_min=0.02, max_grad_norm=5.0,
                 device='cpu',
                 actor_update_every=5, fire_update_every=1):
        self.ac = networks
        self.ac_target = copy.deepcopy(networks)
        self.ac_target.eval()
        for p in self.ac_target.parameters():
            p.requires_grad_(False)
        self.enm_num = int(networks.enm_num)

        self.gamma = gamma          # 论文 γ=0.9
        self.tau = tau              # 式(39) τ=0.05
        self.batch_size = batch_size
        self.device = device
        self.max_grad_norm = max_grad_norm

        # critic 与 actor 各自独立参数、独立优化器（标准 SAC 口径，取代式(35) 的共享反传）
        self.critic_optimizer = torch.optim.Adam(self.ac.critic_parameters(), lr=lr)
        self.actor_optimizer = torch.optim.Adam(self.ac.actor_parameters(), lr=lr)
        # 开火 BCE 专用优化器：只优化独立 fire_head；backbone 在 evaluate_fire 内已 detach，
        # 因此 Lf 不会改动机动表征（与 PPO 的 supervised_fire_update 同口径）
        self.fire_optimizer = torch.optim.Adam(self.ac.fire_parameters(), lr=lr)

        # 温度的 log 参数：αD（式37 Ñ>0）、αJ（式38 按 Ñ 分桶，每桶独立）
        # 温度单独给 lr：α 只在 actor 步（每 actor_update_every 次）更新一次，有效步数远少于网络，
        # 与网络共用 lr 会让 α 几乎不动（H_J 长期高于目标熵）；alpha_lr=None 时退回 lr。
        self.log_alpha_D = torch.tensor(np.log(alpha_init), device=device, requires_grad=True)
        self.log_alpha_J = torch.tensor(np.full(self.enm_num + 1, np.log(alpha_init)),
                                        device=device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha_D, self.log_alpha_J],
            lr=lr if alpha_lr is None else float(alpha_lr))
        # 温度下界：α 会沿"熵高于目标→降 α"一路降到 0，导致组内/联合熵项彻底失效。
        # 每次 α 步后把 log α 夹到 ln(alpha_min)，保证熵正则始终在场。
        self.alpha_min = float(alpha_min)

        # 期望熵：H̄D = kD·log|AD|（论文取 kI=kD=0.8）
        # 联合动作含开火位，故 |AD| = 2·N_TDM（机动 × {不发射, 发射}）；
        # TIM 已删除：kI/ξ 仅为入口兼容保留，不参与计算；
        # 式(48) 在 |AI|→0 的极限下退化为 H̄J_Ñ = log(Ñ) + H̄D（Ñ=0 取 0 占位）。
        self.B_D = float(kD * np.log(2 * N_TDM))
        self.B_J_table = [float(np.log(n) + self.B_D) if n > 0 else 0.0
                          for n in range(self.enm_num + 1)]

        self.buffer = ReplayBuffer(buffer_size)                    # D_BFM
        self.fire_buffer = deque(maxlen=int(fire_buffer_size))     # D_f
        self.fire_batch_size = int(fire_batch_size)
        self.fire_min_samples = int(fire_min_samples)
        self.min_terminal_samples = int(min_terminal_samples)

        # 更新节奏（对齐 PPO 的等效总更新次数后，critic:actor 至少 5:1）
        # actor_update_every=5 → 每 5 次 critic 反传才做 1 次 actor+温度反传；
        # fire_update_every=1 → 每次 critic 更新同时做 1 次开火 BCE 监督。
        self.actor_update_every = max(1, int(actor_update_every))
        self.fire_update_every = max(1, int(fire_update_every))
        self._last_loss_pi = 0.0     # 最近一次 actor 步的策略损失（非 actor 步沿用，供日志）
        self._last_loss_fire = 0.0   # 最近一次开火步的 BCE 损失（同上）

        self.update_count = 0

    # ------------------------------------------------------------------ 数据写入
    def push_stacked(self, stacked_trans):
        self.buffer.push_stacked(stacked_trans)

    def push_fire_records(self, records):
        self.fire_buffer.extend(records)

    # ------------------------------------------------------------------ 内部量
    def _soft_value_terms(self, out, alpha_D, alpha_J):
        """给定 forward_sac 输出，返回 (ΣπQmin 所需 qmin, 熵加权和)（式 32 目标口径）。
        alpha_J 为 per-Ñ 温度向量 (Ne+1,)，按存活敌机数取值；
        熵项按组概率加权——式(32) 指示子在 πJ 采样下的数学期望。"""
        qmin = torch.min(out['q1'], out['q2'])
        n_surv = out['enm_mask'].sum(-1).long()                    # (B,)
        alpha_J_n = alpha_J[n_surv]                                # (B,)
        ent = alpha_J_n * out['H_J'] \
            + alpha_D * (out['pr_g'] * out['H_Dj']).sum(-1)
        return qmin, ent

    # ------------------------------------------------------------------ 主更新
    def update(self, batch: Any = None, n_updates=1) -> UpdateResult:
        """执行 n_updates 次梯度更新；batch 参数仅为抽象接口兼容（数据来自内部回放池）。
        末端门控：回放池不满一批、或末端转移（terminated+truncated）不足 min_terminal_samples 时不更新。"""
        if len(self.buffer) < self.batch_size or self.buffer.n_ended < self.min_terminal_samples:
            return UpdateResult({})
        metrics_acc = {}
        metrics_cnt = {}   # 各标量指标的有效计数（NaN 不计入，避免全禁火批次污染均值）
        n_done = 0
        for _ in range(int(n_updates)):
            if len(self.buffer) < self.batch_size:
                break
            m = self._update_once()
            n_done += 1
            for k, v in m.items():
                if isinstance(v, (int, float)):
                    if isinstance(v, float) and not np.isfinite(v):
                        continue   # 例如 fire_prob_max/min：本批全被禁火时无有效值
                    metrics_acc[k] = metrics_acc.get(k, 0.0) + v
                    metrics_cnt[k] = metrics_cnt.get(k, 0) + 1
                else:
                    metrics_acc[k] = v  # 非标量指标（如 alpha_J_each 列表）取最近一次
        if n_done == 0:
            return UpdateResult({})
        return UpdateResult({k: (v / metrics_cnt.get(k, n_done) if isinstance(v, (int, float)) else v)
                             for k, v in metrics_acc.items()})

    def _update_once(self):
        device = self.device
        ac, ac_t = self.ac, self.ac_target

        # ---------------- 采样 D_BFM ----------------
        trans = self.buffer.sample(self.batch_size)
        obs = _stack_field([t['obs'] for t in trans])
        next_obs = _stack_field([t['next_obs'] for t in trans])
        flat_idx = torch.as_tensor(np.array([t['action']['flat_idx'] for t in trans]),
                                   dtype=torch.long, device=device)
        fire_bit = torch.as_tensor(np.array([int(t['action'].get('fire', 0)) for t in trans]),
                                   dtype=torch.long, device=device)
        # 联合动作索引：j·(2·N_TDM) + a·2 + f，与 pi_joint / Q 头的顺序一致
        joint_idx = (flat_idx // N_TDM) * (2 * N_TDM) + (flat_idx % N_TDM) * 2 + fire_bit
        reward = torch.as_tensor(np.array([t['reward'] for t in trans], dtype=np.float32),
                                 dtype=torch.float32, device=device)
        terminated = torch.as_tensor(np.array([t['terminated'] for t in trans], dtype=np.float32),
                                     dtype=torch.float32, device=device)
        active = torch.as_tensor(np.array([t['active_mask'] for t in trans], dtype=np.float32),
                                 dtype=torch.float32, device=device)
        act_sum = active.sum().clamp(min=1.0)

        alpha_D = self.log_alpha_D.exp().detach()
        alpha_J = self.log_alpha_J.exp().detach()

        # ---------------- 在线/目标网络前向 ----------------
        out = ac.forward_sac(obs['agent_input'], obs['mask_input'], obs['rule_input'])
        with torch.no_grad():
            out_t = ac_t.forward_sac(next_obs['agent_input'], next_obs['mask_input'],
                                     next_obs['rule_input'])

        # 开火概率统计（仅记录、无梯度）：只统计"本批决策实际选中的目标位"（flat_idx→enm_idx）
        # 且 fire_gate=True 的样本。禁火位（冷却中/敌机已死亡）不参与，非选中敌机也不参与，
        # 因此 max/min/mean/std 反映的是部署时真正作用在 0.8 阈值上的那个概率。
        # 本批无有效样本时记 NaN，由 update() 的有限性过滤剔除。
        with torch.no_grad():
            p_raw = torch.sigmoid(out['fire_logits_raw'])                     # (B, Ne) 未掩码
            star_idx = (flat_idx // N_TDM).unsqueeze(1)                       # (B, 1)
            p_star = p_raw.gather(1, star_idx).squeeze(1)                     # (B,)
            gate_star = out['fire_gate'].gather(1, star_idx).squeeze(1)       # (B,)
            if gate_star.any():
                v = p_star[gate_star]
                fire_prob_max = float(v.max().cpu())
                fire_prob_min = float(v.min().cpu())
                fire_prob_mean = float(v.mean().cpu())
                fire_prob_std = float(v.std(unbiased=False).cpu())
            else:
                fire_prob_max = float('nan')
                fire_prob_min = float('nan')
                fire_prob_mean = float('nan')
                fire_prob_std = float('nan')

        # ---------------- 双 Q 的 TD 损失（式 31-33） ----------------
        with torch.no_grad():
            qmin_t, ent_t = self._soft_value_terms(out_t, alpha_D, alpha_J)
            V_t = (out_t['pi_joint'] * qmin_t).sum(-1) + ent_t
            y = reward + self.gamma * (1.0 - terminated) * V_t  # 截断帧照常 bootstrap

        q1_taken = out['q1'].gather(1, joint_idx.unsqueeze(1)).squeeze(1)
        q2_taken = out['q2'].gather(1, joint_idx.unsqueeze(1)).squeeze(1)
        LQ1 = (((q1_taken - y) ** 2) * active).sum() / act_sum
        LQ2 = (((q2_taken - y) ** 2) * active).sum() / act_sum

        # ---------------- critic 步（每次更新都做）：双 Q 反传（式 31-33） ----------------
        # 更新节奏对齐 PPO 等效总更新次数后，Q 每次更新都反传；
        # 策略/温度/开火按下面的节奏降频（critic:actor ≥ 5:1 的 delayed policy update）。
        self.critic_optimizer.zero_grad()
        (LQ1 + LQ2).backward()
        torch.nn.utils.clip_grad_norm_(self.ac.critic_parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        n_surv_now = out['enm_mask'].sum(-1).long()               # (B,)
        # 指标用 Q 均值：反传后计算图已释放，这里只读数值
        with torch.no_grad():
            qmin_metric, _ = self._soft_value_terms(out, alpha_D, alpha_J)

        # ---------------- actor 步（每 actor_update_every 次更新做一次） ----------------
        # 策略损失（式 34：−J；熵项不加组概率权重）
        # Q 反传已释放计算图，这里对同一批数据重新前向一次，供策略/温度反传使用。
        if (self.update_count + 1) % self.actor_update_every == 0:
            out_pi = ac.forward_sac(obs['agent_input'], obs['mask_input'], obs['rule_input'])
            qmin, _ = self._soft_value_terms(out_pi, alpha_D, alpha_J)
            qmin = qmin.detach()   # actor/critic 分步更新：策略梯度不回传进 critic
            n_surv_pi = out_pi['enm_mask'].sum(-1).long()
            ent_pi = alpha_J[n_surv_pi] * out_pi['H_J'] \
                + alpha_D * out_pi['H_Dj'].sum(-1)   # 阵亡敌机组 H_Dj=0，自动只计入存活组
            Lpi = (-((out_pi['pi_joint'] * qmin).sum(-1) + ent_pi) * active).sum() / act_sum

            self.actor_optimizer.zero_grad()
            Lpi.backward()
            torch.nn.utils.clip_grad_norm_(self.ac.actor_parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            self._last_loss_pi = float(Lpi.detach().cpu())

            # ---------------- 温度更新（式 36-38，熵全部 detach；与 actor 同步） ----------------
            # 式(37)：TDM 熵只统计 Ñ>0 的样本；
            # 式(38)：联合熵按存活敌机数 Ñ 分桶，每桶独立温度，期望值查 B_J_table（固定 ξ，式 48）
            # （TIM 已删除：式(36) 的 αI 项不再存在）
            surv = out_pi['enm_mask'].any(-1).to(torch.float32)          # Ñ>0 过滤
            n_surv = (surv * active).sum().clamp(min=1.0)
            L_alpha_D = (self.log_alpha_D * ((out_pi['H_D'] - self.B_D) * active * surv).detach().sum() / n_surv)
            laj_terms = []
            for n in range(self.enm_num + 1):
                mask_n = ((n_surv_now == n).to(torch.float32) * active)
                if mask_n.sum().item() > 0:
                    laj_terms.append(self.log_alpha_J[n]
                                     * ((out_pi['H_J'] - self.B_J_table[n]) * mask_n).detach().sum()
                                     / mask_n.sum())
            L_alpha_J = torch.stack(laj_terms).sum()
            alpha_loss = L_alpha_D + L_alpha_J
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            with torch.no_grad():
                self.log_alpha_D.clamp_(min=float(np.log(self.alpha_min)))
                self.log_alpha_J.clamp_(min=float(np.log(self.alpha_min)))

        # ---------------- 开火监督 Lf（式 28，只更新独立 fire_head；每 fire_update_every 次做一次） ----------------
        if (self.update_count + 1) % self.fire_update_every == 0 \
                and len(self.fire_buffer) >= self.fire_min_samples:
            fire_batch = random.sample(list(self.fire_buffer),
                                       min(self.fire_batch_size, len(self.fire_buffer)))
            fire_obs = _stack_field([r['fire_obs'] for r in fire_batch])
            fire_idx = np.array([r['enm_idx'] for r in fire_batch])
            fire_hit = np.array([r['hit_record'] for r in fire_batch], dtype=np.float32)
            Lf = self.ac.evaluate_fire(fire_obs, fire_idx, fire_hit)
            self.fire_optimizer.zero_grad()
            Lf.backward()
            torch.nn.utils.clip_grad_norm_(self.ac.fire_parameters(), self.max_grad_norm)
            self.fire_optimizer.step()
            self._last_loss_fire = float(Lf.detach().cpu())

        # ---------------- 目标网络软更新（式 39） ----------------
        with torch.no_grad():
            for p, tp in zip(self.ac.parameters(), self.ac_target.parameters()):
                tp.mul_(1 - self.tau).add_(p, alpha=self.tau)

        self.update_count += 1
        return {
            'loss_q': float((LQ1 + LQ2).detach().cpu()),
            'loss_pi': self._last_loss_pi,
            'loss_fire': self._last_loss_fire,
            'H_J': float(out['H_J'].mean().detach().cpu()),
            'H_D': float(out['H_D'].mean().detach().cpu()),
            'alpha_D': float(alpha_D),
            'alpha_J': float(alpha_J.mean()), 'alpha_J_each': [float(a) for a in alpha_J],
            'B_J': float(np.mean([self.B_J_table[n] for n in n_surv_now.tolist()])),
            'q_mean': float(qmin_metric.mean().detach().cpu()),
            'fire_buffer': float(len(self.fire_buffer)),
            'fire_prob_max': fire_prob_max,
            'fire_prob_min': fire_prob_min,
            'fire_prob_mean': fire_prob_mean,
            'fire_prob_std': fire_prob_std,
            'replay_size': float(len(self.buffer)),
            'buffer_ended': float(self.buffer.n_ended),
        }

    # ------------------------------------------------------------------ 存取（断点续训）
    def state_dict(self):
        return {
            'ac': self.ac.state_dict(),
            'ac_target': self.ac_target.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
            'log_alpha_D': float(self.log_alpha_D.detach().cpu()),
            'log_alpha_J': [float(a) for a in self.log_alpha_J.detach().cpu()],
            'update_count': self.update_count,
        }

    def load_state_dict(self, state):
        self.ac.load_state_dict(state['ac'])
        self.ac_target.load_state_dict(state['ac_target'])
        self.critic_optimizer.load_state_dict(state['critic_optimizer'])
        self.actor_optimizer.load_state_dict(state['actor_optimizer'])
        self.alpha_optimizer.load_state_dict(state['alpha_optimizer'])
        self.log_alpha_D.data = torch.tensor(state['log_alpha_D'], device=self.device)
        self.log_alpha_J.data = torch.tensor(state['log_alpha_J'], device=self.device)
        self.update_count = state.get('update_count', 0)
