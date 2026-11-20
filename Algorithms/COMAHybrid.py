"""COMAHybrid：COMA（反事实多智能体策略梯度）的 on-policy learner。

论文对照（Foerster et al., "Counterfactual Multi-Agent Policy Gradients", AAAI 2018）：
- 集中式 critic Q_i(s, u)：actor 只看 o_i，critic 训练时看全队观测与联合动作；
- 反事实优势 A_i = Q_i(s,u) − Σ_{a'_i} π_i(a'_i|o_i) Q_i(s,(u_{-i},a'_i))；
  开火位概率按 A 方案 detach（与 HanSAC 的 pi_joint 口径一致）；
- TD(λ)（默认 λ=0.8）训练 critic；目标 critic 每 target_update_interval 个梯度步硬拷贝；
- on-policy：一次 update 用一段新 rollout，跑 update_epochs 后清空，不做跨 rollout 复用。

已声明偏差（declared_deviations）：
1) 固定熵系数 entropy_coef（默认 0.001）加在策略损失上，COMA 原版无熵正则；
2) 开火仍由独立 fire_head 的 BCE（D_f）监督，不进入反事实策略梯度；
3) critic 的"状态"是各 agent 本地观测的集合编码，不是上帝视角真值状态；
4) 时限截断（truncated）片段末端用"末步联合动作 + 当前策略"的软价值做 bootstrap，
   属近似（正常对局以 terminated 结束，不触发此路径）。

数据约定（一条 decision = 一个决策点的联合记录，N 为本局红方机数）：
decision = {
  'obs': {'agent_input': {'fixed': (N,F), 'enm': (N,Ne,De), 'ally': (N,Na,Da)},
          'mask_input': {'enm': (N,Ne), 'ally': (N,Na)},      # 可缺省
          'rule_input': stack_field 后的 dict}                 # 可缺省
  'next_obs': 同上（只有链路末端需要，其余步可缺省）；
  'actions': (N,) int，联合索引 flat_idx·2 + fire；
  'rewards': (N,) float，每个 agent 自己的奖励 r_i；
  'active': (N,) 0/1，该决策时刻该机存活；
  'terminated': 0/1，该决策点后本局因胜负终结；
  'truncated': 0/1，该决策点后本局因时限截断；
}
按回合（或未跨回合的连续片段）调用 push_chain(decisions)，攒够 rollout_min_decisions
后调用 update()。这样 TD(λ) 只在片段内部回传，跨回合不会串线。
"""

import copy
import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch

from Algorithms.COMAWrapper import pack_team_arrays


@dataclass(frozen=True)
class UpdateResult:
    metrics: Mapping[str, float]


def _stack_tree(items):
    """list[dict-of-arrays / array / None] → dict-of-arrays，沿新的 batch 维堆叠。"""
    sample = next((it for it in items if it is not None), None)
    if sample is None:
        return None
    if isinstance(sample, dict):
        keys = set()
        for it in items:
            if it is not None:
                keys.update(it.keys())
        return {k: _stack_tree([it.get(k) if it is not None else None for it in items])
                for k in keys}
    return np.stack([np.asarray(it) for it in items], axis=0)


def _take_tree(tree, idx):
    """沿 batch 维取子集（idx 可为切片或整型数组）。"""
    if tree is None:
        return None
    if isinstance(tree, dict):
        return {k: _take_tree(v, idx) for k, v in tree.items()}
    return tree[idx]


def _add_batch(tree):
    """(N,...) → (1,N,...)，用于单条 next_obs 的 bootstrap 前向。"""
    if tree is None:
        return None
    if isinstance(tree, dict):
        return {k: _add_batch(v) for k, v in tree.items()}
    return np.asarray(tree)[None, ...]


class RolloutBuffer:
    """on-policy 片段池：以"决策链"为单位存放（一条链 = 一个回合或一段连续决策）。"""

    def __init__(self):
        self.chains = []
        self.n_decisions = 0

    def push_chain(self, decisions):
        if not decisions:
            return
        self.chains.append(list(decisions))
        self.n_decisions += len(decisions)

    def pop_all(self):
        chains = self.chains
        self.chains = []
        self.n_decisions = 0
        return chains

    def clear(self):
        self.chains = []
        self.n_decisions = 0

    def __len__(self):
        return self.n_decisions


class COMAHybrid:
    """COMA learner：集中式单 Q + 反事实基线 + TD(λ)，on-policy 更新。"""

    def __init__(self, networks,
                 lr=3e-4, gamma=0.9, lam=0.8, entropy_coef=0.001,
                 update_epochs=3, minibatch_size=256, rollout_min_decisions=1024,
                 target_update_interval=200, max_grad_norm=10.0,
                 fire_buffer_size=1000, fire_batch_size=128, fire_min_samples=100,
                 device='cpu'):
        self.ac = networks
        self.enm_num = int(networks.enm_num)
        self.ally_num = int(networks.ally_num)
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.entropy_coef = float(entropy_coef)
        self.update_epochs = max(1, int(update_epochs))
        self.minibatch_size = max(1, int(minibatch_size))
        self.rollout_min_decisions = max(1, int(rollout_min_decisions))
        self.target_update_interval = max(1, int(target_update_interval))
        self.max_grad_norm = float(max_grad_norm)
        self.device = device

        # 目标 critic：只拷贝集中式 Q（不含 actor），硬拷贝按梯度步计数
        self.critic_target = copy.deepcopy(networks.critic)
        self.critic_target.to(device)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)
        self.critic_target.eval()

        self.critic_optimizer = torch.optim.Adam(self.ac.critic_parameters(), lr=lr)
        self.actor_optimizer = torch.optim.Adam(self.ac.actor_parameters(), lr=lr)
        self.fire_optimizer = torch.optim.Adam(self.ac.fire_parameters(), lr=lr)

        self.buffer = RolloutBuffer()
        self.fire_buffer = deque(maxlen=int(fire_buffer_size))
        self.fire_batch_size = int(fire_batch_size)
        self.fire_min_samples = int(fire_min_samples)

        self.update_count = 0      # 调用的 update() 次数
        self.n_grad_steps = 0      # 累计梯度步（目标 critic 的同步节拍）
        self._last_loss_pi = 0.0
        self._last_loss_fire = 0.0

    # ------------------------------------------------------------------ 数据写入
    def push_chain(self, decisions):
        self.buffer.push_chain(decisions)

    def push_fire_records(self, records):
        self.fire_buffer.extend(records)

    # ------------------------------------------------------------------ 目标计算（TD(λ)）
    def _target_q(self, obs, actions, active):
        """用目标 critic 取真实动作的 Q：返回 (D,N) numpy。分批前向控制显存。"""
        outs = []
        total = actions.shape[0]
        chunk = 1024
        for s in range(0, total, chunk):
            sub = _take_tree(obs, slice(s, s + chunk))
            fixed, enm, ally, enm_mask, ally_mask = pack_team_arrays(
                sub['agent_input'], sub.get('mask_input'), self.enm_num, self.ally_num, self.device)
            B, N = fixed.shape[0], fixed.shape[1]
            act = torch.as_tensor(actions[s:s + chunk], dtype=torch.long,
                                  device=self.device).reshape(B, N)
            alive = torch.as_tensor(active[s:s + chunk] > 0.5, dtype=torch.bool,
                                    device=self.device).reshape(B, N)
            with torch.no_grad():
                q = self.critic_target(fixed, enm, ally, enm_mask, ally_mask, act, alive)
                q_taken = q.gather(2, act[..., None]).squeeze(-1)      # (B,N)
            outs.append(q_taken.cpu().numpy().astype(np.float32))
        return np.concatenate(outs, axis=0)

    def _soft_boot(self, next_obs, last_action, last_alive):
        """链路末端的 bootstrap：Σ_a π(a|o') · Q_target(s', (u_last_{-i}, a))。

        下一时刻的联合动作在截断时不可知，此处用末步动作近似（见文件头偏差 4）。
        """
        if next_obs is None:
            return np.zeros(last_action.shape[0], dtype=np.float32)
        obs_b = _add_batch(next_obs)
        act1 = np.asarray(last_action)[None, :]
        alive1 = (np.asarray(last_alive) > 0.5)[None, :]
        with torch.no_grad():
            out_pi = self.ac.forward_joint(obs_b['agent_input'], obs_b.get('mask_input'),
                                           obs_b.get('rule_input'))
            fixed, enm, ally, enm_mask, ally_mask = pack_team_arrays(
                obs_b['agent_input'], obs_b.get('mask_input'), self.enm_num, self.ally_num,
                self.device)
            act = torch.as_tensor(act1, dtype=torch.long, device=self.device)
            alive = torch.as_tensor(alive1, dtype=torch.bool, device=self.device)
            q_row = self.critic_target(fixed, enm, ally, enm_mask, ally_mask, act, alive)
            v = (out_pi['pi_joint'] * q_row).sum(-1).squeeze(0)         # (N,)
        return v.cpu().numpy().astype(np.float32)

    def _prepare_targets(self, chains):
        """把若干决策链展平成批，并按链内 TD(λ) 计算逐 agent 目标 y。"""
        decisions = [d for chain in chains for d in chain]
        if not decisions:
            return None
        obs = _stack_tree([d['obs'] for d in decisions])
        actions = np.stack([np.asarray(d['actions'], dtype=np.int64) for d in decisions], axis=0)
        rewards = np.stack([np.asarray(d['rewards'], dtype=np.float32) for d in decisions], axis=0)
        active = np.stack([np.asarray(d['active'], dtype=np.float32) for d in decisions], axis=0)
        term = np.stack([float(d.get('terminated', 0)) for d in decisions], axis=0)

        q_cur = self._target_q(obs, actions, active)                    # (D,N)
        y = np.zeros_like(rewards)
        offset = 0
        for chain in chains:
            T = len(chain)
            qs = q_cur[offset:offset + T]
            rs = rewards[offset:offset + T]
            ds = term[offset:offset + T]
            acc = np.zeros(actions.shape[1], dtype=np.float32)
            # 链路末端若为时限截断，用软价值 bootstrap；终止则为 0
            boot = None
            if ds[-1] < 0.5:
                boot = self._soft_boot(chain[-1].get('next_obs'),
                                       np.asarray(chain[-1]['actions'], dtype=np.int64),
                                       np.asarray(chain[-1]['active'], dtype=np.float32))
            for t in range(T - 1, -1, -1):
                d = float(ds[t])
                if t == T - 1:
                    q_next = np.zeros_like(acc) if boot is None else boot
                else:
                    q_next = qs[t + 1]
                delta = rs[t] + self.gamma * (1.0 - d) * q_next - qs[t]
                acc = delta + self.gamma * self.lam * (1.0 - d) * acc
                y[offset + t] = qs[t] + acc
            offset += T
        return {'obs': obs, 'actions': actions, 'y': y, 'active': active}

    # ------------------------------------------------------------------ 梯度步
    def _grad_step(self, obs, actions, y, active):
        acts_np = np.asarray(actions, dtype=np.int64)
        acts = torch.as_tensor(acts_np, dtype=torch.long, device=self.device)
        alive = torch.as_tensor(np.asarray(active) > 0.5, dtype=torch.bool, device=self.device)
        y_t = torch.as_tensor(np.asarray(y, dtype=np.float32), dtype=torch.float32,
                              device=self.device)
        w = alive.to(torch.float32)
        denom = w.sum().clamp(min=1.0)

        # ---- critic：Q_i(s,u) 回归 TD(λ) 目标 ----
        q_all = self.ac.forward_critic(obs['agent_input'], obs.get('mask_input'),
                                       acts_np, np.asarray(active))
        q_taken = q_all.gather(2, acts[..., None]).squeeze(-1)          # (b,N)
        LQ = (((q_taken - y_t) ** 2) * w).sum() / denom

        self.critic_optimizer.zero_grad()
        LQ.backward()
        torch.nn.utils.clip_grad_norm_(self.ac.critic_parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        # ---- actor：反事实基线策略梯度 + 固定系数熵正则 ----
        out_pi = self.ac.forward_joint(obs['agent_input'], obs.get('mask_input'),
                                       obs.get('rule_input'))
        pi_joint = out_pi['pi_joint']                                   # (b,N,2A)
        log_pi_a = pi_joint.gather(2, acts[..., None]).squeeze(-1).clamp_min(1e-8).log()
        H_J = -(out_pi['pi'] * out_pi['log_pi']).sum(-1)                # (b,N) 机动槽位熵
        adv = q_taken.detach() - (pi_joint.detach() * q_all.detach()).sum(-1)
        Lpi = -(((adv * log_pi_a) * w).sum() / denom) \
            - self.entropy_coef * ((H_J * w).sum() / denom)

        self.actor_optimizer.zero_grad()
        Lpi.backward()
        torch.nn.utils.clip_grad_norm_(self.ac.actor_parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        self._last_loss_pi = float(Lpi.detach().cpu())

        self.n_grad_steps += 1
        if self.n_grad_steps % self.target_update_interval == 0:
            self.critic_target.load_state_dict(self.ac.critic.state_dict())

        return {
            'loss_q': float(LQ.detach().cpu()),
            'loss_pi': float(Lpi.detach().cpu()),
            'H_J': float(H_J.mean().detach().cpu()),
            'adv_abs': float(adv.abs().mean().detach().cpu()),
            'q_mean': float(q_taken.mean().detach().cpu()),
        }

    # ------------------------------------------------------------------ 主更新
    def update(self, n_updates=1) -> UpdateResult:
        """消耗当前 rollout 做 update_epochs×n_updates 轮更新，随后清空缓冲（on-policy）。"""
        if self.buffer.n_decisions < self.rollout_min_decisions:
            return UpdateResult({})
        chains = self.buffer.pop_all()
        data = self._prepare_targets(chains)
        if data is None:
            return UpdateResult({})

        acc, cnt = {}, 0
        D = data['actions'].shape[0]
        for _ in range(max(1, int(n_updates))):
            for _epoch in range(self.update_epochs):
                order = np.random.permutation(D)
                for start in range(0, D, self.minibatch_size):
                    idx = order[start:start + self.minibatch_size]
                    m = self._grad_step(_take_tree(data['obs'], idx),
                                        data['actions'][idx], data['y'][idx], data['active'][idx])
                    for k, v in m.items():
                        acc[k] = acc.get(k, 0.0) + v
                    cnt += 1

        loss_fire = self._fire_step()
        self.update_count += 1
        metrics = {k: v / max(1, cnt) for k, v in acc.items()}
        metrics['loss_fire'] = self._last_loss_fire if loss_fire is None else loss_fire
        metrics['rollout_decisions'] = float(D)
        metrics['fire_buffer'] = float(len(self.fire_buffer))
        print(f"  [COMA] update#{self.update_count} decisions={D} grad_steps={cnt} "
              f"loss_q={metrics.get('loss_q', float('nan')):.4f} "
              f"loss_pi={metrics.get('loss_pi', float('nan')):.4f} "
              f"H_J={metrics.get('H_J', float('nan')):.3f} "
              f"adv_abs={metrics.get('adv_abs', float('nan')):.3f} "
              f"loss_fire={metrics['loss_fire']:.4f}")
        return UpdateResult(metrics)

    def _fire_step(self):
        """开火 BCE（式 28 口径）：只更新独立 fire_head（backbone 已在 evaluate_fire 内 detach）。"""
        if len(self.fire_buffer) < self.fire_min_samples:
            return None
        fire_batch = random.sample(list(self.fire_buffer),
                                   min(self.fire_batch_size, len(self.fire_buffer)))
        fire_obs = _stack_tree([r['fire_obs'] for r in fire_batch])
        fire_idx = np.array([r['enm_idx'] for r in fire_batch], dtype=np.int64)
        fire_hit = np.array([r['hit_record'] for r in fire_batch], dtype=np.float32)
        Lf = self.ac.evaluate_fire(fire_obs, fire_idx, fire_hit)
        self.fire_optimizer.zero_grad()
        Lf.backward()
        torch.nn.utils.clip_grad_norm_(self.ac.fire_parameters(), self.max_grad_norm)
        self.fire_optimizer.step()
        self._last_loss_fire = float(Lf.detach().cpu())
        return self._last_loss_fire

    # ------------------------------------------------------------------ 存取（断点续训）
    def state_dict(self):
        return {
            'ac': self.ac.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'fire_optimizer': self.fire_optimizer.state_dict(),
            'update_count': self.update_count,
            'n_grad_steps': self.n_grad_steps,
        }

    def load_state_dict(self, state):
        self.ac.load_state_dict(state['ac'])
        self.critic_target.load_state_dict(state['critic_target'])
        self.critic_optimizer.load_state_dict(state['critic_optimizer'])
        self.actor_optimizer.load_state_dict(state['actor_optimizer'])
        self.fire_optimizer.load_state_dict(state['fire_optimizer'])
        self.update_count = state.get('update_count', 0)
        self.n_grad_steps = state.get('n_grad_steps', 0)
