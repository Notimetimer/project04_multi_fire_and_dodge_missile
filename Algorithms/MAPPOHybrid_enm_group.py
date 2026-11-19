# =====================================================================
# 副本来源：Algorithms/MAPPOHybrid.py @ commit 39ece5b（原件保持只读）
# 差异：新增 enm_group_update —— 对同时存在 enm_chosen=0/1 的样本，最小化
#       P1−P0 组概率差并最大化选取熵，只更新 actor.selector。
# 原因：让选取器主动避开“已有本方导弹在攻击”的目标，实现协同分工。
# =====================================================================
import numpy as np
import torch
import torch.nn.functional as F
import os
import json
from Algorithms.Utils import compute_advantage
# from Algorithms.ActorMLPHeads import ActorMLPWrapper, ActorMLPHeads

class MAPPOHybrid:
    def __init__(self, actor, critic, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device, 
                 k_entropy=0.001, critic_max_grad=2, actor_max_grad=2, max_std=0.7):
        
        self.actor = actor 
        self.critic = critic
        
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device
        self.k_entropy = k_entropy
        
        self.critic_max_grad = critic_max_grad
        self.actor_max_grad = actor_max_grad
        self.max_std = max_std
        
        self.actor_loss = 0
        self.critic_loss = 0
        # 本批更新的开火概率极值（各 mini_batch 极值的平均，供训练侧按 id 记录 tensorboard）
        self.max_fire_prob = float('nan')
        self.min_fire_prob = float('nan')
        
    def RDistill(self, transition_dict, *args, **kwargs):
        return dict(transition_dict), 0
        
    def ADistill(self, advantage, *args, **kwargs):
        return advantage

    def update(self, transition_dict, adv_normed=True, clip_range=0.2, shuffled=True, mini_batch_size=None):
        if clip_range is not None:
            self.eps = clip_range
        def to_tensor(x, dtype):
            if isinstance(x, np.ndarray):
                return torch.tensor(x, dtype=dtype).to(self.device)
            else:
                return torch.tensor(np.array(x), dtype=dtype).to(self.device)

        # 递归按 batch 索引切片 dict-of-arrays（obs/states/actions 均为嵌套 dict，叶子为 ndarray）
        def index_tree(tree, idx):
            if isinstance(tree, dict):
                return {k: index_tree(v, idx) for k, v in tree.items()}
            return tree[idx]

        # states/next_states 为 {'fixed','ally','enm'} 的 dict-of-arrays，
        # 直接交给 critic.get_value（内部自行转 tensor），value 统一 view 成 (B,1) 与 rewards 对齐
        states = transition_dict['states']
        dones = to_tensor(transition_dict['dones'], torch.float).view(-1, 1)
        rewards = to_tensor(transition_dict['rewards'], torch.float).view(-1, 1)

        if 'active_masks' in transition_dict:
            active_masks = to_tensor(transition_dict['active_masks'], torch.float).view(-1, 1)
        else:
            active_masks = torch.ones_like(dones)

        critic_inputs = states
        actions = transition_dict['actions']
        obs = transition_dict['obs']

        if 'advantages' in transition_dict and 'td_targets' in transition_dict:
            advantage = to_tensor(transition_dict['advantages'], torch.float).view(-1, 1)
            td_target = to_tensor(transition_dict['td_targets'], torch.float).view(-1, 1)
        else:
            truncs = to_tensor(transition_dict['truncs'], torch.float).view(-1, 1) if 'truncs' in transition_dict else None
            with torch.no_grad():
                if 'next_values' in transition_dict:
                    next_vals = to_tensor(transition_dict['next_values'], torch.float).view(-1, 1)
                else:
                    next_states = transition_dict['next_states']
                    next_vals = self.critic.get_value(next_states)['value'].view(-1, 1)
                td_target = rewards + self.gamma * next_vals * (1 - dones)
                td_delta = td_target - self.critic.get_value(critic_inputs)['value'].view(-1, 1)
                advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu(), dones.cpu(), truncs.cpu() if truncs is not None else None).to(self.device)

        with torch.no_grad():
            # [修改] old_log_prob 直接读取采样时存储的行为策略 log_prob（joint = enm + act，
            # 与 evaluate_actions['log_prob'] 同一联合语义），不再重算
            old_log_probs = (
                to_tensor(actions['log_prob_enm'], torch.float)
                + to_tensor(actions['log_prob_act'], torch.float)
            ).view(-1, 1)
            v_pred_old = self.critic.get_value(critic_inputs)['value'].view(-1, 1)

        if adv_normed:
            active_adv = advantage[active_masks.squeeze(-1).bool()]
            if active_adv.numel() > 1:
                advantage = (advantage - active_adv.mean()) / (active_adv.std(unbiased=False) + 1e-8)
        else:
            active_adv = advantage[active_masks.squeeze(-1).bool()]
            if active_adv.numel() > 0:
                advantage = advantage - active_adv.mean()

        actor_loss_list, critic_loss_list, entropy_list, ratio_list = [], [], [], []
        kl_list, clip_frac_list = [], []
        fire_prob_max_list, fire_prob_min_list = [], []
        ppo_samples_total, ppo_valid_samples_total = 0, 0
        mask_eps = 1e-5
        num_samples = rewards.shape[0]
        if mini_batch_size is None: mini_batch_size = num_samples

        for _ in range(self.epochs):
            idx = torch.randperm(num_samples, device=self.device) if shuffled else torch.arange(num_samples, device=self.device)
            for start in range(0, num_samples, mini_batch_size):
                end = min(start + mini_batch_size, num_samples)
                batch_idx = idx[start:end]
                batch_idx_np = batch_idx.cpu().numpy()

                mb_obs = index_tree(obs, batch_idx_np)
                mb_actions = index_tree(actions, batch_idx_np)

                mb_critic_inputs = index_tree(critic_inputs, batch_idx_np)
                mb_advantage = advantage[batch_idx]
                mb_td_target = td_target[batch_idx]
                mb_old_log_probs = old_log_probs[batch_idx]
                mb_active_masks = active_masks[batch_idx]

                actor_out = self.actor.evaluate_actions(
                    mb_obs['agent_input'], mb_obs['mask_input'], mb_obs['rule_input'], mb_actions
                )
                log_probs = actor_out['log_prob'].view(-1, 1)
                entropy = actor_out['entropy'].view(-1, 1)
                # 收集本 mini_batch 未禁火样本的开火概率极值（由 evaluate_actions 提供）
                if 'fire_prob_max' in actor_out:
                    fire_prob_max_list.append(actor_out['fire_prob_max'])
                    fire_prob_min_list.append(actor_out['fire_prob_min'])
                log_ratio = log_probs - mb_old_log_probs
                ratio = torch.exp(log_ratio)

                with torch.no_grad():
                    active_sum = mb_active_masks.sum()
                    approx_kl = (((ratio - 1) - log_ratio) * mb_active_masks).sum() / (active_sum + mask_eps)
                    kl_list.append(approx_kl.item())
                    clip_fracs = (((ratio - 1.0).abs() > self.eps).float() * mb_active_masks).sum() / (active_sum + mask_eps)
                    clip_frac_list.append(clip_fracs.item())

                surr1 = ratio * mb_advantage
                surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * mb_advantage
                actor_loss = -(torch.min(surr1, surr2) * mb_active_masks).sum() / (active_sum + mask_eps)

                v_pred = self.critic.get_value(mb_critic_inputs)['value'].view(-1, 1)
                critic_loss = F.mse_loss(v_pred, mb_td_target, reduction='none')
                critic_loss = (critic_loss * mb_active_masks).sum() / (active_sum + mask_eps)
                entropy = (entropy * mb_active_masks).sum() / (active_sum + mask_eps)

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss = actor_loss - self.k_entropy * entropy + critic_loss
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_max_grad)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.critic_max_grad)
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                actor_loss_list.append(actor_loss.item())
                critic_loss_list.append(critic_loss.item())
                entropy_list.append(entropy.item())
                ratio_list.append(ratio.mean().item())
                ppo_samples_total += len(batch_idx)
                ppo_valid_samples_total += active_sum.item()

        # 各 mini_batch 极值平均到类属性（全被禁火时无有效值，保留 NaN）
        finite_max = [v for v in fire_prob_max_list if np.isfinite(v)]
        finite_min = [v for v in fire_prob_min_list if np.isfinite(v)]
        self.max_fire_prob = float(np.mean(finite_max)) if finite_max else float('nan')
        self.min_fire_prob = float(np.mean(finite_min)) if finite_min else float('nan')

        return {
            'fire_prob_max': self.max_fire_prob,
            'fire_prob_min': self.min_fire_prob,
            'actor_loss': np.mean(actor_loss_list) if actor_loss_list else 0.0,
            'critic_loss': np.mean(critic_loss_list) if critic_loss_list else 0.0,
            'entropy': np.mean(entropy_list) if entropy_list else 0.0,
            'kl': np.mean(kl_list) if kl_list else 0.0,
            'clip_frac': np.mean(clip_frac_list) if clip_frac_list else 0.0,
            'ratio': np.mean(ratio_list) if ratio_list else 0.0,
            'samples': ppo_samples_total,
            'valid_samples': ppo_valid_samples_total,
        }

    # 敌机选取的“组概率”辅助更新：同时存在 enm_chosen=0/1 时生效
    def enm_group_update(self, transition_dict, epochs=1, batch_size=256,
                         group_form='diff', beta_entropy=0.05, eps=1e-6,
                         lr_ratio=1.0, max_grad_norm=None):
        """在回放样本上最小化“被攻击组 / 未被攻击组”的选取概率差，并最大化选取熵。

        对单个样本 b，有效槽位集合 V（enm 掩码为 1），记 A={j:c_j=1}、U={j:c_j=0}，
        p_j 为 selector 的 masked softmax 选取概率：
            P0_b = Σ_{j∈U} p_j,  P1_b = Σ_{j∈A} p_j,  P0+P1=1
            L_group = mean_b(P1_b - P0_b)                     # group_form='diff'
                    = mean_b(log(P1_b+eps) - log(P0_b+eps))   # group_form='log_ratio'
            H_b = -Σ_{j∈V} p_j log p_j
            L = L_group - beta_entropy * mean_b(H_b)
        只更新 actor.selector：冻结 actor 其余参数，梯度不流向 pilot（机动/开火头）。
        """
        obs = transition_dict.get('obs', {}) if isinstance(transition_dict, dict) else {}
        agent_inputs = obs.get('agent_input') if isinstance(obs, dict) else None
        mask_inputs = obs.get('mask_input') if isinstance(obs, dict) else None

        skip = {'loss': float('nan'), 'p0_mean': float('nan'), 'p1_mean': float('nan'),
                'ratio_mean': float('nan'), 'entropy': float('nan'), 'samples': 0}
        if not isinstance(agent_inputs, dict) or 'enm_chosen' not in agent_inputs:
            self.last_enm_group_stats = skip
            return skip

        device = self.device

        def _t(x, dtype):
            if torch.is_tensor(x):
                return x.to(device=device, dtype=dtype)
            return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)

        fixed = _t(agent_inputs['fixed'], torch.float32)
        enm = _t(agent_inputs['enm'], torch.float32)
        ally = _t(agent_inputs['ally'], torch.float32)
        enm_chosen = _t(agent_inputs['enm_chosen'], torch.float32)
        # 单样本输入自动补 batch 维，保持与 evaluate_actions 一致的批量语义
        if fixed.ndim == 1:
            fixed = fixed.unsqueeze(0)
        if enm.ndim == 2:
            enm = enm.unsqueeze(0)
        if ally.ndim == 2:
            ally = ally.unsqueeze(0)
        if enm_chosen.ndim == 1:
            enm_chosen = enm_chosen.unsqueeze(0)

        if isinstance(mask_inputs, dict) and 'enm' in mask_inputs and len(np.asarray(mask_inputs['enm'])) > 0:
            enm_mask = _t(mask_inputs['enm'], torch.bool)
            if enm_mask.ndim == 1:
                enm_mask = enm_mask.unsqueeze(0)
        else:
            enm_mask = torch.ones_like(enm_chosen, dtype=torch.bool)
        if isinstance(mask_inputs, dict) and 'ally' in mask_inputs and len(np.asarray(mask_inputs['ally'])) > 0:
            ally_mask = _t(mask_inputs['ally'], torch.bool)
            if ally_mask.ndim == 1:
                ally_mask = ally_mask.unsqueeze(0)
        else:
            ally_mask = torch.ones(ally.shape[0], ally.shape[1], dtype=torch.bool, device=device)

        # 槽位长度必须与 enm 序列一致，否则分组统计不可信，直接跳过
        if enm_chosen.shape[1] != enm.shape[1] or enm_mask.shape[1] != enm.shape[1]:
            self.last_enm_group_stats = skip
            return skip

        valid = enm_mask.bool()
        c1 = (enm_chosen > 0.5) & valid
        c0 = (~(enm_chosen > 0.5)) & valid
        # 只保留“两组同时存在”的样本：单侧为空时该样本对组差没有意义
        keep = c1.any(dim=1) & c0.any(dim=1)
        samples = int(keep.sum())
        if samples == 0:
            self.last_enm_group_stats = skip
            return skip

        mb_agent = {
            'fixed': fixed[keep],
            'enm': enm[keep],
            'ally': ally[keep],
            # selector 侧要求 (B, Ne, 1)，与 pack_inputs 的口径一致
            'enm_chosen': enm_chosen[keep].unsqueeze(-1),
        }
        mb_mask = {'enm': enm_mask[keep], 'ally': ally_mask[keep]}
        c1_f = c1[keep].to(torch.float32)
        c0_f = c0[keep].to(torch.float32)

        # ── 缩放学习率并冻结 actor 其余参数：本次只更新 selector ─────────────
        current_lr = self.actor_optimizer.param_groups[0]['lr']
        self.actor_optimizer.param_groups[0]['lr'] = current_lr * lr_ratio

        params = list(self.actor.parameters())
        old_requires_grad = [p.requires_grad for p in params]
        for p in params:
            p.requires_grad_(False)
        for p in self.actor.selector.parameters():
            p.requires_grad_(True)

        clip_norm = self.actor_max_grad if max_grad_norm is None else max_grad_norm
        losses, p0_list, p1_list, h_list = [], [], [], []
        mb_size = max(1, min(int(batch_size), samples))
        for _ in range(max(1, int(epochs))):
            order = torch.randperm(samples, device=device)
            for start in range(0, samples, mb_size):
                idx = order[start:start + mb_size]
                logits = self.actor.selector.enm_logits(
                    {k: v[idx] for k, v in mb_agent.items()},
                    {k: v[idx] for k, v in mb_mask.items()},
                    temperature=1.0,
                )
                probs = torch.softmax(logits, dim=-1)
                p1 = (probs * c1_f[idx]).sum(dim=-1)
                p0 = (probs * c0_f[idx]).sum(dim=-1)
                if group_form == 'log_ratio':
                    group_loss = (torch.log(p1 + eps) - torch.log(p0 + eps)).mean()
                else:  # 默认差值形式：有界、梯度平滑
                    group_loss = (p1 - p0).mean()
                # 掩码槽位 p=0，0*log(eps)=0，对熵无贡献
                entropy = -(probs * probs.clamp_min(eps).log()).sum(dim=-1).mean()
                loss = group_loss - beta_entropy * entropy

                self.actor_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.selector.parameters(), clip_norm)
                self.actor_optimizer.step()

                losses.append(float(loss.detach().cpu()))
                p0_list.append(float(p0.mean().detach().cpu()))
                p1_list.append(float(p1.mean().detach().cpu()))
                h_list.append(float(entropy.detach().cpu()))

        for p, requires_grad in zip(params, old_requires_grad):
            p.requires_grad_(requires_grad)
        self.actor_optimizer.param_groups[0]['lr'] = current_lr

        stats = {
            'loss': float(np.mean(losses)) if losses else 0.0,
            'p0_mean': float(np.mean(p0_list)) if p0_list else float('nan'),
            'p1_mean': float(np.mean(p1_list)) if p1_list else float('nan'),
            'ratio_mean': float(np.mean(p1_list) / (np.mean(p0_list) + eps)) if p0_list else float('nan'),
            'entropy': float(np.mean(h_list)) if h_list else float('nan'),
            'samples': samples,
        }
        self.last_enm_group_stats = stats
        return stats

    # 调用evaluate_
    def supervised_fire_update(self, fire_batch, epochs=1, batch_size=128, max_grad_norm=2.0, fire_lr_ratio=1.0):
        """用回合结束后收集的 fire_obs/enm_idx/hit_record 样本有监督更新开火分支。

        参考 PPOHybrid23_0.supervised_fire_update：切断开火分支之前的所有梯度——
        更新前冻结 actor 全部参数、仅放行 pilot.fire_head，并只对 fire_head 建优化器，
        因此 selector/pilot.backbone/机动头均不受该更新影响。
        损失由 ActorWithATTN.ActorWrapper.evaluate_fire 计算（BCE-with-logits，带梯度）。

        Args:
            fire_batch: record 列表，每条含 fire_obs(dict: fixed/enm/ally)、enm_idx、hit_record(0/1)；
                        也兼容已打包的 dict（fire_obs/enm_idx/hit_record 三组键）
            fire_lr_ratio: 本次更新学习率相对 fire_head_optimizer 当前 lr 的缩放系数
        """
        # 解析样本：record 列表拆成三组同序序列；已打包 dict 直接取
        if isinstance(fire_batch, dict):
            fire_obs = fire_batch.get('fire_obs', fire_batch.get('states'))
            enm_idx = fire_batch.get('enm_idx')
            labels = fire_batch.get('hit_record', fire_batch.get('labels'))
        else:
            records = list(fire_batch or [])
            fire_obs = [r.get('fire_obs', r.get('state')) for r in records]
            enm_idx = [r.get('enm_idx') for r in records]
            labels = [r.get('hit_record', r.get('label')) for r in records]
        if fire_obs is None or enm_idx is None or labels is None or len(fire_obs) == 0:
            self.last_supervised_fire_loss = 0.0
            return 0.0

        # list[dict] -> dict-of-arrays（与 worker 侧 stack_field 同一语义，逐键堆叠）
        if not isinstance(fire_obs, dict) and isinstance(fire_obs[0], dict):
            fire_obs = {k: np.stack([o[k] for o in fire_obs]) for k in fire_obs[0]}
        fire_obs = {k: np.asarray(v) for k, v in fire_obs.items()}
        enm_idx = np.asarray(enm_idx)
        labels = np.asarray(labels, dtype=np.float32)

        n = len(labels)
        if fire_obs['fixed'].shape[0] != n or len(enm_idx) != n:
            raise ValueError('fire_obs/enm_idx/hit_record must have the same length')

        # 懒创建开火头优化器（首次调用时以 actor 当前 lr 为基准，只含 fire_head 参数）
        if getattr(self, 'fire_head_optimizer', None) is None:
            base_lr = self.actor_optimizer.param_groups[0]['lr']
            self.fire_head_optimizer = torch.optim.Adam(self.actor.pilot.fire_head.parameters(), lr=base_lr)

        # ── 缩放学习率 ────────────────────────────────────────────────────────
        current_fire_lr = self.fire_head_optimizer.param_groups[0]['lr']
        self.fire_head_optimizer.param_groups[0]['lr'] = current_fire_lr * fire_lr_ratio

        # ── 切断开火分支之前的梯度：冻结 actor 全部参数，仅放行 fire_head ──
        params = list(self.actor.parameters())
        old_requires_grad = [p.requires_grad for p in params]
        for p in params:
            p.requires_grad_(False)
        for p in self.actor.pilot.fire_head.parameters():
            p.requires_grad_(True)

        losses = []
        mb_size = max(1, min(int(batch_size), n))
        for _ in range(max(1, int(epochs))):
            order = np.random.permutation(n)
            for start in range(0, n, mb_size):
                idx = order[start:start + mb_size]
                mb_obs = {k: v[idx] for k, v in fire_obs.items()}
                loss = self.actor.evaluate_fire(mb_obs, enm_idx[idx], labels[idx])
                self.fire_head_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.pilot.fire_head.parameters(), max_grad_norm)
                self.fire_head_optimizer.step()
                losses.append(float(loss.detach().cpu()))

        for p, requires_grad in zip(params, old_requires_grad):
            p.requires_grad_(requires_grad)

        # ── 还原学习率 ────────────────────────────────────────────────────────
        self.fire_head_optimizer.param_groups[0]['lr'] = current_fire_lr

        self.last_supervised_fire_loss = float(np.mean(losses)) if losses else 0.0
        return self.last_supervised_fire_loss