import numpy as np
import torch
import torch.nn.functional as F
import os
import json
from Algorithms.Utils import compute_advantage
from Algorithms.ActorMLPHeads import ActorMLPWrapper, ActorMLPHeads

class MAPPOHybrid:
    def __init__(self, actor, critic, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device, 
                 beta_entropy=0.01, critic_max_grad=2, actor_max_grad=2, max_std=0.7):
        
        self.actor = actor 
        self.critic = critic
        
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device
        self.beta_entropy = beta_entropy
        
        self.critic_max_grad = critic_max_grad
        self.actor_max_grad = actor_max_grad
        self.max_std = max_std
        
        self.actor_loss = 0
        self.critic_loss = 0
        
    def RDistill(self, transition_dict, *args, **kwargs):
        return dict(transition_dict), 0
        
    def ADistill(self, advantage, *args, **kwargs):
        return advantage

    def update(self, transition_dict, adv_normed=True, clip_vf=False, clip_range=0.2, shuffled=True, mini_batch_size=None):
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
            # obs 为 {'agent_input','mask_input','rule_input'} 的 dict-of-arrays，
            # actions 为 {'vert','hor','fire','enm_idx','ally_idx',...} 的 dict-of-arrays
            old_log_probs = self.actor.evaluate_actions(
                obs['agent_input'], obs['mask_input'], obs['rule_input'], actions
            )['log_prob'].view(-1, 1)
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
                if clip_vf:
                    v_pred_clipped = v_pred_old[batch_idx] + torch.clamp(v_pred - v_pred_old[batch_idx], -clip_range, clip_range)
                    critic_loss = torch.max(F.mse_loss(v_pred, mb_td_target, reduction='none'), F.mse_loss(v_pred_clipped, mb_td_target, reduction='none'))
                else:
                    critic_loss = F.mse_loss(v_pred, mb_td_target, reduction='none')
                critic_loss = (critic_loss * mb_active_masks).sum() / (active_sum + mask_eps)
                entropy = (entropy * mb_active_masks).sum() / (active_sum + mask_eps)

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss = actor_loss - self.beta_entropy * entropy + critic_loss
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

        return {
            'actor_loss': np.mean(actor_loss_list) if actor_loss_list else 0.0,
            'critic_loss': np.mean(critic_loss_list) if critic_loss_list else 0.0,
            'entropy': np.mean(entropy_list) if entropy_list else 0.0,
            'kl': np.mean(kl_list) if kl_list else 0.0,
            'clip_frac': np.mean(clip_frac_list) if clip_frac_list else 0.0,
            'ratio': np.mean(ratio_list) if ratio_list else 0.0,
            'samples': ppo_samples_total,
            'valid_samples': ppo_valid_samples_total,
        }
