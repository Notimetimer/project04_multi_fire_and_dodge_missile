from torch.distributions import Normal
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import os, sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from Algorithms.Utils import model_grad_norm, check_weights_bias_nan, moving_average, compute_advantage
from Algorithms.MLP_heads import PolicyNetBernouli, ValueNet

class PPO_bernouli:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, state_dim, hidden_dims, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device, k_entropy=0.01, critic_max_grad=2, actor_max_grad=2):
        self.actor = PolicyNetBernouli(state_dim, hidden_dims, action_dim).to(device)  # 使用 PolicyNetBernouli
        self.critic = ValueNet(state_dim, hidden_dims).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device
        self.k_entropy = k_entropy
        self.critic_max_grad = critic_max_grad
        self.actor_max_grad = actor_max_grad

    def set_learning_rate(self, actor_lr=None, critic_lr=None):
        """动态设置 actor 和 critic 的学习率"""
        if actor_lr is not None:
            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] = actor_lr
        if critic_lr is not None:
            for param_group in self.critic_optimizer.param_groups:
                param_group['lr'] = critic_lr    

    def get_action_probs(self, state):
        """
        获取动作概率分布。
        :param state: 输入状态 (tensor)
        :return: 动作概率 (tensor)
        """
        logits = self.actor(state)  # 获取未经过 sigmoid 的 logits
        probs = torch.sigmoid(logits)  # 应用 sigmoid 激活函数
        return probs

    # take action
    def take_action(self, state, explore=True, mask=None):
        # state -> tensor (1, state_dim)
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        probs = self.get_action_probs(state)  # 调用 get_action_probs 获取概率 (1,action_dim)

        if explore:
            sampled = torch.bernoulli(probs)  # (1, action_dim)
        else:
            sampled = (probs >= 0.5).float()

        # to numpy arrays (flatten for probs, keep action as 1-D ndarray)
        probs_np = probs.detach().cpu().numpy().flatten()           # shape (action_dim,)
        actions_np = sampled.detach().cpu().numpy().reshape(-1)     # shape (action_dim,)

        # mask = np.array([[0,1]] * probs_np.shape[-1]) ###
        if mask is not None:
            actions_np = np.clip(actions_np, mask[:, 0], mask[:, 1])
            # for i, action in enumerate(actions_np):
            #     actions_np[i] = np.clip(action, mask[i][0], mask[i][1])

        # 如果 action_dim == 1，保持返回形状为 (1,) 而不是 python int
        return actions_np, probs_np

    def update(self, transition_dict, adv_normed=False, clip_vf=False, clip_range=0.2):
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions']), dtype=torch.float).to(self.device)
        # # 统一 actions 形状为 (N, action_dim)
        # if actions.dim() == 1:
        #     actions = actions.unsqueeze(1)
        rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']), dtype=torch.float).view(-1, 1).to(self.device)
        
        # 基础检查
        if torch.isnan(states).any(): raise ValueError("NaN in input states")
        probs = self.get_action_probs(states)  # 调用 get_action_probs 获取概率
        log_probs = torch.log(probs) * actions + torch.log(1 - probs) * (1 - actions)
        log_probs = log_probs.sum(dim=1, keepdim=True)  # 对所有动作维度求和
        # 添加Actor NaN检查
        if torch.isnan(log_probs).any():
            raise ValueError("NaN in Actor outputs")
        # 添加Critic NaN检查
        critic_values = self.critic(states)
        if torch.isnan(critic_values).any():
            raise ValueError("NaN in Critic outputs")

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu(), dones.cpu()).to(self.device)
        
        # 优势归一化
        if adv_normed:
            adv_mean, adv_std = advantage.detach().mean(), advantage.detach().std(unbiased=False) 
            advantage = (advantage - adv_mean) / (adv_std + 1e-8)

        # 提前计算一次旧的 value 预测（用于 value clipping）
        v_pred_old = self.critic(states).detach()  # (N,1)

        old_probs = self.get_action_probs(states).detach()  # 调用 get_action_probs 获取概率
        old_log_probs = torch.log(old_probs) * actions + torch.log(1 - old_probs) * (1 - actions)
        old_log_probs = old_log_probs.sum(dim=1, keepdim=True).detach()

        actor_grad_list = []
        actor_loss_list = []
        critic_grad_list = []
        pre_clip_actor_grad = []
        pre_clip_critic_grad = []
        critic_loss_list = []
        entropy_list = []
        ratio_list = []

        for _ in range(self.epochs):
            probs = self.get_action_probs(states)  # 调用 get_action_probs 获取概率
            log_probs = torch.log(probs) * actions + torch.log(1 - probs) * (1 - actions)
            log_probs = log_probs.sum(dim=1, keepdim=True)

            # 添加Actor NaN检查
            if torch.isnan(log_probs).any():
                raise ValueError("NaN in Actor outputs in loop")
            # 添加Critic NaN检查
            critic_values = self.critic(states)
            if torch.isnan(critic_values).any():
                raise ValueError("NaN in Critic outputs in loop")

            # 权重/偏置 NaN 检查（在每次前向后、反向前检查参数）
            check_weights_bias_nan(self.actor, "actor", "update循环中")
            check_weights_bias_nan(self.critic, "critic", "update循环中")

            ratio = torch.exp(log_probs - old_log_probs) # (N,1)
            surr1 = torch.clamp(ratio, -20, 20) * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage

            # 计算熵
            entropy = - probs * torch.log(probs) - (1 - probs) * torch.log(1 - probs)
            entropy_factor = entropy.sum(dim=1).mean()

            actor_loss = torch.mean(-torch.min(surr1, surr2)) - self.k_entropy * entropy_factor # 标量

            # 计算 critic_loss：支持可选的 value clipping（PPO 风格）
            if clip_vf:
                v_pred = self.critic(states)                                  # 当前预测 (N,1)
                v_pred_clipped = torch.clamp(v_pred, v_pred_old - clip_range, v_pred_old + clip_range)
                vf_loss1 = (v_pred - td_target.detach()).pow(2)               # (N,1)
                vf_loss2 = (v_pred_clipped - td_target.detach()).pow(2)       # (N,1)
                critic_loss = torch.max(vf_loss1, vf_loss2).mean()
            else:
                critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
            
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            
            # 裁剪前梯度
            pre_clip_actor_grad.append(model_grad_norm(self.actor))
            pre_clip_critic_grad.append(model_grad_norm(self.critic))  

            # 梯度裁剪
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.actor_max_grad)
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.critic_max_grad)

            self.actor_optimizer.step()
            self.critic_optimizer.step()

            # # 保存用于日志/展示的数值（断开计算图并搬到 CPU）
            actor_grad_list.append(model_grad_norm(self.actor))
            actor_loss_list.append(actor_loss.detach().cpu().item())
            critic_grad_list.append(model_grad_norm(self.critic))            
            critic_loss_list.append(critic_loss.detach().cpu().item())
            entropy_list.append(entropy_factor.detach().cpu().item())
            ratio_list.append(ratio.mean().detach().cpu().item())
        
        self.actor_loss = np.mean(actor_loss_list)
        self.actor_grad = np.mean(actor_grad_list)
        self.critic_loss = np.mean(critic_loss_list)
        self.critic_grad = np.mean(critic_grad_list)
        self.entropy_mean = np.mean(entropy_list)
        self.ratio_mean = np.mean(ratio_list)
        self.pre_clip_critic_grad = np.mean(pre_clip_critic_grad)
        self.pre_clip_actor_grad = np.mean(pre_clip_actor_grad)
        self.advantage = advantage.abs().mean().detach().cpu().item()
        # 权重/偏置 NaN 检查（在每次前向后、反向前检查参数）
        check_weights_bias_nan(self.actor, "actor", "update后")
        check_weights_bias_nan(self.critic, "critic", "update后")

    def MARWIL_update(self, transition_dict, adv_normed=False, clip_vf=False, clip_range=0.2):
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions']), dtype=torch.float).to(self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']), dtype=torch.float).view(-1, 1).to(self.device)
        
        # 1. 初始 NaN 检查
        if torch.isnan(states).any(): raise ValueError("NaN in input states")
        probs = self.get_action_probs(states)  # 调用 get_action_probs 获取概率
        log_probs = torch.log(probs) * actions + torch.log(1 - probs) * (1 - actions)
        if torch.isnan(log_probs).any(): raise ValueError("NaN in Actor outputs")
        critic_values = self.critic(states)
        if torch.isnan(critic_values).any(): raise ValueError("NaN in Critic outputs")

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu(), dones.cpu()).to(self.device)
        
        # 优势归一化
        if adv_normed:
            adv_mean, adv_std = advantage.detach().mean(), advantage.detach().std(unbiased=False) 
            advantage = (advantage - adv_mean) / (adv_std + 1e-8)

        # 提前计算一次旧的 value 预测（用于 value clipping）
        v_pred_old = self.critic(states).detach()  # (N,1)
        for _ in range(self.epochs):
            probs = self.get_action_probs(states)  # 调用 get_action_probs 获取概率
            log_probs = torch.log(probs) * actions + torch.log(1 - probs) * (1 - actions)
            # log_probs = log_probs.sum(dim=1, keepdim=True)
            actor_loss = -torch.mean(torch.exp(self.beta * advantage * log_probs))

            # 计算 critic_loss：支持可选的 value clipping（PPO 风格）
            if clip_vf:
                v_pred = self.critic(states)                                  # 当前预测 (N,1)
                v_pred_clipped = torch.clamp(v_pred, v_pred_old - clip_range, v_pred_old + clip_range)
                vf_loss1 = (v_pred - td_target.detach()).pow(2)               # (N,1)
                vf_loss2 = (v_pred_clipped - td_target.detach()).pow(2)       # (N,1)
                critic_loss = torch.max(vf_loss1, vf_loss2).mean()
            else:
                # critic_loss = F.mse_loss(self.critic(states), td_target.detach()) # 原有
                critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))