'''
混合动作空间空间的PPO改为SAC
'''
import random
import numpy as np
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, Bernoulli
import collections
import copy
import os, sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from Algorithms.Utils import model_grad_norm, check_weights_bias_nan, compute_advantage, SquashedNormal
from Algorithms.MLP_heads import ValueNet

def sigmoid(x):
    return 1/(1+np.exp(-x))

class ReplayBufferHybrid:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.buffer = collections.deque(maxlen=self.capacity)

    def add(self, state, action_dict, reward, next_state, done, active_mask=1.0):
        # active_mask: 智能体存活=1，死亡后=0，死亡样本不参与任何反向传播
        self.buffer.append((state, action_dict, reward, next_state, done, active_mask))

    def save(self, path):
        """持久化经验池内容，支持中断续训。"""
        torch.save({'capacity': self.capacity, 'data': list(self.buffer)}, path)
        print(f"[ReplayBufferHybrid] Saved to {path}. Size: {len(self.buffer)}")

    @staticmethod
    def load(path, map_location='cpu'):
        """从磁盘读取经验池，找不到则返回 None。"""
        if not os.path.exists(path):
            return None
        ckpt = torch.load(path, map_location=map_location)
        buf = ReplayBufferHybrid(ckpt['capacity'])
        buf.buffer = collections.deque(ckpt['data'], maxlen=ckpt['capacity'])
        print(f"[ReplayBufferHybrid] Loaded from {path}. Size: {len(buf.buffer)}")
        return buf

    def sample(self, batch_size):
        # 1. 随机抽样
        transitions = random.sample(self.buffer, batch_size)
        
        # 2. 解包（兼容旧的 5 元组存档：无 active_mask 时默认 1.0=存活）
        states, actions, rewards, next_states, dones, active_masks = [], [], [], [], [], []
        for t in transitions:
            states.append(t[0])
            actions.append(t[1])
            rewards.append(t[2])
            next_states.append(t[3])
            dones.append(t[4])
            active_masks.append(t[5] if len(t) > 5 else 1.0)
        
        # 3. 规整动作字典 (List[Dict] -> Dict[Array])
        actions_dict_np = {}
        if len(actions) > 0:
            for key in actions[0].keys():
                actions_dict_np[key] = np.array([act[key] for act in actions])

        # 4. 【核心】：打包成一个干净的字典返回
        # 此时所有值都是已经对齐好的 NumPy Array
        batch_dict = {
            'states': np.array(states, dtype=np.float32),
            'actions': actions_dict_np,
            'rewards': np.array(rewards, dtype=np.float32).reshape(-1, 1), # 预处理形状
            'next_states': np.array(next_states, dtype=np.float32),
            'dones': np.array(dones, dtype=np.float32).reshape(-1, 1),     # 预处理形状
            'active_masks': np.array(active_masks, dtype=np.float32).reshape(-1, 1)  # 死亡样本=0
        }
        
        return batch_dict

    def size(self):
        return len(self.buffer)

# =============================================================================
# 1. 神经网络定义 (保持不变，只负责 forward 计算)
# =============================================================================


class QNetHybrid(torch.nn.Module):
    def __init__(self, state_dim, hidden_dims, action_dims_dict):
        super(QNetHybrid, self).__init__()
        
        # 计算所有动作展平后的总维度
        act_dim = 0
        if 'cont' in action_dims_dict: act_dim += action_dims_dict['cont']
        if 'cat' in action_dims_dict: act_dim += sum(action_dims_dict['cat']) # Cat 需要转为 One-hot 输入
        if 'bern' in action_dims_dict: act_dim += action_dims_dict['bern']
        
        layers = []
        prev_size = state_dim + act_dim
        for layer_size in hidden_dims:
            layers.append(nn.Linear(prev_size, layer_size))
            layers.append(nn.ReLU())
            prev_size = layer_size
        self.net = nn.Sequential(*layers)
        self.fc_out = nn.Linear(prev_size, 1)

    def forward(self, state, action_dict):
        # 拼接动作
        actions_list = []
        if 'cont' in action_dict and action_dict['cont'] is not None:
            actions_list.append(action_dict['cont'])
        if 'cat' in action_dict and action_dict['cat'] is not None:
            actions_list.append(action_dict['cat']) # 这里必须已经是 one-hot 或 gumbel-softmax 的输出
        if 'bern' in action_dict and action_dict['bern'] is not None:
            actions_list.append(action_dict['bern'])
            
        action_cat = torch.cat(actions_list, dim=-1)
        x = torch.cat([state, action_cat], dim=-1)
        return self.fc_out(self.net(x))

from Algorithms.PPOHybrid23_0 import PolicyNetHybrid, HybridActorWrapper as PPOHybridActorWrapper

# =============================================================================
# 2. Actor 适配器 (Wrapper) - 核心重构点
# =============================================================================

class HybridActorWrapper(PPOHybridActorWrapper):
    """
    继承自 PPOHybrid23_0 的统一接口适配器。
    为 SAC 额外提供可导采样的 sample_for_sac 方法。
    """

    def sample_for_sac(self, states, action_masks=None, gumbel_tau=1.5):
        """
        专门为 SAC 提供的采样方法。
        返回可导的 actions，以及按动作头拆分的 log_prob 字典。
        """
        actor_outputs = self.net(states, action_masks=action_masks)

        actions_differentiable = {}
        log_probs_total = torch.zeros(states.size(0), 1).to(self.device)
        log_probs_cont = torch.zeros(states.size(0), 1).to(self.device)
        log_probs_cat = torch.zeros(states.size(0), 1).to(self.device)
        log_probs_bern = torch.zeros(states.size(0), 1).to(self.device)

        # --- Cont (连续动作，使用 rsample) ---
        if actor_outputs['cont'] is not None:
            mu, std = actor_outputs['cont']
            # 注意：此处需确保 SquashedNormal 支持 rsample 并且正确计算了 tanh 的 log_prob
            dist = Normal(mu, std)
            u = dist.rsample() # 重参数化采样
            a_norm = torch.tanh(u)
            # 计算 Squash 的 log_prob
            log_prob_cont = dist.log_prob(u) - torch.log(1 - a_norm.pow(2) + 1e-7)
            log_prob_cont_sum = log_prob_cont.sum(-1, keepdim=True)
            log_probs_cont += log_prob_cont_sum
            log_probs_total += log_prob_cont_sum

            actions_differentiable['cont'] = a_norm # 直接输出 -1~1 的范围给 Q 网络

        # --- Cat (离散动作，使用 Gumbel-Softmax) ---
        if actor_outputs['cat'] is not None:
            # PPOHybrid23_0.PolicyNetHybrid 返回的是 softmax 后的 probs，
            # 取 log 后作为 gumbel_softmax 的 logits 使用
            cat_logits_list = [torch.log(probs + 1e-8) for probs in actor_outputs['cat']]
            cat_actions = []
            log_p_cat_sum = torch.zeros(states.size(0), 1).to(self.device)
            for logits in cat_logits_list:
                # hard=True 表示前向传播输出 One-hot(例如[0,1,0])，反向传播用 softmax 的梯度
                gumbel_out = F.gumbel_softmax(logits, tau=gumbel_tau, hard=True)
                cat_actions.append(gumbel_out)

                # 计算 log_prob (近似)
                probs = F.softmax(logits, dim=-1)
                dist = Categorical(probs=probs)
                # 由于 hard=True 返回的是 one-hot，可以通过与 log_probs 相乘来提取选中项的 log_prob
                log_p = torch.sum(torch.log(probs + 1e-8) * gumbel_out, dim=-1, keepdim=True)
                log_p_cat_sum += log_p

            log_probs_cat += log_p_cat_sum
            log_probs_total += log_p_cat_sum
            actions_differentiable['cat'] = torch.cat(cat_actions, dim=-1)

        # --- Bern (伯努利动作，使用 Binary Gumbel-Softmax / 缓和的 Sigmoid) ---
        if actor_outputs['bern'] is not None:
            bern_logits = actor_outputs['bern']
            # 将 logits 转换为 [prob_0, prob_1] 的形式以便使用 gumbel_softmax
            logits_2d = torch.stack([-bern_logits, bern_logits], dim=-1)
            gumbel_out = F.gumbel_softmax(logits_2d, tau=1.0, hard=True)
            bern_action = gumbel_out[..., 1] # 取出代表 1(True) 的那一列

            actions_differentiable['bern'] = bern_action

            probs = torch.sigmoid(bern_logits)
            # 计算每个 bern 维度的 log_prob
            log_p_bern = torch.log(probs + 1e-8) * bern_action + torch.log(1 - probs + 1e-8) * (1 - bern_action)

            # 用 fire_mask 屏蔽被 can_fire=False 的位置：乘以 0 使其对梯度无贡献
            # 这样被 mask 的位置既不影响 actor_loss、alpha_loss，也不影响 Q 目标中的熵正则项
            fire_mask = actor_outputs.get('fire_mask', None)
            if fire_mask is not None:
                log_p_bern = log_p_bern * fire_mask  # shape: (batch, bern_dim)，masked 位置乘 0

            log_p_bern_sum = log_p_bern.view(states.size(0), -1).sum(-1, keepdim=True)
            log_probs_bern += log_p_bern_sum
            log_probs_total += log_p_bern_sum

        log_probs = {
            'cont': log_probs_cont,
            'cat': log_probs_cat,
            'bern': log_probs_bern,
            'total': log_probs_total,
        }
        return actions_differentiable, log_probs
# =============================================================================
# 3. SAC 算法类 (精简版)
# =============================================================================
class SACHybrid:
    def __init__(self, actor, critic_temp, critic_1, critic_2, target_critic_1, target_critic_2, 
                 actor_lr, critic_lr, alpha_lr, action_dims_dict, gamma, tau, device,
                 k_entropy={'cont':0.01, 'cat':0.005, 'bern':0.05}, critic_max_grad=2, actor_max_grad=2, max_std=0.7,
                 gumbel_tau=1.5):
        self.actor = actor
        # MARWIL_update 内部引用 self.critic，这里让其指向预训练用的 ValueNet
        self.critic = critic_temp # 仅给预训练(MARWIL)使用，在线SAC阶段弃置不用
        self.critic_1 = critic_1
        self.critic_2 = critic_2
        self.target_critic_1 = target_critic_1
        self.target_critic_2 = target_critic_2

        # 保存超参，供学习率调整 / 梯度裁剪 / 重建优化器使用
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.alpha_lr = alpha_lr
        self.k_entropy = k_entropy
        self.max_std = max_std
        self.actor_max_grad = actor_max_grad
        self.critic_max_grad = critic_max_grad
        self.gumbel_tau = gumbel_tau
        
        # 初始化目标网络
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)
        # 预训练 (MARWIL) 阶段优化 ValueNet 的优化器
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # 自动调节温度参数 Alpha
        # 针对 Hybrid，可以设一个全局 Alpha，也可以为 cont, cat, bern 各设一个。这里用一个全局的演示。
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        
        # target_entropy 由外部在 update() 中传入，此处不做预设
        
        self.gamma = gamma
        self.tau = tau
        self.device = device

    def take_action(self, state, explore=True, check_obs=None, **kwargs):
        # 推理时仍然使用 get_action，用于环境交互并包含动作还原
        # 透传 mask_on / temperature 等额外参数给 wrapper
        return self.actor.get_action(state, explore=explore, check_obs=check_obs, **kwargs)

    def set_learning_rate(self, actor_lr=None, critic_lr=None):
        """动态调整学习率，兼容主训练脚本的调用接口。"""
        if actor_lr is not None:
            self.actor_lr = actor_lr
            for g in self.actor_optimizer.param_groups:
                g['lr'] = actor_lr
        if critic_lr is not None:
            self.critic_lr = critic_lr
            for opt in (self.critic_1_optimizer, self.critic_2_optimizer, self.critic_optimizer):
                for g in opt.param_groups:
                    g['lr'] = critic_lr

    def reset_optimizer(self):
        """重建优化器以清除动量（中断续训/恢复崩溃时使用）。"""
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=self.critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=self.critic_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)

    def save_critics(self, path):
        """保存在线SAC的Q网络与温度参数（弃置ValueNet）。"""
        torch.save({
            'critic_1': self.critic_1.state_dict(),
            'critic_2': self.critic_2.state_dict(),
            'target_critic_1': self.target_critic_1.state_dict(),
            'target_critic_2': self.target_critic_2.state_dict(),
            'log_alpha': self.log_alpha.detach().cpu(),
            'gumbel_tau': self.gumbel_tau,
        }, path)

    def load_critics(self, path, map_location='cpu'):
        ckpt = torch.load(path, map_location=map_location)
        # 兼容旧的 ValueNet critic.pt（只有 state_dict，没有Q网络键）
        if not isinstance(ckpt, dict) or 'critic_1' not in ckpt:
            print(f"[SACHybrid] {path} 不是SAC critic格式，跳过加载Q网络。")
            return
        self.critic_1.load_state_dict(ckpt['critic_1'])
        self.critic_2.load_state_dict(ckpt['critic_2'])
        self.target_critic_1.load_state_dict(ckpt['target_critic_1'])
        self.target_critic_2.load_state_dict(ckpt['target_critic_2'])
        if 'log_alpha' in ckpt:
            with torch.no_grad():
                self.log_alpha.copy_(ckpt['log_alpha'].to(self.log_alpha.device))

    def save_optimizers(self, path):
        torch.save({
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_1_optimizer': self.critic_1_optimizer.state_dict(),
            'critic_2_optimizer': self.critic_2_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
        }, path)

    def load_optimizers(self, path, map_location='cpu'):
        s = torch.load(path, map_location=map_location)
        try:
            self.actor_optimizer.load_state_dict(s['actor_optimizer'])
            if 'critic_1_optimizer' in s:
                self.critic_1_optimizer.load_state_dict(s['critic_1_optimizer'])
                self.critic_2_optimizer.load_state_dict(s['critic_2_optimizer'])
            if 'alpha_optimizer' in s:
                self.alpha_optimizer.load_state_dict(s['alpha_optimizer'])
        except Exception as e:
            print(f"[SACHybrid] Failed to load optimizers: {e}")

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, batch, target_entropy=2.0, alpha_clip=(0.001, 0.1), freeze_actor=False):
        """
        接收 ReplayBuffer 返回的字典 batch
        target_entropy : 目标熵（正数，由外部传入）。None 则不更新 alpha。
        alpha_clip      : (min, max) 对 alpha=exp(log_alpha) 的截断范围。
        freeze_actor   : True 时只更新 Q 网络，跳过 actor 和 alpha 更新（Q 预热阶段使用）。
        """
        # --- A. 数据搬运与类型转换 (NumPy -> Tensor) ---
        device = self.device
        
        states = torch.from_numpy(batch['states']).to(device)
        next_states = torch.from_numpy(batch['next_states']).to(device)
        rewards = torch.from_numpy(batch['rewards']).to(device)
        dones = torch.from_numpy(batch['dones']).to(device)
        if 'active_masks' in batch:
            active_masks = torch.from_numpy(np.array(batch['active_masks'])).float().to(device).view(-1, 1)
        else:
            active_masks = torch.ones(states.size(0), 1, device=device)
        
        # 处理动作 (从字典中提取并转为 Tensor)
        raw_actions = batch['actions']
        actions_for_q = {}
        
        if 'cont' in raw_actions:
            actions_for_q['cont'] = torch.from_numpy(raw_actions['cont']).to(device)
            
        if 'cat' in raw_actions:
            cat_idx = torch.from_numpy(raw_actions['cat']).to(device).long()
            # 转换为 One-hot 供 Q 网络输入
            cat_onehots = []
            for i, dim in enumerate(self.actor.action_dims['cat']):
                cat_onehots.append(F.one_hot(cat_idx[:, i], num_classes=dim).float())
            actions_for_q['cat'] = torch.cat(cat_onehots, dim=-1)
            
        if 'bern' in raw_actions:
            actions_for_q['bern'] = torch.from_numpy(raw_actions['bern']).to(device)

        # --- B. SAC 计算逻辑 (逻辑保持不变，但变量名已对齐) ---
        
        # [诊断] 统计当前 batch 中 replay buffer 存储的 bern 动作分布
        if not hasattr(self, '_diag_update_count'):
            self._diag_update_count = 0
        self._diag_update_count += 1
        
        if self._diag_update_count % 200 == 1:
            if 'bern' in raw_actions:
                bern_buf = raw_actions['bern']  # shape: (batch, bern_dim)
                n_fire = (bern_buf > 0.5).sum()
                n_no_fire = (bern_buf <= 0.5).sum()
                print(f"[SAC diag #{self._diag_update_count}] replay buffer bern: fire={n_fire}, no_fire={n_no_fire}, ratio={n_fire/(n_fire+n_no_fire+1e-8):.3f}")

        # 1. 更新 Q 网络 (Critic)
        with torch.no_grad():
            # 获取下一状态的动作 (可导采样) 和 log_prob
            next_actions_diff, next_log_probs = self.actor.sample_for_sac(next_states, gumbel_tau=self.gumbel_tau)
            
            # [诊断] 统计 next_states 里被 mask 和未被 mask 的样本数
            if self._diag_update_count % 200 == 1:
                _outs_diag = self.actor.net(next_states)
                _fm = _outs_diag.get('fire_mask', None)
                if _fm is not None:
                    n_can_fire = (_fm > 0.5).sum().item()
                    n_masked = (_fm <= 0.5).sum().item()
                    bern_lp = next_log_probs['bern']
                    # 统计 sample_for_sac 采出的 bern_action 在 can_fire 位置的均值（接近1=偏开火，接近0=偏不开火）
                    valid_mask_1d = (_fm > 0.5).view(-1)
                    bern_act = next_actions_diff.get('bern', None)
                    if bern_act is not None:
                        bern_act_flat = bern_act.view(-1)
                        canfire_mean = bern_act_flat[valid_mask_1d].mean().item() if valid_mask_1d.any() else float('nan')
                    else:
                        canfire_mean = float('nan')
                    bern_logit_canfire = _outs_diag['bern'].view(-1)[valid_mask_1d]
                    logit_mean = bern_logit_canfire.mean().item() if valid_mask_1d.any() else float('nan')
                    logit_max = bern_logit_canfire.max().item() if valid_mask_1d.any() else float('nan')
                    print(f"[SAC diag #{self._diag_update_count}] next_states fire_mask: can_fire={n_can_fire}, masked={n_masked}, "
                          f"bern_logprob mean={bern_lp.mean().item():.4f}, "
                          f"bern_action[can_fire] mean={canfire_mean:.3f}, "
                          f"bern_logit[can_fire] mean={logit_mean:.3f} max={logit_max:.3f}")
            
            # 目标 Q 值
            q1_target = self.target_critic_1(next_states, next_actions_diff)
            q2_target = self.target_critic_2(next_states, next_actions_diff)
            alpha = self.log_alpha.exp()
            k_bern = self.k_entropy.get('bern', 0.003)
            entropy_reg = alpha * (next_log_probs['cont'] + next_log_probs['cat']) + k_bern * next_log_probs['bern']
            min_q_target = torch.min(q1_target, q2_target) - entropy_reg
            
            # TD 目标
            y_target = rewards + self.gamma * (1 - dones) * min_q_target
            
        # 当前 Q 值预测
        q1_pred = self.critic_1(states, actions_for_q)
        q2_pred = self.critic_2(states, actions_for_q)
        
        mask_eps = 1e-5
        active_sum = active_masks.sum()
        critic_loss = (F.mse_loss(q1_pred, y_target, reduction='none') + F.mse_loss(q2_pred, y_target, reduction='none')).mean()
        
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad = nn.utils.clip_grad_norm_(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()), self.critic_max_grad)
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # 2. 更新 策略网络 (Actor) —— freeze_actor=True 时跳过
        if not freeze_actor:
            # 重新对当前状态采样
            curr_actions_diff, curr_log_probs = self.actor.sample_for_sac(states, gumbel_tau=self.gumbel_tau)
            
            q1_pi = self.critic_1(states, curr_actions_diff)
            q2_pi = self.critic_2(states, curr_actions_diff)
            min_q_pi = torch.min(q1_pi, q2_pi)
            
            alpha = self.log_alpha.exp().detach()
            # 机动部分 (cont+cat) 使用自适应 alpha；开火部分 (bern) 使用固定初始熵系数
            k_bern = self.k_entropy.get('bern', 0.05)
            actor_loss = ((alpha * (curr_log_probs['cont'] + curr_log_probs['cat'])
                          + k_bern * curr_log_probs['bern']
                          - min_q_pi) * active_masks).sum() / (active_sum + mask_eps)
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_grad = nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_max_grad)
            self.actor_optimizer.step()

            # 3. 更新 Alpha (熵系数)
            if target_entropy is not None:
                self.target_entropy = target_entropy
                # alpha 只根据机动部分 (cont+cat) 的熵来调节
                # mobility_log_probs 是 log_prob（负数），熵 entropy = -log_prob（正数）
                mobility_log_probs = curr_log_probs['cont'].detach() + curr_log_probs['cat'].detach()
                mobility_entropy = -mobility_log_probs
                alpha_loss = -(self.log_alpha * (mobility_entropy - target_entropy) * active_masks).sum() / (active_sum + mask_eps)
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                # 将 alpha 截断到合法范围
                if alpha_clip is not None:
                    log_alpha_min = np.log(alpha_clip[0])
                    log_alpha_max = np.log(alpha_clip[1])
                    with torch.no_grad():
                        self.log_alpha.clamp_(log_alpha_min, log_alpha_max)
            else:
                alpha_loss = torch.tensor(0.0)
        else:
            # freeze_actor=True：用零值占位，不触碰 actor/alpha 参数
            curr_log_probs = {'cont': torch.zeros(1), 'cat': torch.zeros(1), 'bern': torch.zeros(1), 'total': torch.zeros(1)}
            actor_loss = torch.tensor(0.0)
            actor_grad = torch.tensor(0.0)
            alpha_loss = torch.tensor(0.0)

        # 4. 目标网络软更新
        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)
        
        # --- 监控指标（兼容主训练脚本的 logger 字段） ---
        self.last_actor_loss = actor_loss.item()
        self.last_critic_loss = critic_loss.item()
        self.last_entropy = -curr_log_probs['total'].mean().item()
        self.last_entropy_mobility = -(curr_log_probs['cont'] + curr_log_probs['cat']).mean().item()
        self.last_entropy_bern = -curr_log_probs['bern'].mean().item()
        self.actor_loss = self.last_actor_loss
        self.critic_loss = self.last_critic_loss
        self.entropy_mean = self.last_entropy
        self.alpha = self.log_alpha.exp().item()
        self.k_bern = self.k_entropy.get('bern', 0.003)
        self.pre_clip_actor_grad = float(actor_grad)
        self.pre_clip_critic_grad = float(critic_grad)
        self.td_error_var = (y_target - q1_pred).detach().var().item()

        # 各动作头熵 / 开火概率（基于当前策略分布，便于监控）
        with torch.no_grad():
            outs = self.actor.net(states)
            self.entropy_cat = 0.0
            self.entropy_bern = 0.0
            self.max_fire_prob = 0.0
            self.min_fire_prob = 0.0
            if outs.get('cat') is not None:
                ent_c = 0.0
                for probs in outs['cat']:
                    ent_c += Categorical(probs=probs).entropy().mean().item()
                self.entropy_cat = ent_c
            if outs.get('bern') is not None:
                bern_logits = outs['bern'].clamp(min=-1e8)
                fire_mask = outs.get('fire_mask', None)
                if fire_mask is not None:
                    # fire_mask 由 PolicyNetHybrid.forward 内部计算，标记哪些样本位置允许开火（弹药充足+冷却到位+角度合理）
                    # 只统计未被 mask 的有效位置，避免被强制压成 -1e8 的位置拉低 min_fire_prob 或 entropy_bern
                    valid_mask = (fire_mask > 0.5)
                    if valid_mask.any():
                        valid_probs = torch.sigmoid(bern_logits)[valid_mask]
                        valid_logits = bern_logits[valid_mask]
                        # 熵只在有效位置内求平均，除以有效位置数
                        self.entropy_bern = Bernoulli(logits=valid_logits).entropy().sum().item() / max(valid_mask.sum().item(), 1)
                        # 最大/最小开火概率也只在有效位置内统计
                        self.max_fire_prob = valid_probs.max().item()
                        self.min_fire_prob = valid_probs.min().item()
                    else:
                        # 全 batch 都被 mask，无法开火
                        self.entropy_bern = 0.0
                        self.max_fire_prob = 0.0
                        self.min_fire_prob = 0.0
                else:
                    # 兼容性分支：如果 net 没有返回 fire_mask（理论上不应该发生），则全量统计
                    self.entropy_bern = Bernoulli(logits=bern_logits).entropy().sum(-1).mean().item()
                    fire_probs = torch.sigmoid(bern_logits)
                    self.max_fire_prob = fire_probs.max().item()
                    self.min_fire_prob = fire_probs.min().item()

    # =========================================================================
    #  [New Method] Bernoulli 开火头保护性有监督训练 (防止机动策略被bern崩溃拖累)
    # =========================================================================
    def fire_prob_protection(self, batch, protect_epochs=4, protect_mini_batch=256):
        """
        Bern头概率范围保护器。当开火概率整体崩溃（全高或全低）时，以有监督方式
        强行拉回bern头分布，同时切断backbone和其它动作头的梯度，保护机动策略不被拖垮。

        必要条件1 (比值护栏): max_fire_prob / min_fire_prob >= 10，说明分布仍有分化空间，
                               不需要干预，直接跳过。
        必要条件2 (触发case):
          case1: max_fire_prob < 0.05  → 整体开火概率崩到极低，以0.5为监督信号，拉高熵。
          case2: min_fire_prob > 0.1   → 整体开火概率过高，以1e-3为监督信号，压低概率。

        Args:
            batch           : 与update()相同格式的经验字典（来自ReplayBuffer）。
            protect_epochs  : 保护性训练的epoch数。
            protect_mini_batch: 每个mini-batch的大小。
        """
        # ── 必要条件1：比值护栏 ──────────────────────────────────────────────────
        ratio = self.max_fire_prob / (self.min_fire_prob + 1e-12)
        if ratio >= 10.0:
            return

        # ── 必要条件2：判断触发case ──────────────────────────────────────────────
        if self.max_fire_prob < 0.05:
            target_prob = 0.5
        elif self.min_fire_prob > 0.1:
            target_prob = 1e-3
        else:
            return

        # ── 数据准备 ─────────────────────────────────────────────────────────
        def to_tensor(x, dtype):
            if isinstance(x, np.ndarray):
                return torch.tensor(x, dtype=dtype).to(self.device)
            else:
                return torch.tensor(np.array(x), dtype=dtype).to(self.device)

        if 'obs' in batch:
            actor_inputs = to_tensor(batch['obs'], torch.float)
        else:
            actor_inputs = to_tensor(batch['states'], torch.float)

        if 'active_masks' in batch:
            active_masks_p = to_tensor(batch['active_masks'], torch.float).view(-1, 1)
        else:
            active_masks_p = torch.ones(actor_inputs.size(0), 1, device=self.device)

        num_samples = actor_inputs.size(0)
        mb_size = min(protect_mini_batch, num_samples)

        # ── 冻结除bern头以外的所有actor模块 ───────────────────────────────────
        net = self.actor.net

        def set_requires_grad(module_or_param, flag):
            if isinstance(module_or_param, nn.Module):
                for p in module_or_param.parameters():
                    p.requires_grad_(flag)
            else:
                module_or_param.requires_grad_(flag)

        set_requires_grad(net.net, False)
        if hasattr(net, 'fc_mu'):
            set_requires_grad(net.fc_mu, False)
        if hasattr(net, 'log_std_cont'):
            set_requires_grad(net.log_std_cont, False)
        if hasattr(net, 'fc_cat'):
            set_requires_grad(net.fc_cat, False)
        if hasattr(net, 'fc_bern'):
            set_requires_grad(net.fc_bern, True)

        # ── 监督训练循环 ──────────────────────────────────────────────────────
        target_tensor = torch.tensor(target_prob, device=self.device)

        for _ in range(protect_epochs):
            perm = torch.randperm(num_samples, device=self.device)
            for start in range(0, num_samples, mb_size):
                end = min(start + mb_size, num_samples)
                batch_idx = perm[start:end]

                mb_states = actor_inputs[batch_idx]
                mb_active = active_masks_p[batch_idx]
                active_sum_p = mb_active.sum()

                actor_out = self.actor.net(mb_states)
                if actor_out['bern'] is None:
                    break

                bern_probs = torch.sigmoid(actor_out['bern'].clamp(min=-1e8))
                target_full = target_tensor.expand_as(bern_probs)
                bern_loss_per = F.binary_cross_entropy(bern_probs, target_full, reduction='none').sum(dim=-1, keepdim=True)
                bern_loss = (bern_loss_per * mb_active).sum() / (active_sum_p + 1e-5)

                self.actor_optimizer.zero_grad()
                bern_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.actor_max_grad)
                self.actor_optimizer.step()

        # ── 恢复所有actor模块的梯度反向传播 ──────────────────────────────────
        set_requires_grad(net.net, True)
        if hasattr(net, 'fc_mu'):
            set_requires_grad(net.fc_mu, True)
        if hasattr(net, 'log_std_cont'):
            set_requires_grad(net.log_std_cont, True)
        if hasattr(net, 'fc_cat'):
            set_requires_grad(net.fc_cat, True)
        if hasattr(net, 'fc_bern'):
            set_requires_grad(net.fc_bern, True)

        return

    # --- 修改后的 MARWIL_update， 注意原先是0 ---
    def MARWIL_update(self, il_transition_dict, beta=1.0, batch_size=64, alpha=1.0, c_v=1.0, shuffled=1, label_smoothing=0.3, max_weight=100.0,
                      tau=0.8, no_bern=1):
        """
        MARWIL 离线更新函数
        输入 actions 结构支持: [{'cat': array([v]), 'bern': array([v])}, ...]
        tau: 非对称损失权重 (Expectile Regression). tau=0.5 为 MSE; tau>0.5 (如0.9) 倾向于高估 Value (拟合好样本)
        """
        # 1. 数据准备
        if 'obs' in il_transition_dict and len(il_transition_dict['obs']) > 0:
            obs_all = torch.tensor(np.array(il_transition_dict['obs']), dtype=torch.float).to(self.device)
            use_obs = True
        else:
            use_obs = False
            
        # 预训练阶段通常不训练探索 std
        if hasattr(self.actor.net, 'log_std_cont'):
            self.actor.net.log_std_cont.requires_grad = False

        # 1. 提取全量数据并转为 Tensor
        states_all = torch.tensor(np.array(il_transition_dict['states']), dtype=torch.float).to(self.device)
        returns_all = torch.tensor(np.array(il_transition_dict['returns']), dtype=torch.float).view(-1, 1).to(self.device)
        
        # 统一处理 Actions：List of Dicts -> Dict of Tensors
        raw_actions = il_transition_dict['actions']
        actions_all = {}
        
        # 1. 如果是列表 (List of Dicts)，先堆叠成 Dict of Numpy Arrays
        if isinstance(raw_actions, list):
            keys = raw_actions[0].keys()
            temp_dict = {}
            for k in keys:
                # np.stack 会把 [array([1]), array([2])] 变成 array([[1], [2]]) -> (N, 1)
                temp_dict[k] = np.stack([d[k] for d in raw_actions], axis=0)
            raw_actions = temp_dict # 现在变成了 Dict of Arrays

        # 2. Dict of Arrays -> Dict of Tensors
        if isinstance(raw_actions, dict):
            for k, v in raw_actions.items():
                if k == 'cat':
                    actions_all[k] = torch.tensor(v, dtype=torch.long).to(self.device)
                else:
                    actions_all[k] = torch.tensor(v, dtype=torch.float).to(self.device)
        # ============================================================

        # 2. 准备 Batch 索引
        total_size = states_all.size(0)
        indices = np.arange(total_size)
        if shuffled:
            np.random.shuffle(indices)

        total_actor_loss = 0
        total_critic_loss = 0
        total_c = 0
        batch_count = 0

        # [新增] 权重与 advantage 监控累加器
        total_weight_mean = 0.0
        total_weight_max = 0.0
        total_weight_min = 0.0
        total_clip_frac = 0.0
        total_adv_std = 0.0
        total_adv_p95 = 0.0
        total_adv_max = 0.0
        total_adv_mean = 0.0

        # 3. Mini-batch 循环
        for start in range(0, total_size, batch_size):
            end = min(start + batch_size, total_size)
            batch_indices = indices[start:end]
            
            s_batch = states_all[batch_indices] 
            r_batch = returns_all[batch_indices]
            
            if use_obs:
                actor_input_batch = obs_all[batch_indices]
            else:
                actor_input_batch = s_batch 
            
            # 动作字典切片
            actions_batch = {}
            for k, v in actions_all.items():
                actions_batch[k] = v[batch_indices]

            # A. Advantage & Weights
            with torch.no_grad():
                values = self.critic(s_batch)
                residual = r_batch - values
                
                if not hasattr(self, 'c_sq'): 
                    self.c_sq = torch.tensor(1.0, device=self.device)
                
                batch_mse = (residual ** 2).mean().item()
                self.c_sq = self.c_sq + 1e-8 * (batch_mse - self.c_sq)
                c = torch.sqrt(self.c_sq)
                
                advantage = residual / (c + 1e-8)
                raw_weights = torch.exp(beta * advantage)
                weights = torch.clamp(raw_weights, max=max_weight)

                # [新增] 记录本 batch 的权重统计
                total_weight_mean += weights.mean().item()
                total_weight_max += weights.max().item()
                total_weight_min += weights.min().item()
                total_clip_frac += (weights >= max_weight - 1e-6).float().mean().item()

                # [新增] 记录本 batch advantage 分布统计
                adv = advantage.detach()
                total_adv_std += adv.std().item()
                total_adv_p95 += torch.quantile(adv, 0.95).item()
                total_adv_max += adv.max().item()
                total_adv_mean += adv.mean().item()

            # B. Actor Loss
            raw_il_loss = self.actor.compute_il_loss(
                actor_input_batch,
                actions_batch,
                label_smoothing,
                no_bern=no_bern,
                good_samples=1,
                pre_training=1,
            )
            actor_loss = torch.mean(alpha * weights * raw_il_loss)

            # C. Critic Loss
            v_pred = self.critic(s_batch)
            
            # 原有
            critic_loss = F.mse_loss(v_pred, r_batch) * c_v
            


            # D. Optimize
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.actor_max_grad)
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.critic_max_grad)
            self.actor_optimizer.step()
            self.critic_optimizer.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_c += c.item()
            batch_count += 1

        avg_weight_mean = total_weight_mean / batch_count if batch_count > 0 else 0
        avg_weight_max = total_weight_max / batch_count if batch_count > 0 else 0
        avg_weight_min = total_weight_min / batch_count if batch_count > 0 else 0
        avg_clip_frac = total_clip_frac / batch_count if batch_count > 0 else 0
        avg_adv_std = total_adv_std / batch_count if batch_count > 0 else 0
        avg_adv_p95 = total_adv_p95 / batch_count if batch_count > 0 else 0
        avg_adv_max = total_adv_max / batch_count if batch_count > 0 else 0
        avg_adv_mean = total_adv_mean / batch_count if batch_count > 0 else 0

        avg_actor_loss = total_actor_loss / batch_count if batch_count > 0 else 0
        avg_critic_loss = total_critic_loss / batch_count if batch_count > 0 else 0
        avg_c = total_c / batch_count if batch_count > 0 else 0

        # ============================================================
        # [新增] 监控：在固定的全量 batch 上独立统计每个动作头的 NLL 与熵
        # 全程 no_grad，不参与反传，因此不会干扰现有的网络更新。
        # ============================================================
        if use_obs:
            monitor_input_all = obs_all
        else:
            monitor_input_all = states_all
        with torch.no_grad():
            values_all = self.critic(states_all)
            residual_all = returns_all - values_all
            advantage_all = residual_all / (torch.sqrt(self.c_sq) + 1e-8)
        monitor_metrics = self.actor.compute_marwil_monitor(monitor_input_all, actions_all, advantages=advantage_all)
        # 缓存到 agent 属性，便于训练脚本拉取写入 logger
        self.marwil_nll_cont = monitor_metrics['nll_cont']
        self.marwil_nll_cat = monitor_metrics['nll_cat']
        self.marwil_nll_bern = monitor_metrics['nll_bern']
        self.marwil_entropy_cont = monitor_metrics['entropy_cont']
        self.marwil_entropy_cat = monitor_metrics['entropy_cat']
        self.marwil_entropy_bern = monitor_metrics['entropy_bern']
        self.marwil_accuracy_cont = monitor_metrics['accuracy_cont']
        self.marwil_accuracy_cat = monitor_metrics['accuracy_cat']
        self.marwil_accuracy_bern = monitor_metrics['accuracy_bern']
        self.marwil_weight_mean = avg_weight_mean
        self.marwil_weight_max = avg_weight_max
        self.marwil_weight_min = avg_weight_min
        self.marwil_weight_clip_frac = avg_clip_frac
        self.marwil_adv_std = avg_adv_std
        self.marwil_adv_p95 = avg_adv_p95
        self.marwil_adv_max = avg_adv_max
        self.marwil_adv_mean = avg_adv_mean
        self.marwil_adv_positive_frac = monitor_metrics['adv_positive_frac']

        return avg_actor_loss, avg_critic_loss, avg_c
    