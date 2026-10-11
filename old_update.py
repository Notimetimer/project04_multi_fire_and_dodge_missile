    def update(self, transition_dict, adv_normed=False, 
                clip_vf=False, clip_range=0.2, shuffled=1, 
                mini_batch_size=None, alpha_logit_reg=0.05,
                v_trace=None, target_p1=0.65, target_p1_b=0.8, 
                k_nonlinear=0.89, mask_on=0, actor_frozen=0, bern_max_logits=4.0, alpha_distill=0, teacher_actor=None,
                AFiltered=0, conf_thres=0.7, bern_included=0): 
                # [新增] target_p1 默认“一超”概率，剩下来的留给“多强”)
                # [修改] 增加 target_p1_b 参数，对应开火控制的“笃定程度”

        # RL 更新阶段：确保所有分布参数都参与梯度更新
        if hasattr(self.actor.net, 'log_std_cont'):
            self.actor.net.log_std_cont.requires_grad = True
        # if hasattr(self.actor.net, 'log_temp_cat'):
        #     self.actor.net.log_temp_cat.requires_grad = False
        # if hasattr(self.actor.net, 'log_temp_bern'):
        #     self.actor.net.log_temp_bern.requires_grad = False

        #  6. 智能数据转换：如果已经是 np.ndarray (来自 HybridReplayBuffer)，直接转 Tensor
        # 否则 (来自 list append)，先转 np 再转 Tensor
        def to_tensor(x, dtype):
            if isinstance(x, np.ndarray):
                return torch.tensor(x, dtype=dtype).to(self.device)
            else:
                return torch.tensor(np.array(x), dtype=dtype).to(self.device)

        states = to_tensor(transition_dict['states'], torch.float)
        
        dones = to_tensor(transition_dict['dones'], torch.float).view(-1, 1)
        rewards = to_tensor(transition_dict['rewards'], torch.float).view(-1, 1)

        #  处理 active_masks (可选输入)
        # 如果 transition_dict 中没有 active_masks，则默认所有样本均有效
        if 'active_masks' in transition_dict:
            active_masks = to_tensor(transition_dict['active_masks'], torch.float).view(-1, 1)
        else:
            # 创建与 dones 形状一致的全 1 张量
            active_masks = torch.ones_like(dones)

        # 处理 obs (如果存在)
        if 'obs' in transition_dict:
            actor_inputs = to_tensor(transition_dict['obs'], torch.float)
            critic_inputs = states
        else:
            actor_inputs = states
            critic_inputs = states
        
        # 1. 准备动作数据
        actions_from_buffer = transition_dict['actions']
        
        # todo action_mask 防止“死后动作”干扰决策
        # todo truncs
        # todo global states，适配集中式Critic

        # 1. 准备动作数据 (转 Tensor)
        actions_on_device = {}
        
        # Buffer 传来的 actions 已经是 dict of arrays
        if isinstance(actions_from_buffer, dict):
            for key, val in actions_from_buffer.items():
                if val is None or (isinstance(val, (np.ndarray, torch.Tensor, list)) and len(val) == 0):
                    continue
                if key == 'cat':
                    actions_on_device[key] = to_tensor(val, torch.long)
                else:
                    actions_on_device[key] = to_tensor(val, torch.float)
        else:
            # 兼容旧代码 (list of dicts)
            # 旧逻辑：List of Dicts (较慢)
            all_keys = actions_from_buffer[0].keys()
            for key in all_keys:
                vals = [d[key] for d in actions_from_buffer if key in d]
                if len(vals) == 0:
                    continue
                if key == 'cat':
                    actions_on_device[key] = torch.tensor(np.array(vals), dtype=torch.long).to(self.device)
                else:
                    actions_on_device[key] = torch.tensor(np.array(vals), dtype=torch.float).to(self.device)


        # 2. 获取 Advantage (优先使用 Buffer 算好的)
        if 'advantages' in transition_dict and 'td_targets' in transition_dict:
            advantage = to_tensor(transition_dict['advantages'], torch.float).view(-1, 1)
            td_target = to_tensor(transition_dict['td_targets'], torch.float).view(-1, 1)
        else:
            # 现场计算 GAE (不推荐用于并行展平后的数据)
            
            #  处理 truncs
            if 'truncs' in transition_dict:
                truncs = to_tensor(transition_dict['truncs'], torch.float).view(-1, 1)
            else:
                truncs = None # 或者 torch.zeros_like(dones)

            # 以下为公共部分
            # 如果没有预计算，则现场计算 (注意：如果是并行数据直接展平进来的，这里计算会有偏差)
            with torch.no_grad():
                # [修改] 优先使用预计算的 next_values
                if 'next_values' in transition_dict:
                    next_vals = to_tensor(transition_dict['next_values'], torch.float).view(-1, 1)
                else:
                    # Critic 使用全局 next_states 计算 Target
                    # 注意：对于截断的步，next_value 应该是 V(s_t+1) 而不是 0
                    next_states = to_tensor(transition_dict['next_states'], torch.float)
                    # Critic 使用全局 next_states 计算 Target
                    # 注意：对于截断的步，next_value 应该是 V(s_t+1) 而不是 0
                    next_vals = self.critic(next_states)
                
                # td_target的计算不应考虑truncs。仅当dones=1时，next_value才为0。
                # truncs的影响由compute_advantage函数内部处理。
                td_target = rewards + self.gamma * next_vals * (1 - dones)

                # Critic 使用全局 states 计算当前 Value
                td_delta = td_target - self.critic(critic_inputs)
                advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu(), dones.cpu(), truncs.cpu() if truncs is not None else None).to(self.device)

                
        # # 策略蒸馏 (Advantage-based): 提升与 teacher 一致的 cat 动作的优势度
        # if alpha_distill > 0 and teacher_actor is not None:
        #     advantage = self.ADistill(transition_dict, advantage, alpha_distill, teacher_actor, AFiltered=AFiltered, conf_thres=conf_thres, bern_included=bern_included)
        
        # 3. 计算旧策略的 log_probs (使用 Wrapper)
        with torch.no_grad():
            # Actor 使用 actor_inputs (可能是 obs)
            # [修改] 接收 5 个返回值
            old_log_probs, _, _, _ ,_ = self.actor.evaluate_actions(actor_inputs, actions_on_device, h=None, max_std=self.max_std, mask_on=mask_on)
            # Critic 使用 critic_inputs (全局 states)
            v_pred_old = self.critic(critic_inputs)
            
        # --- [2. 优势归一化 - 适配 active_masks] ---
        if adv_normed:
            #  仅使用 active 的数据计算统计量
            active_adv = advantage[active_masks.squeeze(-1).bool()]
            
            if active_adv.numel() > 1: # 防止 active 数据过少导致 NaN
                adv_mean = active_adv.mean()
                adv_std = active_adv.std(unbiased=False)
                advantage = (advantage - adv_mean) / (adv_std + 1e-8)
            else:
                # 降级策略：如果有效数据太少，不进行归一化或者只做去中心化
                pass
        else:
            # 推荐: 即使不归一化，也建议减去均值 (Centering)
            # 这有助于降低方差，且不改变梯度的方向
            #  使用 mask 计算均值
            active_adv = advantage[active_masks.squeeze(-1).bool()]
            if active_adv.numel() > 0:
                adv_mean = active_adv.mean()
                advantage = advantage - adv_mean

        # =====================================================================
        # [新增] 计算 Cat 分布的动态目标熵 (Target Entropy)
        # =====================================================================
        target_entropy_cat_total = 0.0
        if 'cat' in self.actor.action_dims and sum(self.actor.action_dims['cat']) > 0:
            for n_actions in self.actor.action_dims['cat']:
                if n_actions > 1:
                    # “一超”概率为 target_p1，其余 (n_actions - 1) 个动作平分剩余概率
                    p_rest = (1.0 - target_p1) / (n_actions - 1)
                    # 香农熵: -p1*ln(p1) - (N-1) * p_rest*ln(p_rest)
                    h_dim = - target_p1 * np.log(target_p1 + 1e-8) - (n_actions - 1) * p_rest * np.log(p_rest + 1e-8)
                    target_entropy_cat_total += h_dim
        target_entropy_cat_tensor = torch.tensor(target_entropy_cat_total, device=self.device)
        
        # =====================================================================
        # [新增] 计算 Bernoulli 分布的动态目标熵 (Target Entropy)
        # =====================================================================
        target_entropy_bern_total = 0.0
        if 'bern' in self.actor.action_dims and self.actor.action_dims['bern'] > 0:
            # 伯努利熵公式: -p*ln(p) - (1-p)*ln(1-p)
            p = target_p1_b
            h_bern_dim = - p * np.log(p + 1e-8) - (1 - p) * np.log(1 - p + 1e-8)
            # 伯努利熵按动作维度累加
            target_entropy_bern_total = h_bern_dim * self.actor.action_dims['bern']
        target_entropy_bern_tensor = torch.tensor(target_entropy_bern_total, device=self.device)

        # 4. PPO Update Loop
        actor_loss_list, critic_loss_list, entropy_list, ratio_list = [], [], [], []
        actor_grad_list, critic_grad_list = [], []
        pre_clip_actor_grad, pre_clip_critic_grad = [], []

        #  监控列表
        kl_list = []
        clip_frac_list = []
        #  分项 Entropy 列表
        entropy_cat_list = []
        entropy_bern_list = []
        entropy_cont_list = []
        grad_norm_ratio_list = [] # [新增] 范数比列表
        max_fire_prob_list = [] # [新增] 记录未被mask的bern最大触发概率
        min_fire_prob_list = [] # [新增] 记录未被mask的bern最小触发概率
        
        # [新增] 初始化样本统计计数器 (包含重复更新累加)
        ppo_samples_total = 0
        ppo_valid_samples_total = 0

        mask_eps = 1e-5
        
        num_samples = actor_inputs.size(0)
        if mini_batch_size is None:
            mini_batch_size = num_samples

        for _ in range(self.epochs):
            
            # --- [Shuffle 逻辑移入 Epoch 循环] ---
            # 每个 epoch 生成新的随机索引，确保数据打乱
            if shuffled:
                idx = torch.randperm(num_samples, device=self.device)
            else:
                idx = torch.arange(num_samples, device=self.device)
            
            # --- [引入 Mini-Batch 循环] ---
            for start in range(0, num_samples, mini_batch_size):
                end = min(start + mini_batch_size, num_samples)
                batch_idx = idx[start:end]
                
                # 切片 mini-batch 数据
                mb_actor_inputs = actor_inputs[batch_idx]
                mb_critic_inputs = critic_inputs[batch_idx]
                mb_advantage = advantage[batch_idx]
                mb_td_target = td_target[batch_idx]
                mb_old_log_probs = old_log_probs[batch_idx]
                mb_active_masks = active_masks[batch_idx]
                # v_pred_old_batch = v_pred_old[batch_idx] # 如果需要用到旧 Value
                
                # 切片 Actions (Dict 结构)
                mb_actions = {}
                for k, v in actions_on_device.items():
                    mb_actions[k] = v[batch_idx]

                # 计算当前策略的 log_probs 和 entropy (使用 Wrapper)
                #  接收 entropy_details 和 actor_outputs
                log_probs, entropy, entropy_details, actor_outputs, _ = self.actor.evaluate_actions(mb_actor_inputs, mb_actions, h=None, max_std=self.max_std, mask_on=mask_on)
                
                #  计算 log_ratio 用于更精准的 KL 计算
                log_ratio = log_probs - mb_old_log_probs
                ratio = torch.exp(log_ratio)
                
                #  计算 Approximate KL Divergence
                with torch.no_grad():
                    #  KL 计算也最好应用 mask，但为了监控方便，这里先保持全局均值或应用mask均值
                    active_sum = mb_active_masks.sum()
                    approx_kl = (((ratio - 1) - log_ratio) * mb_active_masks).sum() / (active_sum + mask_eps)
                    kl_list.append(approx_kl.item())
                    
                    #  计算 Clip Fraction (有多少样本触发了裁剪)
                    clip_fracs = (((ratio - 1.0).abs() > self.eps).float() * mb_active_masks).sum() / (active_sum + mask_eps)
                    clip_frac_list.append(clip_fracs.item())
                
                # 借用一点IMPALA的经验，防止ratio过大导致梯度被炸飞
                # 建议设置数值：2.0~5.0
                if v_trace is not None:
                    ratio = torch.clamp(ratio, max=v_trace)

                surr1 = ratio * mb_advantage
                surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * mb_advantage
                
                #  Actor Loss 使用 mask 加权
                surrogate_loss = -torch.min(surr1, surr2)
                # active_sum 已经在上面计算过
                
                # [新增] 统计 PPO 样本
                ppo_samples_total += mb_active_masks.numel()
                ppo_valid_samples_total += mb_active_masks.sum().item()

                actor_loss = (surrogate_loss * mb_active_masks).sum() / (active_sum + mask_eps)
                
                # [修改] 分项 Entropy Loss 计算
                e_cont = entropy_details['cont'] if entropy_details['cont'] is not None else torch.tensor(0., device=self.device)
                e_cat = entropy_details['cat'] if entropy_details['cat'] is not None else torch.tensor(0., device=self.device)
                e_bern = entropy_details['bern'] if entropy_details['bern'] is not None else torch.tensor(0., device=self.device)

                loss_ent_cont = (e_cont * mb_active_masks).sum() / (active_sum + mask_eps)
                loss_ent_cat = (e_cat * mb_active_masks).sum() / (active_sum + mask_eps)
                
                # [新增] 使用 fire_mask 过滤 Bernoulli 熵，只统计允许开火时段的熵
                if 'fire_mask' in actor_outputs and actor_outputs['fire_mask'] is not None:
                    bern_fire_mask = actor_outputs['fire_mask']  # shape: (batch, bern_dim)
                    # 对齐到 (batch, 1)
                    if bern_fire_mask.dim() > 1:
                        bern_fire_mask = bern_fire_mask[:, 0:1]  # 取第一个 Bernoulli 维度
                    fireable_sum = (mb_active_masks * bern_fire_mask).sum()
                    loss_ent_bern = (e_bern * mb_active_masks * bern_fire_mask).sum() / (fireable_sum + mask_eps)
                else:
                    # 回退到原有逻辑（兼容旧版本）
                    loss_ent_bern = (e_bern * mb_active_masks).sum() / (active_sum + mask_eps)

                # =====================================================================
                # [修改] 使用动态 k_cat 计算 Actor Loss
                # =====================================================================
                k_cont = self.k_entropy.get('cont', 0.0)
                k_cat = self.k_entropy.get('cat', 0.0)
                k_bern = self.k_entropy.get('bern', 0.0)
                
                # # 当前的 k_cat 是 e^(log_k_cat)
                # curr_k_cat = torch.exp(self.log_k_cat)
                # --- 关键修改：跟踪目标熵, 熵过小，加熵。熵过大，减熵 ---
                # 比平方误差更加平缓的损失函数候选：
                # 1. pseudo-huber loss:     torch.sqrt(1 + torch.square(target - current)) - 1
                # 2. log-cosh loss:         torch.log(torch.cosh(target_entropy_cat_tensor - loss_ent_cat))

                # 裁剪的范围，绝不能太小
                theoretical_ent_cat_max = 0.0
                for dim in self.actor.action_dims['cat']:
                    theoretical_ent_cat_max += np.log(float(dim))
                diff_cat0 = theoretical_ent_cat_max - target_entropy_cat_tensor
                k_nonlinear_max_cat = torch.sqrt(1+diff_cat0**2)/(diff_cat0+1e-8)

                theoretical_ent_bern_max = -np.log(0.5)*self.actor.action_dims['bern']
                diff_bern0 = theoretical_ent_bern_max - target_entropy_bern_tensor
                k_nonlinear_max_bern = torch.sqrt(1+diff_bern0**2)/(diff_bern0+1e-8)

                k_nonlinear_cat = min(max(k_nonlinear, 0.0), k_nonlinear_max_cat)
                k_nonlinear_bern = min(max(k_nonlinear, 0.0), k_nonlinear_max_bern)

                # 1. Categorical 约束项
                cat_constraint_term = k_cat * (
                    - loss_ent_cat # + 
                    # torch.where(loss_ent_cat > target_entropy_cat_tensor, 
                    # min(max(k_nonlinear_cat, 0.0), 1.0)/(2 * diff_cat0) * \
                    #     torch.square(loss_ent_cat - target_entropy_cat_tensor)
                    # , 0.0)
                )
                # # 2. Bernoulli 约束项 (动态系数：熵越高推力越弱，理论最大时归零)
                # # f(H) = -H + H^2/(2*H_max), 导数 f'(H) = -1 + H/H_max
                # # H=0 时全力推，H=H_max 时导数为0
                # bern_constraint_term = k_bern * (
                #     - loss_ent_bern
                #     + torch.where(loss_ent_bern > target_entropy_bern_tensor,
                #     min(max(k_nonlinear_bern, 0.0), 1.0)/(2 * diff_bern0) * \
                #         torch.square(loss_ent_bern - target_entropy_bern_tensor)
                #     , 0.0)
                # )
                # 3. 组合最终 Actor Loss
                # [修改] 移除 bern_constraint_term（熵约束），由 mini-batch 内的 FRR 替代
                actor_loss = actor_loss + cat_constraint_term - (k_cont * loss_ent_cont)
                
                # 原有非目标熵正则项
                # actor_loss = actor_loss - (k_cont * loss_ent_cont + k_cat * loss_ent_cat + k_bern * loss_ent_bern)

                # # =====================================================================
                # # [修改/重构] Bernoulli 开火头的专属正则化、稀疏惩罚与冷却约束
                # # =====================================================================
                if actor_outputs['bern'] is not None:
                    bern_logits = actor_outputs['bern']
                    bern_probs = torch.sigmoid(bern_logits)
                    
                    # 1. 基础 Logit 越界惩罚 (保持 Logit 在可激活区间)
                    over = F.relu(torch.abs(bern_logits) - bern_max_logits) # 4.0  20步至少打一发，那么开火概率至少也是0.05，对应-3的logits
                    # 只惩罚允许开火的位置
                    if 'fire_mask' in actor_outputs and actor_outputs['fire_mask'] is not None:
                        fire_mask = actor_outputs['fire_mask']  # shape: (batch, bern_dim)
                    else:
                        fire_mask = (bern_logits > -1e6).float()
                    # 只对允许开火且存活的样本求平均，避免被 non-fireable 样本稀释
                    fireable_sum = (mb_active_masks * bern_fire_mask).sum()
                    logit_loss = ((over ** 2) * fire_mask * mb_active_masks).sum() / (fireable_sum + mask_eps)
                    actor_loss = actor_loss + alpha_logit_reg * logit_loss
                    
                    # [新增] 记录未被mask的bern触发概率极值
                    valid_fire_mask = (fire_mask * mb_active_masks).bool()
                    if valid_fire_mask.any():
                        valid_probs = bern_probs[valid_fire_mask]
                        max_fire_prob_list.append(valid_probs.max().item())
                        min_fire_prob_list.append(valid_probs.min().item())


                    # [新增] 3. 单阈值均值约束 (FRR) —— 替代 bern 熵正则项
                    # 约束允许开火样本的平均概率不超过 p_target，用 pseudo-Huber loss
                    # 保护弹药节奏：防止中远距离手痒乱射导致后期无弹可用
                    fire_mask_1d = fire_mask[:, 0] if fire_mask.dim() > 1 else fire_mask.squeeze(-1)
                    active_1d = mb_active_masks.squeeze(-1)
                    fireable_frr = (fire_mask_1d > 0.5) & (active_1d > 0.5)
                    if fireable_frr.sum() > 0:
                        p_mean_fireable = bern_probs[:, 0][fireable_frr].mean()
                        p_target = 0.1  # 目标平均开火概率 20%，对应 6 枚 / 30 个有效开火步
                        delta = F.relu(p_mean_fireable - p_target)
                        # pseudo-Huber loss: sqrt(1 + delta^2) - 1，平缓惩罚，对异常值不爆炸
                        frr_loss = torch.sqrt(1.0 + delta * delta) - 1.0
                        # 复用原熵系数 k_bern 作为强度（建议 0.001~0.005）
                        actor_loss = actor_loss + k_bern * frr_loss

                # Critic Loss
                # Critic 使用 critic_inputs
                v_pred = self.critic(mb_critic_inputs)
                # if clip_vf:
                #     v_pred_old_batch = v_pred_old[batch_idx]

                #     # 新的critic clip方法，按标准差倍数缩放限幅
                #     # 1. 动态计算当前批次 TD Target 的标准差，作为价值尺度的基准
                #     with torch.no_grad():
                #         td_target_std = torch.std(mb_td_target).item()
                #     # 2. 自适应计算 clip_range。设定 0.5 倍标准差为窗口，并用 10.0 进行保底
                #     # 防止训练初期或特定批次由于 Target 过于单一导致 std 接近 0 从而锁死更新
                #     adaptive_clip_range = max(td_target_std * 0.5, 10.0)
                #     # 3. 使用动态计算的范围进行截断
                #     v_pred_clipped = torch.clamp(
                #         v_pred, 
                #         v_pred_old_batch - adaptive_clip_range, 
                #         v_pred_old_batch + adaptive_clip_range
                #     )
                #     vf_loss1 = (v_pred - mb_td_target).pow(2)
                #     vf_loss2 = (v_pred_clipped - mb_td_target).pow(2)
                #     critic_loss_per_sample = torch.max(vf_loss1, vf_loss2)

                # else:
                #  reduction='none' 使得我们可以应用 mask
                critic_loss_per_sample = F.mse_loss(v_pred, mb_td_target, reduction='none')
               
                # 1、有序列时间修正的时候， Critic Loss 使用 mask 加权
                # critic_loss = (critic_loss_per_sample * mb_active_masks).sum() / (active_sum + mask_eps)
                # 2、多智能体情况下，critic不传入active_mask
                critic_loss = critic_loss_per_sample.mean()
                
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                # if alpha_loss.requires_grad:    # [新增]
                #     self.k_cat_optim.zero_grad()
                if actor_frozen:
                    self.reset_optimizer(reset_type='actor')

                if not actor_frozen:
                    actor_loss.backward()
                critic_loss.backward()
                # if alpha_loss.requires_grad:    # [新增] 反向传播温度损失
                #     alpha_loss.backward()

                pre_clip_actor_grad.append(model_grad_norm(self.actor))
                pre_clip_critic_grad.append(model_grad_norm(self.critic)) 
                
                # [新增] 计算范数比 (Pre-clip)
                grad_norm_ratio_list.append(pre_clip_actor_grad[-1] / (pre_clip_critic_grad[-1] + 1e-8))

                nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.actor_max_grad)
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.critic_max_grad)
                if not actor_frozen:
                    self.actor_optimizer.step()
                self.critic_optimizer.step()
                # if alpha_loss.requires_grad:    # [新增] 步进自适应熵系数
                #     self.k_cat_optim.step()

                # Logging
                actor_grad_list.append(model_grad_norm(self.actor))
                critic_grad_list.append(model_grad_norm(self.critic))            
                actor_loss_list.append(actor_loss.item())
                critic_loss_list.append(critic_loss.item())
                
                # 记录总 Entropy (未加权)
                entropy_total = (entropy * mb_active_masks).sum() / (active_sum + mask_eps)
                entropy_list.append(entropy_total.item()) 
                ratio_list.append(ratio.mean().item()) 
                
                #  [修改] 记录分项 Entropy (现在是 Tensor，需要 .mean().item())
                if entropy_details['cont'] is not None:
                    entropy_cont_list.append(entropy_details['cont'].mean().item())
                if entropy_details['cat'] is not None:
                    entropy_cat_list.append(entropy_details['cat'].mean().item())
                if entropy_details['bern'] is not None:
                    # [修改] 使用 fire_mask 过滤，只记录 fireable 样本的平均熵
                    if 'fire_mask' in actor_outputs and actor_outputs['fire_mask'] is not None:
                        bern_fire_mask = actor_outputs['fire_mask']  # shape: (batch, bern_dim)
                        if bern_fire_mask.dim() > 1:
                            bern_fire_mask = bern_fire_mask[:, 0:1]  # 对齐到 (batch, 1)
                        fireable_sum = (mb_active_masks * bern_fire_mask).sum()
                        bern_entropy_filtered = (entropy_details['bern'] * mb_active_masks * bern_fire_mask).sum() / (fireable_sum + mask_eps)
                        entropy_bern_list.append(bern_entropy_filtered.item())
                    else:
                        entropy_bern_list.append(entropy_details['bern'].mean().item())

        self.actor_loss = np.mean(actor_loss_list)
        self.actor_grad = np.mean(actor_grad_list)
        self.critic_loss = np.mean(critic_loss_list)
        self.critic_grad = np.mean(critic_grad_list)
        self.entropy_mean = np.mean(entropy_list)
        self.ratio_mean = np.mean(ratio_list)
        self.pre_clip_critic_grad = np.mean(pre_clip_critic_grad)
        self.pre_clip_actor_grad = np.mean(pre_clip_actor_grad)
        
        #  记录 active 的 advantage 均值
        active_sum_total = active_masks.sum().item()
        if active_sum_total > 0:
            self.advantage = (advantage.abs() * active_masks).sum().item() / active_sum_total
        else:
            self.advantage = 0
        
        #  汇总新指标
        self.approx_kl = np.mean(kl_list)
        self.clip_frac = np.mean(clip_frac_list)
        #  计算分项 Entropy 均值
        self.entropy_cont = np.mean(entropy_cont_list) if len(entropy_cont_list) > 0 else 0
        self.entropy_cat = np.mean(entropy_cat_list) if len(entropy_cat_list) > 0 else 0
        self.entropy_bern = np.mean(entropy_bern_list) if len(entropy_bern_list) > 0 else 0
        
        # [新增] 汇总监控项
        self.grad_norm_ratio = np.mean(grad_norm_ratio_list) if len(grad_norm_ratio_list) > 0 else 0
        
        # [新增] 赋值有效样本监控项
        self.PPO_samples = ppo_samples_total
        self.PPO_valid_samples = ppo_valid_samples_total
        
        # [新增] 汇总未被mask的bern触发概率极值
        self.max_fire_prob = np.mean(max_fire_prob_list) if len(max_fire_prob_list) > 0 else 0
        self.min_fire_prob = np.mean(min_fire_prob_list) if len(min_fire_prob_list) > 0 else 0


        #  计算 Explained Variance
        # y_true: td_target, y_pred: v_pred_old (更新前的值) 或 v_pred (更新后的值，通常用更新前比较多，或者直接对比)
        # 这里使用 numpy 计算以防 tensor 维度广播问题
        #  explained_var 最好也只看 active 的，但为了简单起见，这里先保持原样或简单过滤
        mask_bool = active_masks.squeeze(-1).bool().cpu().numpy()
        y_true = td_target.flatten().cpu().numpy()[mask_bool]
        y_pred = v_pred_old.flatten().cpu().numpy()[mask_bool] # 比较更新前的 Value 网络预测能力
        
        # if len(y_true) > 1:
        #     var_y = np.var(y_true)
        #     if var_y == 0:
        #         self.explained_var = np.nan
        #     else:
        #         self.explained_var = 1 - np.var(y_true - y_pred) / var_y
        # else:
        #     self.explained_var = 0

        if len(y_true) > 1:
            var_y = np.var(y_true)
            self.td_error_var = np.var(y_true - y_pred) # [新增] TD error 方差
            if var_y < 1e-8:
                self.explained_var = 0.0
            else:
                self.explained_var = 1 - self.td_error_var / var_y
        else:
            self.td_error_var = 0.0
            self.explained_var = 0.0

        check_weights_bias_nan(self.actor, "actor", "update后")
        check_weights_bias_nan(self.critic, "critic", "update后")

    # =========================================================================
    #  [New Method] Bernoulli 开火头保护性有监督训练 (防止机动策略被bern崩溃拖累)
    # =========================================================================
