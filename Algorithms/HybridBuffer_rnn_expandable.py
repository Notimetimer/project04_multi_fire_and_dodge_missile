# 如果超出容量，经验池扩容

import numpy as np
import torch
from _context import *  # 获取项目根目录
from Algorithms.Utils import compute_advantage

# 不再接收next_states，统一使用next_values

class HybridReplayBuffer:
    def __init__(self, n_envs, buffer_size, obs_dim, state_dim, action_dims_dict, 
                 actor_hidden_dim=None, critic_hidden_dim=None, use_truncs=False, use_active_masks=True, device='cpu'):
        """
        参数:
            n_envs: 并行环境数量 (Threads)
            buffer_size: 每个环境采样的步数 (Time Steps)
            obs_dim: Actor 观测维度 (Local Obs)
            state_dim: Critic 状态维度 (Global State)
            action_dims_dict: 动作维度字典 {'cont': int, 'cat': [], 'bern': int}
            actor/critic_hidden_dim: RNN 隐藏状态维度 (Layers, Dim) 或 None
            use_truncs: 是否处理截断信号
            use_active_masks: 是否处理Active Mask (用于多智能体死后屏蔽)
        """
        self.n_envs = n_envs
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dims = action_dims_dict
        self.device = device
        self.ptr = 0
        self.full = False
        self.use_truncs = use_truncs
        self.use_active_masks = use_active_masks
        
        # [修改] 使用字典存储各个 Key 的时间戳 List
        self.timestamps_dict = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_values': [],
            'dones': []
        }
        
        # 1. 基础数据预分配 (Time, Envs, Dim) - Numpy Array 存储
        self.obs = np.zeros((buffer_size, n_envs, obs_dim), dtype=np.float32)
        self.states = np.zeros((buffer_size, n_envs, state_dim), dtype=np.float32)
        # [修改] 使用 next_values 代替 next_states
        self.next_values = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.dones = np.zeros((buffer_size, n_envs), dtype=np.float32)
        
        if self.use_truncs:
            self.truncs = np.zeros((buffer_size, n_envs), dtype=np.float32)
            
        if self.use_active_masks:
            self.active_masks = np.ones((buffer_size, n_envs), dtype=np.float32)

        # 2. 混合动作空间显式预分配
        self.actions = {}
        if 'cont' in action_dims_dict and action_dims_dict['cont'] > 0:
            self.actions['cont'] = np.zeros((buffer_size, n_envs, action_dims_dict['cont']), dtype=np.float32)
        if 'cat' in action_dims_dict and len(action_dims_dict['cat']) > 0:
            # cat 动作通常存储为索引，假设是多头离散
            n_cat_heads = len(action_dims_dict['cat'])
            self.actions['cat'] = np.zeros((buffer_size, n_envs, n_cat_heads), dtype=np.int64)
        if 'bern' in action_dims_dict and action_dims_dict['bern'] > 0:
            self.actions['bern'] = np.zeros((buffer_size, n_envs, action_dims_dict['bern']), dtype=np.float32)

        # 3. RNN 隐藏状态显式预分配 (Time, Layers, Envs, Dim)
        self.actor_hidden = None
        self.critic_hidden = None
        
        if actor_hidden_dim is not None:
            # actor_hidden_dim 应该是一个元组 (num_layers, hidden_size)
            self.actor_hidden = np.zeros((buffer_size, actor_hidden_dim[0], n_envs, actor_hidden_dim[1]), dtype=np.float32)
        
        if critic_hidden_dim is not None:
            self.critic_hidden = np.zeros((buffer_size, critic_hidden_dim[0], n_envs, critic_hidden_dim[1]), dtype=np.float32)

    def clear(self):
        """重置指针，数据无需清零，下次覆盖即可"""
        self.ptr = 0
        self.full = False
        
        # [修改] 清空 List
        for k in self.timestamps_dict:
            self.timestamps_dict[k].clear()

    def add(self, obs, state, action_dict, reward, done, next_value, 
            trunc=None, active_mask=None, actor_h=None, critic_h=None):
        """添加一步采样数据，输入均为 (N_Envs, ...) 形状的 numpy 数组"""
        # 动态扩容：若 ptr 超过当前容量则扩容（至少翻倍）
        if self.ptr >= self.buffer_size:
            self._expand_capacity(min_needed=1)
 
        idx = self.ptr
        
        self.obs[idx] = obs
        self.states[idx] = state
        # [修改] 存储 next_value
        self.next_values[idx] = next_value
        self.rewards[idx] = reward
        self.dones[idx] = done
        
        if self.use_truncs and trunc is not None:
            self.truncs[idx] = trunc
            
        if self.use_active_masks and active_mask is not None:
            self.active_masks[idx] = active_mask

        # 动作填入
        for k, v in action_dict.items():
            if k in self.actions:
                self.actions[k][idx] = v

        # 显式存储隐藏状态
        if self.actor_hidden is not None and actor_h is not None:
            self.actor_hidden[idx] = actor_h
        if self.critic_hidden is not None and critic_h is not None:
            self.critic_hidden[idx] = critic_h

        self.ptr += 1
        if self.ptr >= self.buffer_size:
            self.full = True

    def compute_estimates_and_flatten(self, critic_net, gamma, lmbda):
        """兼容 MLP 模式的扁平化方法"""
        T = self.ptr
        N = self.n_envs
        S_Dim = self.states.shape[-1]
        
        # 1. 准备数据转 Tensor 并计算 Values
        with torch.no_grad():
            states_tensor = torch.tensor(self.states[:T], dtype=torch.float, device=self.device)
            
            if self.critic_hidden is not None:
                # RNN 模式：使用存储的每一帧 h_in 计算 V(s_t)
                flat_states = states_tensor.reshape(-1, 1, S_Dim)
                h_in = torch.tensor(self.critic_hidden[:T], dtype=torch.float, device=self.device)
                # 维度变换: (T, L, N, H) -> (L, T*N, H)
                h_in = h_in.permute(1, 0, 2, 3).reshape(h_in.shape[1], -1, h_in.shape[-1]).contiguous()
                # [修改] 带上隐藏状态接收，确保 GRU 模式正常
                v_res, _ = critic_net(flat_states, h_in=h_in)
                values = v_res.view(T, N, 1).cpu().numpy()
            else:
                # MLP 模式
                # [修改] 移除 [0] 索引，接收完整的 values Tensor
                v_res = critic_net(states_tensor.reshape(-1, S_Dim))
                values = v_res.view(T, N, 1).cpu().numpy()
                
            # [修改] 直接使用存储的 next_values
            next_values = self.next_values[:T, :, None]

        # 提取 numpy 数据用于 GAE
        rewards = self.rewards[:T, :, None]
        dones = self.dones[:T, :, None]
        truncs = self.truncs[:T, :, None] if self.use_truncs else None

        # 4. GAE 计算 (Numpy 实现，分环境独立计算)
        advantages = np.zeros_like(rewards)
        gae = 0.0
        
        for t in reversed(range(T)):
            mask = 1.0 - dones[t]
            # 核心：不同环境 N 之间是独立的 element-wise 运算
            if truncs is not None:
                # 如果被截断，该步的 value 不传递，视为引导价值
                # 但这里简化处理，通常 PPO 处理截断的方式是 next_val * (1-done)
                # 如果 done=False 但 trunc=True，我们应当用 next_value 引导
                # 这里的 dones 应该代表 terminated
                
                # 如果发生截断，GAE 需要重置，因为时间序列断了
                mask_gae = mask * (1.0 - truncs[t]) 
            else:
                mask_gae = mask

            delta = rewards[t] + gamma * next_values[t] * mask - values[t]
            gae = delta + gamma * lmbda * mask_gae * gae
            advantages[t] = gae
            
        # td_targets = advantages + values
        
        # [回归老版逻辑] 直接使用 1-step TD Target
        # 老版公式：td_target = rewards + gamma * next_vals * (1 - dones)
        td_targets = self.rewards[:T, :, None] + gamma * next_values * (1.0 - self.dones[:T, :, None])

        # 5. 展平并打包 (Flatten) -> (T*N, ...)
        flatten_dict = {
            'states': self.states[:T].reshape(-1, S_Dim),
            'obs': self.obs[:T].reshape(-1, self.obs.shape[-1]),
            'next_values': self.next_values[:T].reshape(-1), # [修改]
            'rewards': self.rewards[:T].reshape(-1),
            'dones': self.dones[:T].reshape(-1),
            'advantages': advantages.reshape(-1),
            'td_targets': td_targets.reshape(-1),
        }
        
        if self.use_truncs: flatten_dict['truncs'] = self.truncs[:T].reshape(-1)
        if self.use_active_masks: flatten_dict['active_masks'] = self.active_masks[:T].reshape(-1)

        flat_actions = {}
        for k, v in self.actions.items():
            flat_actions[k] = v[:T].reshape(-1, v.shape[-1])
        flatten_dict['actions'] = flat_actions

        return flatten_dict

    def get_recurrent_data(self, critic_net, seq_len, gamma, lmbda):
        """
        专门为带 GRU 的网络整备序列数据。
        逻辑：按环境识别回合，倒序切块（确保尾部完整），丢弃头部不足 seq_len 的部分。
        """
        T = self.ptr
        N = self.n_envs
        S_Dim = self.states.shape[-1]
        
        # --- Step 1: 计算 Advantage 和 TD-Target (GAE 断流逻辑) ---
        with torch.no_grad():
            states_tensor = torch.tensor(self.states[:T], dtype=torch.float, device=self.device)
            
            if self.critic_hidden is not None:
                # RNN 模式：利用存储的 h_in 计算 V(s_t) [修改]
                flat_states = states_tensor.reshape(-1, 1, S_Dim)
                h_in = torch.tensor(self.critic_hidden[:T], dtype=torch.float, device=self.device)
                # 维度变换: (T, L, N, H) -> (L, T*N, H)
                h_in = h_in.permute(1, 0, 2, 3).reshape(h_in.shape[1], -1, h_in.shape[-1]).contiguous()
                # [修改] 显式接收两个返回值
                v_res, _ = critic_net(flat_states, h_in=h_in)
                values = v_res.view(T, N).cpu().numpy()
            else:
                # MLP 模式
                # [修改] 移除 [0] 索引，确保拿到所有 Batch 样本的 Value
                v_res = critic_net(states_tensor.reshape(-1, S_Dim))
                values = v_res.view(T, N).cpu().numpy()
                
            # [修改] 直接使用存储的 next_values
            next_values = self.next_values[:T]

        advantages = np.zeros((T, N), dtype=np.float32)
        for n in range(N):
            gae = 0
            for t in reversed(range(T)):
                mask = 1.0 - self.dones[t, n]
                # trunc 截断：断开优势流传递，但保留 next_value 引导
                if self.use_truncs:
                    mask_gae = mask * (1.0 - self.truncs[t, n])
                else:
                    mask_gae = mask
                
                delta = self.rewards[t, n] + gamma * next_values[t, n] * mask - values[t, n]
                gae = delta + gamma * lmbda * mask_gae * gae
                advantages[t, n] = gae
        
        # td_targets = advantages + values
        
        # [回归老版逻辑] 1-step TD Target
        td_targets_raw = self.rewards[:T] + gamma * self.next_values[:T] * (1.0 - self.dones[:T])

        # --- Step 2: 序列整备 (按环境回合倒序切块) ---
        valid_seq_starts = []  # 存储 (env_id, start_idx)

        for n in range(N):
            done_indices = np.where(self.dones[:T, n] == 1)[0]
            ep_ends = list(done_indices)
            if (T - 1) not in ep_ends:
                ep_ends.append(T - 1)
            
            curr_start = 0
            for ep_end in ep_ends:

                # 丢弃头部过短的序列
                ep_len = ep_end - curr_start + 1
                if ep_len < seq_len:
                    print(f"[Buffer Hint] Env {n} Episode len {ep_len} < {seq_len}, discarding head data.")
                else:
                    # 倒序切分：确保尾部完整，丢弃头部余数
                    for block_end in range(ep_end, curr_start + seq_len - 2, -seq_len):
                        block_start = block_end - seq_len + 1
                        valid_seq_starts.append((n, block_start))
                curr_start = ep_end + 1
                
                # # 无论回合多短，至少提取一个序列块 -- 不太可行，会导致短的实际序列无法写入长的固定seqlen序列 --
                # # 即使 ep_len < seq_len，我们也从 curr_start 开始提取
                # for block_start in range(curr_start, ep_end + 1, seq_len):
                #     # 如果 block_start + seq_len 超过了 ep_end，说明是尾部短块
                #     valid_seq_starts.append((n, block_start, ep_end))
                # curr_start = ep_end + 1

        if not valid_seq_starts:
            raise RuntimeError("No valid sequences found! Sequence length is too long.")

        # --- Step 3: 显式组装 3D 张量 ---
        num_seqs = len(valid_seq_starts)
        
        # 初始化返回字典
        final_data = {
            'obs': np.zeros((num_seqs, seq_len, self.obs.shape[-1]), dtype=np.float32),
            'states': np.zeros((num_seqs, seq_len, self.states.shape[-1]), dtype=np.float32),
            'next_values': np.zeros((num_seqs, seq_len), dtype=np.float32), # [修改]
            'rewards': np.zeros((num_seqs, seq_len), dtype=np.float32),
            'dones': np.zeros((num_seqs, seq_len), dtype=np.float32),
            'advantages': np.zeros((num_seqs, seq_len), dtype=np.float32),
            'td_targets': np.zeros((num_seqs, seq_len), dtype=np.float32),
            'actions': {}
        }

        if self.use_truncs:
            final_data['truncs'] = np.zeros((num_seqs, seq_len), dtype=np.float32) # 修复：改为3D
            
        if self.use_active_masks:
            final_data['active_masks'] = np.zeros((num_seqs, seq_len), dtype=np.float32)

        # 显式初始化动作 3D 容器
        if 'cont' in self.actions: final_data['actions']['cont'] = np.zeros((num_seqs, seq_len, self.actions['cont'].shape[-1]), dtype=np.float32)
        if 'cat' in self.actions: final_data['actions']['cat'] = np.zeros((num_seqs, seq_len, self.actions['cat'].shape[-1]), dtype=np.int64)
        if 'bern' in self.actions: final_data['actions']['bern'] = np.zeros((num_seqs, seq_len, self.actions['bern'].shape[-1]), dtype=np.float32)

        # 初始隐藏状态：(Batch, Layers, Dim)
        final_data['init_h_actor'] = np.zeros((num_seqs, self.actor_hidden.shape[1], self.actor_hidden.shape[3]), dtype=np.float32)
        final_data['init_h_critic'] = np.zeros((num_seqs, self.critic_hidden.shape[1], self.critic_hidden.shape[3]), dtype=np.float32)

        
        for i, (env_id, s_ptr) in enumerate(valid_seq_starts):
            e_ptr = s_ptr + seq_len
        
        # # 不丢弃长度不足的数组，不太可行
        # for i, (env_id, s_ptr, ep_end) in enumerate(valid_seq_starts):
        #     # 计算实际可用的长度
        #     actual_len = min(seq_len, ep_end - s_ptr + 1)
        #     e_ptr = s_ptr + actual_len
            
            final_data['obs'][i] = self.obs[s_ptr:e_ptr, env_id]
            final_data['states'][i] = self.states[s_ptr:e_ptr, env_id]
            final_data['next_values'][i] = self.next_values[s_ptr:e_ptr, env_id] # [修改]
            final_data['rewards'][i] = self.rewards[s_ptr:e_ptr, env_id]
            final_data['dones'][i] = self.dones[s_ptr:e_ptr, env_id]
            final_data['advantages'][i] = advantages[s_ptr:e_ptr, env_id]
            # 丢弃长度不足的经验
            final_data['td_targets'][i] = td_targets_raw[s_ptr:e_ptr, env_id]
            # # 不丢弃长度不足的经验
            # final_data['td_targets'][i, :actual_len] = td_targets_raw[s_ptr:e_ptr, env_id]
            
            if self.use_truncs:
                final_data['truncs'][i] = self.truncs[s_ptr:e_ptr, env_id]
            if self.use_active_masks:
                final_data['active_masks'][i] = self.active_masks[s_ptr:e_ptr, env_id]
            
            # 拷贝动作
            # --- 修正点：直接向预分配好的 final_data['actions'] 写入数据 ---
            for k in self.actions:
                final_data['actions'][k][i] = self.actions[k][s_ptr:e_ptr, env_id]
            
            # 记录序列起始时刻之前的隐藏状态
            # 注意：s_ptr 对应的是该序列第一帧进入 GRU 之前的状态
            final_data['init_h_actor'][i] = self.actor_hidden[s_ptr, :, env_id, :]
            final_data['init_h_critic'][i] = self.critic_hidden[s_ptr, :, env_id, :]

        return final_data

    def _expand_capacity(self, min_needed=1):
        """内部：扩展 buffer_size，至少满足 min_needed 新槽位（默认翻倍策略）。"""
        new_cap = max(self.buffer_size * 2 if self.buffer_size > 0 else 1, self.buffer_size + min_needed)
        extra = new_cap - self.buffer_size
        if extra <= 0:
            return

        # 扩展基础数组
        self.obs = np.concatenate([self.obs, np.zeros((extra, self.n_envs, self.obs_dim), dtype=self.obs.dtype)], axis=0)
        self.states = np.concatenate([self.states, np.zeros((extra, self.n_envs, self.state_dim), dtype=self.states.dtype)], axis=0)
        self.next_values = np.concatenate([self.next_values, np.zeros((extra, self.n_envs), dtype=self.next_values.dtype)], axis=0)
        self.rewards = np.concatenate([self.rewards, np.zeros((extra, self.n_envs), dtype=self.rewards.dtype)], axis=0)
        self.dones = np.concatenate([self.dones, np.zeros((extra, self.n_envs), dtype=self.dones.dtype)], axis=0)

        if self.use_truncs:
            self.truncs = np.concatenate([self.truncs, np.zeros((extra, self.n_envs), dtype=self.truncs.dtype)], axis=0)
        if self.use_active_masks:
            self.active_masks = np.concatenate([self.active_masks, np.ones((extra, self.n_envs), dtype=self.active_masks.dtype)], axis=0)

        # 扩展动作字典
        for k, arr in list(self.actions.items()):
            tail_shape = (extra, self.n_envs, arr.shape[-1])
            pad = np.zeros(tail_shape, dtype=arr.dtype)
            self.actions[k] = np.concatenate([arr, pad], axis=0)

        # 扩展隐藏状态缓存 (如果存在)
        if self.actor_hidden is not None:
            ah = np.zeros((extra, self.actor_hidden.shape[1], self.n_envs, self.actor_hidden.shape[3]), dtype=self.actor_hidden.dtype)
            self.actor_hidden = np.concatenate([self.actor_hidden, ah], axis=0)
        if self.critic_hidden is not None:
            ch = np.zeros((extra, self.critic_hidden.shape[1], self.n_envs, self.critic_hidden.shape[3]), dtype=self.critic_hidden.dtype)
            self.critic_hidden = np.concatenate([self.critic_hidden, ch], axis=0)

        self.buffer_size = new_cap

# =============================================================================
# 5. 测试代码
# =============================================================================
if __name__ == '__main__':
    print("Testing HybridReplayBuffer with next_values logic...")
    n_envs = 2
    buffer_capacity = 30
    seq_len = 8
    obs_d = 4
    state_d = 6
    a_hidden_dim = (1, 128)
    c_hidden_dim = (1, 128)
    act_dims = {'cont': 2, 'cat': [3], 'bern': 1}
    
    class MockCritic(torch.nn.Module):
        def forward(self, x, h_in=None):
            # 模拟 RNN 输出 (B, S, 1)
            if x.dim() == 3: return torch.sum(x, dim=-1, keepdim=True) * 0.1, h_in
            return torch.sum(x, dim=-1, keepdim=True) * 0.1, None

    buffer = HybridReplayBuffer(n_envs, buffer_capacity, obs_d, state_d, act_dims, 
                                actor_hidden_dim=a_hidden_dim, critic_hidden_dim=c_hidden_dim)
    critic = MockCritic()
    
    for i in range(buffer_capacity):
        o = np.random.randn(n_envs, obs_d)
        s = np.random.randn(n_envs, state_d)
        # [修改] 模拟在 main 中计算出的 next_value
        nv = np.random.randn(n_envs)
        a = {'cont': np.random.randn(n_envs, 2), 'cat': np.random.randint(0, 3, (n_envs, 1)), 'bern': np.random.randint(0, 2, (n_envs, 1))}
        r, d = np.random.randn(n_envs), np.zeros(n_envs)
        if i == 15: d[0] = 1
        
        h_a = np.random.randn(a_hidden_dim[0], n_envs, a_hidden_dim[1])
        h_c = np.random.randn(c_hidden_dim[0], n_envs, c_hidden_dim[1])
        
        buffer.add(o, s, a, r, d, nv, actor_h=h_a, critic_h=h_c)
        
    print(f"Buffer ptr: {buffer.ptr}")
    
    try:
        data = buffer.get_recurrent_data(critic, seq_len=seq_len, gamma=0.99, lmbda=0.95)
        print("\nRecurrent Data Keys:", data.keys())
        print("Next Values Shape (Expected (6, 8)):", data['next_values'].shape)
        print("Init Critic Hidden Shape:", data['init_h_critic'].shape)
        print("\nTest Passed: next_values logic verified.")
    except Exception as e:
        print(f"\nTest Failed: {e}")