import numpy as np
import torch

class HybridReplayBuffer:
    def __init__(self, n_envs, buffer_size, obs_dim, state_dim, action_dims_dict, 
                 actor_hidden_dim=None, critic_hidden_dim=None, use_truncs=False, use_active_masks=False, device='cpu'):
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

        # 1. 预分配内存 (Time, Envs, Dim)
        self.obs = np.zeros((buffer_size, n_envs, obs_dim), dtype=np.float32)
        self.states = np.zeros((buffer_size, n_envs, state_dim), dtype=np.float32)
        self.next_states = np.zeros((buffer_size, n_envs, state_dim), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.dones = np.zeros((buffer_size, n_envs), dtype=np.float32)
        
        if self.use_truncs:
            self.truncs = np.zeros((buffer_size, n_envs), dtype=np.float32)
            
        if self.use_active_masks:
            self.active_masks = np.ones((buffer_size, n_envs), dtype=np.float32)

        # 动作处理 (支持混合动作)
        self.actions = {}
        if 'cont' in action_dims_dict and action_dims_dict['cont'] > 0:
            self.actions['cont'] = np.zeros((buffer_size, n_envs, action_dims_dict['cont']), dtype=np.float32)
        if 'cat' in action_dims_dict and len(action_dims_dict['cat']) > 0:
            # cat 动作通常存储为索引，假设是多头离散
            n_cat_heads = len(action_dims_dict['cat'])
            self.actions['cat'] = np.zeros((buffer_size, n_envs, n_cat_heads), dtype=np.int64)
        if 'bern' in action_dims_dict and action_dims_dict['bern'] > 0:
            self.actions['bern'] = np.zeros((buffer_size, n_envs, action_dims_dict['bern']), dtype=np.float32)

        # RNN 隐藏状态 (可选)
        # 假设形状为 (Layers, N_Envs, Hidden_Dim)，存储时我们增加 Time 维度 -> (Time, Layers, N_Envs, Hidden_Dim)
        self.actor_hidden = None
        self.critic_hidden = None
        
        if actor_hidden_dim is not None:
            # actor_hidden_dim 应该是一个元组 (num_layers, hidden_size)
            self.actor_hidden = np.zeros((buffer_size, *actor_hidden_dim[0:1], n_envs, actor_hidden_dim[1]), dtype=np.float32)
        
        if critic_hidden_dim is not None:
            self.critic_hidden = np.zeros((buffer_size, *critic_hidden_dim[0:1], n_envs, critic_hidden_dim[1]), dtype=np.float32)

    def clear(self):
        """重置指针，数据无需清零，下次覆盖即可"""
        self.ptr = 0
        self.full = False

    def add(self, obs, state, action_dict, reward, done, next_state, 
            trunc=None, active_mask=None, actor_h=None, critic_h=None):
        """
        添加一步数据。所有输入都应该是 (N_Envs, ...) 形状的 numpy 数组。
        """
        if self.ptr >= self.buffer_size:
            print("Buffer overflow! Clear before adding.")
            return

        idx = self.ptr
        
        self.obs[idx] = obs
        self.states[idx] = state
        self.next_states[idx] = next_state
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

        # RNN 状态填入
        if self.actor_hidden is not None and actor_h is not None:
            self.actor_hidden[idx] = actor_h
        if self.critic_hidden is not None and critic_h is not None:
            self.critic_hidden[idx] = critic_h

        self.ptr += 1
        if self.ptr >= self.buffer_size:
            self.full = True

    def compute_estimates_and_flatten(self, critic_net, gamma, lmbda):
        """
        计算 GAE 和 TD Target，然后展平所有数据返回 Dict。
        需要传入 critic 网络来计算 value。
        """
        if self.ptr < self.buffer_size:
            print(f"Warning: Buffer not full ({self.ptr}/{self.buffer_size}), calculating on partial data.")

        # 1. 准备数据转 Tensor
        # 展平前两个维度 (T*N, D) 以便 Critic 批量推理
        # 注意：这里我们只是为了 Critic 推理方便，GAE 计算时还需要 reshape 回去
        T, N, S_Dim = self.states.shape
        
        flat_states = torch.tensor(self.states.reshape(-1, S_Dim), dtype=torch.float, device=self.device)
        flat_next_states = torch.tensor(self.next_states.reshape(-1, S_Dim), dtype=torch.float, device=self.device)
        
        # 2. Critic 推理 Values
        with torch.no_grad():
            # values shape: (T*N, 1)
            values = critic_net(flat_states)
            next_values = critic_net(flat_next_states)
            
        # 3. 还原维度用于 GAE 计算 (T, N, 1)
        values = values.view(T, N, 1).cpu().numpy()
        next_values = next_values.view(T, N, 1).cpu().numpy()
        
        # 提取 numpy 数据
        rewards = self.rewards[:, :, None] # (T, N, 1)
        dones = self.dones[:, :, None]     # (T, N, 1)
        truncs = self.truncs[:, :, None] if self.use_truncs else None

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
            
        td_targets = advantages + values

        # 5. 展平并打包 (Flatten) -> (T*N, ...)
        flatten_dict = {
            'states': self.states.reshape(-1, self.states.shape[-1]),
            'obs': self.obs.reshape(-1, self.obs.shape[-1]),
            'next_states': self.next_states.reshape(-1, self.next_states.shape[-1]),
            'rewards': self.rewards.reshape(-1), # 1D
            'dones': self.dones.reshape(-1),     # 1D
            'advantages': advantages.reshape(-1), # 1D
            'td_targets': td_targets.reshape(-1)  # 1D
        }
        
        if self.use_truncs:
            flatten_dict['truncs'] = self.truncs.reshape(-1)
            
        if self.use_active_masks:
            flatten_dict['active_masks'] = self.active_masks.reshape(-1)

        # 动作字典展平
        flat_actions = {}
        for k, v in self.actions.items():
            flat_actions[k] = v.reshape(-1, v.shape[-1])
        flatten_dict['actions'] = flat_actions

        # # 隐藏状态展平 (如果存在)
        # if self.actor_hidden is not None:
        #     # (T, Layers, N, D) -> (T*N, Layers, D) ? 
        #     # 这里的展平方式取决于 RNN PPO 的实现。通常 PPO 训练时如果切片，需要保留 N 的独立性
        #     # 但既然我们是 flatten 为 batch 训练，且假设不使用 recurrent slice update (普通 MLP PPO):
        #     # 简单的 reshape 即可。如果是真正的 RNN PPO，这里的 flatten 方式需要配合 DataGenerator
        #     pass 

        return flatten_dict


# --- 5. 测试代码 ---
if __name__ == '__main__':
    print("Testing HybridReplayBuffer...")
    n_envs = 4
    bs = 10
    obs_d = 5
    state_d = 10
    act_dims = {'cont': 3, 'cat': [4], 'bern': 1}
    
    # Mock Critic Net
    class MockCritic(torch.nn.Module):
        def forward(self, x):
            return torch.sum(x, dim=1, keepdim=True) * 0.1 # dummy value

    buffer = HybridReplayBuffer(n_envs, bs, obs_d, state_d, act_dims, use_active_masks=True)
    critic = MockCritic()
    
    # Fill buffer
    for i in range(bs):
        o = np.random.randn(n_envs, obs_d)
        s = np.random.randn(n_envs, state_d)
        ns = np.random.randn(n_envs, state_d)
        a = {
            'cont': np.random.randn(n_envs, 3),
            'cat': np.random.randint(0, 4, (n_envs, 1)),
            'bern': np.random.randint(0, 2, (n_envs, 1))
        }
        r = np.random.randn(n_envs)
        d = np.zeros(n_envs)
        active = np.ones(n_envs)
        if i == bs-1: d[:] = 1 # last step done
        
        buffer.add(o, s, a, r, d, ns, active_mask=active)
        
    print(f"Buffer filled. Pointer: {buffer.ptr}")
    
    # Compute
    data = buffer.compute_estimates_and_flatten(critic, 0.99, 0.95)
    
    print("Keys in output:", data.keys())
    print("States shape (Expected 40, 10):", data['states'].shape)
    print("Actions Cont shape (Expected 40, 3):", data['actions']['cont'].shape)
    print("Advantages shape (Expected 40,):", data['advantages'].shape)
    
    if 'active_masks' in data:
        print("Active masks shape:", data['active_masks'].shape)
    
    assert data['states'].shape == (n_envs * bs, state_d)
    assert data['actions']['cont'].shape == (n_envs * bs, 3)
    
    print("Test Passed.")