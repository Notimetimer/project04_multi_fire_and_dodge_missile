import numpy as np
import torch
from _context import *
from TrainAndTests.Combats.BasicRules_new import basic_rules

'''
暂时不处理多策略软平均的问题，随机匹配1条策略并蒸馏
'''

class UnifiedPolicyWrapper:
    """
    统一策略包装器，支持神经网络和规则策略
    输出格式统一为: {'cat': array([14维概率分布]), 'bern': array([1维])}
    """
    
    def __init__(self, env, agent_info=None, critic_info=None, epsilon=0.3, device=None):
        """
        Args:
            env: 环境实例，用于获取状态缩放等信息
            agent_info: 元组 (agent_type, actor) 或其列表 [(agent_type, actor), ...]
            critic_info: 元组 (agent_type, critic) 或其列表 [(agent_type, critic), ...]
            epsilon: 规则策略时用于平滑动作分布的参数
            device: torch device (用于NN策略)
        """
        self.env = env
        self.agent_info = agent_info
        self.critic_info = critic_info
        self.epsilon = epsilon
        self.device = device if device is not None else torch.device("cpu")
    
    def get_action(self, obs, weights=1, explore=None):
        """
        统一接口获取动作
        
        Args:
            obs: 观测值 (用于NN)
            check_obs: 检查用观测值 (用于规则)
            weights: 权重，默认为1。如果 agent_info 是列表且 weights 形状不符，则平分权重。
            explore: 探索参数字典 (用于NN)
        
        Returns:
            action_exec: {'cat': array([动作标签]), 'bern': array([开火概率])}
            action_check: {'cat': array([14维概率分布]), 'bern': array([开火概率])}
        """
        if self.agent_info is None:
            raise ValueError("agent_info has not been set.")

        if isinstance(self.agent_info, list):
            agent_list = self.agent_info
            if not isinstance(weights, list) or len(weights) != len(agent_list):
                weight_list = [1.0 / len(agent_list)] * len(agent_list)
            else:
                weight_list = weights
        else:
            agent_list = [self.agent_info]
            weight_list = [1.0]

        # 暂时采用列表中的第一个 agent，后续实现权重融合
        agent_type, actor = agent_list[0]

        check_obs = self.env.obs2obs_check(obs)
        
        if agent_type == 'NN':
            return self._get_nn_action(obs, check_obs, actor, explore)
        elif agent_type == 'rule':
            return self._get_rule_action(check_obs, actor)
        else:
            raise ValueError(f"Unknown agent_type: {agent_type}. Expected 'NN' or 'rule'.")

    def get_value(self, obs, weights=1, explore=None):
        """
        获取价值函数值
        输入接口与 get_action 保持一致
        """
        if self.critic_info is None:
            return float('inf')

        if isinstance(self.critic_info, list):
            critic_list = self.critic_info
        else:
            critic_list = [self.critic_info]

        agent_type, critic = critic_list[0]

        if agent_type == 'NN':
            return float('inf')  # 不可靠的旧critic, 使用门控不如全通过
            # with torch.no_grad():
            #     if not isinstance(obs, torch.Tensor):
            #         obs_tensor = torch.tensor(obs, dtype=torch.float, device=self.device).unsqueeze(0)
            #     else:
            #         obs_tensor = obs
            #     val = critic(obs_tensor)
            #     return val.item() if val.numel() == 1 else val
        else:
            # # 纯策略蒸馏临时全通过
            # return float('inf')

            # 正常模式
            fake_value = float('-inf')  # 这里需要负无穷
            check_obs = self.env.obs2obs_check(obs)
            state_check = self.env.unscale_state(check_obs)
            # 规则
            d_hor, leftright = state_check["border"]
            if d_hor < 8e3:
                fake_value = float('inf')

            return fake_value
    
    def _get_nn_action(self, obs, check_obs, actor, explore=None):
        """处理神经网络策略"""
        if explore is None:
            explore = {'cont': 1, 'cat': 1, 'bern': 1}
        
        with torch.no_grad():
            action_exec, _, _, action_check = actor.get_action(
                obs, 
                explore=explore
            )
        
        return action_exec, action_check
    
    def _get_rule_action(self, check_obs, rule_num):
        """处理规则策略"""
        # 将check_obs转换为状态
        state_check = self.env.unscale_state(check_obs)
        
        # 调用规则获取动作标签和开火决策
        action_label, fire = basic_rules(
            state_check, 
            rule_num, 
        )
        
        
        # 构造action_exec格式
        action_exec = {
            'cat': np.array([action_label]),
            'bern': np.array([fire], dtype=np.float32)
        }
        
        # 构造action_check格式 (平滑的概率分布)
        num_actions = 14  # 假设有14个机动动作
        cat_probs = np.ones(num_actions, dtype=np.float32) * (self.epsilon / (num_actions - 1))
        cat_probs[action_label] = 1.0 - self.epsilon
        
        action_check = {
            'cat': cat_probs,
            'bern': np.array([fire], dtype=np.float32)
        }
        
        return action_exec, action_check
    
    def reset(self):
        """重置包装器状态"""
        pass


def create_policy_wrapper(env, agent_type, actor, epsilon=0.3, device=None):
    """
    工厂函数：创建策略包装器
    
    Args:
        env: 环境实例
        agent_type: 'NN' 或 'rule'
        actor: 网络实例或rule_num
        epsilon: 规则策略的平滑参数
        device: torch device
    
    Returns:
        wrapper: UnifiedPolicyWrapper实例
        agent_info: (agent_type, actor) 元组
    """
    agent_info = (agent_type, actor)
    wrapper = UnifiedPolicyWrapper(env, agent_info=agent_info, epsilon=epsilon, device=device)
    return wrapper, agent_info