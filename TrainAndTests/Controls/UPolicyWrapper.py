import numpy as np
import torch
from _context import *
# from Controller.F16PIDController2_1 import *
from Controller.F16PIDController2_1 import *

'''
暂时不处理多策略软平均的问题，随机匹配1条策略并蒸馏
'''

class UnifiedPolicyWrapper:
    """
    统一策略包装器，支持神经网络和规则策略
    输出格式统一为: {'cont': array([4维概率分布])}
    """
    
    def __init__(self, env, agent_info=None, critic_info=None, epsilon=0.3, device=None, dt_decide=0.02):
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
        self.PIDController = F16PIDController()
        self.dt = dt_decide
    
    def get_action(self, obs, weights=1, explore=None):

        check_obs = self.env.obs2obs_check(obs)
        
        return self._get_rule_action(check_obs)
        
    def _get_rule_action(self, check_obs):
        """处理规则策略"""
        # 将check_obs转换为状态
        state_check = self.env.unscale_state(check_obs)
        comand = state_check["flight_cmd"] # 必须合并到 check_obs 中
        cos_delta_psi, sin_delta_psi, delta_height_cmd, delta_speed_cmd = comand
        delta_psi_cmd = np.arctan2(sin_delta_psi, cos_delta_psi)
        ego_height = state_check["ego_main"][1]
        delta_heading = delta_psi_cmd
        theta = np.arcsin(np.clip(state_check["ego_main"][3], -0.9999, 0.9999))
        speed = state_check["ego_main"][0]
        set_speed = speed + delta_speed_cmd
        phi = np.arctan2(state_check["ego_main"][5], state_check["ego_main"][4])
        alpha_air = state_check["ego_control"][5]
        beta_air = state_check["ego_control"][6]
        p = state_check["ego_control"][0]
        q = state_check["ego_control"][1]
        r = state_check["ego_control"][2]
        theta_v = state_check["ego_control"][3]
        delta_psi_v = state_check["ego_control"][4]
        Ny = state_check["ego_control"][7]

        set_height = ego_height + delta_height_cmd

        obs_jsbsim = np.zeros(15)
        obs_jsbsim[0] = set_height / 5000  # 期望高度 # 测试飞行控制器
        obs_jsbsim[1] = delta_heading  # 期望相对航向角
        obs_jsbsim[2] = set_speed / 340  # 期望速度
        obs_jsbsim[3] = theta  # 当前俯仰角
        obs_jsbsim[4] = speed / 340  # 当前速度
        obs_jsbsim[5] = phi  # 当前滚转角
        obs_jsbsim[6] = alpha_air  # 当前迎角
        obs_jsbsim[7] = beta_air  # 当前侧滑角
        obs_jsbsim[8] = p
        obs_jsbsim[9] = q
        obs_jsbsim[10] = r
        obs_jsbsim[11] = theta_v  # 爬升角
        obs_jsbsim[12] = delta_psi_v  # 相对航迹角
        obs_jsbsim[13] = ego_height / 5000  # 高度/5000
        obs_jsbsim[14] = Ny


        norm_act = self.PIDController.flight_output(obs_jsbsim, dt=self.dt)
        # aileron, elevetor, rudder, throttle
        
        # 构造action_exec格式
        action = {
            'cont': norm_act,
        }
        
        return action
    
    def reset(self):
        """重置包装器状态"""
        self.PIDController = F16PIDController()


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