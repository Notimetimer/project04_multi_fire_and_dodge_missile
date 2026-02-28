# 代码用于gymnasium1.0.0版本
import argparse
import time
import sys
import os
import numpy as np
sys.path.append(project_root)

# 环境
from Envs.battle6dof1v1_missile0919 import *

parser = argparse.ArgumentParser("UAV swarm confrontation")
parser.add_argument("--max-episode-len", type=float, default=120, help="maximum episode time length")
parser.add_argument("--R-cage", type=float, default=70e3, help="")
args = parser.parse_args()


# 检查环境的API
from gymnasium.utils.env_checker import check_env
check_env(Battle(args, tacview_show=0))

# 使用 SyncVectorEnv 做轻量向量化兼容性测试（适用于 gymnasium >=0.26 / 1.0）
from gymnasium.vector import SyncVectorEnv

def make_env_factory():
    def _thunk():
        return Battle(args, tacview_show=0)
    return _thunk

n_envs = 4  # 可调整并行环境数量
vec = SyncVectorEnv([make_env_factory() for _ in range(n_envs)])

# 简单向量化兼容性测试，适合 gymnasium 1.0.0
from gymnasium.vector import SyncVectorEnv
import numpy as np

def make_env():
    def _thunk():
        return Battle(args, tacview_show=0)
    return _thunk

vec = SyncVectorEnv([make_env() for _ in range(4)])
obs, infos = vec.reset()
actions = np.stack([vec.single_action_space.sample() for _ in range(4)])
obs, rewards, terminations, truncations, infos = vec.step(actions)
vec.close()
print("vector test OK")
