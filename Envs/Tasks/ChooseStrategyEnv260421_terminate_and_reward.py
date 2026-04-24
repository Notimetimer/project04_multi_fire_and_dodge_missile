'''

'''

import numpy as np
from random import random
import random
import copy
import jsbsim
import sys
import os
import importlib
import copy
from math import *
import torch

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 获取project目录
def get_current_file_dir():
    return os.path.dirname(os.path.abspath(__file__))


current_dir = get_current_file_dir()
sys.path.append(os.path.dirname(os.path.dirname(current_dir)))

from Envs.BattleHomo260421 import *
from .ChooseStrategyEnv260421 import ChooseStrategyEnv as BaseChooseStrategyEnv

# 通过继承构建观测空间、奖励函数和终止条件
# 通过类的组合获取各子策略的观测量裁剪

class ChooseStrategyEnv(BaseChooseStrategyEnv):
    def combat_terminate_and_reward(self, side, action_shoots, action_cycle_multiplier=10):
        
        ego_win=0
        ego_lose=0
        ego_draw=0

        if self.t > self.game_time_limit:
            terminate = 1
        else:
            terminate = 0
        
        return terminate, ego_win, ego_win, ego_win