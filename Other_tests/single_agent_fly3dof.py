'''
只飞一架，测试控制稳定性
分解出env中的单个智能体测试飞行问题
'''

import argparse
import time
from Envs.battle3dof1v1_missile0812 import *  # battle3dof1v1_proportion
from math import pi
import numpy as np
import matplotlib
import socket
import threading
from send2tacview import *

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd

dt_refer = dt

tacview = Tacview()
red_pos_list = np.empty((0, 3))

red_birth_state = {'position': np.array([-38000.0, 8000.0, -1700.0]),
                   'psi': 0
                   }

UAV = UAVModel()
UAV.pos_ = red_birth_state['position']
UAV.speed = (UAV.speed_max - UAV.speed_min) / 2
speed = UAV.speed
mach, _ = calc_mach(speed, UAV.pos_[1])
UAV.mach = mach
UAV.psi = red_birth_state[
    'psi']
UAV.theta = 0 * pi / 180
UAV.gamma = 0 * pi / 180
UAV.vel_ = UAV.speed * np.array([cos(UAV.theta) * cos(UAV.psi),
                                 sin(UAV.theta),
                                 cos(UAV.theta) * sin(UAV.psi)])


def control():
    pass


if __name__ == '__main__':
    # 控制目标

    # 动力学解算

    # 运动学解算

    # 可视化

    pass
