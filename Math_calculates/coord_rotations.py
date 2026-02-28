from math import *
import numpy as np


def active_rotation(vector,heading,theta,gamma):
    # 飞机的轴在外界看来是怎样的， 前上右在北天东看来朝向哪个方向
    # vector是行向量，旋转顺序仍然是 psi, theta, gamma，但是计算顺序相反
    # 主动旋转，-在1下
    # 注意：北天东坐标
    psi = - heading
    Rpsi=np.array([
        [cos(psi), 0, sin(psi)],
        [0, 1, 0],
        [-sin(psi), 0, cos(psi)]
        ])
    Rtheta=np.array([
        [cos(theta), -sin(theta), 0],
        [sin(theta), cos(theta), 0],
        [0, 0, 1]
        ])
    Rgamma=np.array([
        [1, 0, 0],
        [0, cos(gamma), -sin(gamma)],
        [0, sin(gamma), cos(gamma)]
        ])
    return vector@Rgamma.T@Rtheta.T@Rpsi.T


def passive_rotation(vector,heading,theta,gamma):
    # 外界的景物在飞机看来是怎样的，北天东在体轴下看来各自朝向哪个方向（前上右）
    # vector是行向量，根据 psi，theta，gamma 的顺序旋转坐标系，最后输出行向量
    # 被动旋转，-在1上
    # 注意：北天东坐标
    psi = - heading
    Rpsi=np.array([
        [cos(psi), 0, -sin(psi)],
        [0, 1, 0],
        [sin(psi), 0, cos(psi)]
        ])
    Rtheta=np.array([
        [cos(theta), sin(theta), 0],
        [-sin(theta), cos(theta), 0],
        [0, 0, 1]
        ])
    Rgamma=np.array([
        [1, 0, 0],
        [0, cos(gamma), sin(gamma)],
        [0, -sin(gamma), cos(gamma)]
        ])
    return vector@Rpsi.T@Rtheta.T@Rgamma.T