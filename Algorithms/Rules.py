import numpy as np
from math import *
from numpy.linalg import norm
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Math_calculates.sub_of_angles import *

def track_behavior(delta_psi):
    """
    追踪行为：返回 (heading_cmd, speed_cmd)
    """
    heading_cmd = delta_psi
    speed_cmd = 1.5 * 340
    return heading_cmd, speed_cmd


def crank_behavior(delta_psi):
    """
    crank 行为：返回 (heading_cmd, speed_cmd)
    """
    if 0 <= delta_psi:
        temp = 0.4 * (delta_psi - pi / 4) / (pi / 4) * 2
        heading_cmd = np.clip(temp, -0.4, 0.4)
    else:
        temp = 0.4 * (delta_psi + pi / 4) / (pi / 4) * 2
        heading_cmd = np.clip(temp, -0.4, 0.4)
    speed_cmd = 1.1 * 340
    return heading_cmd, speed_cmd


def escape_behavior(rel_psi_m):
    """
    逃逸行为：返回 (heading_cmd, speed_cmd)
    rel_psi_m 是导弹相对方位
    """
    heading_cmd = np.clip(sub_of_radian(rel_psi_m, pi), -pi/2, pi/2)
    # temp = 0.4 * sub_of_radian(rel_psi_m, pi) / pi * 4
    # heading_cmd = np.clip(temp, -0.4, 0.4)
    speed_cmd = 1.5 * 340
    return heading_cmd, speed_cmd


# def attack_again_behavior(delta_psi):
#     """
#     反击/继续攻击行为：返回 (heading_cmd, speed_cmd)
#     """
#     heading_cmd = delta_psi
#     speed_cmd = 1.5 * 340
#     return heading_cmd, speed_cmd


def wander_behavior():
    """
    wander 随机漫步行为：返回 (alt_cmd, heading_cmd, speed_cmd)
    """
    alt_cmd = 3000 * np.random.uniform(-1, 1)
    heading_cmd = np.random.normal(0, 25 * pi / 180)
    speed_cmd = 300
    return alt_cmd, heading_cmd, speed_cmd


def decision_rule(ego_pos_, ego_psi, enm_pos_, distance, ally_missiles, enm_missiles, o00, R_cage, wander=0, set_height=7e3, set_speed=None):
    # 输出为所需的绝对高度、相对方位和绝对速度
    # 是否有导弹可用
    has_missile_in_the_air = 0
    for missile in ally_missiles:
        if not missile.dead:
            has_missile_in_the_air = 1
            break
    # 是否被敌机导弹锁定（存在敌机导弹而且距离在20km内，此时导弹雷达可以锁定）
    should_escape = 0
    rel_psi_m = None
    for missile in enm_missiles:
        if not missile.dead:
            line_m_ = missile.pos_ - ego_pos_
            dist_mt = norm(line_m_)
            if dist_mt <= 20e3:  # 假设为导弹雷达的锁定距离
                should_escape = 1
                rel_psi_m = sub_of_radian(atan2(line_m_[2], line_m_[0]), ego_psi)
                break  # trick：先发射的导弹被当做距离最近的，没有在所有导弹中判断距离 todo 判断距离

    action_n = np.array([0.0, 0.0, 400])
    ego_height = ego_pos_[1]

    # 默认高度指令（固定高度7000m）
    action_n[0] = set_height - ego_height

    L_ = enm_pos_ - ego_pos_
    beta = atan2(L_[2], L_[0])
    delta_psi = sub_of_radian(beta, ego_psi)

    # 行为决策：按原逻辑分支调用独立函数
    if distance > 40e3:
        heading_cmd, speed_cmd = track_behavior(delta_psi)
    elif not should_escape and has_missile_in_the_air:
        heading_cmd, speed_cmd = crank_behavior(delta_psi)
    if should_escape:
        # rel_psi_m 在 should_escape 时应已被设置
        heading_cmd, speed_cmd = escape_behavior(rel_psi_m if rel_psi_m is not None else 0.0)
    else:
        heading_cmd, speed_cmd = track_behavior(delta_psi)

    action_n[1] = heading_cmd
    action_n[2] = speed_cmd if set_speed is None else set_speed

    # 追踪任务的目标在散步
    if wander:
        action_n[0], action_n[1], action_n[2] = wander_behavior()

    # 最高优先级：不许出圈
    # R_to_o00 = np.linalg.norm([ego_pos_[0], ego_pos_[2]])
    # if R_cage - R_to_o00 < 8e3:
    #     beta_of_o00 = atan2(-ego_pos_[2], -ego_pos_[0])
    #     action_n[1] = sub_of_radian(beta_of_o00, ego_psi)
    if ego_height > 13e3:
        action_n[0] = -5000
    elif ego_height < 3e3:
        action_n[0] = 5000

    return action_n

