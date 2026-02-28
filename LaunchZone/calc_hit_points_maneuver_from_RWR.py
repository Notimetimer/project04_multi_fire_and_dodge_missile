'''
遍历机动样式，根据命中的结果数占的比例获取发射概率
'''

import numpy as np
from math import *
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from Envs.MissileModel1 import *
from Envs.MissileModel1112 import *


g = 9.81
theta_limit_target = 30 * pi / 180

dt = 0.2
t = 0
# g_ = np.array([0, -g, 0])



# 提取最后一行
def latest(vectors):
    if vectors.ndim == 1:  # 一维数组
        return vectors  # 将一维数组转换为二维数组
    elif vectors.ndim == 2:  # 二维数组
        return vectors[-1]  # 直接返回最后一行


# 定义大气计算等通用部分
# 大气密度计算
def rho(h):
    # 输入高度(m)
    answer = 1.225 * np.exp(-h / 9300)
    return answer


# 马赫数计算：
def calc_mach(v, height):
    sound_speed = 20.0463 * np.sqrt(288.15 - 0.00651122 * height) if height <= 11000 else 295.069
    return v / sound_speed, sound_speed


def sub_of_radian(input1, input2):
    # 计算两个弧度的差值，范围为[-pi, pi]
    diff = input1 - input2
    diff = (diff + pi) % (2 * pi) - pi
    return diff


# 定义目标模型
class target:
    def __init__(self, pos0_, vel0_):
        super(target, self).__init__()
        self.pos_ = pos0_.copy()
        self.vel_ = vel0_.copy()
        self.av = 0
        self.ah = 0
        self.a_refer = 6 * g  # 参考加速度大小
        self.target_moves = np.array(range(5)) + 1

    def step(self, pm_, dt=dt, target_move=1, RWR_dist=20e3):
        theta = atan2(self.vel_[1], sqrt(self.vel_[0] ** 2 + self.vel_[2] ** 2))
        heading = atan2(self.vel_[2], self.vel_[0])
        v = norm(self.vel_)
        line_ = pm_ - self.pos_

        rel_theta = atan2(line_[1], sqrt(line_[0] ** 2 + line_[2] ** 2))
        rel_heading = atan2(line_[2], line_[0])

        # 平飞
        if target_move == 1:
            ay = -np.clip(2 * theta / (pi / 2) * self.a_refer, -self.a_refer, self.a_refer)
            az = 0

        # 水平置尾机动
        elif target_move == 2:
            ay = -np.clip(2 * theta / (pi / 2) * self.a_refer, -self.a_refer, self.a_refer)
            az = 4 * self.a_refer * sub_of_radian(rel_heading, sub_of_radian(heading + pi, 0)) / pi
            az = np.clip(az, -self.a_refer, self.a_refer)
            normal = sqrt(ay ** 2 + az ** 2)
            ay = 0 if normal < 1e-3 else ay / normal * self.a_refer
            az = 0 if normal < 1e-3 else az / normal * self.a_refer

        # 水平39机动
        elif target_move == 3:
            ay = -np.clip(2 * theta / (pi / 2) * self.a_refer, -self.a_refer, self.a_refer)
            temp_9 = sub_of_radian(rel_heading, sub_of_radian(heading - pi / 2, 0))  # 9点钟方向和导弹威胁方向的差角
            temp_3 = sub_of_radian(rel_heading, sub_of_radian(heading + pi / 2, 0))  # 3点钟方向和导弹威胁方向的差角
            if norm(temp_3) < norm(temp_9):
                # 三点钟对目标
                az = 4 * self.a_refer * temp_3 / pi * 2
            else:
                # 九点钟对目标
                az = 4 * self.a_refer * temp_9 / pi * 2
            az = np.clip(az, -self.a_refer, self.a_refer)
            normal = sqrt(ay ** 2 + az ** 2)
            ay = 0 if normal < 1e-3 else ay / normal * self.a_refer
            az = 0 if normal < 1e-3 else az / normal * self.a_refer

        # 置尾下高机动
        elif target_move == 4:
            ay = -self.a_refer
            az = 4 * self.a_refer * sub_of_radian(rel_heading, sub_of_radian(heading + pi, 0)) / pi
            az = np.clip(az, -self.a_refer, self.a_refer)
            normal = sqrt(ay ** 2 + az ** 2)
            ay = 0 if normal < 1e-3 else ay / normal * self.a_refer
            az = 0 if normal < 1e-3 else az / normal * self.a_refer

        # 39下高机动
        elif target_move == 5:
            ay = -self.a_refer
            temp_9 = sub_of_radian(rel_heading, sub_of_radian(heading - pi / 2, 0))  # 9点钟方向和导弹威胁方向的差角
            temp_3 = sub_of_radian(rel_heading, sub_of_radian(heading + pi / 2, 0))  # 3点钟方向和导弹威胁方向的差角
            if norm(temp_3) < norm(temp_9):
                # 三点钟对目标
                az = 4 * self.a_refer * temp_3 / pi * 2
            else:
                # 九点钟对目标
                az = 4 * self.a_refer * temp_9 / pi * 2
            az = np.clip(az, -self.a_refer, self.a_refer)
            normal = sqrt(ay ** 2 + az ** 2)
            ay = 0 if normal < 1e-3 else ay / normal * self.a_refer
            az = 0 if normal < 1e-3 else az / normal * self.a_refer

        # 告警距离之外保持平飞
        if norm(line_) > RWR_dist:
            ay = -np.clip(2 * theta / (pi / 2) * self.a_refer, -self.a_refer, self.a_refer)
            az = 0

        # 根据加速度解算速度和位置
        theta += ay / v * dt
        heading += az / v / cos(theta) * dt

        theta = np.clip(theta, -theta_limit_target, theta_limit_target)
        heading = sub_of_radian(heading, 0)

        # 根据高度限制俯仰角,北天东坐标
        if self.pos_[1] < 3000:
            theta_min = -(self.pos_[1] - 500) / 2500 * theta_limit_target
            theta = max(theta, theta_min)  # 将theta作用在目标上，不允许目标撞地
        if self.pos_[1] > 11000:
            theta_max = (15000 - self.pos_[1]) / 4000 * theta_limit_target
            theta = min(theta, theta_max)  # 将theta作用在目标上，不允许目标飞向太空

        self.vel_ = v * np.array([cos(theta) * cos(heading), sin(theta), cos(theta) * sin(heading)])  # 考虑加速度矢量
        self.pos_ += self.vel_ * dt

        return self.pos_, self.vel_


def sim_hit(pm0_, vm0_, pt0_, vt0_, target_move, datalink=1, show=0):
    g = 9.81
    theta_limit_target = 85 * pi / 180
    t = 0
    # 初始化导弹和目标实例
    missile1 = missile_class(pm0_, vm0_, pt0_, vt0_, t)
    Target = target(pt0_, vt0_)
    t_max = 120  # 假设电池工作120s
    dt_small = 0.08
    dt_big = 0.2
    break_flag = 0
    ptt_ = pt0_.copy()
    pmt_ = pm0_.copy()
    vtt_ = vt0_.copy()
    vmt_ = vm0_.copy()

    while t < t_max:
        distance = norm(pmt_ - ptt_)
        if distance > 5e3:
            dt = dt_big
        else:
            dt = dt_small
        t += dt

        target_information = missile1.observe(vmt_, vtt_, pmt_, ptt_)
        # 导弹移动
        vmt_, pmt_, v_dot, nyt, nzt, line_t_, q_beta_t, q_epsilon_t, theta_mt, psi_mt = missile1.step(
            target_information, dt=dt, datalink=datalink, record=False)
        vmt = norm(vmt_)

        # 目标移动
        ptt_, vtt_ = Target.step(pmt_, dt=dt, target_move=target_move)

        # 毁伤判定
        # 判断命中情况并终止运行
        if vmt < missile1.speed_min and t > 0.5 + missile1.stage1_time + missile1.stage2_time:
            missile1.dead = True
            break_flag = 1
        if pmt_[1] < missile1.minH_m:  # 高度小于限高自爆
            missile1.dead = True
            break_flag = 1
        if missile1.t > missile1.t_max:  # 超时自爆
            missile1.dead = True
            break_flag = 1
        if norm(missile1.vel_) < missile1.speed_min:  # 低速自爆
            missile1.dead = True
            break_flag = 1
        if t >= 0 + dt:
            hit, point1, point2 = hit_target(missile1.pos_, missile1.vel_, Target.pos_, Target.vel_, dt)
            if hit:
                # print('Target hit')
                missile1.dead = True
                missile1.hit = True
                break_flag = 1

        if break_flag: break
    return missile1.hit

def hit_prob(p_carrier_, v_carrier_, p_target_, v_target_):
    hit_counts = 0
    move_patterns = np.array(range(5)) + 1
    for move_pattern in move_patterns:
        # print("pattern", move_pattern)
        hit = sim_hit(p_carrier_, v_carrier_, p_target_, v_target_,
                      target_move=move_pattern, datalink=1, show=0)
        if hit:
            hit_counts += 1
    output = hit_counts / len(move_patterns)  # 可命中比例
    return output


if __name__ == '__main__':
    p_carrier_ = np.array([0, 10000, 0], dtype='float64')
    v_carrier_ = np.array([350, 0, 0], dtype='float64')
    p_target_ = np.array([12e3, 10000, 0e3], dtype='float64')
    v_target_ = np.array([-300, 0, 0], dtype='float64')

    # 变步长解算导弹和目标的运动直到目标被命中或导弹超时/速度过低/错过目标

    import time
    start_time = time.time()
    output = hit_prob(p_carrier_, v_carrier_, p_target_, v_target_)
    print('命中概率:', output)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"程序运行时长: {execution_time} 秒")

    # move_patterns = np.array(range(5)) + 1
    # start_time = time.time()
    # for run in range(100):  # 算100次需要多久
    #     print("run", run + 1)
    #     hit_counts = 0
    #     for move_pattern in move_patterns:
    #         # print("pattern", move_pattern)
    #         hit = sim_hit(p_carrier_, v_carrier_, p_target_, v_target_, target_move=move_pattern, datalink=1,
    #                       show=0)
    #         if hit:
    #             hit_counts += 1
    #
    #     output = hit_counts / len(move_patterns)  # 可命中比例
    #     print(output)
    #
    # end_time = time.time()
    # execution_time = end_time - start_time
    # print(f"程序运行时长: {execution_time} 秒")
