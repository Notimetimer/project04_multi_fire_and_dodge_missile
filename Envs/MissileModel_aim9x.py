"""
尚未更改使用矢量计算导弹导引率，避免欧拉角死锁
发现observe的调用逻辑、脱锁的效果需要重写
取消根据距离决定步长的逻辑，导弹固定用小步长，飞机固定用大步长

欠缺重构：脱锁的逻辑没有写在observe中，而是在step里面反映在输出结果中
"""

import numpy as np
from math import cos, sin, tan, pi, atan, atan2, acos, asin
from numpy.linalg import norm
import socket
import threading
import time

dt_test = 0.05
g = 9.81
theta_limit = 85 * pi / 180

# 提取第一行
def earlist(vectors):
    if vectors.ndim == 1:  # 一维数组
        return vectors  # 将一维数组转换为二维数组
    elif vectors.ndim == 2:  # 二维数组
        return vectors[0]  # 直接返回第一行

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
    # # 拟合公式1（高空误差大）
    # answer = 1.225 * np.exp(-h / 9300)
    # 拟合公式2（高空误差小）
    answer = 101325*(1-h/44300)**5.256/(286.9*(288.15-0.0065*h))
    return answer


# 马赫数计算：
def calc_mach(v, height):
    sound_speed = 20.0463 * np.sqrt(288.15 - 0.00651122 * height) if height <= 11000 else 295.069
    return v / sound_speed, sound_speed

# 计算命中情况
def hit_target(pmt_1, vmt_1, ptt_1, vtt_1, dt=0.02, kill_range=85):
    # # 计算点与点距离
    # distance1 = np.linalg.norm(pmt_ - ptt_)
    # return distance1 <= kill_range, pmt_

    # 解析计算
    killed = False
    # 基于当前点和线性外推得的dt后下一点的判据
    M1 = pmt_1
    T1 = ptt_1
    M2 = pmt_1 + dt * vmt_1
    T2 = ptt_1 + dt * vtt_1

    t1 = 0
    delta_t = dt
    T1T2_ = T2 - T1
    M1M2_ = M2 - M1
    M1T1_ = T1 - M1
    M2T2_ = T2 - M2

    # 调大杀伤半径后直接根据两点距离取最小判别
    d1 = norm(M1-T1)
    d2 = norm(M2-T2)
    if d1 <= kill_range:
        return 1, M1, T1
    elif d2 <= kill_range:
        return 1, M2, T2
    else:
        return 0, M1, T1
    
    # # 解析计算命中情况
    # # 超出5km跳过计算
    # if np.linalg.norm(M1-T1) > 5e3:
    #     return False, M1, T1

    # if np.linalg.norm(T1T2_ - M1M2_) < 1e-3:
    #     # 平行移动了，另一套公式
    #     d_min = np.linalg.norm(M1T1_)
    #     posm_ = M1
    #     post_ = T1
    # else:
    #     # 至少运动的方向和大小中有一项是不一样的
    #     t_min = t1 - (np.dot(M1T1_, T1T2_) - np.dot(M1T1_, M1M2_)) / np.linalg.norm(
    #         T1T2_ - M1M2_) ** 2 * delta_t
    #     if t_min < t1:
    #         t_min = t1
    #     elif t_min > t1 + delta_t:
    #         t_min = t1 + delta_t
    #     posm_ = M1 + (t_min - t1) / delta_t * (M2 - M1)
    #     post_ = T1 + (t_min - t1) / delta_t * (T2 - T1)
    #     d_min = np.linalg.norm(M1T1_ + (t_min - t1) / delta_t * (T1T2_ - M1M2_))
        
    # if d_min <= kill_range:
    #     # print(d_min)
    #     killed = True
    # return killed, posm_, post_

def sub_of_radian(input1, input2):
    # 计算两个弧度的差值，范围为[-pi, pi]
    diff = input1 - input2
    diff = (diff + pi) % (2 * pi) - pi
    return diff

# 导弹模型
class missile_class:
    def __init__(self, pos0_, vel0_, pt0_, vt0_, launch_time=0):
        super(missile_class, self).__init__()
        self.WeaponType = "AIM-9X"
        self.dead = False  # 导弹是否死亡
        self.hit = False  # 导弹是否命中目标
        self.launch_time = launch_time  # 导弹发射时间
        self.lock_time = None
        self.pos_ = np.array(pos0_, dtype=float).copy()
        self.vel_ = np.array(vel0_, dtype=float).copy()
        self.pt0_ = np.array(pt0_, dtype=float).copy()
        self.vt0_ = np.array(vt0_, dtype=float).copy()
        self.theta = 0.0
        self.psi = 0.0
        self.phi = 0.0
        self.latest_time_of_target = launch_time  # 上一次观测到目标的时间 todo 没有考虑载机转走的情况

        # 最高和最低高度
        self.maxH_m = 25000
        self.minH_m = 100 # 500
        # 重量参数和燃烧时间参数
        self.max_weight = 85.3 # kg 
        self.stage1_weight = 34 # kg
        self.stage2_weight = 1e-8 # 44.9 # kg
        self.empty_weight = self.max_weight - self.stage1_weight - self.stage2_weight # kg
        self.stage1_time = 5.0 # s
        self.stage2_time = 1e-8 # 10.5 # 10.5 11  # s
        self.stage1_thrust = 12802 # N
        self.stage2_thrust = 0 # 9360.5  # N
        self.stage1_start = 0.05  # s
        self.stage2_start = self.stage1_start + self.stage1_time  # s
        self.stage2_end = self.stage1_start + self.stage1_time + self.stage2_time  # s
        # 杀伤半径
        self.kill_range = 30 # 85 # 20  # 为了提高训练效率，这个半径可以被放大，从而避免变步长
        # 最大过载
        self.max_g0 = 50
        self.max_g1 = 35
        self.max_g = self.max_g0
        # 最大马赫数
        self.max_mach = 4.0
        # 特征面积
        self.area = pi*0.127**2 /4 # 425 435 0.405  # m2
        # 阻力系数是一个函数，不在这里定义
        # 最小速度
        self.speed_min = 0.65 * 340  # m/s
        # 最大视角
        self.sight_angle_max = np.radians(90) # 极限离轴可能接近90度
        # 最大跟踪视角速度
        self.sight_angle_rate_max = np.radians(75) # 0.7  # rad/s
        # 被告警距离
        self.detect_range = 8e3  # 15~20km之间, 14km有些激进了  # m todo 计算截获距离
        self.distance = 100e3
        self.t = 0  # 导弹初始计时
        self.t_max = self.stage1_start + 65  # 电池工作时间
        self.t_go = 120
        self.trajectory = np.empty((0, 7))  # 导弹轨迹, 结构为时间、位置（3）、速度（3）
        self.guidance_stage = 2  # 2为中制导，3为末制导
        self.in_angle = 0
        self.in_distance = 0
        self.q_beta = 0
        self.q_epsilon = 0
        self.nzt = 0  # 过载量记忆值
        self.nyt = 0
        self.aN_target_required_ = np.array([0, 0, 0])
        self.mid_guid_rate = 4
        self.term_guid_rate = 4
        self.last_target_pos = None
        self.last_target_t = None
        self.last_target_v = None
        self.radar_on = False
        self.lock_on = False
        self.side = None
        self.datalink = None
        self.A_pole_moment = None
        self.g_ = g * np.array([0, -1, 0])
        self.delta_height = 0
        self.target_close_v = None
        self.off_lock = False
        self.ground_close_rate = None
        self.dt = 0.04 # 初始值 
        self.v_dot = 0
        self.gliding = 0
        self.spotted_distance = 10e3

    def Cd(self, mach):
        lis0 = np.array([
            [0,   0.5],
            [0.6, 0.5],
            [0.8, 0.5],
            [1.0, 0.55],
            [1.2, 0.77],
            [1.6, 0.65],
            [2.4, 0.5],
            [3,   0.45],
            [4,   0.36],
        ])
        return np.interp(mach, lis0[:, 0], lis0[:, 1]) # 不支持外差值，外差值要用scipy


    # 计算推力和质量
    def calc_burn(self):
        # 一级点火
        if self.stage1_start < self.t <= self.stage1_start + self.stage1_time:
            Fp = self.stage1_thrust
            m_missile = self.empty_weight + self.stage2_weight + \
                        self.stage1_weight * (1 - (self.t - self.stage1_start) / self.stage1_time)
        # 二级点火
        elif self.stage1_start + self.stage1_time < self.t <= self.stage1_start + self.stage1_time + self.stage2_time:
            Fp = self.stage2_thrust
            m_missile = self.empty_weight + \
                        self.stage2_weight * (1 - (self.t - self.stage1_start - self.stage1_time) / self.stage2_time)
        # 滑翔段
        else:
            Fp = 0
            m_missile = self.empty_weight
        return Fp, m_missile

    def observe(self, v_missile_, v_target_, p_missile_, p_target_):
        # todo 产生target_information，包含visable，为0时后续所有都是0,在step中引用无目标时的导引率（尚未写的类方法），否则为1时正常调用制导律
        # 当前写法后面还要改
        target_information = np.zeros(7)
        target_information[0] = 1
        target_information[1:4] = p_target_
        target_information[4:7] = v_target_
        # 输入参数
        vmt_ = v_missile_
        ptt_ = p_target_
        pmt_ = p_missile_
        line_t_ = ptt_ - pmt_
        vrt_ = v_target_ - v_missile_
        distance = np.linalg.norm(line_t_)
        self.distance = distance
        vmt = np.linalg.norm(vmt_)
        vm_hor = np.linalg.norm([vmt_[2], vmt_[0]])  # 导弹水平分速度
        distance_hor = np.linalg.norm([line_t_[2], line_t_[0]])  # 弹目线水平距离
        theta_mt1 = np.arctan2(vmt_[1], vm_hor)

        # # 超出距离、跟踪角度或是跟踪角速度脱锁
        # if np.linalg.norm(distance) > self.detect_range or \
        #         np.dot(v_missile_, line_t_) / vmt / distance < cos(self.sight_angle_max) or \
        #         np.linalg.norm(np.cross(line_t_, vrt_)) / distance ** 2 > self.sight_angle_rate_max:
        #     target_information = np.zeros(7)

        return target_information

    def get_max_g(self, current_rho, v):
        rho0 = rho(5000) # 假设的取得最大过载量的高度
        v0 = 340 * 3.2 # 假设取得最大过载量的速度
        presssure0 = 0.5*rho0*v0**2 # 假设取得最大过载量的动压
        pressure = 0.5*current_rho*v**2 # 当前动压
        self.max_g = self.max_g0 * min(pressure / presssure0, 1)

    # 中制导
    def mid_term_guidance(self, v_missile_, v_target_, p_missile_, p_target_, datalink=True):
        "原位置预测模型，没有考虑误差，精度过高被钻了空子"
        # # 线性递推预测目标位置
        if self.last_target_t is None:
            vtt_predict = self.vt0_
            ptt_predict = self.pt0_ + self.vt0_ * (self.t - self.latest_time_of_target)
        else:
            vtt_predict = self.last_target_v * np.exp(-(self.t - self.last_target_t) / 15.0)  # * 0.5 #  0.8 # 预测速度带衰减
            ptt_predict = self.last_target_pos + vtt_predict*(self.t - self.last_target_t)
            ptt_predict[1] = max(ptt_predict[1], 200) # 禁止往地底下线性递推

        # 输入参数
        vmt_ = v_missile_
        ptt_ = ptt_predict
        pmt_ = p_missile_
        line_t_ = ptt_ - pmt_
        vrt_ = vtt_predict - v_missile_
        distance = np.linalg.norm(line_t_)
        vmt = np.linalg.norm(vmt_)
        vm_hor = np.linalg.norm([vmt_[2], vmt_[0]])  # 导弹水平分速度
        distance_hor = np.linalg.norm([line_t_[2], line_t_[0]])  # 弹目线水平距离
        theta_mt1 = np.arctan2(vmt_[1], vm_hor)
        
        # 制导指令算法，矢量方程
        # 制导指令算法，矢量方程
        g_ = self.g_
        g_parallel_ = np.dot(g_, vmt_) / vmt ** 2 * vmt_
        distance_dot = np.abs(np.dot(vrt_, line_t_) / distance)
        
        vrL_ = np.dot(vrt_, line_t_) / distance ** 2 * line_t_  # 径向相对速度
        vr_perpendicular_ = vrt_ - vrL_
        n2 = vr_perpendicular_ - np.dot(vr_perpendicular_, vmt_) / vmt ** 2 * vmt_  # 周向相对速度在导弹法平面的投影

        n_normalized_ = n2 / (np.linalg.norm(n2) + 1e-6) # if np.linalg.norm(n2) > 0 else np.array([0, 0, 0])

        aN_target_required_ = self.mid_guid_rate * np.linalg.norm(np.cross(line_t_, vrt_)) / (
                distance ** 2) * vmt * n_normalized_  # PPG
        aN_course_required_ = -(g_ - g_parallel_)
        aN1_ = aN_target_required_ + aN_course_required_
        
        return 2, aN1_


    # 末制导
    def terminal_guidance(self, v_missile_, v_target_, p_missile_, p_target_):
        
        # 输入参数
        vmt_ = v_missile_
        vtt_ = v_target_
        ptt_ = p_target_
        pmt_ = p_missile_
        line_t_ = ptt_ - pmt_
        vrt_ = v_target_ - v_missile_
        distance = np.linalg.norm(line_t_)
        vmt = np.linalg.norm(vmt_)
        vm_hor = np.linalg.norm([vmt_[2], vmt_[0]])  # 导弹水平分速度
        vt_hor = np.linalg.norm([vtt_[2], vtt_[0]])  # 目标水平分速度
        distance_hor = np.linalg.norm([line_t_[2], line_t_[0]])  # 弹目线水平距离
        theta_mt1 = np.arctan2(vmt_[1], vm_hor)
        
        # 制导指令算法，矢量方程
        g_ = self.g_
        g_parallel_ = np.dot(g_, vmt_) / vmt ** 2 * vmt_
        distance_dot = np.abs(np.dot(vrt_, line_t_) / distance)
        
        vrL_ = np.dot(vrt_, line_t_) / distance ** 2 * line_t_  # 径向相对速度
        vr_perpendicular_ = vrt_ - vrL_
        n2 = vr_perpendicular_ - np.dot(vr_perpendicular_, vmt_) / vmt ** 2 * vmt_  # 周向相对速度在导弹法平面的投影

        n_normalized_ = n2 / (np.linalg.norm(n2) + 1e-6) # if np.linalg.norm(n2) > 0 else np.array([0, 0, 0])

        aN_target_required_ = self.term_guid_rate * np.linalg.norm(np.cross(line_t_, vrt_)) / (
                distance ** 2) * vmt * n_normalized_  # PPG
        aN_course_required_ = -(g_ - g_parallel_)
        
        # 导引头脱锁模拟, 速度方向当做导弹头部方向
        off_lock = False
        # 超出导引头视角导致脱锁
        if np.dot(v_missile_, line_t_) / vmt / distance < cos(self.sight_angle_max):
            off_lock = True
        # 超出跟踪角速度导致脱锁
        if np.linalg.norm(np.cross(line_t_, vrt_)) / distance ** 2 > self.sight_angle_rate_max:
            off_lock = True

        # 矢量方程，脱锁处理
        if off_lock:
            aN_target_required_ = self.aN_target_required_
        else:
            self.aN_target_required_ = aN_target_required_
        aN1_ = aN_target_required_ + aN_course_required_

        return 3, aN1_

    # 制导律选择
    def guidance(self, v_missile_, v_target_, p_missile_, p_target_, datalink):
        if self.lock_on:
            self.A_pole_moment = int(self.guidance_stage == 2)
            self.guidance_stage = 3
            if self.lock_time is None:
                self.lock_time = self.t
            return self.terminal_guidance(v_missile_, v_target_, p_missile_, p_target_)

        self.A_pole_moment = 0
        self.guidance_stage = 2
        return self.mid_term_guidance(v_missile_, v_target_, p_missile_, p_target_, datalink)

    def step(self, target_information, dt=0.02, datalink=True, record=False):
        '''
        输入结构：是否看到目标(1)，目标的位置(3)、目标的速度(3)
        '''
        self.dt = dt
        # 根据目标信息产生制导指令
        visable = target_information[0]
        # if visable==0:
        #     nzt=
        ptt_ = target_information[1:4]
        vtt_ = target_information[4:7]
        pmt_ = self.pos_
        vmt_ = self.vel_

        # test
        line_t_ = ptt_ - pmt_
        distance = np.linalg.norm(line_t_)
        self.distance = distance
        self.delta_height = line_t_[1] # 与目标的高度差
        self.radar_on = self.t <= self.t_max and self.distance < self.spotted_distance
        # print('导弹感知到的距离:' + str(distance))

        vmt = np.linalg.norm(vmt_)
        if np.dot(vmt_, line_t_) / vmt / distance > cos(self.sight_angle_max):
            self.in_angle = 1
        else:
            self.in_angle = 0
        if self.distance < self.detect_range:
            self.in_distance = 1
        else:
            self.in_distance = 0
        self.lock_on = self.in_angle and self.in_distance and (not self.off_lock)

        case, temp = self.guidance(vmt_, vtt_, pmt_, ptt_, datalink=datalink)

        v_rel_ = vtt_-vmt_
        L_dot = np.dot(v_rel_, line_t_)/distance
        self.t_go = -distance/L_dot if abs(L_dot) > 1e-6 else self.t_max # 弹目距离
        self.t_go = max(self.t_go, 0.1)

        # 试验：多普勒速度窗口相关计算
        self.target_close_v = - np.dot(vtt_, line_t_)/distance        
        self.ground_close_rate = vmt if self.theta < np.radians(-15) else 0

        vtt = np.linalg.norm(vtt_)
        psi_mt = np.arctan2(vmt_[2], vmt_[0])
        psi_tt = np.arctan2(vtt_[2], vtt_[0])
        vm_hor = np.linalg.norm([vmt_[2], vmt_[0]])  # 导弹水平分速度
        vt_hor = np.linalg.norm([vtt_[2], vtt_[0]])  # 目标水平分速度
        distance_hor = np.linalg.norm([line_t_[2], line_t_[0]])  # 弹目线水平距离
        q_beta_t = np.arctan2(line_t_[2], line_t_[0])  # 目标线偏角
        q_epsilon_t = np.arctan2(line_t_[1], distance_hor)  # 目标线倾角
        theta_mt = np.arctan2(vmt_[1], vm_hor)
        theta_tt = np.arctan2(vtt_[1], vt_hor)
        g_parallel_ = np.dot(self.g_, vmt_) / vmt ** 2 * vmt_
        
        # theta = atan2(vmt_[1], np.linalg.norm([vmt_[0], vmt_[2]]))  # 速度倾角
        # psi = atan2(vmt_[2], vmt_[0])  # 速度航向角
        

        self.theta = theta_mt
        self.psi = psi_mt

        self.q_beta = q_beta_t
        self.q_epsilon = q_epsilon_t

        if case == 1:
            aN1_ = np.zeros([1, 3])
        else:
            aN1_ = temp
        

        # 根据制导指令更新动力学
        Fp, m_missile1 = self.calc_burn()
        # 导弹马赫数
        mach, sound_speed = calc_mach(vmt, pmt_[1])  # 速率，高度
        # 导弹阻力计算
        cd = self.Cd(mach)
        # 空气密度
        Rho = rho(pmt_[1])
        # 空气阻力
        Fx = 1 / 2 * Rho * vmt ** 2 * self.area * cd
        # 速率更新
        v_dot = (Fp - Fx) / m_missile1 - g * sin(theta_mt)
        vmt += v_dot * dt
        
        # # 限马赫数
        # vmt = min(vmt, self.max_mach * sound_speed)
        # # 过载限制2
        # nt = np.clip(np.linalg.norm([nyt, nzt]), 0, 40)
        
        # 动力学矢量方程
        aT_ = v_dot * vmt_ / (vmt + 1e-6) # if vmt > 0 else np.zeros([1, 3])

        N_max = self.max_g0 if self.t < self.stage2_end else self.max_g1

        aN2 = np.clip(np.linalg.norm(aN1_), 0, N_max * g)
        aN2_ = aN1_ * aN2 / (np.linalg.norm(aN1_) + 1e-6) # if np.linalg.norm(aN1_) > 0 else np.array([0, 0, 0])  # 动作引起的加速度
        aN_ = aN2_ + (self.g_ - g_parallel_)  # 加速度在法向的合量
        am_ = aT_ + aN_
        
        # 废弃变量，但是不能没有
        nyt = 0
        nzt = 0
        
        
        # 欧拉角反奇异
        theta_mt = np.clip(theta_mt, -theta_limit, theta_limit)
        if psi_mt > pi:
            psi_mt -= 2 * pi
        if psi_mt < -pi:
            psi_mt += 2 * pi
        
        # 运动方程
        vmt_ += am_ * dt
        vmt_ = vmt_ * vmt / (np.linalg.norm(vmt_) + 1e-6) # if np.linalg.norm(vmt_) > 0 else np.zeros([1, 3])
        
        self.vel_ = vmt_

        # print(vmt_)
        # print(self.pos_)

        self.pos_ += vmt_ * dt  # 欧拉积分
        # 更新时间
        self.t += dt
        self.t = round(self.t, 2)  # 保留两位小数

        if self.t < self.stage2_end or self.t > self.t_max:
            self.gliding = 0
        else:
            self.gliding = 1

        # print(self.t)
        # 记录运行轨迹
        if record:
            traj_add = np.array(
                [self.t + self.launch_time, self.pos_[0], self.pos_[1], self.pos_[2], self.vel_[0], self.vel_[1],
                 self.vel_[2]])
            self.trajectory = np.vstack((self.trajectory, traj_add.copy()))  # 记录当前位置
        
        return vmt_, pmt_, v_dot, nyt, nzt, line_t_, q_beta_t, q_epsilon_t, theta_mt, psi_mt



if __name__ == '__main__':
    list_missiles = []
    import os
    import sys
    def get_current_file_dir():
    # 判断是否在 Jupyter Notebook 环境
        # try:
        #     shell = get_ipython().__class__.__name__  # ← 误报，不用管
        #     if shell == 'ZMQInteractiveShell':  # Jupyter Notebook 或 JupyterLab
        #         # 推荐用 os.getcwd()，指向启动 Jupyter 的目录
        #         return os.getcwd()
        #     else:  # 其他 shell
        #         return os.path.dirname(os.path.abspath(__file__))
        # except NameError:
        #     # 普通 Python 脚本
        return os.path.dirname(os.path.abspath(__file__))
    current_dir = get_current_file_dir()
    sys.path.append(os.path.dirname(current_dir))
    dt = dt_test # 0.02 # 0.02
    from Visualize.tacview_visualize import *

    # 联动tacview
    import socket
    import threading
    import time

    # 地理原点
    Longitude0 = 144 + 43 / 60
    Latitude0 = 13 + 26 / 60
    Height0 = 0
    mark = np.array([Longitude0, Latitude0, Height0])  # 地理原点


    def ENU2LLH(mark, NUE):
        # 东北天单位为m，经纬度单位是角度
        N, U, E = NUE
        # E, N, U = ENU
        longit0, latit0, height0 = mark
        R_earth = 6371004  # ???
        dlatit = N / R_earth * 180 / pi
        dlongit = E / (R_earth * cos(latit0 * pi / 180)) * 180 / pi
        dheight = U
        out = np.array([longit0 + dlongit, latit0 + dlatit, height0 + dheight])
        return out


    # 目标运动类,不考虑机动
    class target:
        def __init__(self, pos0_, vel0_):
            super(target, self).__init__()
            self.pos_ = pos0_
            self.vel_ = vel0_

        def step(self, dt=dt):
            self.pos_ += self.vel_ * dt
            self.vel_ = self.vel_
            return self.pos_, self.vel_


    # 调用导弹模型仿真
    t_max = 60 * 2  # 最大仿真时间
    i_list = np.arange(0, int(t_max / dt), 1)
    t_range = np.round(i_list * dt, 2)

    # 平射
    p_carrier_ = np.array([0, 5e3, 0])
    v_carrier_ = np.array([300, 0, 0])
    list_pt_ = np.array([14e3, 5e3, 0e3])  # 目标
    list_vt_ = np.array([300, 0, 0])

    # # 抛射
    # p_carrier_ = np.array([0, 11e3, 0])
    # v_carrier_ = np.array([cos(pi/6), sin(pi/6), 0]) * 1.2 *340
    # list_pt_ = np.array([140e3, 9e3, 0])  # 目标 3e3
    # list_vt_ = np.array([-300, 0, 40])

    missile_used = 0
    hit = False

    list_pm_ = p_carrier_
    list_vm_ = v_carrier_
    list_vm = np.linalg.norm(list_vm_)
    # missile1 = missile_class(list_pm_, list_vm_, t)
    Target = target(list_pt_, list_vt_)

    list_line_ = list_pt_ - list_pm_
    list_distance = norm(list_pm_ - list_pt_)
    vmt_ = list_vm_
    vmt = list_vm
    pmt_ = list_pm_
    ptt_ = list_pt_
    vtt_ = list_vt_

    # 多枚导弹连续发
    # 遍历导弹列表，如果导弹炸了，则跳过
    # 如果导弹没炸，则获取目标位置、动力学计算以及毁伤判定、记录轨迹
    end_flag = 0
    t_count_start = -10
    tacview = Tacview()
    for i in i_list:
        t = np.round(i * dt, 2)
        # 更新载机位置和速度
        theta_carrier = np.arcsin(v_carrier_[1]/norm(v_carrier_))
        # theta_carrier = (15000-pmt_[1])/5000 * pi/2 # 不管载机了，定高不好写
        psi_carrier = atan2(v_carrier_[2], v_carrier_[0])
        v_carrier_ = norm(v_carrier_)*np.array([
            cos(theta_carrier)*cos(psi_carrier),
            sin(theta_carrier),
            cos(theta_carrier)*sin(psi_carrier),
        ])
        p_carrier_ += v_carrier_ * dt
        # 更新目标位置和速度
        ptt_, vtt_ = Target.step(dt=dt)
        L_t_ = ptt_ - p_carrier_
        distance_of_planes = norm(L_t_)

        off_axis_angle_radian = np.arccos(np.dot(v_carrier_, L_t_) / norm(v_carrier_) / distance_of_planes)

        in_range = 1 # 1e3 < distance_of_planes < 80e3 and off_axis_angle_radian < 60 * pi / 180

        # 最快每隔5s发射一枚导弹，一共4枚导弹
        if t - t_count_start >= 10 and in_range and missile_used < 1: # 4
            missile1 = missile_class(p_carrier_, v_carrier_, ptt_, vtt_, t)
            list_missiles.append(missile1)
            t_count_start = t  # 重置计时器
            missile_used += 1
            print('missile launched')
        # 对每一枚导弹做判断
        for missile1 in list_missiles:
            if not missile1.dead:
                # 目标位置传给导弹

                target_information = missile1.observe(vmt_, vtt_, pmt_, ptt_)
                # 导弹移动
                vmt_, pmt_, v_dot, nyt, nzt, line_t_, q_beta_t, q_epsilon_t, theta_mt, psi_mt = missile1.step(
                    target_information, dt=dt)
                vmt = norm(vmt_)

                # 毁伤判定
                # 判断命中情况并终止运行
                if vmt < missile1.speed_min and t > missile1.stage1_start + missile1.stage1_time + missile1.stage2_time:
                    missile1.dead = True
                if pmt_[1] < missile1.minH_m:  # 高度小于限高自爆
                    missile1.dead = True
                if missile1.t > missile1.t_max:  # 超时自爆
                    missile1.dead = True
                if t >= 0 + dt:
                    # if distance < missile1.kill_range*50:  # 启动变步长计算命中情况的条件
                    #     pass
                    #     # 假定目标在dt的时间内线性运动，导弹可以以更细分的时间取得观测信息和机动决策

                    hit, _, _ = hit_target(pmt_, vmt_, ptt_, vtt_)
                    if hit:
                        print('Target hit')
                        missile1.dead = True
                        missile1.hit = True

                        end_flag = 1

        # 在tacview上显示运动
        send_t = t
        name_R = '001'
        name_B = '002'
        loc_r = ENU2LLH(mark, p_carrier_)
        loc_b = ENU2LLH(mark, ptt_)
        data_to_send = f"#{send_t:.2f}\n{name_R},T={loc_r[0]:.6f}|{loc_r[1]:.6f}|{loc_r[2]:.6f},Name=F16,Color=Red\n" + \
                       f"#{send_t:.2f}\n{name_B},T={loc_b[0]:.6f}|{loc_b[1]:.6f}|{loc_b[2]:.6f},Name=F16,Color=Blue\n"
        # tacview.send_data_to_client(
        #     "#0.00\nA0100,T=119.99999999999999|59.999999999999986|8902.421354242131|5.124908336161374e-15|2.6380086088911072e-15|92.1278924460462,Name=F16,Color=Red\n")
        # tacview.send_data_to_client(data_to_send)

        # alive_missiles = [missile1 for missile1 in list_missiles if not missile1.dead]
        if not len(list_missiles) == 0:
            for j, missile1 in enumerate(list_missiles):
                if not missile1.dead:
                    loc_m = ENU2LLH(mark, missile1.pos_)
                    data_to_send += f"#{send_t:.2f}\n{10+j},T={loc_m[0]:.6f}|{loc_m[1]:.6f}|{loc_m[2]:.6f}|" \
                                    f"{0.0:.6f}|{missile1.theta * 180 / pi:.6f}|{missile1.psi * 180 / pi:.6f}," \
                                    f"Name=AIM-9X,Color=Orange\n"

        tacview.send_data_to_client(data_to_send)
        if t % 1 == 0:
            time.sleep(0.5)

        if end_flag:
            break
