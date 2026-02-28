"""
假设水平面上圆心为(0,0)，那么飞机坐标(x,z)可以导出极坐标下的(ρ,η),以北为正方向,则
ρ=sqrt(x**2+z**2)
η=atan2(z,x)
设飞机速度矢量v_水平分量vh_延长线还有dh到达圆形边界，航向角为ψ
那么由圆心, 碰撞点和飞机坐标构成的三角形可确定半径R, 飞机ρ还有R的对角 π+η-ψ
由余弦定理可以得到
R**2=ρ**2+dh**2-2*ρ*dh*cos(π+η-ψ)
dh = rho*cos(pi+eta-psi)±sqrt(R**2-rho**2*sin(pi+eta-psi)**2)
两个值要保留[0,R]之间的那个
最后考虑到飞机的俯仰角θ，
d=dh/cos(theta)

"""
from math import *
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Math_calculates.sub_of_angles import *

def sub_of_radian(input1, input2):
    # 弧度减法
    # 计算两个弧度的差值，范围为[-pi, pi]
    diff = input1 - input2
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    return diff

def calc_intern_dist2cylinder(R, pos_, psi, theta):
    """
    计算飞机到圆柱形边界的斜距离
    
    参数:
    R: float, 圆柱形边界半径
    rho: float, 飞机到圆心的距离
    eta: float, 飞机相对于圆心的方位角（弧度）
    psi: float, 飞机航向角（弧度）
    theta: float, 飞机俯仰角（弧度）
    
    返回:
    d: float, 飞机到边界的斜距离
    dh: float, 飞机到边界的水平距离
    pos_: ndarray, 飞机位置坐标 [北、天、东]
    """
    # 计算飞机位置
    pos_on_floor_ = np.array([pos_[0], 0, pos_[2]])
    rho = norm(pos_on_floor_)
    eta = atan2(pos_[2], pos_[0])
    
    # 计算水平距离
    if rho<R:
        dh_list = rho*cos(pi+eta-psi) + sqrt(R**2-rho**2*sin(pi+eta-psi)**2)
        dh = dh_list
    else:
        dh = 0
    
    # 计算斜距离
    d = dh/(cos(theta)+1e-5)

    # 边界在飞机的左边还是右边
    left_or_right = np.sign(sub_of_radian(eta, psi)) # -1 左边，0 中间，1 右边
    
    return d, dh, left_or_right

if __name__ == '__main__':
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 参数设置
    R = 50
    rho = 30
    eta = 150 * pi/180
    psi = 180 * pi/180
    theta = 89 * pi/180
    
    pos_ = np.array([rho*cos(eta), 0, rho*sin(eta)])

    # 计算距离
    d, dh, left_or_right = calc_intern_dist2cylinder(R, pos_, psi, theta)
    
    # 可视化
    print(f"水平距离: {dh:.2f}")
    print(f"斜距离: {d:.2f}")
    print(left_or_right)
    
    # 绘制图形
    plt.figure(figsize=(8, 6))
    
    # 绘制圆形边界
    theta_circle = np.linspace(0, 2*pi, 100)
    x_circle = R * np.cos(theta_circle)
    z_circle = R * np.sin(theta_circle)
    plt.plot(x_circle, z_circle, 'b-', label='边界')
    
    # 绘制圆心到飞机位置的线段
    plt.plot([0, pos_[2]], [0, pos_[0]], 'g-', label='圆心到飞机')
    
    # 计算速度向量终点
    vx = pos_[0] + dh * cos(psi)
    vz = pos_[2] + dh * sin(psi)
    
    # 绘制飞机位置到速度向量终点的线段
    plt.plot([pos_[2], vz], [pos_[0], vx], 'r-', label='速度向量')
    
    # 绘制速度向量终点到圆心的线段
    plt.plot([vz, 0], [vx, 0], 'k-', label='终点到圆心')
    
    # 标记关键点
    plt.plot(0, 0, 'ko', label='圆心')
    plt.plot(pos_[2], pos_[0], 'ro', label='飞机位置')
    plt.plot(vz, vx, 'bo', label='碰撞点')
    
    # 设置图形属性
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.title('飞机与圆柱形边界的相对位置')
    plt.xlabel('东')
    plt.ylabel('北')
    
    plt.show()


