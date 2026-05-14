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
from shapely.geometry import Point, Polygon, LineString


import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Math_calculates.sub_of_angles import *

# def sub_of_radian(input1, input2=0):
#     # 弧度减法
#     # 计算两个弧度的差值，范围为[-pi, pi]
#     diff = input1 - input2
#     diff = (diff + np.pi) % (2 * np.pi) - np.pi
#     return diff

def calc_intern_dist2circle(R, pos_, psi):
    pos_on_floor_ = np.array([pos_[0], 0, pos_[2]])
    rho = norm(pos_on_floor_)
    eta = atan2(pos_[2], pos_[0])
    
    # 计算水平距离
    if rho<R:
        dh_list = rho*cos(pi+eta-psi) + sqrt(R**2-rho**2*sin(pi+eta-psi)**2)
        dh = dh_list
    else:
        dh = 0

    # 边界在飞机的左边还是右边
    # left_or_right = np.sign(sub_of_radian(eta, psi)) # -1 左边，0 中间，1 右边
    left_or_right = sub_of_radian(psi, pi+eta)/pi*2 # -1 左边，0 中间，1 右边
    
    return dh, left_or_right   


def get_velocity_intercept_info(poly, p, v):
    """计算沿速度矢量方向与多边形边界的交点"""
    v_norm = np.linalg.norm(v)
    if v_norm == 0: return None, None
    
    v_unit = v / v_norm
    minx, miny, maxx, maxy = poly.bounds
    max_dim = np.sqrt((maxx - minx)**2 + (maxy - miny)**2)
    ray_end = np.array([p.x, p.y]) + v_unit * max_dim
    ray_line = LineString([(p.x, p.y), (ray_end[0], ray_end[1])])
    
    intersection = ray_line.intersection(poly.boundary)
    if intersection.is_empty: return None, None
    
    if intersection.geom_type == 'Point':
        hit_pt = np.array([intersection.x, intersection.y])
    else:
        pts = []
        if intersection.geom_type == 'MultiPoint':
            pts = [np.array([pt.x, pt.y]) for pt in intersection.geoms]
        elif intersection.geom_type == 'GeometryCollection':
            pts = [np.array([pt.x, pt.y]) for pt in intersection.geoms if pt.geom_type == 'Point']
        if not pts: return None, None
        hit_pt = pts[np.argmin([np.linalg.norm(pt - np.array([p.x, p.y])) for pt in pts])]
        
    dist_along_v = np.linalg.norm(hit_pt - np.array([p.x, p.y]))
    return dist_along_v, hit_pt

class polygon_fences:
    def __init__(self, vertices):
        # 计算几何重心：所有边界顶点的坐标加权平均
        self.poly = Polygon(vertices)
        self.vertices_arr = np.array(vertices)
        self.center = np.mean(self.vertices_arr, axis=0)
        
        self.test_point = np.array([2.5, 5.0]) 
        self.target_point = np.array([7.0, 7.0])
    
    def calc_intern_dist(self, pos_, psi):
        p = Point(np.array([pos_[0], pos_[2]]))
        is_inside = self.poly.contains(p)
        v = np.array([cos(psi), sin(psi)])
        dist_along_v, _ = get_velocity_intercept_info(self.poly, p, v)
        vec2center = self.center - np.array([pos_[0], pos_[2]])
        if not is_inside:
            dist_along_v = -100
        psi_of_vec2center = np.arctan2(vec2center[1], vec2center[0])
        side = sub_of_radian(psi, psi_of_vec2center)/pi*2
        # # 给出当前方向相对中心的偏移角度归一化值
        # crossed = np.array([
        #     v[0]*vec2center[1]/(norm(vec2center)+1e-8) - 
        #     v[1]*vec2center[0]/(norm(vec2center)+1e-8)
        #     ])
        # side = np.arcsin(crossed) / pi * 2
        return dist_along_v, side

if __name__ == '__main__':
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 参数设置
    R = 50
    rho = 80
    eta = 150 * pi/180
    psi = 180 * pi/180
    theta = 89 * pi/180
    
    pos_ = np.array([rho*cos(eta), 0, rho*sin(eta)])

    # 计算距离
    d_hor, left_or_right = calc_intern_dist2circle(R,pos_,psi)
    
    # 可视化
    print(f"水平距离: {d_hor:.2f}")
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
    vx = pos_[0] + d_hor * cos(psi)
    vz = pos_[2] + d_hor * sin(psi)
    
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


