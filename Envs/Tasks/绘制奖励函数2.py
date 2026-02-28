import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import *


def onedim_reward(x):
    # 函数实现
    output = x

    return output

# def twodim_rewward(x, y):
#     # 使用 np.where 或 np.select 进行条件选择
#     a0 = np.select(
#         [x <= 8000, (x > 8000) & (x <= 12000), x > 12000],
#         [10, 6, 3]
#     )
#
#     b0 = np.select(
#         [y <= 22.5, (y > 22.5) & (y <= 45), y > 45],
#         [3, 6, 10]
#     )
#
#     a = 1 + 100 - np.maximum(0, 100 - x)
#     b = 1 + 100 - np.maximum(0, 100 - y)
#
#     output = (a + a0) / (a + a0 + b + b0)
#
#     return output
def twodim_rewward(x, y):
    # 使用 np.where 处理条件判断
    distance = 30e3
    angle = 0

    a0 = np.where(distance <= 8000, 10,
                  np.where(distance <= 12000, 6, 3))

    b0 = np.where(angle <= 22.5, 3,
                  np.where(angle <= 45, 6, 10))

    a = 1 + 100 - np.maximum(0, 100 - x)
    b = 1 + 100 - np.maximum(0, 100 - y)

    output = np.clip((a + a0) / (a + a0 + b + b0), 0, 1)

    # todo 相对俯仰角允许一个小平台，相对方位角缺少一个“角度过小”的惩罚，alpha的作用还不是很清楚

    return output


def draw_onedim(x_range, spacing):
    """
    绘制一维奖励函数

    参数:
    x_range: [x下界, x上界]
    spacing: 间距
    """
    # 创建x轴数据点
    x = np.arange(x_range[0], x_range[1], spacing)
    # 计算对应的y值（奖励值）
    y = onedim_reward(x)

    # 绘制曲线
    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.xlabel('X')
    plt.ylabel('Reward')
    plt.title('One-dimensional Reward Function')
    plt.grid(True)
    plt.show()


def draw_twodim(x_range, y_range, spacing):
    """
    绘制二维奖励函数

    参数:
    x_range: [x下界, x上界]
    y_range: [y下界, y上界]
    spacing: 间距
    """
    # 创建网格点
    x = np.arange(x_range[0], x_range[1], spacing[0])
    y = np.arange(y_range[0], y_range[1], spacing[1])
    X, Y = np.meshgrid(x, y)

    # 计算对应的Z值（奖励值）
    Z = twodim_rewward(X, Y)

    # 创建3D图形
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Reward')
    ax.set_title('Two-dimensional Reward Function')

    plt.show()
    return ax


if __name__ == "__main__":
    # draw_onedim([-10, 10], 0.1)
    ax = draw_twodim([-100, 100], [-100, 100], [1, 1])
