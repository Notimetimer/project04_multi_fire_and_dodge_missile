import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import numpy as np

def show_trajectory(red_pos_list, blue_pos_list, min_east, min_north, min_height, max_east, max_north, max_height, r_show=1, b_show=1):
    red_p_show_show = np.array(red_pos_list).T
    blue_p_show_show = np.array(blue_pos_list).T

    fig = plt.figure(2)
    ax3d = fig.add_subplot(projection='3d')
    if r_show == 1:
        ax3d.plot(red_p_show_show[2], red_p_show_show[0], red_p_show_show[1], c='r', label='MissileTrack')
        ax3d.scatter(red_p_show_show[2][0], red_p_show_show[0][0], red_p_show_show[1][0], c='r', s=5)
    if b_show == 1:
        ax3d.plot(blue_p_show_show[2], blue_p_show_show[0], blue_p_show_show[1], c='b', label='TargetTrack')
        ax3d.scatter(blue_p_show_show[2][0], blue_p_show_show[0][0], blue_p_show_show[1][0], c='b', s=5)

    # 绘制箭头
    startpoint = np.array([min_east, min_north, min_height])  # 起点
    endpoint_E = np.array([max_east, min_north, min_height])
    endpoint_N = np.array([min_east, max_north, min_height])
    endpoint_U = np.array([min_east, min_north, max_height])

    # 计算方向向量
    direction_E = endpoint_E - startpoint
    direction_N = endpoint_N - startpoint
    direction_U = endpoint_U - startpoint

    # 绘制箭头
    class Arrow3D(FancyArrowPatch):
        def __init__(self, xs, ys, zs, *args, **kwargs):
            super().__init__((0, 0), (0, 0), *args, **kwargs)
            self._verts3d = xs, ys, zs

        def do_3d_projection(self, renderer=None):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            return min(zs)

    # 绘制箭头
    arrow1 = Arrow3D([startpoint[0], endpoint_E[0]],
                     [startpoint[1], endpoint_E[1]],
                     [startpoint[2], endpoint_E[2]],
                     mutation_scale=20, arrowstyle="-|>", color="b")
    arrow2 = Arrow3D([startpoint[0], endpoint_N[0]],
                     [startpoint[1], endpoint_N[1]],
                     [startpoint[2], endpoint_N[2]],
                     mutation_scale=20, arrowstyle="-|>", color="r")
    arrow3 = Arrow3D([startpoint[0], endpoint_U[0]],
                     [startpoint[1], endpoint_U[1]],
                     [startpoint[2], endpoint_U[2]],
                     mutation_scale=20, arrowstyle="-|>", color="g")
    ax3d.add_artist(arrow1)
    ax3d.add_artist(arrow2)
    ax3d.add_artist(arrow3)

    # 在终点处添加带颜色的文本
    ax3d.text(startpoint[0], startpoint[1], startpoint[2], 'O', color='black')
    ax3d.text(endpoint_E[0], endpoint_E[1], endpoint_E[2], 'E', color='blue')
    ax3d.text(endpoint_N[0], endpoint_N[1], endpoint_N[2], 'N', color='red')
    ax3d.text(endpoint_U[0], endpoint_U[1], endpoint_U[2], 'U', color='green')

    # 设置坐标轴等比例
    def set_axes(ax):
        """确保3D图的坐标轴单位长度相等。"""
        x_range = abs(max_east - min_east)
        y_range = abs(max_north - min_north)
        z_range = abs(max_height - min_height)
        ax.set_xlim3d([min_east, max_east])
        ax.set_ylim3d([min_north, max_north])
        ax.set_zlim3d([min_height, max_height])
        # 设置等显示缩放比例
        ax.set_box_aspect([x_range, y_range, z_range])

    # 设置坐标轴标签
    # ax3d.set_xlabel('E')
    # ax3d.set_ylabel('N')
    # ax3d.set_zlabel('U')

    # ax3d.grid(False)

    # 调用函数设置等比例坐标轴
    set_axes(ax3d)

    # 定义滚轮事件处理函数
    def on_scroll(event):
        ax3d = event.inaxes
        if ax3d is not None:
            # 获取当前坐标轴的限制
            xlim = ax3d.get_xlim()
            ylim = ax3d.get_ylim()
            zlim = ax3d.get_zlim()
            # 缩放因子
            scale_factor = 1 / 1.1 if event.button == 'up' else 1.1
            # 计算新的坐标轴限制
            new_xlim = [(xlim[0] + xlim[1]) / 2 - (xlim[1] - xlim[0]) / 2 * scale_factor,
                        (xlim[0] + xlim[1]) / 2 + (xlim[1] - xlim[0]) / 2 * scale_factor]
            new_ylim = [(ylim[0] + ylim[1]) / 2 - (ylim[1] - ylim[0]) / 2 * scale_factor,
                        (ylim[0] + ylim[1]) / 2 + (ylim[1] - ylim[0]) / 2 * scale_factor]
            new_zlim = [(zlim[0] + zlim[1]) / 2 - (zlim[1] - zlim[0]) / 2 * scale_factor,
                        (zlim[0] + zlim[1]) / 2 + (zlim[1] - zlim[0]) / 2 * scale_factor]
            # 设置新的坐标轴限制
            ax3d.set_xlim(new_xlim)
            ax3d.set_ylim(new_ylim)
            ax3d.set_zlim(new_zlim)
            # 重绘图形
            fig.canvas.draw_idle()

    # 绑定滚轮事件
    fig.canvas.mpl_connect('scroll_event', on_scroll)

    plt.show()  # block=False  不阻塞运行