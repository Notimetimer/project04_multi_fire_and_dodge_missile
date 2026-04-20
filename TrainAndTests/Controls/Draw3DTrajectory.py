from math import *
import numpy as np
import matplotlib.pyplot as plt
from _context import *
from Visualize.plot_tools import *
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh
import trimesh.transformations as tf

# 导入自定义的模型变换函数
from Visualize.plot_F16_in_matplotlib import get_transformed_mesh
# 设置字体以支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def start_drawing(file_name, interval=30, model_scale=400, x_limits=None, y_limits=None, z_limits=None):
    csv_path = os.path.join(project_root, "logs", "control_test_results", file_name)
    if not os.path.exists(csv_path):
        print(f"找不到轨迹文件: {csv_path}，请检查文件名。")
        # 尝试列出目录下的文件以便调试
        print("可用文件列表:")
        print(os.listdir(os.path.join(project_root, "logs", "control_test_results")))
        exit()

    df = pd.read_csv(csv_path)

    # 如果 CSV 中包含 round 列，则自动过滤出第一轮的数据，防止轨迹重叠
    if 'round' in df.columns:
        df = df[df['round'] == df['round'].iloc[0]]

    # 提取数据
    time = df['time'].values
    uav_n = df['uav_N'].values
    uav_u = df['uav_U'].values
    uav_e = df['uav_E'].values
    uav_psi = df['uav_psi'].values * pi/180
    uav_theta = df['uav_theta'].values * pi/180
    uav_phi = df['uav_phi'].values * pi/180
    target_n = df['target_N'].values
    target_u = df['target_U'].values
    target_e = df['target_E'].values

    # 加载基础 3D 模型
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # obj_path = os.path.join(script_dir, "F-16.obj")
    obj_path = os.path.join(project_root,"Visualize", "F-16.obj")
    base_mesh = trimesh.load(obj_path)
    if hasattr(base_mesh, 'geometry'): 
        base_mesh = trimesh.util.concatenate(list(base_mesh.geometry.values()))

    # 绘制轨迹
    fig = plt.figure(1, figsize=(14, 10))
    ax3d = fig.add_subplot(projection='3d')

    # 1. 绘制连续的无人机轨迹 (红色)
    ax3d.plot(uav_e, uav_n, uav_u, c='r', label='UAV Trajectory', linewidth=2, alpha=0.6)
    # 2. 绘制目标轨迹 (已按需移除)
    # ax3d.plot(target_e, target_n, target_u, 'b--', label='Target Trajectory', linewidth=1, alpha=0.5)

    # 3. 每隔 30s 绘制姿态模型和对应指示
    # 定义模型缩放倍数，使得模型在公里级视野中可见

    for i in range(0, len(time)):
        # 每 30s 抽样一次，首位必取
        if i == 0 or (round(time[i], 2) % interval == 0 and time[i] != time[i-1]):
            u_p = np.array([uav_e[i], uav_n[i], uav_u[i]])
            t_p = np.array([target_n[i], target_u[i], target_e[i]]) # 注意输入给 get_transformed_mesh 的顺序是 N, U, E
            
            # 调用函数获取变换后的模型定点
            verts, faces = get_transformed_mesh(base_mesh, 
                                                uav_n[i], uav_u[i], uav_e[i], 
                                                uav_psi[i], uav_theta[i], uav_phi[i], 
                                                scale=model_scale)
            
            # 将物理 NUE 映射到绘图视野 ENU
            plot_verts = np.zeros_like(verts)
            plot_verts[:, 0] = -verts[:, 2]  # E -> Plot X
            plot_verts[:, 1] = -verts[:, 0]  # N -> Plot Y
            plot_verts[:, 2] = verts[:, 1]   # U -> Plot Z
            
            # 绘制模型实体
            poly3d = Poly3DCollection(plot_verts[faces], alpha=0.9, facecolor='red', edgecolor='darkred', linewidth=0.1)
            ax3d.add_collection3d(poly3d)
            
            # 绘制指向目标的线
            target_plot_p = np.array([target_e[i], target_n[i], target_u[i]])
            ax3d.plot([u_p[0], target_plot_p[0]], [u_p[1], target_plot_p[1]], [u_p[2], target_plot_p[2]], 
                    'b', alpha=0.6, linewidth=2)
            
            # 画个蓝色散点标出目标位置 (增加图例标签)
            ax3d.scatter(target_plot_p[0], target_plot_p[1], target_plot_p[2], c='b', s=10, label='Flight Cmd')
            
            # 标注时间点 (已按需移除)
            # ax3d.text(u_p[0], u_p[1], u_p[2], f"{int(time[i])}s", fontsize=9)

    import matplotlib.ticker as ticker
    # ax3d.set_zlim3d(0, 20000)
    set_axes_equal_manual(ax3d, x_limits=x_limits, y_limits=y_limits, z_limits=z_limits)

    # 设置坐标轴显示单位为 km (将原始米数值除以 1000)
    ax3d.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x/1000:.0f}'))
    ax3d.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x/1000:.0f}'))
    ax3d.zaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x/1000:.0f}'))

    ax3d.set_xlabel('East (km)')
    ax3d.set_ylabel('North (km)')
    ax3d.set_zlabel('Up (km)')

    # 限制图例：使用 unique labels 逻辑，并开启鼠标拖拽功能
    handles, labels = ax3d.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    leg = ax3d.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize='small')
    
    # 开启像 MATLAB 那样的图例可拖动功能，并防止拖动时背景跟着转
    if leg:
        leg.set_draggable(True)
        
        # 内部处理函数：当点击图例时，屏蔽掉 3D 背景的自动旋转
        def on_legend_pick(event):
            # 如果点中的是图例框或其子组件
            if event.artist == leg or event.artist == leg.get_frame():
                ax3d.mouse_init(rotate_btn=None, pan_btn=None, zoom_btn=None)

        def on_mouse_release(event):
            # 鼠标释放后，恢复正常的 3D 交互功能
            ax3d.mouse_init()

        # 将事件绑定到画布
        fig.canvas.mpl_connect('pick_event', on_legend_pick)
        fig.canvas.mpl_connect('button_release_event', on_mouse_release)
        
        # 必须显式给图例框设置 picker，否则 pick_event 不会对矩形框生效
        leg.get_frame().set_picker(True)

    plt.title(f"3D Trajectory & Attitude Visualization (per 30s)\nFile: {file_name}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 读取 CSV 文件
    current_dir = os.path.join(project_root, "TrainAndTests/Controls")

    # 使用最新生成的轨迹数据
    file_name = "FlightControl_parallel无课程无蒸馏_有过载限制_动态lr_wave__trajectory.csv"
    start_drawing(file_name, z_limits = (0, 15000))