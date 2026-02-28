import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator
from mpl_toolkits.mplot3d import Axes3D

# ----------------------------
# 1. 原始散点数据
# ----------------------------
x = np.array([-60, 50, 50, 50, 60, -180, 180, 50, 50, -180, -180, 180, 180, -60,-60])
y = np.array([0, -30, 0, 30, 0, 0, 0, -90, 90, 90,-90, 90,-90, -90, 90])
z = np.array([-1, -1, 1, -1, -1, -5, -5, -5, -5, -5,-5,-5,-5, -5, -5])

# ----------------------------
# 2. 建立线性插值器
# ----------------------------
# LinearNDInterpolator 会对不在凸包内的点返回 NaN
# 我们将 NaN 替换为线性外插值（用最近点外推）
interp = LinearNDInterpolator(list(zip(x, y)), z, fill_value=np.nan)

# ----------------------------
# 3. 构建绘制网格
# ----------------------------
x_lin = np.linspace(min(x) - 20, max(x) + 20, 100)
y_lin = np.linspace(min(y) - 20, max(y) + 20, 100)
X, Y = np.meshgrid(x_lin, y_lin)

Z = interp(X, Y)

# 简单的线性外插：对 NaN 区域用最近点值填充
mask = np.isnan(Z)
if np.any(mask):
    from scipy.spatial import cKDTree
    points = np.vstack((x, y)).T
    tree = cKDTree(points)
    # 找每个网格点最近的原始点
    nearest_dist, nearest_idx = tree.query(np.vstack((X[mask], Y[mask])).T)
    Z[mask] = z[nearest_idx]

# ----------------------------
# 4. 绘图
# ----------------------------
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# 绘制曲面
surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none')

# 绘制原始散点
ax.scatter(x, y, z, color='r', s=50, label='Data Points')

# ----------------------------
# 5. 图形设置
# ----------------------------
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('线性插值/外插生成的曲面')
ax.legend()
fig.colorbar(surf, shrink=0.5, aspect=10)

plt.tight_layout()
plt.show()
