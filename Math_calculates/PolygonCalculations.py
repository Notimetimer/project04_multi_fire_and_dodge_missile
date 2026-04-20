import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, LineString

def calc_dist2polygon_border(vertices, test_pt, velocity):
    """
    纯计算函数：计算点到多边形边界的距离矢量(d_vec)和左右方位(side)
    vertices: 多边形顶点列表 (N, 2)
    test_pt: 待测点坐标 [x, y]
    velocity: 速度矢量 [vx, vy]
    """
    poly = Polygon(vertices)
    p = Point(test_pt)
    v = np.array(velocity, dtype=float)
    v_norm = np.linalg.norm(v)
    v_unit = v / (v_norm + 1e-6)
    
    is_inside = poly.contains(p)
    boundary = poly.boundary
    
    # 获取最近的边界点坐标
    nearest_p_coords = boundary.interpolate(boundary.project(p))
    nearest_p = np.array([nearest_p_coords.x, nearest_p_coords.y])
    
    # 判定左右方位 (利用指向最近边界点的最短矢量)
    d_vec_shortest = nearest_p - np.array(test_pt)
    cross_product = v[0] * d_vec_shortest[1] - v[1] * d_vec_shortest[0]
    side = -1 if cross_product > 0 else 1 if cross_product < 0 else 0

    # 计算沿速度方向到边界的矢量 (射线法)
    d_vec = np.zeros(2)
    if is_inside:
        minx, miny, maxx, maxy = poly.bounds
        max_dist = np.hypot(maxx - minx, maxy - miny) * 2  # 足够长的射线段
        ray_end = test_pt + v_unit * max_dist
        ray = LineString([test_pt, ray_end])
        intersection = boundary.intersection(ray)

        if not intersection.is_empty:
            if intersection.geom_type == 'Point':
                int_pt = np.array([intersection.x, intersection.y])
            else:
                pts = []
                if hasattr(intersection, 'geoms'):
                    for geom in intersection.geoms:
                        if geom.geom_type == 'Point':
                            pts.append([geom.x, geom.y])
                        elif geom.geom_type == 'LineString':
                            pts.extend(list(geom.coords))
                elif intersection.geom_type == 'LineString':
                    pts.extend(list(intersection.coords))
                
                if pts:
                    pts = np.array(pts)
                    dists = np.linalg.norm(pts - test_pt, axis=1)
                    # 避免微小数值误差导致取到起点自身（距离接近0）
                    valid_idx = dists > 1e-5
                    if np.any(valid_idx):
                        int_pt = pts[valid_idx][np.argmin(dists[valid_idx])]
                    else:
                        int_pt = pts[np.argmin(dists)]
                else:
                    int_pt = np.array([intersection.centroid.x, intersection.centroid.y])
            
            d_vec = int_pt - test_pt
        
    return d_vec, side

if __name__=="__main__":
    # 1. 示例数据
    poly_verts = np.array([[0, 0], [10, 0], [10, 10], [5, 4], [0, 10]])
    test_point = np.array([2,-1])
    velocity_vec = np.array([-0.4, -1])

    # 2. 调用核心计算接口
    d_vec, side = calc_dist2polygon_border(poly_verts, test_point, velocity_vec)
    print("到边界矢量为：", d_vec, "方向为：", side)

    # 3. 在 main 函数中实现独立可视化
    fig, ax = plt.subplots(figsize=(7, 7))
    poly = Polygon(poly_verts)
    x, y = poly.exterior.xy
    ax.plot(x, y, c='#6699cc', alpha=0.7, lw=3, label='Boundary')
    ax.fill(x, y, c='#6699cc', alpha=0.3)
    
    ax.scatter([test_point[0]], [test_point[1]], color='red', s=80, zorder=5, label='Point')
    ax.quiver(test_point[0], test_point[1], velocity_vec[0], velocity_vec[1], 
              color='green', angles='xy', scale_units='xy', scale=1, label='Velocity')
    
    if np.linalg.norm(d_vec) > 1e-6:
        intersect_p = test_point + d_vec
        ax.plot([test_point[0], intersect_p[0]], [test_point[1], intersect_p[1]], 
                'r:', lw=2, label=f'Vector to Border along V (Side: {side})')
        
    ax.set_title(f"Distance to Border (Calculated Vector)")
    ax.set_aspect('equal')
    plt.grid(True); ax.legend(); plt.show()