import numpy as np

# language: python
def f(table, x, extrapolate=False):
    """
    线性插值/（可选）外推函数。
    输入:
      table: np.array shape (n,2) 表示 (x_i, y_i) 对，x_i 不必有序。
      x: 标量或一维数组，要求与 table 的 x 维相同类型（数值）。
      extrapolate: False -> 区间外使用端点值；True -> 使用两端相邻线性外推。
    返回:
      插值/外推后的 y（标量或 np.ndarray，与 x 的形状对应）
    """
    import numpy as np

    table = np.asarray(table)
    if table.ndim != 2 or table.shape[1] != 2:
        raise ValueError("table must be shape (n,2)")

    xs = table[:, 0].astype(float)
    ys = table[:, 1].astype(float)

    # 排序 x
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]

    x_arr = np.asarray(x)
    # 使用 np.interp 处理区间内插值（默认对区间外返回端点）
    y = np.interp(x_arr, xs, ys, left=ys[0], right=ys[-1])

    if extrapolate:
        # 左侧线性外推
        if xs.size >= 2:
            slope_left = (ys[1] - ys[0]) / (xs[1] - xs[0])
            slope_right = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
            left_mask = x_arr < xs[0]
            right_mask = x_arr > xs[-1]
            if left_mask.any():
                y[left_mask] = ys[0] + (x_arr[left_mask] - xs[0]) * slope_left
            if right_mask.any():
                y[right_mask] = ys[-1] + (x_arr[right_mask] - xs[-1]) * slope_right
        else:
            # 只有一个点，外推仍然返回该点的 y
            pass

    # 返回标量或数组与输入 x 一致
    if np.isscalar(x):
        return float(y)  # 返回标量
    else:
        return y