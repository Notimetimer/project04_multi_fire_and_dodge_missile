import numpy as np
# 处理观测由dict转为nparray的问题

def flatten_obs1(obs: dict, keys: list):
    """
    高效展平 obs（keys 指定顺序）为一维 np.array（dtype=float32）。
    不使用 parts = parts + arr，改为收集小数组后一次性 np.concatenate。
    """
    parts = []
    for k in keys:
        v = obs[k]
        if np.isscalar(v):
            parts.append(np.array([float(v)], dtype=np.float32))
        elif isinstance(v, (list, tuple, np.ndarray)):
            parts.append(np.asarray(v, dtype=np.float32).reshape(-1))
        else:
            # 保险 fallback：尝试转为数组
            parts.append(np.asarray(v, dtype=np.float32).reshape(-1))
    if len(parts) == 0:
        return np.array([], dtype=np.float32)
    return np.concatenate(parts, axis=0)


def flatten_obs2(obs: dict, keys: list):
    parts = []
    for k in keys:
        v = obs[k]
        # 把标量 / list / ndarray 都转换为一维 ndarray
        arr = np.atleast_1d(np.asarray(v, dtype=np.float32))
        parts.append(arr)
    if not parts:
        return np.array([], dtype=np.float32)
    return np.concatenate(parts, axis=0)