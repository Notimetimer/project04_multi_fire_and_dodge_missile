import numpy as np
# elo相似度优先
def simular_proritized(elo_ratings, target_elo=None, sigma=400):
    """返回与 elo_ratings.keys() 顺序对应的概率数组。
    - target_elo: 要优先靠近的 ELO（传入 main_agent_elo）。
    - sigma: 高斯核标准差，越小越只选接近 target_elo 的对手。
    """
    keys = list(elo_ratings.keys())
    if len(keys) == 0:
        return np.array([]), keys # 返回 keys 以便索引
    elos = np.array([elo_ratings[k] for k in keys], dtype=np.float64)

    if target_elo is None:
        target_elo = np.mean(elos)

    # 以高斯核度量相似度（基于差的平方）
    diffs = elos - float(target_elo)
    # 数值稳定性：将 exponent 的常数项减去 max
    scores = np.exp(-0.5 * (diffs / float(sigma))**2)
    probs = scores / (scores.sum() + 1e-12)
    return probs, keys # 修改为返回 probs 和 keys


# 高手优先
def higher_proritized(elo_ratings, target_elo=None, sigma=400):
    keys = list(elo_ratings.keys())
    if len(keys) == 0:
        return np.array([]), keys # 返回 keys 以便索引
    elos = np.array([elo_ratings[k] for k in keys], dtype=np.float64)

    if target_elo is None:
        target_elo = np.mean(elos)

    # 以高斯核度量相似度（基于差的平方）
    diffs = elos - float(target_elo)
    # 数值稳定性：将 exponent 的常数项减去 max
    scores = np.exp(-0.5 * (diffs / float(sigma))**2)
    probs = scores / (scores.sum() + 1e-12)
    return probs, keys # 修改为返回 probs 和 keys

