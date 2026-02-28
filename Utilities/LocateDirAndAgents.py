import sys
import os

import re
import glob



# 找出日期最大的目录
def get_latest_log_dir(pre_log_dir, mission_name=None):
    # 匹配 run-YYYYMMDD-HHMMSS 目录
    # pattern = re.compile(r"run-(\d{8})-(\d{6})")
    if mission_name:
        pattern = re.compile(rf"{re.escape(mission_name)}-run-(\d{{8}})-(\d{{6}})")
    else:
        pattern = re.compile(r"run-(\d{8})-(\d{6})")
    max_dt = None
    latest_dir = None
    for d in os.listdir(pre_log_dir):
        m = pattern.match(d)
        if m:
            dt_str = m.group(1) + m.group(2)  # 'YYYYMMDDHHMMSS'
            if max_dt is None or dt_str > max_dt:
                max_dt = dt_str
                latest_dir = d
    if latest_dir:
        return os.path.join(pre_log_dir, latest_dir)
    else:
        return None

# 新增：按数字编号选择 actor 文件（只返回路径，不加载）
def load_actor_from_log(log_dir, number=None, rein_prefix="actor_rein", sup_prefix="actor_sup"):
    """
    在 log_dir 中查找 actor 权重文件并返回选定的文件路径（不执行加载）。
    参数:
      log_dir: 日志目录路径
      number: 若指定整数，则优先查找尾号等于该数的 checkpoint（rein 或 sup）
              若 None，则选择编号最大的 rein（存在则）否则选择编号最大的 sup。
      rein_prefix / sup_prefix: 文件名前缀（默认 actor_rein / actor_sup）
    返回: 选中的文件全路径（str），未找到则返回 None
    """
    if not log_dir or not os.path.isdir(log_dir):
        return None

    pattern_rein = re.compile(rf"{re.escape(rein_prefix)}(\d+)\.pt$")
    pattern_sup = re.compile(rf"{re.escape(sup_prefix)}(\d+)\.pt$")

    rein_files = glob.glob(os.path.join(log_dir, f"{rein_prefix}*.pt"))
    sup_files = glob.glob(os.path.join(log_dir, f"{sup_prefix}*.pt"))

    if number is not None:
        # 优先在 rein 中找指定编号，再在 sup 中找
        for p in rein_files:
            m = pattern_rein.search(os.path.basename(p))
            if m and int(m.group(1)) == int(number):
                return p
        for p in sup_files:
            m = pattern_sup.search(os.path.basename(p))
            if m and int(m.group(1)) == int(number):
                return p
        return None

    # 未指定编号：选编号最大的 rein，否则最大的 sup
    rein_cands = []
    for p in rein_files:
        m = pattern_rein.search(os.path.basename(p))
        if m:
            rein_cands.append((int(m.group(1)), p))
    if rein_cands:
        return max(rein_cands, key=lambda x: x[0])[1]

    sup_cands = []
    for p in sup_files:
        m = pattern_sup.search(os.path.basename(p))
        if m:
            sup_cands.append((int(m.group(1)), p))
    if sup_cands:
        return max(sup_cands, key=lambda x: x[0])[1]

    return None


# 修改 __main__ 测试调用：只让 load_actor_from_log 返回路径，实际加载留在 main
if __name__ =='__main__':
    import torch as th
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from Trains.PPOAttackTrain import *

    # 测试训练效果
    agent = PPOContinuous(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                    lmbda, epochs, eps, gamma, device)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pre_log_dir = os.path.join(project_root, "logs")
    log_dir = get_latest_log_dir(pre_log_dir, mission_name=mission_name)

    # 只查找路径（示例：强制查找 1000 尾号）
    actor_path = load_actor_from_log(log_dir, number=1000)
    if not actor_path:
        print(f"No actor checkpoint found in {log_dir}")
    else:
        sd = th.load(actor_path, map_location=device, weights_only=True)
        agent.actor.load_state_dict(sd)
        print(f"Loaded actor for test from: {actor_path}")
