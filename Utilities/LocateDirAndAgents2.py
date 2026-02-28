import sys
import os
import re
import glob
import datetime

# --- 2. 辅助函数 ---
def get_latest_log_dir(pre_log_dir, mission_name):
    """在日志根目录中查找最新的一个训练文件夹（使用 datetime 比较更可靠）"""
    pattern = re.compile(rf"{re.escape(mission_name)}-run-(\d{{8}})-(\d{{6}})")
    latest_dir = None
    latest_dt = None
    for d in os.listdir(pre_log_dir):
        m = pattern.match(d)
        if not m:
            continue
        dt_str = m.group(1) + m.group(2)  # YYYYMMDD + HHMMSS -> YYYYMMDDHHMMSS
        try:
            dt = datetime.datetime.strptime(dt_str, "%Y%m%d%H%M%S")
        except ValueError:
            continue
        if latest_dt is None or dt > latest_dt:
            latest_dt = dt
            latest_dir = d
    return os.path.join(pre_log_dir, latest_dir) if latest_dir else None

def find_latest_agent_path(log_dir, agent_id=None, prefix="actor_rein"):
    """在指定的日志文件夹中查找智能体权重文件，prefix 默认为 'actor_rein'"""
    if agent_id is not None:
        path = os.path.join(log_dir, f"{prefix}{agent_id}.pt")
        return path if os.path.exists(path) else None

    search_pattern = os.path.join(log_dir, f"{prefix}*.pt")
    files = glob.glob(search_pattern)
    if not files:
        return None

    latest_file = None
    max_id = -1
    for f in files:
        match = re.search(rf'{re.escape(prefix)}(\d+)\.pt', os.path.basename(f))
        if match:
            current_id = int(match.group(1))
            if current_id > max_id:
                max_id = current_id
                latest_file = f
    return latest_file


# 修改 __main__ 测试调用：只让 load_actor_from_log 返回路径，实际加载留在 main
if __name__ =='__main__':
    import torch as th
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pre_log_dir = os.path.join(project_root, "logs/combat")
    log_dir = get_latest_log_dir(pre_log_dir, mission_name="ILRL_combat_打rule0带导弹")

    print("\n智能体所在目录：", log_dir)
    
    agent_path = find_latest_agent_path(log_dir, agent_id=None)
    
    print("\n智能体位置", agent_path, "\n")
    
    