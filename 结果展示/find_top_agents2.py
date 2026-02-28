import os
import json

def find_top_agents(log_dir, num_needed=10):
    """
    输入：程序仿真记录路径，需要寻找的智能体数量
    逻辑：优先从名人堂找（代表确定性测试全胜），数量不足再从精英Elo列表补充。
    """
    # 路径对齐
    full_json_path = os.path.join(log_dir, "elo_ratings.json")
    elite_json_path = os.path.join(log_dir, "elite_elo_ratings.json")
    hof_json_path = os.path.join(log_dir, "hall_of_fame.json")

    # 1. 加载所有池子数据
    def load_json_safe(p, default):
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                return json.load(f)
        return default

    # 包含所有 Agent 的最新 Elo
    global_elo_dict = load_json_safe(full_json_path, {})
    # 当前活跃的精英池
    elite_elo_ratings = load_json_safe(elite_json_path, {})
    # 达成全胜成就的名单
    hall_of_fame_keys = load_json_safe(hof_json_path, [])

    selected_agents = []
    seen_keys = set()

    # --- 策略 A: 优先检索名人堂 (HoF) ---
    # 这些是测试模式下的“满分选手”，按它们在全量表中的 Elo 排序
    hof_candidates = [k for k in hall_of_fame_keys if k in global_elo_dict]
    hof_sorted = sorted(hof_candidates, key=lambda k: global_elo_dict[k], reverse=True)

    for k in hof_sorted:
        if len(selected_agents) < num_needed:
            selected_agents.append(k)
            seen_keys.add(k)

    # --- 策略 B: 补充精英池 (Elite) ---
    # 如果 HoF 没填满，按精英池分数从高到低找
    elite_candidates = [k for k in elite_elo_ratings.keys() if k.startswith("actor_rein")]
    elite_sorted = sorted(elite_candidates, key=lambda k: elite_elo_ratings[k], reverse=True)

    for k in elite_sorted:
        if k not in seen_keys and len(selected_agents) < num_needed:
            selected_agents.append(k)
            seen_keys.add(k)

    return selected_agents

def main():
    # 示例：遍历多个实验文件夹
    logs_root = "./logs/combat" 
    target_count = 5 # 每个实验取前 5
    
    if not os.path.exists(logs_root):
        print(f"Path not found: {logs_root}")
        return

    exp_folders = [os.path.join(logs_root, d) for d in os.listdir(logs_root) 
                   if os.path.isdir(os.path.join(logs_root, d))]

    for folder in exp_folders:
        top_agents = find_top_agents(folder, target_count)
        print(f"\nExperiment: {os.path.basename(folder)}")
        if not top_agents:
            print("  No qualified agents found.")
            continue
            
        for idx, agent in enumerate(top_agents):
            source = "HoF" if "step" in agent else "Elite" # 简单通过 key 判定来源
            print(f"  [{idx+1}] {agent:<30} (Source: {source})")

if __name__ == "__main__":
    main()