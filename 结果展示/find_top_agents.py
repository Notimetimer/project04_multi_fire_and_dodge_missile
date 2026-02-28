import os
import json

def find_top_agents(log_dir, num_needed=10):
    """
    在单个实验文件夹下搜索最优智能体列表
    策略：HoF(按Elo排序) > Elite(按Elo排序)
    """
    # 定义文件路径
    full_json_path = os.path.join(log_dir, "elo_ratings.json")
    elite_json_path = os.path.join(log_dir, "elite_elo_ratings.json")
    hof_json_path = os.path.join(log_dir, "hall_of_fame.json")

    # 1. 加载数据
    def load_json(p, default):
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                return json.load(f)
        return default

    global_elos = load_json(full_json_path, {})
    elite_pool = load_json(elite_json_path, {})
    hof_keys = load_json(hof_json_path, [])

    # 2. 提取并排序名人堂成员
    # 过滤掉可能存在的 Rule 键，只保留 Agent
    hof_agents = [k for k in hof_keys if not k.startswith("Rule")]
    # 按 global_elos 中的分值降序排列
    hof_sorted = sorted(hof_agents, key=lambda k: global_elos.get(k, 0), reverse=True)

    # 3. 提取并排序精英池成员
    elite_agents = [k for k in elite_pool.keys() if k.startswith("actor_rein")]
    elite_sorted = sorted(elite_agents, key=lambda k: elite_pool.get(k, 0), reverse=True)

    # 4. 组合列表并去重
    top_list = []
    seen = set()

    # 先加 HoF
    for agent in hof_sorted:
        if agent not in seen and len(top_list) < num_needed:
            top_list.append(agent)
            seen.add(agent)

    # 数量不够再加 Elite
    for agent in elite_sorted:
        if agent not in seen and len(top_list) < num_needed:
            top_list.append(agent)
            seen.add(agent)

    return top_list, global_elos

if __name__ == "__main__":
    # 待扫描的实验根目录
    experiments_root = "./logs/combat" 
    num_per_exp = 5  # 每个实验找几个最强的
    
    # 自动获取所有实验子文件夹
    if not os.path.exists(experiments_root):
        print(f"Directory not found: {experiments_root}")
    else:
        exp_folders = [os.path.join(experiments_root, d) for d in os.listdir(experiments_root) 
                       if os.path.isdir(os.path.join(experiments_root, d))]

        print(f"{'Experiment Folder':<50} | {'Top Agents Selected'}")
        print("-" * 100)

        for folder in exp_folders:
            candidates, elo_map = find_top_agents(folder, num_per_exp)
            
            if not candidates:
                continue
            
            # 格式化输出：名字(Elo)
            results = [f"{c}({elo_map.get(c, 0):.0f})" for c in candidates]
            print(f"{os.path.basename(folder):<50} | {', '.join(results)}")