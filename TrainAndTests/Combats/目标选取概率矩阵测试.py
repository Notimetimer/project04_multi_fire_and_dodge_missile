import os
import glob
import re
import numpy as np
import torch
import argparse
from math import pi
from _context import *

from Envs.Tasks.ChooseStrategyEnv_terminate_and_reward import ChooseStrategyEnv
from Utilities.LocateDirAndAgents2 import get_latest_log_dir, find_latest_agent_path

# ======================= 可配置参数区 =======================
# 模型来源：MAPPO 同构训练产出的 ActorWrapper（dict 观测：fixed/enm/ally）
from Algorithms.ActorWithATTN import ActorWrapper

DIR_NAME = 'MAPPO_homo_SLA_1_blackboard_1-run-20261104-105610'

EXPERIMENT_NAME = None

# 抽取进度为几%的actor参数 / 弹药量（本测试只跑单点，不遍历）
PROGRESS = 100  # select_progress_percentage (%)
AMMO = 4        # 初始弹药量（训练 red_init_ammo=4）

DEVICE = 'cpu'

# 场地半径：与开火概率脚本一致，放大到 105km 保证设定距离不被 reset 裁剪
R_CAGE = 105e3
BLACKBOARD_VISABLE = 1 # 黑板是否可见
# ---------- 红方 id=1 朝蓝方 id=201 发射一枚 AIM-120 ----------
FIRE_RED_ID, FIRE_BLUE_ID = 1, 201
# ===========================================================


def create_initial_states():
    # 飞机出生状态指定（照 BasicRules_Homo.py 的对头场景：红向东、蓝向西，相距 110km）
    blue_height = 9000
    red_height = 9000
    red_psi = pi/2
    blue_psi = - red_psi
    red_N = 0
    red_E = - 20e3
    blue_N = red_N
    blue_E = -red_E
    DEFAULT_RED_BIRTH_STATE = {'position': np.array([red_N, red_height, red_E]),
                        'psi': red_psi
                        }
    DEFAULT_BLUE_BIRTH_STATE = {'position': np.array([blue_N, blue_height, blue_E]),
                                'psi': blue_psi
                                }
    return DEFAULT_RED_BIRTH_STATE, DEFAULT_BLUE_BIRTH_STATE


def select_agent_by_progress(log_dir, percentage):
    """
    扫描目录中 actor_rein*.pt，按编号排序，返回最接近指定进度百分比的文件。
    进度按最大编号计算：target = max_step * percentage / 100，取编号差最小的文件。
    """
    files = glob.glob(os.path.join(log_dir, "actor_rein*.pt"))
    step_files = []
    for f in files:
        m = re.fullmatch(r'actor_rein(\d+(?:\.\d+)?)\.pt', os.path.basename(f))
        if m:
            step_files.append((float(m.group(1)), f))
    if not step_files:
        return None
    step_files.sort(key=lambda x: x[0])
    target = step_files[-1][0] * percentage / 100.0
    step, best = min(step_files, key=lambda x: abs(x[0] - target))
    print(f"选择进度 {percentage}%: 目标步数 {target:.0f}, 选中 {os.path.basename(best)}")
    return best


def load_trained_actor(model_path, device='cpu'):
    """加载训练好的actor模型"""
    # MAPPO 同构训练的 actor_rein{batch}.pt 由 torch.save(actor.state_dict()) 落盘，
    # 网络维度统一从 env_obs_dims.json 读取（与训练 worker 的 ActorWrapper 完全一致）
    fixed_dim, enm_shape, ally_shape = load_obs_dims()
    actor = ActorWrapper(fixed_dim=fixed_dim, enm_shape=enm_shape,
                         ally_shape=ally_shape, device=device)

    # 加载权重
    actor.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    actor.eval()

    return actor


def get_enm_selection_probs(actor, env, red_id, device='cpu', enm_chosen_vec=None):
    """
    取单架红机对所有蓝机槽位的目标选取概率行。

    做法：遍历每个有效（存活可见）enm 槽位，把该槽位作为 enm_idx 传入 selector
    独立做一次前向，读取返回的 log_prob_enm（= log_softmax(enm_scores)[该槽位]），
    最后 exp 还原为概率；被 mask 的槽位保持 0。
    与训练时 evaluate_actions 的 log_prob 口径完全一致，无需改动 ActorWithATTN。

    enm_chosen_vec: 主循环风格的黑板向量（在途导弹目标槽位置 1），None 时用全零占位。
    """
    agent_input, mask_input, rule_input = env.get_obs(red_id, pomdp=1)
    # 决策前把“已被在途导弹选中的敌机”写进本机观测（与 VsBaseline_MA 主循环同口径）
    if enm_chosen_vec is not None:
        agent_input['enm_chosen'] = enm_chosen_vec
    actor_rule = stack_field([take_some_of_rule_inputs(rule_input)])

    n_enm_slots = len(mask_input['enm'])
    log_probs = np.full(n_enm_slots, -np.inf)  # 死蓝机槽位保持 -inf -> exp 后为 0

    with torch.no_grad():
        packed_agent, packed_mask = actor.pack_inputs(agent_input, mask_input)
        for slot in range(n_enm_slots):
            if mask_input['enm'][slot] <= 0.5:
                continue
            enm_idx = torch.as_tensor([slot], dtype=torch.long, device=device)
            _, _, _, log_prob_enm, _ = actor.selector(
                packed_agent, packed_mask, actor_rule, enm_idx=enm_idx
            )
            log_probs[slot] = log_prob_enm.item()

    probs = np.exp(log_probs)
    total = probs.sum()
    if total > 0:
        probs = probs / total  # 数值兜底归一（理论上 exp(log_softmax) 行和已为 1）
    return probs, log_probs


def compute_prob_matrix(actor, env, red_alive_ids, n_enm_slots, enm_chosen_of=None, device='cpu'):
    """逐架红机取概率行，拼成矩阵；enm_chosen_of: {red_id: enm_chosen_vec}，缺省全零"""
    prob_matrix = np.zeros((len(red_alive_ids), n_enm_slots))
    for row, red_id in enumerate(red_alive_ids):
        vec = enm_chosen_of.get(red_id) if enm_chosen_of else None
        prob_matrix[row], _ = get_enm_selection_probs(actor, env, red_id, device=device,
                                                      enm_chosen_vec=vec)
    return prob_matrix


def print_prob_matrix(title, prob_matrix, red_alive_ids, blue_uavs):
    """打印矩阵：行=红方存活机（agent），列=蓝机槽位（规则机）"""
    blue_ids = [u.id for u in blue_uavs]
    print(f"\n========== {title} ==========")
    col_labels = [f"id={u.id}{'(活)' if not u.dead else '(死)'}" for u in blue_uavs]
    print("红机\\蓝机 | " + " | ".join(f"{c:^12}" for c in col_labels))
    print("-" * (13 + 15 * len(blue_ids)))
    for red_id, row in zip(red_alive_ids, prob_matrix):
        argmax_slot = int(np.argmax(row))
        print(f"红机 id={red_id:<3d}| " + " | ".join(f"{p:^12.4f}" for p in row) +
              f"  | argmax -> 蓝机 id={blue_ids[argmax_slot]}, 行和={row.sum():.4f}")


def build_enm_chosen_vec(env, red_id, n_enm_slots):
    """照 VsBaseline_MA 主循环：把本侧在途导弹的目标槽位写入黑板向量"""
    vec = np.zeros(n_enm_slots)
    for msl in env.alive_r_missiles:
        chosen_i = env.id2index(red_id, msl.target_id)
        if chosen_i is not None and 0 <= chosen_i < n_enm_slots:
            vec[chosen_i] = 1
    return vec


def main():
    # 查找日志目录
    logs_root_dir = os.path.join(project_root, "logs/combat")
    latest_log_dir = os.path.join(logs_root_dir, DIR_NAME) if DIR_NAME else \
        get_latest_log_dir(logs_root_dir, EXPERIMENT_NAME)
    if not latest_log_dir:
        raise FileNotFoundError(f"No log directory found for mission '{EXPERIMENT_NAME}'")

    # 加载模型
    agent_path = select_agent_by_progress(latest_log_dir, PROGRESS)
    if not agent_path:
        raise FileNotFoundError(f"'{latest_log_dir}' 中没有 actor_rein 文件")
    print(f"找到模型: {agent_path}")
    actor = load_trained_actor(agent_path, device=DEVICE)
    print("模型加载成功")

    # 环境只建一次，单点 reset（初始数量/存活数量与训练一致）
    init_states = load_init_states()
    env_args = argparse.Namespace(max_episode_len=15*60, R_cage=R_CAGE)
    env = ChooseStrategyEnv(env_args, tacview_show=False, vertices=None)
    env.dt_move = 0.07
    env.dt_maneuver = 0.2

    # random_pos=False：关闭队内出生点顺序打乱，保证相对几何与设定一致
    red_state, blue_state = create_initial_states()
    env.reset(red_birth_state=red_state, blue_birth_state=blue_state,
              red_init_num=init_states['red_init_num'],
              blue_init_num=init_states['blue_init_num'],
              red_init_ammo=AMMO, blue_init_ammo=AMMO,
              seed=(0, 0), options={"random_pos": False})

    # 刷新可见性：reset 时可见性矩阵仍为全 1（被当作“仅 RWR 方位可见”），
    # 先按当前几何算一次原始可见性，保证观测到的是真实几何
    env.update_entity_states()

    # 蓝机槽位顺序与 get_state 中 enm_infos 的顺序一致（env.BUAVs 列表序）
    blue_uavs = env.BUAVs
    n_enm_slots = len(blue_uavs)
    red_alive_ids = [i for i in env.RUAV_ids if not env.find_uav_with_id(i).dead]

    # ---------- 决策前：无在途导弹，enm_chosen 全零 ----------
    mat_before = compute_prob_matrix(actor, env, red_alive_ids, n_enm_slots, device=DEVICE)
    print_prob_matrix("决策前（enm_chosen 全零）", mat_before, red_alive_ids, blue_uavs)

    
    # ---- 执行假设的开火（没有step，只创生新的导弹） ----
    # 开火 id 校验：红机须在存活红机中、蓝机须在存活蓝机中，任一越界则不放行
    blue_alive_ids = [u.id for u in blue_uavs if not u.dead]
    if FIRE_RED_ID in red_alive_ids and FIRE_BLUE_ID in blue_alive_ids:
        msl = env.launch_120(FIRE_RED_ID, FIRE_BLUE_ID)
        if msl is not None:
            # 刷新存活列表（让新弹进入 alive_r_missiles）与可见性
            env.update_entity_states()
            print(f"\n已发射：红机 id={FIRE_RED_ID} -> 蓝机 id={FIRE_BLUE_ID}，"
                  f"导弹 id={msl.id}，红{FIRE_RED_ID} 剩余弹药={env.find_uav_with_id(FIRE_RED_ID).ammo}")
        else:
            print(f"\n发射未执行：红{FIRE_RED_ID} 无弹药或已阵亡")
    else:
        print(f"\n开火 id 越界，未发射：红{FIRE_RED_ID}（存活红机={red_alive_ids}），"
              f"蓝{FIRE_BLUE_ID}（存活蓝机={blue_alive_ids}）")

    # ---------- 决策后：按主循环口径给每架红机填 enm_chosen，再算矩阵 ----------
    # BLACKBOARD_VISABLE=0 时黑板不可见：不填 enm_chosen，保持 get_obs 的全零占位
    if BLACKBOARD_VISABLE:
        enm_chosen_of = {red_id: build_enm_chosen_vec(env, red_id, n_enm_slots)
                         for red_id in red_alive_ids}
    else:
        enm_chosen_of = None
    mat_after = compute_prob_matrix(actor, env, red_alive_ids, n_enm_slots,
                                    enm_chosen_of=enm_chosen_of, device=DEVICE)
    print_prob_matrix(f"决策后（黑板{'可见，标记 '+str(FIRE_BLUE_ID)+' 已被导弹选中' if BLACKBOARD_VISABLE else '不可见'}）",
                      mat_after, red_alive_ids, blue_uavs)

    # ---------- 前后差值 ----------
    print_prob_matrix("差值（后 - 前）", mat_after - mat_before, red_alive_ids, blue_uavs)


if __name__ == '__main__':
    main()
