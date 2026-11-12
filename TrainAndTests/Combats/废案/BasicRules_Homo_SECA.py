# -*- coding: utf-8 -*-
"""
BasicRules_Homo_SECA.py
=======================

以 ``TrainAndTests/Combats/BasicRules_Homo.py``（commit 236f832，2026-11-07，
“【环境层面大变动】拆除action_cycle_multiplier层级”）为骨架，复现

    Hou 等, 2023, "Hierarchical decision-making framework for multiple UCAVs
    autonomous confrontation", IEEE TVT.
    （状态机见该文 Sec. IV / Table III~VII，参数见 Table IX）

中的 SECA（State-Event-Condition-Action）空战规则状态机。

本文件是**新增文件**，不修改、不覆盖、不重命名任何现有文件；原有
``BasicRules_Homo.py``、``Envs/`` 与统一评测代码保持只读。

--------------------------------------------------------------------------
一、场景裁剪：全部为 HV-UCAV
--------------------------------------------------------------------------
论文的 SECA 同时容纳 HV-UCAV（高价值，跟踪/防御/发射）与 LV-UCAV
（低价值，U 形回转/协同制导）。按任务要求，本场景内所有飞机都是 HV-UCAV，
因此凡是以 LV-UCAV 为实体、或以“被分配 LV 目标/协同制导任务”为触发源的
状态、事件、条件、动作与规则**全部删除，不保留占位**。

保留（HV-only）：
    状态 Q : q1 编队飞行、q2 跟踪 HV-UCAV、q4 防御拖拽
    事件 E : e1 越过中间线、e2 分配到 HV 目标、e4 与敌距离 < D_A、e5 被导弹攻击
    条件 C : c1 剩余弹量 > 0、c3 小于机动距离
    动作 A : a1 执行编队航路点、a2 跟踪 HV 目标、a3 发射导弹、a5 相对视线转 ±90°

删除（13 条 LV 相关规则）：
    R3  (q1, e3 -> q3)                LV 目标分配
    R6  (q2, e3,c2 -> q3)             LV 目标分配
    R8  (q2, e5,c3 -> q5)             LV 规避
    R9  (q2, e6,c4 -> q6)             协同制导任务
    R10 (q3, - -> a4,q3)              跟踪 LV
    R11 (q3, e4,c1 -> a3,q2)          跟踪 LV 时发射
    R12 (q3, e5,c3 -> q4)             源状态 q3 在 HV-only 下不存在
    R13 (q3, e5,c3 -> q5)             LV 规避
    R14 (q3, e6,c4 -> q6)             协同制导
    R17 (q5, - -> a6,q5)              LV U 形回转
    R18 (q5, ¬e5 -> q2)               LV 退出规避
    R19 (q6, ¬e7,c4 -> a7,q6)         LV 协同制导
    R20 (q6, ¬e7,c5 -> q2)            LV 协同制导结束

保留的 7 条规则（严格对应 Table VII 的 Q/E/C/A/Q'/Entity 六列）：

    R1 : q1, --,   --, a1, q1   （默认动作：编队航路点）
    R2 : q1, e1∨e2, --, --, q2   （无动作，落到 q2 默认动作 a2）
    R4 : q2, --,   --, a2, q2   （默认动作：跟踪 HV 目标）
    R5 : q2, e4,   c1, a3, q2   （进入 D_A 且有余弹 -> 发射）
    R7 : q2, e5,   c3, --, q4   （被导弹攻击且进入机动距离 -> 防御）
    R15: q4, --,   --, a5, q4   （默认动作：相对视线 ±90° 拖拽）
    R16: q4, ¬e5,  --, --, q2   （威胁消失 -> 回到跟踪）

事件语义：SECA 的事件是“发生”而非“当前为真”（论文 Sec. IV：“state
transition is only triggered by events rather than conditions”），因此 e1/e4
按上升沿触发；e5 是告警出现，按上升沿统计，但 R16 的 ¬e5 用告警电平判断。
e4 的距离取“当前被分配/跟踪的 HV 目标”，c3 的距离取“最近的在告警导弹”，
与 Table VII 中 e4/c3 的指向一致。R2/R7/R16 在 Table VII 中动作列为 “-”，
本实现按“同一决策步内落到目标状态的默认动作”处理（a2/a5/a2），不额外跨步。
q2 内若 R5（可发射）与 R7（须防御）在同一步同时成立，先记录发射，再以 a5
作为本步飞行动作（安全优先），与 Table VII 的规则排列顺序一致。

--------------------------------------------------------------------------
二、与 6DOF 环境（Envs/BattleHomo.py -> ChooseStrategyEnv）的换算
--------------------------------------------------------------------------
调用链（已确认）：
    ``ChooseStrategyEnv_terminate_and_reward.ChooseStrategyEnv``（BattleHomo）
    由 ``TrainAndTests/Combats/MA_rulerule_test.py``、
    ``CombatMAPPO_parall_homo.py``、``CombatHanSAC_parall_homo.py`` 等调用；
    这些脚本按 ``basic_rules(rule_inputs[id], mask_inputs[id], rules_num)``
    逐机调用。``Envs/Tasks/battle6dof1v1_missile0309_hierarchical.py`` 走的是
    ``BasicRules_new_hierarchical`` 的数组接口，不用本文件。

坐标与单位：
    论文 3DOF 状态 (x, y, z, ψ, γ, φ) 与项目 6DOF 的对应关系：
    水平面 (x, y) <-> [N, E]（北、东，m）；高度 z <-> h（m）；
    偏航 ψ 同义（与 JSBSim psi-true 一致：0=正北, π/2=正东，绕天向顺时针）；
    航迹倾角 γ <-> theta（rad，相对水平面）；滚转角 φ 同义。
    本文件只使用“相对距离/相对方位角/相对俯仰角”，不依赖绝对 ψ，
    因此在 ego 航向系（x=前向, y=天向, z=右向）内重建几何即可。

    论文            项目（6DOF，位置 [N, h, E]，单位 m；角度 rad；速度 m/s）
    D_A-HV 55 km    ``D_attack_hv   = 55e3``
    D_M-HV 60 km    ``D_maneuver_hv = 60e3``（c3 机动距离）
    D_D-HV 80 km    ``D_detect_hv   = 80e3``（雷达发现距离）
    D_HV-HV 100 km  ``D_track_hv_hv= 100e3``（动态目标分配候选半径）
    N_HV = 4        ``N_track_per_target = 4``（单目标最多被 4 架 HV 同时分配）
    β_HV-HV = 1     ``beta_hv_hv = 1.0``（威胁度 = β/d）
    V_max 400 m/s   环境锁定 340 m/s（UAVModel/maneuver_for_1_enm 内部指令），
                    不属于规则可调项，故不在此处建模。
    Z_max 15 km     环境初始高度 9 km、min_alt_safe=2 km，规则不做额外高度约束。

动作映射（论文动作 -> 环境 [action_v, action_h]，取值见
``ChooseStrategyEnv_action_spaces.maneuver_for_1_enm``）：
    a1 执行编队航路点 -> [1, 7] 平飞直飞（项目无航路点接口；红蓝初始航向
       相对，直飞即沿编队预定方向进入交战区，见“差异说明”）
    a2 跟踪 HV 目标   -> [1, 0] 平飞 + 指向目标（+1° 高度差，保持锁定）
    a3 发射导弹       -> 飞行仍按 a2 跟踪，另发 fire_action.do=1（fire_120）
    a5 相对视线 ±90°  -> [1, 2] 或 [1, 4]；h=2 的航向指令为
       ``delta_psi - 90°``，h=4 为 ``delta_psi + 90°``。为避免“朝威胁转”，
       选与威胁方位角异号的一侧：威胁方位 >0 用 h=2，否则 h=4。
       两者都能把威胁压在正/负 90° 波束上（见 maneuver_for_1_enm 第 392-399 行）。

事件 e1 “越过中间线”：
    rule_inputs 不提供绝对坐标与地图中线，只有相对几何。若调用方传入
    ``ego_pos=(N,E)`` 与 ``midline={'point':(N,E),'normal':(N,E)}``，则用
    几何判据 ``dot(ego_pos-point, normal) > 0``；否则退化为
    “最近存活敌机距离 ≤ D_D-HV（80 km，进入对方雷达/交战区）”。
    注意：本项目初始间距 110 km、目标分配半径 100 km，因此 e2 会先于 e1
    触发，e1 主要作为无目标分配时的兜底入口。

--------------------------------------------------------------------------
三、相对基线的差异（其余接口/张量化风格/环境调用方式保持一致）
--------------------------------------------------------------------------
1. 基线 ``basic_rules`` 是**无记忆**的编号策略（rules_num=0..5），本文件是
   **有记忆**的每机状态机。状态按唯一 ``agent_id`` 存在模块级字典中，必须在
   每回合 ``env.reset`` 前调用 ``reset_seca()``（本文件 __main__ 已示范）。
2. 基线只挑“最近活敌”并固定 70 km 开火；本文件按论文 Table IX 用
   β/d 匈牙利分配（D_HV-HV=100 km、单目标容量 4），开火距离 D_A=55 km。
3. 基线用 45 km 触发逃跑、h=3 置尾/俯冲；本文件严格用 e5（RWR 告警）+
   c3（威胁距离 < 60 km）进 q4，动作只用 a5 的 ±90° 波束拖拽，不额外俯冲。
   注意：本项目 aim120 的 RWR 探测距离为 15 km（MissileModel_aim120.
   detect_range），小于 D_M-HV=60 km，因此实际对局中 c3 随 e5 恒真；这是
   环境探测模型的限制，规则本身仍按 Table VII 判定并在调试信息中记录 c3。
4. 论文未给出目标分配周期 τ 的数值，本文件以“决策步”计
   （``tau_steps=5``，默认 dt_maneuver=2 s 时约 10 s），可通过 params 覆盖。
5. 论文 Table VII 的 c1 只检查余弹；但论文 Sec. II-A 要求发射机在导弹
   中制导段持续照射目标，且本项目 ``fire_120`` 没有冷却，因此新增
   “发射通道保护”：仅当本机最新一枚导弹已飞行 ≥40 s 且不在中制导时才允许
   再发（与基线 ``t_fired>=40 and not on_guiding`` 完全一致）。该保护不是
   新增条件列，而是 c1 的物理可执行性展开，已在 get_seca_debug 中记录。
6. e1 的几何/兜底映射、a1 的直飞映射见上节；这些是环境能力差异，不是规则改动。
7. 观测里没有队友 id，无法保证所有友机按同一行序运行匈牙利；本实现按
   “相对本机的队友顺序”构造代价矩阵。除对称并列外，各机解一致；容量 4
   在 3v3 下通常不紧，并列概率极低。

--------------------------------------------------------------------------
四、运行与验证
--------------------------------------------------------------------------
快速纯逻辑自检（不依赖 jsbsim，几秒完成）：
    python TrainAndTests/Combats/BasicRules_Homo_SECA.py --self-test

真实 6DOF 环境对抗（红蓝双方都用 SECA，逐决策步打印状态/规则/事件）：
    conda run -n 38 python TrainAndTests/Combats/BasicRules_Homo_SECA.py \
        --episodes 2 --dt-maneuver 2 --max-time 300

自检会断言：q1 编队 -> e2 分配后 q2 跟踪 -> e4 上升沿发射一次（重复步不再
发射）-> e5&c3 进 q4 且动作为 90° 波束 -> 告警消失 R16 回 q2 -> 双目标时
分配较近者。真实环境运行会在每回合结束打印每机状态驻留、规则命中、事件计数
与前若干次状态转移（含时间、规则、事件、距离），可据此逐条对照 Table VII。
"""

import argparse
import math
import os
import sys

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - 环境无 scipy 时走确定性贪心
    linear_sum_assignment = None


# ---------------------------------------------------------------------------
# 参数（Table IX，HV 列；距离统一换算为 m）
# ---------------------------------------------------------------------------
SECA_PARAMS = {
    "beta_hv_hv": 1.0,           # β_HV-HV
    "D_track_hv_hv": 100.0e3,    # D_HV-HV：动态目标分配候选半径
    "D_attack_hv": 55.0e3,       # D_A-HV：e4 开火距离
    "D_maneuver_hv": 60.0e3,     # D_M-HV：c3 机动距离
    "D_detect_hv": 80.0e3,       # D_D-HV：e1 兜底用的雷达发现距离
    "N_track_per_target": 4,     # N_HV：单目标最大分配数
    "tau_steps": 5,              # 论文未给 τ，按决策步计（2 s/步 ≈ 10 s）
    "fire_min_flight": 40.0,     # 发射通道保护，与基线 t_fired>=40 对齐
    "enable_launch_guard": True,
}

STATE_Q1 = "q1"   # 编队飞行
STATE_Q2 = "q2"   # 跟踪 HV-UCAV
STATE_Q4 = "q4"   # 防御拖拽

_RULE_NAMES = ("R1", "R2", "R4", "R5", "R7", "R15", "R16")

# 每架飞机的状态机记忆：key 必须是环境内唯一 id（红 1..N，蓝 201..）
_AGENTS = {}


def _new_agent_state():
    return {
        "state": STATE_Q1,
        "assigned_idx": None,
        "decision_count": 0,
        "alloc_step": -10 ** 9,
        "e4_prev": False,
        "warn_prev": False,
        "state_steps": {STATE_Q1: 0, STATE_Q2: 0, STATE_Q4: 0},
        "rule_counts": {name: 0 for name in _RULE_NAMES},
        "event_counts": {"e1": 0, "e2": 0, "e4": 0, "e5": 0},
        "transitions": [],
        "fires": [],
        "last_t": 0.0,
        "last_dist": float("inf"),
        "last_threat_dist": float("inf"),
        "last_rule": None,
        "last_rules_fired": [],
        "last_events": {},
        "last_action": None,
        "last_fire": 0,
        "launch_blocked": 0,
    }


def reset_seca(agent_id=None):
    """清空状态机记忆。每回合 env.reset 前必须调用；agent_id=None 表示全部。"""
    if agent_id is None:
        _AGENTS.clear()
    else:
        _AGENTS.pop(agent_id, None)


def get_seca_debug(agent_id):
    """返回某架飞机的状态机快照（只读用途，勿修改内部字段）。"""
    return _AGENTS.get(agent_id)


def _agent_state(agent_id):
    if agent_id not in _AGENTS:
        _AGENTS[agent_id] = _new_agent_state()
    return _AGENTS[agent_id]


# ---------------------------------------------------------------------------
# 观测解析（兼容 rule_inputs 中标量或长度 1 序列两种形态）
# ---------------------------------------------------------------------------
def _flat(value):
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _scalar(value, default=0.0):
    arr = _flat(value)
    return float(arr[0]) if arr.size else float(default)


def _as_dict_list(value):
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    return list(value)


def _info_vec(info, ego_theta, prefix):
    """把 (距离, 相对方位角, 相对俯仰角) 还原到以本机航向为 +x 的右手系。

    距离不变性：任意两机之间的距离只依赖各自在本机坐标系中的相对矢量，
    因此不需要绝对 psi。x=前向, y=天向, z=右向。
    """
    dist = _scalar(info.get(prefix + "_dist"))
    dpsi = math.atan2(
        _scalar(info.get("sin_" + prefix + "_delta_psi")),
        _scalar(info.get("cos_" + prefix + "_delta_psi")),
    )
    dtheta = _scalar(info.get(prefix + "_delta_theta"))
    q_eps = ego_theta + dtheta
    horiz = dist * math.cos(q_eps)
    return np.array([
        horiz * math.cos(dpsi),
        dist * math.sin(q_eps),
        horiz * math.sin(dpsi),
    ], dtype=np.float64)


def _parse_inputs(rule_inputs, mask_inputs):
    ego_main = rule_inputs.get("ego_main", {})
    sin_theta = float(np.clip(_scalar(ego_main.get("sin_theta")), -1.0, 1.0))
    ego_theta = math.asin(sin_theta)

    friend_vecs = [np.zeros(3)]  # 下标 0 恒为本机
    ally_infos = _as_dict_list(rule_inputs.get("ally_infos", []))
    ally_mask = _flat(mask_inputs.get("ally", []))
    for k, info in enumerate(ally_infos):
        if k < ally_mask.size and ally_mask[k] <= 0.5:
            continue
        friend_vecs.append(_info_vec(info, ego_theta, "ally"))

    enm_infos = _as_dict_list(rule_inputs.get("enm_infos", []))
    enm_mask = _flat(mask_inputs.get("enm", []))
    enm_alive, enm_vecs, enm_dist, enm_dtheta, enm_dpsi = [], [], [], [], []
    for k, info in enumerate(enm_infos):
        alive = bool(enm_mask[k] > 0.5) if k < enm_mask.size else True
        enm_alive.append(alive)
        enm_dist.append(_scalar(info.get("enm_dist")))
        enm_dtheta.append(_scalar(info.get("enm_delta_theta")))
        enm_dpsi.append(math.atan2(
            _scalar(info.get("sin_enm_delta_psi")),
            _scalar(info.get("cos_enm_delta_psi")),
        ))
        enm_vecs.append(_info_vec(info, ego_theta, "enm") if alive else None)

    rwr = rule_inputs.get("RWR_infos", {})
    warn_arr = _flat(rwr.get("warning", 0.0))
    warning = bool(np.any(warn_arr > 0.5)) if warn_arr.size else False
    if warning:
        td = _flat(rwr.get("threat_distance", 999e3))
        active = td[warn_arr > 0.5] if td.size == warn_arr.size else td
        threat_dist = float(np.min(active)) if active.size else 999e3
        threat_dpsi = math.atan2(
            _scalar(rwr.get("sin_threat_delta_psi")),
            _scalar(rwr.get("cos_threat_delta_psi")),
        )
    else:
        threat_dist = 999e3
        threat_dpsi = 0.0

    weapon = rule_inputs.get("weapon_infos", {})
    return {
        "friend_vecs": friend_vecs,
        "enm_alive": enm_alive,
        "enm_vecs": enm_vecs,
        "enm_dist": enm_dist,
        "enm_dtheta": enm_dtheta,
        "enm_dpsi": enm_dpsi,
        "warning": warning,
        "threat_dist": threat_dist,
        "threat_dpsi": threat_dpsi,
        "ammo": _scalar(rule_inputs.get("ego_ammo", 0.0)),
        "weapon_t_fly": _scalar(weapon.get("missile_t_fly", 120.0), 120.0),
        "weapon_mid_term": _scalar(weapon.get("mid_term", 0.0)),
    }


# ---------------------------------------------------------------------------
# 动态目标分配（论文 Sec. IV-B：β/d 匈牙利，候选半径 D_HV-HV，容量 N_HV）
# ---------------------------------------------------------------------------
def _dynamic_target_allocation(parsed, cfg):
    alive = [j for j, ok in enumerate(parsed["enm_alive"]) if ok]
    if not alive:
        return None

    n_friend = len(parsed["friend_vecs"])
    n_enm = len(alive)
    utility = np.zeros((n_friend, n_enm), dtype=np.float64)
    for a, j in enumerate(alive):
        for f in range(n_friend):
            d = float(np.linalg.norm(parsed["friend_vecs"][f] - parsed["enm_vecs"][j]))
            if 1e-6 < d <= cfg["D_track_hv_hv"]:
                utility[f, a] = cfg["beta_hv_hv"] / d

    capacity = max(1, int(cfg["N_track_per_target"]))

    if linear_sum_assignment is not None:
        # 每个目标复制 capacity 个槽位；再补 n_friend 个“不分配”哑槽位，
        # 保证矩形/不可行情形都能求解，只有 utility>0 的槽位才算真分配。
        n_slot = n_enm * capacity
        mat = np.zeros((n_friend, n_slot + n_friend), dtype=np.float64)
        for a in range(n_enm):
            for k in range(capacity):
                mat[:, a * capacity + k] = utility[:, a]
        rows, cols = linear_sum_assignment(-mat)
        for r, c in zip(rows, cols):
            if r != 0:
                continue
            if c < n_slot and mat[r, c] > 0.0:
                return alive[c // capacity]
            return None
        return None

    # 无 scipy 时的确定性贪心（按 β/d 降序，同样受容量约束）
    pairs = sorted(
        [(utility[f, a], f, a) for f in range(n_friend) for a in range(n_enm) if utility[f, a] > 0],
        key=lambda item: (-item[0], item[1], item[2]),
    )
    used_f, used_cnt, assign = set(), [0] * n_enm, {}
    for _, f, a in pairs:
        if f in used_f or used_cnt[a] >= capacity:
            continue
        used_f.add(f)
        used_cnt[a] += 1
        assign[f] = a
    return alive[assign[0]] if 0 in assign else None


def _launch_channel_free(parsed, cfg):
    """论文 Sec. II-A：发射机需持续照射中制导目标；同一时刻只引导一枚。

    与基线 ``t_fired >= 40 and not on_guiding`` 一致。
    """
    if not cfg["enable_launch_guard"]:
        return True
    return (parsed["weapon_t_fly"] >= cfg["fire_min_flight"]
            and parsed["weapon_mid_term"] < 0.5)


def _beam_action(threat_dpsi):
    """a5：相对视线转 ±90°。

    h=2 的航向指令为 delta_psi-90°，h=4 为 delta_psi+90°。选与威胁方位
    异号的一侧，保证是“背离威胁转向”而不是朝威胁压过去。
    """
    return [1, 2 if threat_dpsi > 0 else 4]


# ---------------------------------------------------------------------------
# SECA 主入口：接口与 basic_rules(rule_inputs, mask_inputs, rules_num, p_random) 对齐
# ---------------------------------------------------------------------------
def basic_rules_seca(rule_inputs, mask_inputs, rules_num=None, p_random=0,
                     agent_id=0, params=None, ego_pos=None, midline=None,
                     sim_time=None):
    """单机 SECA 决策。

    Parameters
    ----------
    rule_inputs, mask_inputs : dict
        与 ``basic_rules`` 完全一致的 env.get_obs 输出。
    rules_num, p_random :
        仅为兼容基线调用签名保留，SECA 不使用编号策略/随机动作。
    agent_id : int/str
        环境内唯一 id（红 1..N、蓝 201..），用于保存每机状态机记忆。
    params : dict | None
        覆盖 ``SECA_PARAMS`` 中的任意键。
    ego_pos : (N, E) | None
        可选绝对水平位置，用于 e1 的真实中线判据。
    midline : dict | None
        ``{'point': (N, E), 'normal': (N, E)}``，normal 指向“敌方半区”。
    sim_time : float | None
        仅用于调试记录。

    Returns
    -------
    (flight_action, fire_action, radar_action)
        与 ``basic_rules`` 返回值同构。
    """
    cfg = dict(SECA_PARAMS)
    if params:
        cfg.update(params)

    st = _agent_state(agent_id)
    ego_alive = _scalar(mask_inputs.get("ego", 1), 1.0) > 0.5
    if not ego_alive:
        return {"do": [1, 7], "at": 0}, {"do": 0, "at": 0}, 1

    parsed = _parse_inputs(rule_inputs, mask_inputs)
    st["decision_count"] += 1
    if sim_time is not None:
        st["last_t"] = float(sim_time)

    # ---------------- 1. 动态目标分配（R2 的 e2 来源） ----------------
    target_dead = (
        st["assigned_idx"] is not None
        and (st["assigned_idx"] >= len(parsed["enm_alive"])
             or not parsed["enm_alive"][st["assigned_idx"]])
    )
    alloc_due = (st["decision_count"] - st["alloc_step"]) >= int(cfg["tau_steps"])
    prev_assigned = st["assigned_idx"]
    alloc_result = None
    alloc_ran = False
    if st["assigned_idx"] is None or target_dead or alloc_due:
        alloc_ran = True
        alloc_result = _dynamic_target_allocation(parsed, cfg)
        st["alloc_step"] = st["decision_count"]
        if alloc_result is not None:
            if st["assigned_idx"] != alloc_result:
                st["e4_prev"] = False  # 新目标视为新交战，重新允许 e4 上升沿
            st["assigned_idx"] = alloc_result
        elif target_dead:
            st["assigned_idx"] = None
        # 否则保持粘性目标：目标逃出 100 km 仍继续跟踪，直到死亡

    assigned = st["assigned_idx"]
    target_ok = assigned is not None and assigned < len(parsed["enm_alive"]) \
        and parsed["enm_alive"][assigned]
    dist_target = parsed["enm_dist"][assigned] if target_ok else float("inf")

    # ---------------- 2. 事件与条件 ----------------
    if ego_pos is not None and midline is not None:
        pt = midline["point"]
        nv = midline["normal"]
        e1 = ((float(ego_pos[0]) - pt[0]) * nv[0]
              + (float(ego_pos[1]) - pt[1]) * nv[1]) > 0.0
    else:
        alive_dist = [parsed["enm_dist"][j] for j, ok in enumerate(parsed["enm_alive"]) if ok]
        e1 = bool(alive_dist) and min(alive_dist) <= cfg["D_detect_hv"]

    # e2 是“分配到（新）HV 目标”这一发生型事件，仅在分配结果出现/改变时触发，
    # 不能把 q2 内每 τ 步的重复分配都记成一次事件。
    e2 = bool(alloc_ran and alloc_result is not None and alloc_result != prev_assigned)

    e4_now = target_ok and dist_target < cfg["D_attack_hv"]
    if st["state"] == STATE_Q2:
        e4_edge = e4_now and not st["e4_prev"]
        st["e4_prev"] = e4_now
    else:
        e4_edge = False
        st["e4_prev"] = False  # 进入 q2 的第一步重新起沿

    e5 = parsed["warning"]
    c1 = parsed["ammo"] > 0
    c3 = parsed["threat_dist"] < cfg["D_maneuver_hv"]
    channel_free = _launch_channel_free(parsed, cfg)

    # ---------------- 3. HV-only SECA 状态转移 ----------------
    prev_state = st["state"]
    new_state = prev_state
    rules_fired = []
    action = [1, 7]
    fire = 0

    if prev_state == STATE_Q1:
        if e1 or e2:
            new_state = STATE_Q2
            rules_fired.append("R2")
            action = [1, 0]  # R2 无动作，落到 q2 默认 a2
        else:
            rules_fired.append("R1")
            action = [1, 7]  # a1 编队航路点（本项目=平飞直飞）

    elif prev_state == STATE_Q2:
        rules_fired.append("R4")
        action = [1, 0]  # a2 跟踪 HV

        if e4_edge and c1 and target_ok:
            if channel_free:
                rules_fired.append("R5")
                fire = 1  # a3 发射，飞行仍按 a2 跟踪
            else:
                st["launch_blocked"] += 1

        if e5 and c3:
            rules_fired.append("R7")
            new_state = STATE_Q4
            action = _beam_action(parsed["threat_dpsi"])  # 同一步执行 q4 默认 a5

    else:  # STATE_Q4
        if e5:
            rules_fired.append("R15")
            action = _beam_action(parsed["threat_dpsi"])
        else:
            rules_fired.append("R16")
            new_state = STATE_Q2
            action = [1, 0]  # R16 无动作，落到 q2 默认 a2
            st["e4_prev"] = False  # 退出防御视为重新交战

    st["state"] = new_state

    # ---------------- 4. 统计与调试 ----------------
    st["state_steps"][new_state] += 1
    for name in rules_fired:
        st["rule_counts"][name] += 1
    st["event_counts"]["e1"] += int(e1)
    st["event_counts"]["e2"] += int(e2)
    st["event_counts"]["e4"] += int(e4_edge)
    st["event_counts"]["e5"] += int(e5 and not st["warn_prev"])
    st["warn_prev"] = e5
    st["last_dist"] = dist_target
    st["last_threat_dist"] = parsed["threat_dist"]
    st["last_rule"] = rules_fired[-1] if rules_fired else None
    st["last_rules_fired"] = rules_fired
    st["last_events"] = {
        "e1": bool(e1), "e2": bool(e2),
        "e4": bool(e4_now), "e4_edge": bool(e4_edge), "e5": bool(e5),
        "c1": bool(c1), "c3": bool(c3), "channel_free": bool(channel_free),
    }
    st["last_action"] = list(action)
    st["last_fire"] = int(fire)
    if fire:
        st["fires"].append({
            "t": st["last_t"],
            "target": int(assigned) if target_ok else 0,
            "dist": dist_target,
        })

    if new_state != prev_state:
        st["transitions"].append({
            "t": st["last_t"],
            "from": prev_state,
            "to": new_state,
            "rules": list(rules_fired),
            "events": dict(st["last_events"]),
            "dist": dist_target,
            "threat_dist": parsed["threat_dist"],
        })
        if len(st["transitions"]) > 300:
            st["transitions"].pop(0)

    target_index = int(assigned) if target_ok else 0
    flight_action = {"do": [int(action[0]), int(action[1])], "at": target_index}
    fire_action = {"do": int(fire), "at": target_index}
    return flight_action, fire_action, 1


# ---------------------------------------------------------------------------
# 纯逻辑自检：用合成 rule_inputs 驱动状态机
# ---------------------------------------------------------------------------
def _synth_inputs(dists, warning=0, threat_dist=999e3, threat_dpsi=0.0,
                  ammo=4, t_fly=120.0, mid_term=0.0, ego_theta=0.0):
    """dists: list[float]，每架敌机到本机的距离；方位角统一 0（正前方）。"""
    enm_infos = []
    for d in dists:
        enm_infos.append({
            "enm_dist": float(d),
            "enm_delta_theta": 0.0,
            "sin_enm_delta_psi": 0.0,
            "cos_enm_delta_psi": 1.0,
            "enm_ATA": 0.0,
            "cos_enm_AA": 1.0,
            "enm_locked": 1,
        })
    return {
        "ego_main": {"v_ego": 340.0, "h_ego": 9000.0, "sin_theta": math.sin(ego_theta)},
        "border": {"d_hor2border": 50e3, "border_LR": 0, "dist_from_center": 50e3},
        "ally_infos": [],
        "enm_infos": enm_infos,
        "weapon_infos": {"missile_t_fly": t_fly, "mid_term": mid_term,
                         "cos_msl_delta_psi": 1.0},
        "RWR_infos": {"warning": warning,
                      "sin_threat_delta_psi": math.sin(threat_dpsi),
                      "cos_threat_delta_psi": math.cos(threat_dpsi),
                      "threat_distance": threat_dist},
        "ego_ammo": ammo,
    }


def _synth_mask(n_enm=1, n_ally=0):
    return {"ego": 1,
            "ally": np.ones(n_ally),
            "enm": np.ones(n_enm)}


def _run_self_test():
    print("=" * 78)
    print("BasicRules_Homo_SECA 纯逻辑自检（不启动 jsbsim）")
    print("=" * 78)
    ok = True

    def check(name, cond, info):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name}: {info}")

    params = {"tau_steps": 1}
    agent = "selftest_A"
    reset_seca()

    def step(dists, **kw):
        return basic_rules_seca(_synth_inputs(dists, **kw), _synth_mask(len(dists)),
                                agent_id=agent, params=params)

    # 1) 110 km：无分配 -> q1 编队直飞
    fly, fire, _ = step([110e3])
    dbg = get_seca_debug(agent)
    check("R1 q1 编队飞行", dbg["state"] == STATE_Q1 and fly["do"] == [1, 7] and not fire["do"],
          f"state={dbg['state']} action={fly['do']} fire={fire['do']}")

    # 2) 95 km：落入 D_HV-HV=100 km -> e2 -> q2 跟踪
    fly, fire, _ = step([95e3])
    dbg = get_seca_debug(agent)
    check("R2 e2 分配 HV 目标 -> q2", dbg["state"] == STATE_Q2 and fly["do"] == [1, 0],
          f"state={dbg['state']} action={fly['do']} target={fly['at']}")

    # 3) 仍 95 km：q2 默认跟踪，不开火
    fly, fire, _ = step([95e3])
    check("R4 q2 默认跟踪且不发射", not fire["do"], f"fire={fire['do']}")

    # 4) 54 km：e4 上升沿 + c1 -> R5 发射一次
    fly, fire, _ = step([54e3])
    dbg = get_seca_debug(agent)
    check("R5 e4&c1 发射", bool(fire["do"]) and dbg["state"] == STATE_Q2,
          f"state={dbg['state']} fire={fire['do']} rules={dbg['last_rules_fired']}")

    # 5) 仍 54 km：事件非电平，不应连发
    fly, fire, _ = step([54e3])
    check("e4 非上升沿不连发", not fire["do"], f"fire={fire['do']}")

    # 6) 告警出现且威胁 10 km < D_M=60 km -> R7 进 q4，a5 转 90°
    fly, fire, _ = step([54e3], warning=1, threat_dist=10e3, threat_dpsi=0.3)
    dbg = get_seca_debug(agent)
    check("R7 e5&c3 -> q4 且 a5 为 90° 波束",
          dbg["state"] == STATE_Q4 and fly["do"] == [1, 2],
          f"state={dbg['state']} action={fly['do']}")

    # 7) 告警消失 -> R16 回 q2 跟踪
    fly, fire, _ = step([54e3])
    dbg = get_seca_debug(agent)
    check("R16 ¬e5 -> q2", dbg["state"] == STATE_Q2 and fly["do"] == [1, 0],
          f"state={dbg['state']} action={fly['do']}")

    # 8) 双目标：应分配较近的 1 号槽位
    reset_seca()
    fly, fire, _ = basic_rules_seca(_synth_inputs([120e3, 60e3]), _synth_mask(2),
                                    agent_id="selftest_B", params=params)
    dbg = get_seca_debug("selftest_B")
    check("β/d 分配较近目标", dbg["assigned_idx"] == 1 and fly["at"] == 1,
          f"assigned={dbg['assigned_idx']} at={fly['at']} state={dbg['state']}")

    # 9) 余弹为 0：e4 也不发射
    reset_seca()
    basic_rules_seca(_synth_inputs([95e3]), _synth_mask(1), agent_id="selftest_C", params=params)
    fly, fire, _ = basic_rules_seca(_synth_inputs([54e3], ammo=0), _synth_mask(1),
                                    agent_id="selftest_C", params=params)
    check("c1 无余弹不发射", not fire["do"], f"fire={fire['do']}")

    # 10) 中制导占用：e4 上升沿但通道忙 -> 不发射
    reset_seca()
    basic_rules_seca(_synth_inputs([95e3]), _synth_mask(1), agent_id="selftest_D", params=params)
    fly, fire, _ = basic_rules_seca(
        _synth_inputs([54e3], t_fly=10.0, mid_term=1.0), _synth_mask(1),
        agent_id="selftest_D", params=params)
    dbg = get_seca_debug("selftest_D")
    check("发射通道占用不发射", not fire["do"] and dbg["launch_blocked"] >= 1,
          f"fire={fire['do']} blocked={dbg['launch_blocked']}")

    print("-" * 78)
    print("自检结果：", "全部通过" if ok else "存在失败项")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# 真实环境运行（红蓝双方均使用 SECA）
# ---------------------------------------------------------------------------
def _print_episode_stats(env, episode):
    print("\n" + "=" * 78)
    print(f"[SECA] 第 {episode} 回合结束  t={env.t:.1f}s  "
          f"红存活={sum(not u.dead for u in env.RUAVs)}/{env.Rnum}  "
          f"蓝存活={sum(not u.dead for u in env.BUAVs)}/{env.Bnum}")
    for uav in env.UAVs:
        dbg = get_seca_debug(uav.id)
        if dbg is None:
            continue
        total = sum(dbg["state_steps"].values()) or 1
        state_txt = " ".join(f"{k}={dbg['state_steps'][k]}({dbg['state_steps'][k]/total:.0%})"
                             for k in (STATE_Q1, STATE_Q2, STATE_Q4))
        rule_txt = " ".join(f"{k}={v}" for k, v in dbg["rule_counts"].items() if v)
        event_txt = " ".join(f"{k}={v}" for k, v in dbg["event_counts"].items() if v)
        print(f"  id={uav.id:<4d} side={uav.side} final={dbg['state']:<2s} | {state_txt}")
        print(f"       rules: {rule_txt or '-'} | events: {event_txt or '-'}")
        if dbg["fires"]:
            shoot_txt = ", ".join(f"t={f['t']:.0f}s@{f['dist']/1e3:.1f}km" for f in dbg["fires"])
            print(f"       R5 launches: {shoot_txt}")
        for tr in dbg["transitions"][:6]:
            ev = [k for k, v in tr["events"].items() if v]
            d_txt = "-" if not np.isfinite(tr["dist"]) else f"{tr['dist']/1e3:.1f}km"
            dt_txt = "-" if not np.isfinite(tr["threat_dist"]) else f"{tr['threat_dist']/1e3:.1f}km"
            print(f"       t={tr['t']:7.1f}s  {tr['from']}->{tr['to']}  "
                  f"rule={','.join(tr['rules'])}  event={','.join(ev) or '-'}  "
                  f"d={d_txt}  d_threat={dt_txt}")
        if len(dbg["transitions"]) > 6:
            print(f"       ... 另有 {len(dbg['transitions']) - 6} 次转移")
    print("=" * 78 + "\n")


def _run_env(args):
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    combats_dir = os.path.dirname(os.path.abspath(__file__))
    for path in (combats_dir, project_root):
        if path not in sys.path:
            sys.path.insert(0, path)

    import random
    import torch
    from Envs.Tasks.ChooseStrategyEnv_terminate_and_reward import ChooseStrategyEnv
    from _context import load_init_states

    init_cfg = load_init_states()
    vertices = init_cfg["vertices"]

    env_args = argparse.Namespace()
    env_args.max_episode_len = args.max_time
    env_args.R_cage = 60.3e3
    env = ChooseStrategyEnv(env_args, tacview_show=2, vertices=vertices)
    env.dt_move = args.dt_move

    red_birth = {"position": np.array([0.0, 9000.0, -55e3]), "psi": math.pi / 2}
    blue_birth = {"position": np.array([0.0, 9000.0, 55e3]), "psi": -math.pi / 2}

    mid_point = np.array([
        (red_birth["position"][0] + blue_birth["position"][0]) / 2.0,
        (red_birth["position"][2] + blue_birth["position"][2]) / 2.0,
    ])
    axis = np.array([
        blue_birth["position"][0] - red_birth["position"][0],
        blue_birth["position"][2] - red_birth["position"][2],
    ])
    axis = axis / (np.linalg.norm(axis) + 1e-12)

    try:
        for episode in range(args.episodes):
            seed = args.seed + episode
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            reset_seca()
            env.reset(
                red_birth_state=red_birth,
                blue_birth_state=blue_birth,
                red_init_num=init_cfg["red_init_num"],
                blue_init_num=init_cfg["blue_init_num"],
                red_init_ammo=init_cfg["red_init_ammo"],
                blue_init_ammo=init_cfg["blue_init_ammo"],
                seed=seed,
            )
            env.dt_maneuver = args.dt_maneuver
            env.min_alt_safe = 2000

            fly_action_all = {uid: {"do": [1, 7], "at": 0} for uid in env.UAV_ids}
            done = False

            for _ in range(int(round(args.max_time / args.dt_maneuver))):
                if not env.running or done:
                    break

                rule_inputs_, mask_inputs_ = {}, {}
                for uid in env.UAV_ids:
                    _, mask_input, rule_input = env.get_obs(uid, pomdp=1)
                    rule_inputs_[uid] = rule_input
                    mask_inputs_[uid] = mask_input

                for uid in env.UAV_ids:
                    uav = env.find_uav_with_id(uid)
                    # 红机越线方向为红->蓝，蓝机相反
                    normal = axis if uav.side == "r" else -axis
                    fly, fire, _radar = basic_rules_seca(
                        rule_inputs_[uid], mask_inputs_[uid],
                        rules_num=0, agent_id=uid,
                        ego_pos=(uav.pos_[0], uav.pos_[2]),
                        midline={"point": mid_point, "normal": normal},
                        sim_time=env.t,
                    )
                    fly_action_all[uid] = fly
                    if fire["do"]:
                        try:
                            env.fire_120(uid, fire["at"], tabu=0)
                        except ValueError as exc:
                            print(f"[fire_120 警告] {exc}")
                    env.radar_action(uid, 1)

                flight_cmds = {}
                for uid in env.UAV_ids:
                    act = fly_action_all[uid]
                    flight_cmds[uid] = env.maneuver_for_1_enm(
                        uid, act["at"], act["do"], rule_inputs_[uid])

                env.step(flight_cmds.values())
                for uid in env.UAV_ids:
                    done_i, _ = env.combat_terminate_and_reward(uid, action_cycle_multiplier=1)
                    done = done or done_i

            _print_episode_stats(env, episode)
    finally:
        env.end_render()


def main():
    parser = argparse.ArgumentParser("BasicRules_Homo_SECA")
    parser.add_argument("--self-test", action="store_true", help="只跑纯逻辑状态机自检")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--dt-move", type=float, default=0.07)
    parser.add_argument("--dt-maneuver", type=float, default=2.0)
    parser.add_argument("--max-time", type=float, default=300.0, help="单回合仿真时长(s)")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if args.self_test:
        return _run_self_test()
    _run_env(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
