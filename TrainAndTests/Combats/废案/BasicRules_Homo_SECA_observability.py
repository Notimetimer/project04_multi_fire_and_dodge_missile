# -*- coding: utf-8 -*-
"""
BasicRules_Homo_SECA_observability.py
=====================================

SECA 规则（``BasicRules_Homo_SECA.py``）的**可观测性配套脚本**。不修改任何
现有文件，只新增本脚本。默认只保留 ACMI 录制这一条主链路：

  1. ACMI 回放文件（Tacview 可直接打开，含全部飞机/导弹航迹与雷达波束）
  2. acmi_check.txt（文件大小、飞机/导弹对象数、时间帧数、移除事件数）
  3. run_info.json（参数、种子、git commit、产物路径）

调试产物改为显式开启（默认关闭）：

  --trace   每个决策步的 CSV + 定宽文本表 + events.csv + summary.txt
            （状态、规则、事件、距离、余弹、动作、机动指令、导弹状态）
  --plots   四张静态图（PNG，matplotlib Agg，无弹窗）：
              trajectory.png     水平面 N-E 航迹，标出起点/发射点/转防御点
              distance_time.png  目标距离与威胁距离曲线，叠加 D_A/D_M 阈值
              state_timeline.png 每架飞机的 q1/q2/q4 状态时序
              events.png         发射/转防御/回跟踪/命中/击毁事件散点

ACMI 接线方式与项目现有做法一致（见 Visualize/tacview_visualize2.py 与
BasicRules_Homo.py 主循环）：
  - ``ChooseStrategyEnv(..., tacview_show=2)`` 构造时 ``Tacview_s.handshake()``
    打开 ``acmi_records/<YYYYmmdd_HHMMSS>.acmi`` 并写入文件头 + 战场边界；
  - 每个决策步（本项目 action_cycle_multiplier=1，dt_maneuver=2 s）在
    ``env.step`` 之后调用 ``env.render(t_bias)``，把所有 UAV（T=经|纬|高|
    滚转|俯仰|偏航, Name=F16, Color=Red/Blue）与所有导弹（T=..., Name=AIM-120C,
    Color=Orange/Green）以及雷达/锁定 Beam 写入当前时间帧；
  - 每回合结束 ``env.clear_render(t_bias)`` 写入 ``-id`` 移除行，随后
    ``t_bias += env.t``；全部回合结束 ``env.end_render()`` 关闭文件。
  - 项目没有单独的 ACMI Event 行：发射 = 新导弹对象出现；命中/击毁 =
    导弹或飞机被 ``-id`` 移除。为便于核查，本脚本另写 events.csv。

坐标/时间换算沿用 BattleHomo.render：内部 [N, h, E] 经 NUE2LLH 转经纬高，
时间轴为 ``env.t + t_bias``（秒），初始帧 ``#0.00``。

运行：
    conda run -n 38 python TrainAndTests/Combats/BasicRules_Homo_SECA_observability.py
    # 可选：--episodes 1 --max-time 220 --dt-maneuver 2 --seed 2026
    # 产物默认在 结果展示/SECA_observability/<run_id>/

注：ACMI 由环境按秒级时间戳命名并写入项目根目录 ``acmi_records/``，本脚本
只读取、校验并记录路径，不搬移该文件。
"""

import argparse
import datetime
import json
import math
import os
import re
import shutil
import subprocess
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from BasicRules_Homo_SECA import (
    SECA_PARAMS,
    basic_rules_seca,
    get_seca_debug,
    reset_seca,
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

STATE_COLORS = {"q1": "#9e9e9e", "q2": "#1e88e5", "q4": "#fb8c00", "dead": "#37474f"}
SIDE_COLORS = {"r": "#d62728", "b": "#1f77b4"}
SIDE_NAMES = {"r": "红方", "b": "蓝方"}

CSV_FIELDS = [
    "t", "t_global", "episode", "step", "side", "id", "alive", "state",
    "rules", "events", "action_v", "action_h", "target_idx", "target_id",
    "dist_target_km", "threat_km", "warning", "ammo", "msl_alive",
    "msl_stages", "msl_min_t", "msl_targets", "ego_msl_t", "ego_msl_mid",
    "dh_cmd", "dpsi_cmd", "spd_cmd", "pos_n_km", "pos_e_km", "alt_km",
    "psi_deg", "theta_deg", "fired", "fired_missile_id",
]


def _log(msg):
    print(msg, flush=True)


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _finite(value, default=np.nan):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except Exception:
        return default


def _target_id_of(env, side, idx):
    if idx is None or idx < 0:
        return -1
    table = env.BUAVs if side == "r" else env.RUAVs
    if idx >= len(table):
        return -1
    return table[idx].id


def _missile_stats(env, uid):
    own = [m for m in env.missiles
           if getattr(m, "launcher_id", None) == uid and not getattr(m, "dead", False)]
    if not own:
        return 0, "", np.nan, ""
    stages = "".join("M" if getattr(m, "guidance_stage", 3) < 3 else "T" for m in own)
    min_t = min(float(getattr(m, "t", 0.0)) for m in own)
    targets = ",".join(str(getattr(m, "target_id", -1)) for m in own)
    return len(own), stages, min_t, targets


def _events_str(ev):
    return "+".join(k for k in ("e1", "e2", "e4_edge", "e5", "c1", "c3", "channel_free")
                    if ev.get(k))


def _collect_step(env, rule_inputs, mask_inputs, fly_actions, fire_actions,
                  fire_missile_ids, flight_cmds, episode, step, t_bias,
                  prev_states):
    rows, events = [], []
    for uid in env.UAV_ids:
        uav = env.find_uav_with_id(uid)
        dbg = get_seca_debug(uid)
        ri = rule_inputs[uid]
        fly = fly_actions[uid]
        fire = fire_actions[uid]
        cmd = flight_cmds[uid]
        ev = dbg["last_events"] if dbg else {}
        target_idx = int(fly["at"])
        target_id = _target_id_of(env, uav.side, target_idx)
        n_msl, stages, min_t, msl_targets = _missile_stats(env, uid)
        weapon = ri.get("weapon_infos", {})
        # 已阵亡的飞机不再决策，SECA 只返回固定直飞占位；状态统一记为 dead，
        # 避免把死亡前最后一步的 q4/R15 当成当前判定结果。
        alive = int(not uav.dead)
        row = {
            "t": round(float(env.t), 2),
            "t_global": round(float(env.t + t_bias), 2),
            "episode": episode,
            "step": step,
            "side": uav.side,
            "id": uid,
            "alive": alive,
            "state": (dbg["state"] if dbg else "-") if alive else "dead",
            "rules": ("+".join(dbg["last_rules_fired"]) if dbg else "") if alive else "",
            "events": _events_str(ev) if alive else "",
            "action_v": int(fly["do"][0]),
            "action_h": int(fly["do"][1]),
            "target_idx": target_idx,
            "target_id": target_id,
            "dist_target_km": round(_finite(dbg["last_dist"] if dbg else np.nan) / 1e3, 3),
            "threat_km": round(_finite(dbg["last_threat_dist"] if dbg else np.nan) / 1e3, 3),
            "warning": int(bool(ev.get("e5"))),
            "ammo": int(_finite(ri.get("ego_ammo", 0), 0)),
            "msl_alive": int(n_msl),
            "msl_stages": stages,
            "msl_min_t": round(_finite(min_t), 2),
            "msl_targets": msl_targets,
            "ego_msl_t": round(_finite(weapon.get("missile_t_fly", np.nan)), 2),
            "ego_msl_mid": int(_finite(weapon.get("mid_term", 0), 0)),
            "dh_cmd": round(float(cmd[0]), 3),
            "dpsi_cmd": round(float(cmd[1]), 4),
            "spd_cmd": round(float(cmd[2]), 3),
            "pos_n_km": round(float(uav.pos_[0]) / 1e3, 4),
            "pos_e_km": round(float(uav.pos_[2]) / 1e3, 4),
            "alt_km": round(float(uav.pos_[1]) / 1e3, 4),
            "psi_deg": round(math.degrees(float(uav.psi)), 2),
            "theta_deg": round(math.degrees(float(uav.theta)), 2),
            "fired": int(fire["do"]),
            "fired_missile_id": fire_missile_ids.get(uid, -1),
        }
        rows.append(row)

        prev_state = prev_states.get(uid)
        if prev_state is not None and prev_state != row["state"]:
            kind = {("q1", "q2"): "track_entry",
                    ("q2", "q4"): "defense_entry",
                    ("q4", "q2"): "defense_exit"}.get((prev_state, row["state"]))
            if kind:
                events.append({
                    "t": row["t"], "side": row["side"], "id": uid,
                    "event": kind, "rule": row["rules"], "target": target_id,
                    "detail": row["events"], "n_km": row["pos_n_km"],
                    "e_km": row["pos_e_km"], "alt_km": row["alt_km"],
                })
        prev_states[uid] = row["state"]

        if row["fired"]:
            events.append({
                "t": row["t"], "side": row["side"], "id": uid,
                "event": "launch", "rule": row["rules"], "target": target_id,
                "detail": f"msl={row['fired_missile_id']},d={row['dist_target_km']:.1f}km",
                "n_km": row["pos_n_km"], "e_km": row["pos_e_km"], "alt_km": row["alt_km"],
            })
    return rows, events


def _write_csv(path, rows):
    import csv
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_text_trace(path, rows):
    header = (
        "SECA decision trace（每一步一行；msl=M中制导/T末制导；cmd=(v,h)+dh/dpsi/spd）\n"
        f"{'t(s)':>7} {'side':>4} {'id':>4} {'state':>5} {'rules':>10} "
        f"{'events':>22} {'d_tgt':>7} {'d_thr':>7} {'ammo':>4} {'msl':>4} "
        f"{'cmd(v,h)':>9} {'dh':>7} {'dpsi':>7} {'spd':>7} {'fire':>4}\n"
    )
    lines = [header, "-" * 138 + "\n"]
    for r in rows:
        d_tgt = "-" if not np.isfinite(r["dist_target_km"]) else f"{r['dist_target_km']:.1f}"
        # 环境在无威胁/无目标时给 999 km 占位，文本表中显示为 "-"
        d_thr = "-" if (not np.isfinite(r["threat_km"]) or r["threat_km"] >= 900) \
            else f"{r['threat_km']:.1f}"
        msl = f"{r['msl_alive']}{r['msl_stages']}" if r["msl_alive"] else "0"
        if r["alive"]:
            cmd = f"({r['action_v']},{r['action_h']})"
            dh, dpsi, spd = (f"{r['dh_cmd']:7.2f}", f"{r['dpsi_cmd']:7.3f}",
                             f"{r['spd_cmd']:7.1f}")
        else:
            # 阵亡后 SECA 返回的直飞占位不是真实决策，距离/动作在表中留空
            d_tgt = d_thr = cmd = "-"
            dh, dpsi, spd = f"{'-':>7}", f"{'-':>7}", f"{'-':>7}"
        lines.append(
            f"{r['t']:7.1f} {r['side']:>4} {r['id']:>4} {r['state']:>5} "
            f"{r['rules'][:10]:>10} {r['events'][:22]:>22} {d_tgt:>7} {d_thr:>7} "
            f"{r['ammo']:>4} {msl:>4} {cmd:>9} "
            f"{dh} {dpsi} {spd} "
            f"{'FIRE' if r['fired'] else '':>4}\n"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def _write_events_csv(path, events):
    import csv
    fields = ["t", "side", "id", "event", "rule", "target", "detail", "n_km", "e_km", "alt_km"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(events, key=lambda e: (e["t"], e["id"])))


def _uid_order(env):
    return [u.id for u in env.UAVs]


def _uid_side(env):
    return {u.id: u.side for u in env.UAVs}


def _plot_trajectory(path, rows, events, env):
    fig, ax = plt.subplots(figsize=(10, 7.2))
    ids = _uid_order(env)
    side = _uid_side(env)
    for uid in ids:
        pts = [r for r in rows if r["id"] == uid and r["alive"]]
        if not pts:
            continue
        es = [r["pos_e_km"] for r in pts]
        ns = [r["pos_n_km"] for r in pts]
        ax.plot(es, ns, color=SIDE_COLORS[side[uid]], lw=1.4, alpha=0.75)
        ax.scatter(es[0], ns[0], marker="o", s=40, facecolor="white",
                   edgecolor=SIDE_COLORS[side[uid]], zorder=4)
        ax.text(es[-1], ns[-1], f" {uid}", color=SIDE_COLORS[side[uid]],
                fontsize=9, va="center")
    for ev in events:
        if ev["event"] == "launch":
            ax.scatter(ev["e_km"], ev["n_km"], marker="^", s=90,
                       facecolor="#ffd54f", edgecolor="black", zorder=5)
        elif ev["event"] == "defense_entry":
            ax.scatter(ev["e_km"], ev["n_km"], marker="s", s=70,
                       facecolor="#fb8c00", edgecolor="black", zorder=5)
    handles = [
        Line2D([], [], color=SIDE_COLORS["r"], label="红方航迹"),
        Line2D([], [], color=SIDE_COLORS["b"], label="蓝方航迹"),
        Line2D([], [], marker="^", color="none", markerfacecolor="#ffd54f",
               markeredgecolor="black", markersize=10, label="R5 发射点"),
        Line2D([], [], marker="s", color="none", markerfacecolor="#fb8c00",
               markeredgecolor="black", markersize=8, label="R7 转防御点"),
    ]
    ax.legend(handles=handles, loc="best")
    ax.set_xlabel("E / km（东向）")
    ax.set_ylabel("N / km（北向）")
    ax.set_title("SECA 水平面航迹（起点空心圆，标注终点 id）")
    ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_distance(path, rows):
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    ids = sorted({r["id"] for r in rows})
    cmap = plt.get_cmap("tab10")
    for k, uid in enumerate(ids):
        pts = [r for r in rows if r["id"] == uid and r["alive"]]
        if not pts:
            continue
        ts = [r["t"] for r in pts]
        d_tgt = [r["dist_target_km"] if np.isfinite(r["dist_target_km"]) else np.nan for r in pts]
        d_thr = [r["threat_km"] if np.isfinite(r["threat_km"]) and r["threat_km"] < 900 else np.nan
                 for r in pts]
        axes[0].plot(ts, d_tgt, color=cmap(k % 10), lw=1.3, label=f"id {uid}")
        axes[1].plot(ts, d_thr, color=cmap(k % 10), lw=1.3, label=f"id {uid}")
    axes[0].axhline(SECA_PARAMS["D_attack_hv"] / 1e3, color="crimson", ls="--",
                    label="D_A = 55 km（开火）")
    axes[0].axhline(SECA_PARAMS["D_track_hv_hv"] / 1e3, color="gray", ls=":",
                    label="D_HV-HV = 100 km（分配）")
    axes[0].set_ylabel("到所跟踪目标距离 / km")
    axes[0].set_title("目标距离（上）与最近导弹威胁距离（下）")
    axes[0].legend(ncol=2, fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].axhline(SECA_PARAMS["D_maneuver_hv"] / 1e3, color="#fb8c00", ls="--",
                    label="D_M = 60 km（c3 机动阈值）")
    axes[1].set_ylabel("RWR 最近威胁距离 / km")
    axes[1].set_xlabel("t / s")
    axes[1].legend(ncol=2, fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_state_timeline(path, rows):
    ids = sorted({r["id"] for r in rows})
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(ids) + 2.2))
    for row_idx, uid in enumerate(ids):
        pts = sorted([r for r in rows if r["id"] == uid], key=lambda r: r["t"])
        if not pts:
            continue
        start = pts[0]["t"]
        last_t = pts[0]["t"]
        last_state = pts[0]["state"]
        for r in pts[1:]:
            if r["state"] != last_state:
                ax.broken_barh([(start, max(last_t - start, 1e-6))],
                               (row_idx - 0.35, 0.7), facecolors=STATE_COLORS.get(last_state, "#cccccc"))
                start, last_state = r["t"], r["state"]
            last_t = r["t"]
        ax.broken_barh([(start, max(last_t - start, 1e-6))],
                       (row_idx - 0.35, 0.7), facecolors=STATE_COLORS.get(last_state, "#cccccc"))
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels([f"id {uid}" for uid in ids])
    ax.set_xlabel("t / s")
    ax.set_title("SECA 状态时间线")
    ax.grid(axis="x", alpha=0.3)
    handles = [Patch(color=STATE_COLORS[s], label=s) for s in ("q1", "q2", "q4", "dead")]
    ax.legend(handles=handles, loc="upper right", ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_events(path, events):
    styles = {
        "track_entry": ("D", "#1e88e5", "进入跟踪 q1→q2"),
        "launch": ("^", "#ffb300", "R5 发射"),
        "defense_entry": ("s", "#fb8c00", "R7 转防御 q2→q4"),
        "defense_exit": ("o", "#2e7d32", "R16 回跟踪 q4→q2"),
    }
    ids = sorted({e["id"] for e in events})
    fig, ax = plt.subplots(figsize=(11, 0.5 * max(len(ids), 3) + 2.0))
    for ev in events:
        if ev["event"] not in styles:
            continue
        marker, color, _ = styles[ev["event"]]
        ax.scatter(ev["t"], ids.index(ev["id"]), marker=marker, s=90,
                   facecolor=color, edgecolor="black", zorder=4)
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels([f"id {uid}" for uid in ids])
    ax.set_xlabel("t / s")
    ax.set_title("SECA 事件散点（发射 / 转防御 / 回跟踪）")
    ax.grid(alpha=0.3)
    handles = [Line2D([], [], marker=m, color="none", markerfacecolor=c,
                      markeredgecolor="black", markersize=9, label=lab)
               for m, c, lab in styles.values()]
    # 图例放到坐标轴下方，避免遮住右侧的事件点
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _defense_intervals(rows, uid, end_t, dt):
    pts = sorted([r for r in rows if r["id"] == uid], key=lambda r: r["t"])
    intervals, start = [], None
    for r in pts:
        if r["state"] == "q4" and start is None:
            start = r["t"]
        elif r["state"] != "q4" and start is not None:
            intervals.append((start, r["t"]))
            start = None
    if start is not None:
        intervals.append((start, end_t))
    return intervals


def _write_summary(path, rows, events, episode, end_t, dt, acmi_path):
    lines = [f"SECA 可观测性回合 {episode}", "=" * 60]
    ids = sorted({r["id"] for r in rows})
    sides = {r["id"]: r["side"] for r in rows}

    tracks = [e for e in events if e["event"] == "track_entry"]
    launches = [e for e in events if e["event"] == "launch"]
    if tracks:
        first = min(tracks, key=lambda e: e["t"])
        lines.append(f"最先进入跟踪(q1→q2): id={first['id']} ({SIDE_NAMES.get(first['side'])}) "
                     f"t={first['t']:.1f}s rule={first['rule']} 事件={first['detail']}")
    else:
        lines.append("最先进入跟踪: 无")
    if launches:
        first = min(launches, key=lambda e: e["t"])
        lines.append(f"最先开火(R5): id={first['id']} ({SIDE_NAMES.get(first['side'])}) "
                     f"t={first['t']:.1f}s {first['detail']}")
    else:
        lines.append("最先开火: 无")

    lines.append("-" * 60)
    for uid in ids:
        pts = sorted([r for r in rows if r["id"] == uid], key=lambda r: r["t"])
        if not pts:
            continue
        own_launch = [e for e in events if e["event"] == "launch" and e["id"] == uid]
        track = [e for e in events if e["event"] == "track_entry" and e["id"] == uid]
        intervals = _defense_intervals(pts, uid, end_t, dt)
        dur = sum(b - a for a, b in intervals)
        alive = pts[-1]["alive"] > 0
        launch_ts = ",".join(f"{e['t']:.0f}s" for e in own_launch)
        lines.append(
            f"id={uid:<4d} {SIDE_NAMES.get(sides[uid]):<2s} "
            f"进入跟踪={track[0]['t'] if track else '-'}s "
            f"发射={len(own_launch)}次[{launch_ts}] "
            f"防御段={ ['%.1f-%.1fs' % iv for iv in intervals] } 防御总时长={dur:.1f}s "
            f"余弹={pts[-1]['ammo']} 最终状态={pts[-1]['state']} {'存活' if alive else '被击毁'}"
        )
    total_def = 0.0
    for uid in ids:
        total_def += sum(b - a for a, b in _defense_intervals(rows, uid, end_t, dt))
    lines.append("-" * 60)
    lines.append(f"全队防御总时长（逐机累计）: {total_def:.1f}s")
    lines.append(f"ACMI: {acmi_path}")
    lines.append("提示：发射点=新导弹对象出现；命中/击毁=导弹或飞机被 -id 移除。")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _inspect_acmi(path):
    text = open(path, "r", encoding="utf-8", errors="replace").read()
    ac_ids = sorted(set(re.findall(r"^(\d+),T=.*?Name=F16", text, re.M)),
                    key=lambda s: int(s))
    msl_ids = sorted(set(re.findall(r"^(\d+),T=.*?Name=AIM-120C", text, re.M)),
                     key=lambda s: int(s))
    frames = len(re.findall(r"^#", text, re.M))
    removals = len(re.findall(r"^-\d+\s*$", text, re.M))
    size = os.path.getsize(path)
    lines = [
        f"file={path}",
        f"size_bytes={size}",
        f"header_ok={text.startswith('FileType=text/acmi/tacview')}",
        f"reference_frame_ok={'#0.00' in text}",
        f"time_frames={frames}",
        f"aircraft_objects={len(ac_ids)} ids={ac_ids}",
        f"missile_objects={len(msl_ids)} ids={msl_ids}",
        f"removal_lines={removals}",
    ]
    ok = (size > 2000 and lines[2].endswith("True") and lines[3].endswith("True")
          and len(ac_ids) >= 2 and len(msl_ids) >= 1 and frames > 10)
    lines.append(f"verdict={'PASS' if ok else 'FAIL'}")
    return "\n".join(lines) + "\n", ok, len(ac_ids), len(msl_ids), frames


def run(args):
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    combats_dir = os.path.dirname(os.path.abspath(__file__))
    for path in (combats_dir, project_root):
        if path not in sys.path:
            sys.path.insert(0, path)

    import random
    import torch
    from Envs.Tasks.ChooseStrategyEnv_terminate_and_reward import ChooseStrategyEnv
    from _context import load_init_states

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = args.out_root or os.path.join(project_root, "结果展示", "SECA_observability")
    run_dir = os.path.join(out_root, run_id)
    os.makedirs(run_dir, exist_ok=True)

    init_cfg = load_init_states()
    env_args = argparse.Namespace()
    env_args.max_episode_len = args.max_time
    env_args.R_cage = 60.3e3
    tacview_show = 2 if args.acmi else 0
    env = ChooseStrategyEnv(env_args, tacview_show=tacview_show,
                            vertices=init_cfg["vertices"])
    env.dt_move = args.dt_move
    acmi_path = getattr(getattr(env, "tacview", None), "filename", None)
    _log(f"[observability] 输出目录: {run_dir}")
    _log(f"[observability] ACMI: {acmi_path}")

    red_birth = {"position": np.array([0.0, 9000.0, -55e3]), "psi": math.pi / 2}
    blue_birth = {"position": np.array([0.0, 9000.0, 55e3]), "psi": -math.pi / 2}
    mid_point = np.array([0.0, 0.0])
    axis = np.array([0.0, 1.0])  # 红 -> 蓝（东向）
    t_bias = 0.0
    all_ok = True

    try:
        for episode in range(args.episodes):
            seed = args.seed + episode
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            reset_seca()
            env.reset(
                red_birth_state=red_birth, blue_birth_state=blue_birth,
                red_init_num=init_cfg["red_init_num"],
                blue_init_num=init_cfg["blue_init_num"],
                red_init_ammo=init_cfg["red_init_ammo"],
                blue_init_ammo=init_cfg["blue_init_ammo"],
                seed=seed,
            )
            env.dt_maneuver = args.dt_maneuver
            env.min_alt_safe = 2000

            ep_dir = os.path.join(run_dir, f"ep{episode:02d}")
            debug = args.trace or args.plots
            if debug:
                os.makedirs(ep_dir, exist_ok=True)
            rows, events = [], []
            prev_states = {}
            hit_ids, death_ids = set(), set()
            fly_action_all = {uid: {"do": [1, 7], "at": 0} for uid in env.UAV_ids}
            done = False
            n_steps = int(round(args.max_time / args.dt_maneuver))

            for step in range(n_steps):
                if not env.running or done:
                    break
                rule_inputs, mask_inputs = {}, {}
                for uid in env.UAV_ids:
                    _, mask_input, rule_input = env.get_obs(uid, pomdp=1)
                    rule_inputs[uid] = rule_input
                    mask_inputs[uid] = mask_input

                fire_actions, fire_ids, flight_cmds = {}, {}, {}
                for uid in env.UAV_ids:
                    uav = env.find_uav_with_id(uid)
                    normal = axis if uav.side == "r" else -axis
                    fly, fire, _ = basic_rules_seca(
                        rule_inputs[uid], mask_inputs[uid],
                        rules_num=0, agent_id=uid,
                        ego_pos=(uav.pos_[0], uav.pos_[2]),
                        midline={"point": mid_point, "normal": normal},
                        sim_time=float(env.t),
                    )
                    fly_action_all[uid] = fly
                    fire_actions[uid] = fire
                    if fire["do"]:
                        missile = env.fire_120(uid, fire["at"], tabu=0)
                        # fire_120 成功时返回导弹对象，失败返回 None
                        fire_ids[uid] = -1 if missile is None else int(getattr(missile, "id", -1))
                        if missile is None:
                            fire_actions[uid] = {"do": 0, "at": fire["at"]}
                    else:
                        fire_ids[uid] = -1
                    env.radar_action(uid, 1)
                    flight_cmds[uid] = env.maneuver_for_1_enm(
                        uid, fly["at"], fly["do"], rule_inputs[uid])

                if debug:
                    step_rows, step_events = _collect_step(
                        env, rule_inputs, mask_inputs, fly_action_all, fire_actions,
                        fire_ids, flight_cmds, episode, step, t_bias, prev_states)
                    rows.extend(step_rows)
                    events.extend(step_events)

                env.step(flight_cmds.values())
                for uid in env.UAV_ids:
                    done_i, _ = env.combat_terminate_and_reward(
                        uid, action_cycle_multiplier=1)
                    done = done or done_i

                # 命中/击毁在发生的这一仿真步就地记录（ACMI 中体现为对象被 -id 移除）
                if debug:
                    for m in env.missiles:
                        if getattr(m, "hit", False) and m.id not in hit_ids:
                            hit_ids.add(m.id)
                            events.append({
                                "t": round(float(env.t), 2),
                                "side": getattr(m, "side", "-"),
                                "id": int(getattr(m, "launcher_id", -1)), "event": "hit",
                                "rule": "-", "target": int(getattr(m, "target_id", -1)),
                                "detail": f"msl={m.id}", "n_km": np.nan, "e_km": np.nan,
                                "alt_km": np.nan,
                            })
                    for uav in env.UAVs:
                        if uav.dead and uav.id not in death_ids:
                            death_ids.add(uav.id)
                            events.append({
                                "t": round(float(env.t), 2), "side": uav.side, "id": uav.id,
                                "event": "kill", "rule": "-", "target": uav.id,
                                "detail": f"side={uav.side}",
                                "n_km": round(uav.pos_[0] / 1e3, 4),
                                "e_km": round(uav.pos_[2] / 1e3, 4),
                                "alt_km": round(uav.pos_[1] / 1e3, 4),
                            })
                if args.acmi:
                    env.render(t_bias=t_bias)

            if args.acmi:
                env.clear_render(t_bias=t_bias)
            end_t = float(env.t)

            if args.trace:
                _write_csv(os.path.join(ep_dir, "decision_log.csv"), rows)
                _write_text_trace(os.path.join(ep_dir, "decision_trace.txt"), rows)
                _write_events_csv(os.path.join(ep_dir, "events.csv"), events)
                _write_summary(os.path.join(ep_dir, "summary.txt"), rows, events,
                               episode, end_t, args.dt_maneuver, acmi_path or "-")
            if args.plots:
                _plot_trajectory(os.path.join(ep_dir, "trajectory.png"), rows, events, env)
                _plot_distance(os.path.join(ep_dir, "distance_time.png"), rows)
                _plot_state_timeline(os.path.join(ep_dir, "state_timeline.png"), rows)
                _plot_events(os.path.join(ep_dir, "events.png"), events)

            red_alive = sum(not u.dead for u in env.RUAVs)
            blue_alive = sum(not u.dead for u in env.BUAVs)
            _log(f"[observability] ep{episode}: t={end_t:.1f}s 红存活={red_alive} "
                 f"蓝存活={blue_alive}")
            if args.trace:
                _log(f"[observability] 决策行={len(rows)} 事件={len(events)}")
                _log(open(os.path.join(ep_dir, "summary.txt"), encoding="utf-8").read())
            t_bias += end_t
    finally:
        if args.acmi:
            env.end_render()

    if args.acmi and acmi_path and os.path.exists(acmi_path):
        report, ok, n_ac, n_msl, n_frames = _inspect_acmi(acmi_path)
        with open(os.path.join(run_dir, "acmi_check.txt"), "w", encoding="utf-8") as f:
            f.write(report)
        _log("[observability] ACMI 校验:\n" + report)
        all_ok = all_ok and ok
        if not ok:
            _log("[observability][ERROR] ACMI 校验未通过（空壳或对象缺失）")
    else:
        _log("[observability][ERROR] 未生成 ACMI 文件")
        all_ok = False

    info = {
        "run_id": run_id,
        "git_commit": _git_commit(),
        "seed": args.seed,
        "episodes": args.episodes,
        "max_time_s": args.max_time,
        "dt_maneuver": args.dt_maneuver,
        "dt_move": args.dt_move,
        "seca_params": SECA_PARAMS,
        "acmi": acmi_path,
        "acmi_ok": all_ok,
        "outputs": {"acmi": args.acmi, "trace": args.trace, "plots": args.plots},
        "run_dir": run_dir,
    }
    with open(os.path.join(run_dir, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    _log(f"[observability] 完成，产物目录: {run_dir}")
    return 0 if all_ok else 1


def main():
    parser = argparse.ArgumentParser("BasicRules_Homo_SECA_observability")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-time", type=float, default=220.0)
    parser.add_argument("--dt-maneuver", type=float, default=2.0)
    parser.add_argument("--dt-move", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out-root", type=str, default=None)
    parser.add_argument("--acmi", action="store_true", default=True,
                        help="启用 ACMI 录制（默认开启）")
    parser.add_argument("--no-acmi", dest="acmi", action="store_false")
    parser.add_argument("--trace", action="store_true",
                        help="输出逐决策 CSV/文本轨迹/events.csv/summary.txt（默认关闭）")
    parser.add_argument("--plots", action="store_true",
                        help="输出四张静态 PNG（默认关闭）")
    return run(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
