# -*- coding: utf-8 -*-
"""复现 VsBaseline_MA.test_worker 的决策循环，打印红方每个决策点的开火相关量：
距离/锁定/有效开火概率/是否被规则禁火/采样到的 fire/实际发射数。
"""
import os
import sys
import argparse
import numpy as np
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, script_dir)
sys.path.insert(0, project_root)

from VsBaseline_MA import creat_initial_state, POMDP
from Envs.Tasks.ChooseStrategyEnv_terminate_and_reward import ChooseStrategyEnv
from BasicRules_Homo import basic_rules
from Algorithms.ActorWithATTN import ActorWrapper
from _context import take_some_of_rule_inputs, load_obs_dims, stack_field, load_init_states

ACTOR = os.path.join(project_root, 'logs', 'combat', 'MAPPO_homo1v1-run-20261028-201129', 'actor_rein50.pt')

args = argparse.Namespace(max_episode_len=15 * 60, R_cage=60.3e3)
vertices = load_init_states()['vertices']
env = ChooseStrategyEnv(args, tacview_show=0, vertices=vertices)
env.dt_move = 0.07
env.dt_maneuver = 0.2
env.min_alt_safe = 2000

fixed_dim, enm_shape, ally_shape = load_obs_dims()
actor = ActorWrapper(fixed_dim, enm_shape, ally_shape, device='cpu')
actor.load_state_dict(torch.load(ACTOR, map_location='cpu', weights_only=True))
actor.eval()

red_state, blue_state = creat_initial_state()
env.reset(red_birth_state=red_state, blue_birth_state=blue_state,
          red_init_num=2, blue_init_num=2, red_init_ammo=2, blue_init_ammo=2,
          seed=(42, 43))

maneuver_cmds_all, target_index_all = {}, {}
steps = -1
done = False
n_sample_fire = 0
n_launch = 0
n_decision = 0
n_mask_zero = 0
min_dist = 1e9
max_p = 0.0
m1 = None
while steps < round(args.max_episode_len / env.dt_maneuver):
    steps += 1
    if not env.running or done:
        break
    agent_inputs_, mask_inputs_, rule_inputs_ = {}, {}, {}
    for uid in env.UAV_ids:
        ai, mi, ri = env.get_obs(uid, pomdp=POMDP)
        agent_inputs_[uid], mask_inputs_[uid], rule_inputs_[uid] = ai, mi, ri
        if m1 is None:
            m1 = mi
    if steps % 10 == 0:
        for uid in env.UAV_ids:
            uav = env.find_uav_with_id(uid)
            if uav.side == 'r':
                actor_rule = stack_field([take_some_of_rule_inputs(rule_inputs_[uid])])
                with torch.no_grad():
                    out = actor.get_action(agent_inputs_[uid], mask_inputs_[uid], actor_rule)
                    p, mask_fire, logits = None, None, None
                    if not uav.dead:
                        # 同步用同一条前向路径算有效开火概率（含禁火规则 delta）
                        pa, pm = actor.pack_inputs(agent_inputs_[uid], mask_inputs_[uid])
                        enm_idx = out['enm_idx'].reshape(-1)
                        sel, _, mini, _, _ = actor.selector(pa, pm, actor_rule, enm_idx=enm_idx)
                        logits = actor.pilot(sel, mini)
                        p = torch.sigmoid(logits['fire'].squeeze(-1))[0].item()
                        mask_fire = int(logits['fire_mask'][0])
                maneuver_cmds_all[uid] = [int(out['vert_idx'].item()), int(out['hor_idx'].item())]
                target_index_all[uid] = int(out['enm_idx'].item())
                fire = int(out['fire'].item())
                if not uav.dead:
                    n_sample_fire += fire
                    n_decision += 1
                    n_mask_zero += (1 - mask_fire)
                    max_p = max(max_p, p)
                    alive_b = [u for u in env.BUAVs if not u.dead]
                    d = (np.linalg.norm(alive_b[0].pos_ - uav.pos_) / 1e3) if alive_b else float('nan')
                    min_dist = min(min_dist, d)
                    info = env.get_state(uid)['enm_infos'][int(out['enm_idx'].item())]
                    print(f't={env.t:6.1f}s rid={uid} 采样fire={fire} 概率={p:.3f} fire_mask={mask_fire} '
                          f'距离={d:6.1f}km locked={info["enm_locked"]:.0f} obs={info["enm_observable"]:.0f} '
                          f'ATA={np.degrees(info["enm_ATA"]):+6.1f}° AA={np.arctan2(info["sin_enm_AA"], info["cos_enm_AA"]):+.3f} '
                          f'R弹数={len(env.RMissileTable)} ammo={uav.ammo}')
                    if fire:
                        try:
                            if env.fire_120(uid, int(out['enm_idx'].item()), tabu=0) is not None:
                                n_launch += 1
                        except ValueError as e:
                            print('  fire_120 警告', e)
                env.radar_action(uid, 1)
            else:
                fly_action, fire_action, radar_action = basic_rules(
                    rule_inputs_[uid], mask_inputs_[uid], rules_num=2)
                maneuver_cmds_all[uid] = fly_action['do']
                target_index_all[uid] = int(fly_action['at'])
                if bool(fire_action['do']):
                    try:
                        env.fire_120(uid, int(fire_action['at']), tabu=0)
                    except ValueError:
                        pass
                env.radar_action(uid, 1)
    flight_cmds = {}
    for uid in env.UAV_ids:
        try:
            flight_cmds[uid] = env.maneuver_for_1_enm(uid, target_index_all[uid],
                                                      maneuver_cmds_all[uid], rule_inputs_[uid])
        except ValueError:
            flight_cmds[uid] = [0.0, 0.0, 0.0]
    env.step(flight_cmds.values())
    for uid in env.UAV_ids:
        done, _ = env.combat_terminate_and_reward(id=uid, action_cycle_multiplier=10)

print(f'\n汇总: 决策次数={n_decision}, 采样到fire=1的次数={n_sample_fire}, 实际发射={n_launch}, '
      f'fire_mask=0的点数={n_mask_zero}, 概率max={max_p:.3f}, 最近距离={min_dist:.1f}km, '
      f'R弹数={len(env.RMissileTable)}, t={env.t:.1f}s, running={env.running}, done={done}')
print('初始 mask_enm =', m1['enm'] if m1 is not None else None)
