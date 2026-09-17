import sys
import os
import numpy as np
from math import *

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from Envs.Tasks.ChooseStrategyEnv260421_terminate_and_reward import ChooseStrategyEnv
from BasicRules_Homo import basic_rules
import argparse

parser = argparse.ArgumentParser("UAV swarm confrontation")
args = parser.parse_args()
args.max_episode_len = 15*60
args.R_cage = 80e3

env = ChooseStrategyEnv(args, tacview_show=0)
env.dt_move = 0.04
env.shielded = 1

def creat_initial_state():
    blue_height = 9000
    red_height = 9000
    red_psi = -np.pi/2
    blue_psi = np.pi/2
    red_N = 0
    red_E = 55e3
    blue_N = red_N
    blue_E = -red_E
    DEFAULT_RED_BIRTH_STATE = {'position': np.array([red_N, red_height, red_E]), 'psi': red_psi}
    DEFAULT_BLUE_BIRTH_STATE = {'position': np.array([blue_N, blue_height, blue_E]), 'psi': blue_psi}
    return DEFAULT_RED_BIRTH_STATE, DEFAULT_BLUE_BIRTH_STATE

DEFAULT_RED_BIRTH_STATE, DEFAULT_BLUE_BIRTH_STATE = creat_initial_state()

old_rand = np.random.rand
np.random.rand = lambda: 0.0 # Force no swap
env.reset(red_birth_state=DEFAULT_RED_BIRTH_STATE, blue_birth_state=DEFAULT_BLUE_BIRTH_STATE)
np.random.rand = old_rand

dt_maneuver = 1.0
action_cycle_multiplier = 10
env.dt_maneuver = dt_maneuver

for count in range(500):
    if not env.running: break
    
    agent_inputs_ = {}
    mask_inputs_ = {}
    rule_inputs_ = {}

    for id in env.UAV_ids:
        agent_input, mask_input, rule_input  = env.get_obs(id, pomdp=0)
        agent_inputs_[id] = agent_input
        mask_inputs_[id] = mask_input
        rule_inputs_[id] = rule_input

    if count % action_cycle_multiplier == 0:
        fly_actions = {}
        for id in env.UAV_ids:
            rule_inputs = rule_inputs_[id]
            speed = rule_inputs["ego_main"]["v_ego"]
            alt = rule_inputs["ego_main"]["h_ego"]
            sin_theta = rule_inputs["ego_main"]["sin_theta"]
            enm_info_list = rule_inputs["enm_infos"]

            selected_index = 0
            enm_info = enm_info_list[selected_index]
            distance = enm_info["enm_dist"]
            delta_theta = enm_info["enm_delta_theta"]
            delta_psi = np.arctan2(enm_info["sin_enm_delta_psi"], enm_info["cos_enm_delta_psi"])
            ATA = enm_info["enm_ATA"]
            AA_hor = enm_info["enm_AA"]
            weapon_infos = rule_inputs["weapon_infos"]
            t_fired = weapon_infos["missile_t_fly"]
            on_guiding = weapon_infos["mid_term"]
            
            print(f"Step {count} UAV {id}: dist={distance/1000:.1f}km, sin_theta={sin_theta:.2f}, ATA={ATA*180/pi:.1f}deg, dPsi={delta_psi*180/pi:.1f}deg, AA={AA_hor*180/pi:.1f}deg, t_fired={t_fired}")
            
            fly_action, fire_action, radar_action = basic_rules(rule_inputs, mask_inputs_[id], rules_num=4)
            fly_actions[id] = fly_action
            
            if fire_action['do']:
                print(f"!!! Step {count}: UAV {id} fired! !!!")
                env.fire_120(id, fire_action['at'], tabu=0)
            if fly_action['do'] in [5, 6]:
                print(f"!!! Step {count}: UAV {id} is cranking! (action {fly_action['do']}) !!!")
                
            env.radar_action(id, radar_action)

    flight_cmds = {}
    for id in env.UAV_ids:
        flight_cmds[id] = env.maneuver_for_1_enm(id, fly_actions[id]['at'], fly_actions[id]['do'], rule_inputs_[id])

    env.step(flight_cmds.values())
