import sys
import os
import argparse
import torch

sys.path.append(r'd:\3_Machine_Learning_in_Python\project04_multi_fire_and_dodge_missile')
from Envs.Tasks.ChooseStrategyEnv_terminate_and_reward import ChooseStrategyEnv
from Algorithms.ActorMLPHeads import ActorMLPWrapper

args = argparse.Namespace(max_episode_len=3000)
env = ChooseStrategyEnv(args)
env.reset(red_init_num=2, blue_init_num=2)

actor = ActorMLPWrapper()

obs = {}
rule_obs = {}
mask_obs = {}
for id in env.UAVsTable.keys():
    agent_in, mask_in, rule_in = env.get_obs(id)
    obs[id] = agent_in
    mask_obs[id] = mask_in
    rule_obs[id] = rule_in

print("Running one step with ActorMLPWrapper...")
# We need to batch the inputs
batch_agent_inputs = {
    'fixed': torch.tensor([obs[id]['fixed'] for id in obs.keys()]),
    'enm': torch.tensor([obs[id]['enm'] for id in obs.keys()]),
    'ally': torch.tensor([obs[id]['ally'] for id in obs.keys()])
}
batch_rule_inputs = [rule_obs[id] for id in obs.keys()]
batch_mask_inputs = [mask_obs[id] for id in obs.keys()]

env_actions, raw_actions, diags = actor.get_action(batch_agent_inputs, batch_rule_inputs, batch_mask_inputs, explore=True)
print("Got env_actions:", env_actions)

flight_cmds = []
for i, id in enumerate(obs.keys()):
    action = env_actions[i]
    env.radar_action(id, action['radar_action'])
    if action['fire_action']['do']:
        env.fire_120(id, action['fire_action']['at'])
    
    # Maneuver
    cmd = env.maneuver_for_1_enm(id, action['flight_action']['at'], action['flight_action']['do'], rule_obs[id])
    flight_cmds.append(cmd)

print("Flight cmds:", flight_cmds)
env.step(flight_cmds)
print("Step successful!")
