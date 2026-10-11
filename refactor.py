import sys
import re

file_path = r'd:\3_Machine_Learning_in_Python\project04_multi_fire_and_dodge_missile\TrainAndTests\Combats\CombatPPOWithIL3_parallel_hierarch.py'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace imports
content = content.replace('from Envs.Tasks.ChooseStrategyEnv2_2_hierarchical import *', 'from Envs.Tasks.ChooseStrategyEnv260421_terminate_and_reward import *')
content = content.replace('from Algorithms.PPOHybrid23_0 import PPOHybrid, PolicyNetHybrid, HybridActorWrapper', 'from Algorithms.PPOHybrid23_0 import PPOHybrid, PolicyNetHybrid, HybridActorWrapper\nfrom Algorithms.ActorMLPHeads import ActorMLPWrapper')

# Replace worker_process
worker_regex = re.compile(r'def worker_process\(worker_id.*?return\n', re.DOTALL)

new_worker = '''def worker_process(worker_id, pipe, args, device_worker, num_workers, shared_dict):
    import torch
    import numpy as np
    from Algorithms.ActorMLPHeads import ActorMLPWrapper
    from Algorithms.MLP_heads import ValueNet
    from Envs.Tasks.ChooseStrategyEnv260421_terminate_and_reward import ChooseStrategyEnv
    from copy import deepcopy

    env = ChooseStrategyEnv(args, tacview_show=False)
    
    local_agent = ActorMLPWrapper().to(device_worker)
    adv_agent = ActorMLPWrapper().to(device_worker)

    while True:
        cmd, packet = pipe.recv()
        if cmd == 'RUN_EPISODE':
            local_agent.load_state_dict(packet['agent_state'])
            adv_agent.load_state_dict(packet['adv_state'])
            
            blue_birth = packet['blue_birth']
            red_birth = packet['red_birth']
            env.reset(red_birth_state=red_birth, blue_birth_state=blue_birth, red_init_num=2, blue_init_num=2)
            
            blue_trans = {'obs': [], 'actions': [], 'rewards': [], 'dones': [], 'states': []}
            
            ep_return = 0
            event_return = 0
            wins = 0
            loss_cnt = 0
            draw_cnt = 0
            bvr_perish_together_cnt = 0

            for count in range(args.max_episode_len):
                blue_ids = [id for id, uav in env.UAVsTable.items() if uav.side == 'b' and not uav.dead]
                red_ids = [id for id, uav in env.UAVsTable.items() if uav.side == 'r' and not uav.dead]
                
                if not blue_ids and not red_ids:
                    break
                    
                flight_cmds = [None] * len(env.UAVsTable)
                
                # Forward for blue
                if blue_ids:
                    blue_obs_all = [env.get_obs(id) for id in blue_ids]
                    blue_agent_in = [o[0] for o in blue_obs_all]
                    blue_mask_in = [o[1] for o in blue_obs_all]
                    blue_rule_in = [o[2] for o in blue_obs_all]
                    
                    batch_blue_agent = {
                        'fixed': torch.tensor(np.array([inp['fixed'] for inp in blue_agent_in]), dtype=torch.float32).to(device_worker),
                        'enm': torch.tensor(np.array([inp['enm'] for inp in blue_agent_in]), dtype=torch.float32).to(device_worker),
                        'ally': torch.tensor(np.array([inp['ally'] for inp in blue_agent_in]), dtype=torch.float32).to(device_worker)
                    }
                    env_actions, raw_actions, _ = local_agent.get_action(batch_blue_agent, blue_rule_in, blue_mask_in, explore=True)
                    
                    for i, id in enumerate(blue_ids):
                        act = env_actions[i]
                        env.radar_action(id, act['radar_action'])
                        if act['fire_action']['do']:
                            env.fire_120(id, act['fire_action']['at'])
                        flight_cmds[id-1] = env.maneuver_for_1_enm(id, act['flight_action']['at'], act['flight_action']['do'], blue_rule_in[i])
                        
                        # Store transition
                        obs_tuple = (blue_agent_in[i], blue_rule_in[i], blue_mask_in[i])
                        blue_trans['obs'].append(obs_tuple)
                        blue_trans['actions'].append({k: v[i].detach().cpu().numpy() for k,v in raw_actions.items()})
                        
                        # Get flat state for ValueNet
                        flat_state, _, _ = env.get_obs(id, reward_fn=1)
                        # We use the fixed part of flat state for critic
                        state_t = torch.tensor([flat_state['fixed']]).to(device_worker)
                        flat_s = local_agent.flatten_agent_inputs({'fixed': state_t, 'enm': torch.zeros((1, 2, 10)).to(device_worker), 'ally': torch.zeros((1, 1, 9)).to(device_worker)})
                        blue_trans['states'].append(flat_s[0].cpu().numpy())
                
                # Forward for red
                if red_ids:
                    red_obs_all = [env.get_obs(id) for id in red_ids]
                    red_agent_in = [o[0] for o in red_obs_all]
                    red_mask_in = [o[1] for o in red_obs_all]
                    red_rule_in = [o[2] for o in red_obs_all]
                    
                    batch_red_agent = {
                        'fixed': torch.tensor(np.array([inp['fixed'] for inp in red_agent_in]), dtype=torch.float32).to(device_worker),
                        'enm': torch.tensor(np.array([inp['enm'] for inp in red_agent_in]), dtype=torch.float32).to(device_worker),
                        'ally': torch.tensor(np.array([inp['ally'] for inp in red_agent_in]), dtype=torch.float32).to(device_worker)
                    }
                    env_actions, _, _ = adv_agent.get_action(batch_red_agent, red_rule_in, red_mask_in, explore=True)
                    
                    for i, id in enumerate(red_ids):
                        act = env_actions[i]
                        env.radar_action(id, act['radar_action'])
                        if act['fire_action']['do']:
                            env.fire_120(id, act['fire_action']['at'])
                        flight_cmds[id-1] = env.maneuver_for_1_enm(id, act['flight_action']['at'], act['flight_action']['do'], red_rule_in[i])

                env.step(flight_cmds)

                # Get rewards for blue
                if blue_ids:
                    for i, id in enumerate(blue_ids):
                        rew, is_done, _ = env.combat_terminate_and_reward(id, [])
                        blue_trans['rewards'].append(rew)
                        blue_trans['dones'].append(is_done)
                        ep_return += rew

            pipe.send({
                'cmd': 'EPISODE_DONE',
                'blue_trans': blue_trans,
                'ep_return': ep_return,
                'event_return': event_return,
                'wins': 1 if len(blue_ids) > 0 and len(red_ids) == 0 else 0,
                'loss_cnt': 1 if len(blue_ids) == 0 and len(red_ids) > 0 else 0,
                'draw_cnt': 1 if (len(blue_ids) > 0 and len(red_ids) > 0) or (len(blue_ids)==0 and len(red_ids)==0) else 0,
                'bvr_perish_together_cnt': 0
            })
        elif cmd == 'EXIT':
            break

    return
'''

content = worker_regex.sub(new_worker, content)

# Modify PPOHybrid initialization
ppo_init_regex = re.compile(r'student_agent = PPOHybrid\([\s\S]*?target_kl=target_kl\n    \)')

new_ppo_init = '''student_agent = PPOHybrid(
        state_dim=57, hidden_dims=[256, 256], action_dims_dict={'cat': [5, 7], 'bern': 1},
        actor_lr=actor_lr, critic_lr=critic_lr,
        gamma=gamma, lmbda=lmbda, eps_clip=eps,
        device=device,
        target_kl=target_kl
    )
    student_agent.actor = ActorMLPWrapper().to(device)
    student_agent.actor_optimizer = torch.optim.Adam(student_agent.actor.parameters(), lr=actor_lr)
'''

content = ppo_init_regex.sub(new_ppo_init, content)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Refactored CombatPPOWithIL3_parallel_hierarch.py successfully.")
