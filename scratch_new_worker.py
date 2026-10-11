def worker_process(worker_id, pipe, args, device_worker, num_workers, shared_dict):
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
            
            blue_trans = {'agent_inputs': [], 'rule_inputs': [], 'mask_inputs': [], 'actions': [], 'rewards': [], 'dones': []}
            red_trans = {'agent_inputs': [], 'rule_inputs': [], 'mask_inputs': [], 'actions': [], 'rewards': [], 'dones': []}
            
            blue_ids = [id for id, uav in env.UAVsTable.items() if uav.side == 'b']
            red_ids = [id for id, uav in env.UAVsTable.items() if uav.side == 'r']
            
            ep_return = 0
            event_return = 0
            wins = 0
            loss_cnt = 0
            draw_cnt = 0
            bvr_perish_together_cnt = 0

            for count in range(args.max_episode_len):
                active_blue = [id for id in blue_ids if not env.UAVsTable[id].dead]
                active_red = [id for id in red_ids if not env.UAVsTable[id].dead]
                
                if not active_blue and not active_red:
                    break
                    
                flight_cmds = [None] * len(env.UAVsTable)
                
                # Forward for blue
                if active_blue:
                    blue_agent_in, blue_mask_in, blue_rule_in = zip(*[env.get_obs(id) for id in active_blue])
                    batch_blue_agent = {
                        'fixed': torch.tensor(np.array([inp['fixed'] for inp in blue_agent_in]), dtype=torch.float32).to(device_worker),
                        'enm': torch.tensor(np.array([inp['enm'] for inp in blue_agent_in]), dtype=torch.float32).to(device_worker),
                        'ally': torch.tensor(np.array([inp['ally'] for inp in blue_agent_in]), dtype=torch.float32).to(device_worker)
                    }
                    env_actions, raw_actions, _ = local_agent.get_action(batch_blue_agent, blue_rule_in, blue_mask_in, explore=True)
                    
                    for i, id in enumerate(active_blue):
                        act = env_actions[i]
                        env.radar_action(id, act['radar_action'])
                        if act['fire_action']['do']:
                            env.fire_120(id, act['fire_action']['at'])
                        flight_cmds[id-1] = env.maneuver_for_1_enm(id, act['flight_action']['at'], act['flight_action']['do'], blue_rule_in[i])
                        
                        # Store transition
                        blue_trans['agent_inputs'].append({k: v[i].cpu().numpy() for k,v in batch_blue_agent.items()})
                        blue_trans['rule_inputs'].append(blue_rule_in[i])
                        blue_trans['mask_inputs'].append(blue_mask_in[i])
                        blue_trans['actions'].append({k: v[i].detach().cpu().numpy() for k,v in raw_actions.items()})
                
                # Forward for red
                if active_red:
                    red_agent_in, red_mask_in, red_rule_in = zip(*[env.get_obs(id) for id in active_red])
                    batch_red_agent = {
                        'fixed': torch.tensor(np.array([inp['fixed'] for inp in red_agent_in]), dtype=torch.float32).to(device_worker),
                        'enm': torch.tensor(np.array([inp['enm'] for inp in red_agent_in]), dtype=torch.float32).to(device_worker),
                        'ally': torch.tensor(np.array([inp['ally'] for inp in red_agent_in]), dtype=torch.float32).to(device_worker)
                    }
                    env_actions, raw_actions, _ = adv_agent.get_action(batch_red_agent, red_rule_in, red_mask_in, explore=True)
                    
                    for i, id in enumerate(active_red):
                        act = env_actions[i]
                        env.radar_action(id, act['radar_action'])
                        if act['fire_action']['do']:
                            env.fire_120(id, act['fire_action']['at'])
                        flight_cmds[id-1] = env.maneuver_for_1_enm(id, act['flight_action']['at'], act['flight_action']['do'], red_rule_in[i])

                env.step(flight_cmds)

                # Get rewards for blue
                if active_blue:
                    for i, id in enumerate(active_blue):
                        rew, is_done, _ = env.combat_terminate_and_reward(id, [])
                        blue_trans['rewards'].append(rew)
                        blue_trans['dones'].append(is_done)
                        ep_return += rew

            pipe.send({
                'cmd': 'EPISODE_DONE',
                'blue_trans': blue_trans,
                'ep_return': ep_return,
                'event_return': event_return,
                'wins': 1 if len(active_blue) > 0 and len(active_red) == 0 else 0,
                'loss_cnt': 1 if len(active_blue) == 0 and len(active_red) > 0 else 0,
                'draw_cnt': 1 if (len(active_blue) > 0 and len(active_red) > 0) or (len(active_blue)==0 and len(active_red)==0) else 0,
                'bvr_perish_together_cnt': 0
            })
        elif cmd == 'EXIT':
            break
