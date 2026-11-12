def worker_process(rank, pipe, args, device_worker, dt_maneuver, seed, dt_move=0.07, pomdp=1, vertices=None):
    """采样 worker：红方学生共享一个 ActorWrapper（同构），蓝方为规则或历史 nn（对手）。
    经验只按学生 id 分组保存在内存，堆叠后通过 pipe 传回 master，不写磁盘。"""
    import torch
    import numpy as np
    from Envs.Tasks.ChooseStrategyEnv_terminate_and_reward import ChooseStrategyEnv
    from Algorithms.ActorWithATTN import ActorWrapper
    from BasicRules_Homo import basic_rules
    from _context import take_some_of_rule_inputs, stack_field, load_obs_dims, load_init_states, append_experience

    env = ChooseStrategyEnv(args, tacview_show=0, vertices=vertices)
    env.dt_move = dt_move
    env.dt_maneuver = dt_maneuver

    # 网络维度统一从 json 读
    fixed_dim, enm_shape, ally_shape = load_obs_dims()
    # settings 未携带时的初始数量/挂载兜底值（与 init_states.json 一致）
    init_cfg = load_init_states()
    # 红方学生共享一个 ActorWrapper（同构），蓝方历史 nn 对手按蓝机 id 存放
    actor_dict = {}
    student_actor = None

    while True:
        try:
            cmd, packet = pipe.recv()
            if cmd == 'RUN_EPISODE':
                student_actor_weights = packet[0]   # 同构单份 state_dict，全队共享
                opp_name, opp_type, opp_data = packet[1]
                settings = packet[2]
                action_cycle_multiplier = settings['action_cycle_multiplier']
                red_init_num = settings.get('red_init_num', init_cfg['red_init_num'])
                blue_init_num = settings.get('blue_init_num', init_cfg['blue_init_num'])
                red_init_ammo = settings.get('red_init_ammo', init_cfg['red_init_ammo'])
                blue_init_ammo = settings.get('blue_init_ammo', init_cfg['blue_init_ammo'])

                # 加载红方学生共享 actor 权重（学生 id 与环境红机 id 一致，全队共用一个实例）
                if student_actor is None:
                    student_actor = ActorWrapper(
                        fixed_dim=fixed_dim, enm_shape=enm_shape, ally_shape=ally_shape,
                        device=device_worker,
                    )
                student_actor.load_state_dict(student_actor_weights)
                student_actor.eval()

                # 蓝方历史 nn 对手权重（master 加载时已把学生 id 重映射为蓝机 id 打包）
                blue_nn_weights = opp_data if (opp_type == 'nn' and opp_data) else {}
                for id, sd in blue_nn_weights.items():
                    if id not in actor_dict:
                        actor_dict[id] = ActorWrapper(
                            fixed_dim=fixed_dim, enm_shape=enm_shape, ally_shape=ally_shape,
                            device=device_worker,
                        )
                    actor_dict[id].load_state_dict(sd)
                    actor_dict[id].eval()

                rb, bb = settings['red_birth'], settings['blue_birth']
                red_shuffle_seed = settings.get('red_shuffle_seed')
                blue_shuffle_seed = settings.get('blue_shuffle_seed')
                env.reset(red_birth_state=rb, blue_birth_state=bb,
                          red_init_num=red_init_num, blue_init_num=blue_init_num,
                          red_init_ammo=red_init_ammo, blue_init_ammo=blue_init_ammo,
                          seed=(red_shuffle_seed, blue_shuffle_seed))

                # 经验只按学生（红方）id 收集（蓝方为规则或历史 nn，不参与训练）
                student_ids = [id for id in env.UAV_ids if env.find_uav_with_id(id).side == 'r']
                LocalTrans = {id: {'obs': [], 'states': [], 'actions': [],
                                   'next_states': [], 'rewards': [], 'dones': [], 'active_masks': []}
                              for id in student_ids}
                Last_Obs = {id: None for id in student_ids}
                Last_States = {id: None for id in student_ids}
                Last_Actions = {id: None for id in student_ids}
                Rewards = {id: 0.0 for id in student_ids}

                maneuver_cmds_all = {}
                target_index_all = {}
                done = False
                ep_return = 0.0
                ep_return_by_id = {id: 0.0 for id in student_ids} # 个体回报（tensorboard 按 id 记录）

                for count in range(round(args.max_episode_len / dt_maneuver)):
                    if not env.running or done:
                        break

                    agent_inputs_, mask_inputs_, rule_inputs_ = {}, {}, {}
                    Curr_States = {}
                    for id in env.UAV_ids:
                        agent_input, mask_input, rule_input = env.get_obs(id, pomdp=pomdp)
                        agent_inputs_[id] = agent_input
                        mask_inputs_[id] = mask_input
                        rule_inputs_[id] = rule_input
                        Curr_States[id], _, _ = env.get_obs(id, reward_fn=1)

                    # 决策点
                    if count % action_cycle_multiplier == 0:
                        for id in env.UAV_ids:
                            UAV = env.find_uav_with_id(id)
                            is_student = UAV.side == 'r'
                            # 保存上一步经验（仅学生，决策步对齐，统一走 append_experience）
                            if is_student and Last_Obs[id] is not None and count > 0:
                                LocalTrans[id] = append_experience(LocalTrans[id], Last_Obs[id], Last_States[id],
                                                                   Last_Actions[id], Rewards[id], Curr_States[id],
                                                                   done=0, active_mask=not UAV.dead)

                            if is_student or id in blue_nn_weights:
                                # 神经网络决策（红方学生走共享实例 / 蓝方历史 nn 按 id 取）
                                act_model = student_actor if is_student else actor_dict[id]
                                rule_input4actor = stack_field([take_some_of_rule_inputs(rule_inputs_[id])])
                                with torch.no_grad():
                                    actor_out = act_model.get_action(
                                        agent_inputs_[id], mask_inputs_[id], rule_input4actor)
                                vert = int(actor_out['vert_idx'].item())
                                hor = int(actor_out['hor_idx'].item())
                                fire = int(actor_out['fire'].item())
                                enm_idx = int(actor_out['enm_idx'].item())
                                maneuver_cmds_all[id] = [vert, hor]
                                target_index_all[id] = enm_idx
                                fire_target = enm_idx
                                fire_flag = bool(fire)
                                # 记录动作供经验池使用（仅学生，log_prob 为行为策略真实值，update 直接读取）
                                if is_student:
                                    Last_Actions[id] = {
                                        "vert": vert, "hor": hor, "fire": fire,
                                        "enm_idx": enm_idx,
                                        "ally_idx": int(actor_out['ally_idx'].item()),
                                        "log_prob_enm": float(actor_out['log_prob_enm'].item()),
                                        "log_prob_ally": float(actor_out['log_prob_ally'].item()),
                                        "log_prob_act": float(actor_out['log_prob_act'].item()),
                                    }
                            else:
                                # 蓝方规则对手
                                fly_action, fire_action, _ = basic_rules(
                                    rule_inputs_[id], mask_inputs_[id], rules_num=opp_data)
                                maneuver_cmds_all[id] = fly_action['do']
                                target_index_all[id] = int(fly_action['at'])
                                fire_target = int(fire_action['at'])
                                fire_flag = bool(fire_action['do'])

                            # 决策瞬间的 obs/states 只按学生记录
                            if is_student:
                                Last_Obs[id] = {
                                    'agent_input': agent_inputs_[id],
                                    'mask_input': mask_inputs_[id],
                                    'rule_input': take_some_of_rule_inputs(rule_inputs_[id]),
                                }
                                Last_States[id] = Curr_States[id]

                            if fire_flag:
                                try:
                                    env.fire_120(id, fire_target, tabu=0)
                                except ValueError as e:
                                    pass
                            env.radar_action(id, 1)

                    # 执行机动
                    flight_cmds = {}
                    for id in env.UAV_ids:
                        try:
                            flight_cmds[id] = env.maneuver_for_1_enm(
                                id, target_index_all[id], maneuver_cmds_all[id], rule_inputs_[id])
                        except ValueError:
                            flight_cmds[id] = [0.0, 0.0, 0.0]
                    env.step(flight_cmds.values())

                    # 奖励结算（combat_terminate_and_reward 每机都要调，回报只按学生累计）
                    ep_return = 0.0
                    Curr_States = {}
                    for id in env.UAV_ids:
                        done, reward_dict = env.combat_terminate_and_reward(
                            id=id, action_cycle_multiplier=action_cycle_multiplier)
                        if id in Rewards:
                            Rewards[id] = sum(reward_dict.values())
                            ep_return += Rewards[id]
                            ep_return_by_id[id] += Rewards[id]

                    for id in env.UAV_ids:
                        Curr_States[id], _, _ = env.get_obs(id, reward_fn=1)

                # 最后一帧经验（done=1，仅学生，统一走 append_experience）
                for id in student_ids:
                    UAV = env.find_uav_with_id(id)
                    LocalTrans[id] = append_experience(LocalTrans[id], Last_Obs[id], Last_States[id],
                                                        Last_Actions[id], Rewards[id],
                                                        Curr_States.get(id, Last_States[id]),
                                                        done=1, active_mask=not UAV.dead)

                # 堆叠各 id 经验（dict-of-arrays，只有学生 id）
                stacked_trans = {}
                for id, lt in LocalTrans.items():
                    stacked_trans[id] = {k: stack_field(v) for k, v in lt.items()}

                # 胜负判定：直接读环境端导出的 r_outcome（1 胜 / -1 负 / 0 平，学生为红方）
                outcome = env.r_outcome if env.r_outcome is not None else 0
                win = outcome == 1
                lose = outcome == -1
                draw = outcome == 0
                all_dead = all(uav.dead for uav in env.UAVs)

                # 汇总所有飞机的开火样本
                fire_hit_records = []
                for uav in env.UAVs:
                    fire_hit_records += uav.fire_hit_records
                
                pipe.send({
                    'trans': stacked_trans,
                    'metrics': {
                        'steps': count,
                        'return': ep_return,
                        'return_by_id': ep_return_by_id, # 个体回报
                        'event_return': 0,
                        'win': win,
                        'lose': lose,
                        'draw': draw,
                        'BVR_perish_together': all_dead and env.t <= args.max_episode_len - 10,
                    },
                    'opp_name': opp_name,
                    'fire_hit_records': fire_hit_records,
                })
            elif cmd == 'EXIT':
                break
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            pipe.send({'error': tb})
            break

