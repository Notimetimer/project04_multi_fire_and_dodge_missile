from numpy import argmin
import numpy as np
from math import *
import torch
import argparse
from numpy.linalg import norm
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)


from Envs.Tasks.ChooseStrategyEnv260421_terminate_and_reward import *
from Math_calculates.sub_of_angles import *
import re
import csv
import glob
import pickle

use_tacview = 2 # 1 动态显示 2 静默写入 acmi_records
action_cycle_multiplier = 10

from BasicRules_Homo import basic_rules
from Algorithms.ActorWithATTN import ActorWrapper
from _context import take_some_of_rule_inputs, load_obs_dims, stack_field, append_experience, load_init_states
from Algorithms.CriticWrapper import critic_wrapper
from torch.utils.tensorboard import SummaryWriter

# 是否考虑视野限制
POMDP = 1

if __name__=='__main__':
    print("\n根目录为：", project_root, "\n")
    # 在这里调用规则(编号)下的策略
    parser = argparse.ArgumentParser("UAV swarm confrontation")
    # # Environment
    # parser.add_argument("--max-episode-len", type=float, default=15*60,  # 8 * 60,
    #                     help="maximum episode time length")  # test 真的中远距空战可能会持续20分钟那么长
    # parser.add_argument("--R-cage", type=float, default=71e3,
    #                     help="")
    args = parser.parse_args()
    
    args.max_episode_len = 15*60
    args.R_cage = 60.3e3

    # 构建场地边界，南北长54km，东西宽100km的长方形边界
    vertices = [[60e3, 60e3], [-60e3, 60e3], [-60e3, -60e3], [60e3, -60e3]]
    env = ChooseStrategyEnv(args, tacview_show=use_tacview, vertices=vertices)
    # test
    env.dt_move = 0.07 # 0.04 # 4 # 0.025 0.02

    device = torch.device("cpu")  # torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


    def creat_initial_state():
        # 飞机出生状态指定
        blue_height = 9000
        red_height = 9000
        red_psi = pi/2
        blue_psi = - red_psi 
        red_N = 0
        red_E = - 55e3
        blue_N = red_N
        blue_E = -red_E
        DEFAULT_RED_BIRTH_STATE = {'position': np.array([red_N, red_height, red_E]),
                            'psi': red_psi
                            }
        DEFAULT_BLUE_BIRTH_STATE = {'position': np.array([blue_N, blue_height, blue_E]),
                                    'psi': blue_psi
                                    }
        return DEFAULT_RED_BIRTH_STATE, DEFAULT_BLUE_BIRTH_STATE

    dt_action_cycle = dt_maneuver * action_cycle_multiplier # Agent takes action every dt_action_cycle seconds

    transition_dict_threshold = env.args.max_episode_len//dt_action_cycle + 1 # Adjusted capacity

    steps_count = 0

    total_steps = 0

    training_start_time = time.time()
    launch_time_count = 0

    t_bias = 0

    decide_steps_after_update = 0

    try:
        r_action_list = []
        b_action_list = []
        r_guide_list = []
        b_guide_list = []
        
        # 启动仿真前清空上次遗留的pkl和csv
        event_log_dir = os.path.join(project_root, 'acmi_records')
        os.makedirs(event_log_dir, exist_ok=True)
        for pattern in ('*.pkl', '*.csv'):
            for f in glob.glob(os.path.join(event_log_dir, pattern)):
                os.remove(f)

        
        # 初始化经验回放池（延后到 reset 之后，因为 env.UAV_ids 在 reset 前为 None）
        LocalTrans = {}
        last_Actions = {}
        Rewards = {}
        
        # 采集不同轨迹的动作
        episode_times = []  # 记录每场耗时
        # 红方每架飞机各用一个独立随机初始化的 actor（不加载权重），按 id 索引；蓝方保持规则
        actor_dict = None
        for i_episode in range(3): # 5
            ep_start = time.time()
            prev_total = sum(episode_times)
            print(f"开始第{i_episode+1}场对抗，前{i_episode}场对抗耗时：{prev_total:.1f}s")
            episode_return = 0
            DEFAULT_RED_BIRTH_STATE, DEFAULT_BLUE_BIRTH_STATE = creat_initial_state()
            init_cfg = load_init_states() # 初始数量/挂载统一从 init_states.json 读取
            env.reset(
                red_birth_state=DEFAULT_RED_BIRTH_STATE, blue_birth_state=DEFAULT_BLUE_BIRTH_STATE,
                red_init_num=init_cfg['red_init_num'],
                blue_init_num=init_cfg['blue_init_num'],
                red_init_ammo=init_cfg['red_init_ammo'],
                blue_init_ammo=init_cfg['blue_init_ammo'],
                )
            r_action_label=0
            b_action_label=0

            # 首场 reset 后为每架红机各建一个独立随机初始化的 ActorWrapper，按 id 索引
            if actor_dict is None:
                actor_dict = {}
                red_ids = [i for i in env.UAV_ids if env.find_uav_with_id(i).side == 'r']
                # 网络维度统一从 env_obs_dims.json 读取（由 读取环境参数并存json.py 生成）
                fixed_dim, enm_shape, ally_shape = load_obs_dims()
                for rid in red_ids:
                    actor_dict[rid] = ActorWrapper(
                        fixed_dim=fixed_dim, enm_shape=enm_shape, ally_shape=ally_shape, device=device,
                    )

            # 各机上一决策步的动作缓存（按 id 保存，供执行阶段逐机取用）
            maneuver_cmds_all = {}
            target_index_all = {}

            # 首个 episode 时根据 reset 后的 UAV_ids 初始化各 id 独立的经验池
            if not LocalTrans:
                for _id in env.UAV_ids:
                    LocalTrans[_id] = {'obs': [], 'states': [], 'actions': [],
                                       'next_states': [], 'rewards': [], 'dones': [], 'active_masks': []}
                last_Actions = {id: None for id in env.UAV_ids}
                Rewards = {id: None for id in env.UAV_ids}

            done = False
            Last_Obs = {id: None for id in env.UAV_ids}
            Last_States = {id: None for id in env.UAV_ids}
            Curr_States = {id: None for id in env.UAV_ids}
            
            env.dt_move = 0.07
            env.dt_maneuver = 0.2
            env.min_alt_safe = 2000
            
            episode_start_time = time.time()

            # 环境运行一轮的情况
            steps_of_this_eps = -1 # 没办法了
            
            # 一轮仿真开始
            for count in range(round(args.max_episode_len / dt_maneuver)):
                # 回合结束判断
                current_t = count * dt_maneuver
                steps_of_this_eps += 1
                if env.running == False or done:
                    # print('回合结束，时间为：', env.t, 's')
                    break
                # 获取观测信息
                Obs = {id: None for id in env.UAV_ids}
                agent_inputs_ = {}
                mask_inputs_ = {}
                rule_inputs_ = {}
                prim_rule_inputs_ = {}

                for id in env.UAV_ids:
                    agent_input, mask_input, rule_input = env.get_obs(id, pomdp=POMDP)
                    Curr_States[id], _, _ = env.get_obs(id, reward_fn=1)
                    agent_inputs_[id] = agent_input
                    mask_inputs_[id] = mask_input
                    prim_rule_inputs_[id] = copy.deepcopy(rule_input)
                    rule_inputs_[id] = take_some_of_rule_inputs(rule_input)
                    Obs[id] = copy.deepcopy({"agent_input":agent_input, "mask_input":mask_input, "rule_input":rule_inputs_[id]})
                # # TODO CurrStates从Obs获取fixed信息

                # --- 智能体决策 ---
                # 判断是否到达了决策点（每 10 步）
                if steps_of_this_eps % action_cycle_multiplier == 0:
                    for id in env.UAV_ids:
                        # debug
                        UAV = env.find_uav_with_id(id)
                        # side_word = 'red' if UAV.side == 'r' else 'blue'
                        # num_word = id if UAV.side == 'r' else id-200
                        # print("我是", side_word, num_word)
                        if Last_Obs[id] is not None and steps_of_this_eps > 0:
                            LocalTrans[id] = append_experience(LocalTrans[id], Last_Obs[id], Last_States[id], 
                                                               last_Actions[id], Rewards[id], Curr_States[id], 
                                                               done=0, active_mask=not UAV.dead)
                        
                        # Agent 产生动作：红方走随机初始化的 actor，蓝方走规则
                        if UAV.side == 'r':
                            # rule_inputs_ 是 take_some_of_rule_inputs 的子集（list-of-dict），
                            # 用 stack_field 单样本堆叠成 selector 期望的 dict-of-array（[B=1]）格式
                            actor_rule = stack_field([rule_inputs_[id]])
                            with torch.no_grad():
                                actor_out = actor_dict[id].get_action(agent_inputs_[id], mask_inputs_[id], actor_rule)
                            vert = int(actor_out['vert_idx'].item())
                            hor = int(actor_out['hor_idx'].item())
                            fire = int(actor_out['fire'].item())
                            enm_idx = int(actor_out['enm_idx'].item())
                            ally_idx = int(actor_out['ally_idx'].item())
                            log_prob_enm = float(actor_out['log_prob_enm'].item())
                            log_prob_ally = float(actor_out['log_prob_ally'].item())
                            log_prob_act = float(actor_out['log_prob_act'].item())
                            maneuver_cmds_all[id] = [vert, hor]
                            target_index_all[id] = enm_idx
                            fire_target = enm_idx
                            fire_flag = bool(fire)
                        else:
                            # 如果类型是规则智能体而非神经网络
                            fly_action, fire_action, radar_action = basic_rules(prim_rule_inputs_[id], mask_inputs_[id], rules_num=5)
                            vert = int(fly_action["do"][0])
                            hor = int(fly_action["do"][1])
                            fire = int(bool(fire_action["do"]))
                            enm_idx = int(fly_action["at"])
                            ally_idx = 0
                            log_prob_enm = -1e-6
                            log_prob_ally = -1e-6
                            log_prob_act = -1e-6
                            maneuver_cmds_all[id] = fly_action['do']
                            target_index_all[id] = enm_idx
                            fire_target = int(fire_action['at'])
                            fire_flag = bool(fire_action['do'])
                        
                        # 决策瞬间的states和obs
                        Last_Obs[id] = copy.deepcopy(Obs[id])
                        Last_States[id] = copy.deepcopy(Curr_States[id])
                        last_Actions[id] = copy.deepcopy({
                                "vert": vert,
                                "hor":  hor ,
                                "fire": fire,
                                "enm_idx": enm_idx,
                                "ally_idx": ally_idx,
                                "log_prob_enm": log_prob_enm,
                                "log_prob_ally": log_prob_ally,
                                "log_prob_act": log_prob_act,
                            })
                        
                        # ---- 1 号机张量化数据采集已由 LocalTrans 统一处理 ----

                        # 暂不考虑同时攻击多个目标
                        if fire_flag:
                            try:
                                env.fire_120(id, fire_target, tabu=0)
                            except ValueError as e:
                                # 环境端对虚目标/越界索引抛错，这里仅打印不中断（规则智能体会输出 null_token 索引）
                                print(f"[fire_120 警告] {e}")

                        # 雷达全程开
                        env.radar_action(id, 1)

                        # debug
                        # if side_word == 'red' and \
                        #     (
                        #     # 47 < env.t < 64 or\
                        #         3*60 < env.t < 3*60+12
                        #     ):
                        #     print('id', UAV.id)
                        #     print('enm_infos', rule_inputs_[id]['enm_infos'][fly_action['at']])
                        #     print('weapon_infos', rule_inputs_[id]['weapon_infos'])
                        #     print('RWR_infos', rule_inputs_[id]['RWR_infos'])
                        #     print('机动动作是：', fly_action)
                        #     print('开火动作是:', fire_action)
                        #     print()
                    decide_steps_after_update += 1

                # 循环调用底层控制器，环境更新
                flight_cmds = {}
                for id in env.UAV_ids:
                    maneuver_cmds = maneuver_cmds_all[id]
                    target_index = target_index_all[id]
                    try:
                        flight_cmds[id] = env.maneuver_for_1_enm(id, target_index, maneuver_cmds, prim_rule_inputs_[id])
                    except ValueError as e:
                        # 环境端对虚目标/越界索引抛错，规则智能体全灭时会输出 null_token 索引，用默认平飞指令兜底
                        print(f"[maneuver_for_1_enm 警告] {e}")
                        flight_cmds[id] = [0.0, 0.0, 0.0]
                env.step(flight_cmds.values())  # 环境统一更新
                
                # # DEBUG
                # if steps_of_this_eps % action_cycle_multiplier == 0:
                #     for id in env.UAV_ids:
                #         # debug
                #         UAV = env.find_uav_with_id(id)
                #         if UAV.dead:
                #             print("检查导弹的目标")
                
                Obs = {id: None for id in env.UAV_ids}
                for id in env.UAV_ids: # 每架飞机都调用一遍，保证各自的周期事件都被积累
                    # action_shoots是过时接口，不再通过这个判断开火了
                    done, reward_dict = env.combat_terminate_and_reward(id=id, action_cycle_multiplier=action_cycle_multiplier)
                    Rewards[id] = sum(reward_dict.values())
                    agent_input, mask_input, rule_input = env.get_obs(id, pomdp=POMDP)
                    Curr_States[id], _, _ = env.get_obs(id, reward_fn=1)
                    agent_inputs_[id] = agent_input
                    mask_inputs_[id] = mask_input
                    prim_rule_inputs_[id] = copy.deepcopy(rule_input)
                    rule_inputs_[id] = take_some_of_rule_inputs(rule_input)
                    Obs[id] = copy.deepcopy({"agent_input":agent_input, "mask_input":mask_input, "rule_input":rule_inputs_[id]})
                    # 1 号机 done 采集（仅在决策步，与 obs/actions 对齐）
                    # done 采集已由 LocalTrans 统一处理
                # # TODO CurrStates从Obs获取fixed信息
                
                '''显示运行轨迹'''
                # 可视化
                env.render(t_bias=t_bias)

            # 这里已经是最后的next_states
            for id in env.UAV_ids:
                # buffer再保存一次
                UAV = env.find_uav_with_id(id)
                LocalTrans[id] = append_experience(LocalTrans[id], Last_Obs[id], Last_States[id], 
                                                    last_Actions[id], Rewards[id], Curr_States[id], 
                                                    done=1, active_mask=not UAV.dead)
            
            # 仿真残余清理
            env.clear_render(t_bias=t_bias)
            t_bias += env.t
            
            # 时间记录
            episode_end_time = time.time()  # 记录结束时间
            ep_cost = episode_end_time - ep_start
            episode_times.append(ep_cost)
            print(f"第{i_episode+1}场对抗耗时：{ep_cost:.1f}s, 仿真倍率：{env.t/ep_cost:.1f}")
            
            

            # 保存本回合各机的事件日志和mask调试信息，每架飞机一个csv
            for uav in env.UAVs:
                events = getattr(uav, 'cycle_events', [])
                if not events:
                    continue
                # 从 LocalTrans 提取 mask_input（obs 字段里每条经验的 mask_input）
                masks = [obs['mask_input'] for obs in LocalTrans[uav.id]['obs']
                         if obs is not None and 'mask_input' in obs]
                event_rows = []
                for index, event in enumerate(events):
                    mask = masks[index] if index < len(masks) else {}
                    mask_ally = np.asarray(mask.get('ally', [])).tolist()
                    mask_enm = np.asarray(mask.get('enm', [])).tolist()
                    event_rows.append((*event, mask_ally, mask_enm))
                csv_path = os.path.join(event_log_dir, f'uav_{uav.id}_events.csv')
                write_header = not os.path.exists(csv_path)
                with open(csv_path, 'a', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f)
                    if write_header:
                        writer.writerow(['time', 'id', 'dense_reward', 'end_reward', 'event_reward', 'event',
                                         'mask_ally', 'mask_enm'])
                    writer.writerows(event_rows)

            # 保存本回合各 agent 的 LocalTrans 张量化版本（增量覆盖，累积所有 episode）
            for id in env.UAV_ids:
                lt = LocalTrans[id]
                if not lt['obs']:
                    continue
                tensor_lt = {k: stack_field(lt[k]) for k in lt}
                save_path = os.path.join(event_log_dir, f'agent{id}_local_trans.pkl')
                with open(save_path, 'wb') as f:
                    pickle.dump(tensor_lt, f)
                obs_len = len(lt['obs'])
                # 打印关键维度供快速核对
                _obs = tensor_lt.get('obs')
                _act = tensor_lt.get('actions')
                _obs_enm = _obs.get('agent_input', {}).get('enm') if isinstance(_obs, dict) else None
                _act_hor = _act.get('hor') if isinstance(_act, dict) else None
                print(f"[LocalTrans] 已保存: {save_path} "
                      f"(obs {obs_len} 步, "
                      f"obs.enm {getattr(_obs_enm, 'shape', '?')}, "
                      f"actions.hor {getattr(_act_hor, 'shape', '?')})")

        training_end_time = time.time()  # 记录结束时间
        total_cost = training_end_time - training_start_time
        n_eps = len(episode_times)
        avg_cost = total_cost / n_eps if n_eps > 0 else 0
        print(f"总耗时：{total_cost:.1f}s，平均每场耗时：{avg_cost:.1f}s")

        # ===== 结尾附加：从 LocalTrans[第一架红机] 抽取一个 batch，各做一次 Actor / Critic 正向 =====
        # Actor 走 evaluate_actions（回放动作上的 log_prob 与策略熵），Critic 走 get_value（全局 states）；
        # 输出保存为 numpy(.npz)，正向计算图分别用 tensorboard 记录到 CalcGraphRecords 下的子目录
        calc_graph_dir = os.path.join(project_root, 'CalcGraphRecords')
        os.makedirs(calc_graph_dir, exist_ok=True)
        npz_dir = os.path.join(project_root, 'acmi_records')
        os.makedirs(npz_dir, exist_ok=True)
        
        # LocalTrans 的键是 UAV id（红方 1/2，蓝方 201/202），取第一架红机的经验
        rec_id = [i for i in env.UAV_ids if env.find_uav_with_id(i).side == 'r'][0]
        BATCH_SIZE = 64

        # 递归截取 dict-of-array 的前 n 条作为一个 batch
        def take_batch(tree, n):
            if isinstance(tree, dict):
                return {k: take_batch(v, n) for k, v in tree.items()}
            if isinstance(tree, np.ndarray):
                return tree[:n]
            return tree

        # 堆叠该机全部经验为 dict-of-array，再截取前 BATCH_SIZE 条
        stacked = {k: stack_field(LocalTrans[rec_id][k]) for k in LocalTrans[rec_id]}
        batch_obs = take_batch(stacked['obs'], BATCH_SIZE)          # {'agent_input','mask_input','rule_input'}
        batch_states = take_batch(stacked['states'], BATCH_SIZE)    # {'fixed','ally','enm'} 全局状态
        batch_actions = take_batch(stacked['actions'], BATCH_SIZE)  # {'vert','hor','fire','enm_idx','ally_idx',...}
        real_B = batch_states['fixed'].shape[0]
        print(f"[CalcGraph] 取 id={rec_id} 前 {real_B} 条经验做一次正向计算")

        # --- Actor：一次批量 evaluate_actions，保存 log_prob 与策略熵的 numpy 形式 ---
        actor = actor_dict[rec_id]
        with torch.no_grad():
            actor_out = actor.evaluate_actions(
                batch_obs['agent_input'], batch_obs['mask_input'],
                batch_obs['rule_input'], batch_actions,
            )
        actor_np = {k: v.detach().cpu().numpy() for k, v in actor_out.items() if torch.is_tensor(v)}
        actor_np.update({f'entropy_{k}': v.detach().cpu().numpy()
                         for k, v in actor_out['entropy_details'].items()})
        actor_np_path = os.path.join(npz_dir, 'actor_evaluate_outputs.npz')
        np.savez(actor_np_path, **actor_np)
        print(f"[CalcGraph] Actor 输出已保存: {actor_np_path} (keys: {sorted(actor_np.keys())})")

        # --- Critic：随机初始化 critic_wrapper 对全局 states 做一次 get_value，保存输出 ---
        critic = critic_wrapper(
            fixed_dim=batch_states['fixed'].shape[1],
            enm_shape=batch_states['enm'].shape[1:],
            ally_shape=batch_states['ally'].shape[1:],
            device=device,
        )
        with torch.no_grad():
            critic_out = critic.get_value(batch_states)
        critic_np = {k: v.detach().cpu().numpy() for k, v in critic_out.items()}
        critic_np_path = os.path.join(npz_dir, 'critic_value_outputs.npz')
        np.savez(critic_np_path, **critic_np)
        print(f"[CalcGraph] Critic 输出已保存: {critic_np_path} (keys: {sorted(critic_np.keys())})")

        # --- tensorboard：分别记录两者的正向计算图 ---
        class ActorGraphWrapper(torch.nn.Module):
            # ActorWrapper 未定义 forward，包一层供 add_graph 追踪 evaluate_actions；
            # rule dict 顶层值类型混杂（标量 Tensor + 嵌套 dict），torch 1.11 tracer 无法推断其类型，
            # 故拆成 4 个独立图输入，在 forward 内部重组回 rule dict（图内部的混合 dict 可正常追踪）
            def __init__(self, actor):
                super().__init__()
                self.actor = actor

            def forward(self, agent_inputs, mask_inputs, actions, ammo, RWR_infos, weapon_infos, enm_infos):
                rule_inputs = {'ammo': ammo, 'RWR_infos': RWR_infos,
                               'weapon_infos': weapon_infos, 'enm_infos': enm_infos}
                out = self.actor.evaluate_actions(agent_inputs, mask_inputs, rule_inputs, actions)
                # trace 输出不能是 dict（混合类型无法推断，且 torch 1.11 要求输出用 tuple）：
                # 摊平 entropy_details 后按固定键序返回 tuple
                flat = {k: v for k, v in out.items() if torch.is_tensor(v)}
                flat.update({f'entropy_{k}': v for k, v in out['entropy_details'].items()})
                key_order = ['log_prob', 'log_prob_enm', 'log_prob_ally', 'log_prob_act', 'entropy',
                             'entropy_enm', 'entropy_ally', 'entropy_vert', 'entropy_hor', 'entropy_fire',
                             'entropy_act']
                return tuple(flat[k] for k in key_order)

        class CriticGraphWrapper(torch.nn.Module):
            # critic_wrapper 同样未定义 forward，包一层供 add_graph 追踪 get_value
            def __init__(self, critic):
                super().__init__()
                self.critic = critic

            def forward(self, agent_inputs):
                out = self.critic.get_value(agent_inputs)
                # trace 输出用 tuple（同 ActorGraphWrapper，dict 输出会被 torch 1.11 tracer 拒绝）
                return (out['enm_logits'], out['ally_logits'], out['value'])

        # 递归把 numpy 数组转为 tensor（追踪时 dict 内非 tensor 值不会被记录为图输入）
        def to_tensor_tree(x):
            if isinstance(x, dict):
                return {k: to_tensor_tree(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [to_tensor_tree(v) for v in x]
            return torch.as_tensor(np.asarray(x))

        rule_t = to_tensor_tree(batch_obs['rule_input'])
        actor_graph_inputs = (
            to_tensor_tree(batch_obs['agent_input']),
            to_tensor_tree(batch_obs['mask_input']),
            to_tensor_tree(batch_actions),
            rule_t['ammo'], rule_t['RWR_infos'], rule_t['weapon_infos'], rule_t['enm_infos'],
        )
        critic_graph_inputs = (to_tensor_tree(batch_states),)

        actor_writer = SummaryWriter(log_dir=os.path.join(calc_graph_dir, 'actor_graph'))
        actor_writer.add_graph(ActorGraphWrapper(actor), actor_graph_inputs)
        actor_writer.close()

        critic_writer = SummaryWriter(log_dir=os.path.join(calc_graph_dir, 'critic_graph'))
        critic_writer.add_graph(CriticGraphWrapper(critic), critic_graph_inputs)
        critic_writer.close()
        print(f"[CalcGraph] 正向计算图已写入 tensorboard: "
              f"{os.path.join(calc_graph_dir, 'actor_graph')} 与 {os.path.join(calc_graph_dir, 'critic_graph')}")

    except KeyboardInterrupt:
        print("\n检测到 KeyboardInterrupt")
    finally:
        env.end_render() # 停止发送

