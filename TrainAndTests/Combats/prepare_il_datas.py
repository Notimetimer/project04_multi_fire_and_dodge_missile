# 混合机动与发射
from BasicRules_new import *  # 可以直接读同一级目录
from Algorithms.Utils import compute_monte_carlo_returns

def run_rules(gamma=0.995, weight_reward=np.array([1,1,0]), action_cycle_multiplier=30, shielded=0, current_rule=4):
    gamma = gamma
    
    # 在这里调用规则(编号)下的策略
    parser = argparse.ArgumentParser("UAV swarm confrontation")
    # Environment
    parser.add_argument("--max-episode-len", type=float, default=10*60,  # 8 * 60,
                        help="maximum episode time length")  # test 真的中远距空战可能会持续20分钟那么长
    parser.add_argument("--R-cage", type=float, default=45e3,  # 70e3 还是太大了
                        help="")
    args = parser.parse_args()

    env = ChooseStrategyEnv(args, tacview_show=0)

    env.shielded = shielded

    def creat_initial_state():
        # 飞机出生状态指定
        # todo: 随机出生点，确保蓝方能躲掉但不躲就会被打到
        blue_height = 9000
        red_height = 9000
        red_psi = -pi/2
        blue_psi = pi/2
        red_N = 0
        red_E = 45e3
        blue_N = red_N
        blue_E = -red_E
        DEFAULT_RED_BIRTH_STATE = {'position': np.array([red_N, red_height, red_E]),
                            'psi': red_psi
                            }
        DEFAULT_BLUE_BIRTH_STATE = {'position': np.array([blue_N, blue_height, blue_E]),
                                    'psi': blue_psi
                                    }
        return DEFAULT_RED_BIRTH_STATE, DEFAULT_BLUE_BIRTH_STATE

    t_bias = 0

    decide_steps_after_update = 0
    try:
        r_action_list = []
        b_action_list = []
        
        transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
        il_transition_dict = {'states':[], 'actions': [], 'returns': []}
        
        # 示范数据采集
        for i_episode in range(5):

            last_r_action_label = 0
            last_b_action_label = 0

            episode_return = 0
            
            DEFAULT_RED_BIRTH_STATE, DEFAULT_BLUE_BIRTH_STATE = creat_initial_state()

            env.reset(red_birth_state=DEFAULT_RED_BIRTH_STATE, blue_birth_state=DEFAULT_BLUE_BIRTH_STATE,
                    red_init_ammo=6, blue_init_ammo=6)
            r_action_label=0
            b_action_label=0
            last_decision_state = None
            current_action = None

            done = False

            env.dt_maneuver = dt_maneuver

            # 环境运行一轮的情况
            steps_of_this_eps = -1 # 没办法了
            for count in range(round(args.max_episode_len / dt_maneuver)):
                current_t = count * dt_maneuver
                steps_of_this_eps += 1
                if env.running == False or done: # count == round(args.max_episode_len / dt_maneuver) - 1:
                    # print('回合结束，时间为：', env.t, 's')
                    break
                # 获取观测信息
                r_obs, r_check_obs = env.obs_1v1('r', pomdp=1)
                b_obs, b_check_obs = env.obs_1v1('b', pomdp=1)

                # --- 智能体决策 ---
                # 判断是否到达了决策点（每 10 步）
                if steps_of_this_eps % action_cycle_multiplier == 0:
                    # # **关键点 1: 完成并存储【上一个】动作周期的经验**
                    # 如果这不是回合的第0步，说明一个完整的动作周期已经过去了
                    if steps_of_this_eps > 0:
                        transition_dict['states'].append(last_decision_state)
                        transition_dict['actions'].append(current_action)
                        transition_dict['rewards'].append(b_reward)
                        transition_dict['next_states'].append(b_obs) # 当前状态是上个周期的 next_state
                        transition_dict['dones'].append(False) # 没结束，所以是 False

                    # **关键点 2: 开始【新的】一个动作周期**
                    # 1. 记录新周期的起始状态
                    last_decision_state = b_obs
                    # 2. Agent 产生一个动作
                    
                    # 红方改变规则
                    r_state_check = env.unscale_state(r_check_obs)
                    r_action_label, r_fire = basic_rules(r_state_check, i_episode)
                    if r_fire:
                        launch_missile_immediately(env, 'r')

                    # 蓝方维持最优规则
                    b_state_check = env.unscale_state(b_check_obs)
                    b_action_label, b_fire = basic_rules(b_state_check, current_rule)
                    if b_fire:
                        launch_missile_immediately(env, 'b')

                    decide_steps_after_update += 1
                    
                    b_action_list.append(np.array([env.t + t_bias, b_action_label]))
                    current_action = {'cat': np.array([b_action_label]), 'bern': np.array([b_fire])}
                    # current_action = np.array([b_action_label, b_fire])

                r_action = env.maneuver14LR(env.RUAV, r_action_label)
                b_action = env.maneuver14LR(env.BUAV, b_action_label)

                env.step(r_action, b_action) # Environment updates every dt_maneuver
                done, b_reward_event, b_reward_constraint, b_reward_shaping = env.combat_terminate_and_reward('b', b_action_label, b_fire)
                
                b_reward = sum(np.array([b_reward_event, b_reward_constraint, b_reward_shaping]) * weight_reward)

                # Accumulate rewards between agent decisions
                episode_return += b_reward * env.dt_maneuver

                next_b_check_obs = env.base_obs('b')
                next_b_obs = flatten_obs(next_b_check_obs, env.key_order)

                '''显示运行轨迹'''
                # 可视化
                env.render(t_bias=t_bias)
            
            # # --- 回合结束处理 ---
            # **关键点 3: 存储【最后一个】不完整的动作周期的经验**
            # 循环结束后，最后一个动作周期因为 done=True 而中断，必须在这里手动存入
            if last_decision_state is not None:
                transition_dict['states'].append(last_decision_state)
                transition_dict['actions'].append(current_action)
                transition_dict['rewards'].append(b_reward)
                transition_dict['next_states'].append(next_b_obs) # 最后的 next_state 是环境的最终状态
                transition_dict['dones'].append(True)
            
            # print(t_bias)
            env.clear_render(t_bias=t_bias)
            t_bias += env.t
            r_action_list = np.array(r_action_list)


        # 计算蒙特卡洛回报
        il_transition_dict['states'] = transition_dict['states']
        il_transition_dict['actions'] = transition_dict['actions']
        il_transition_dict['returns'] = compute_monte_carlo_returns(gamma, \
                                                                    transition_dict['rewards'], \
                                                                    transition_dict['dones'])

        # 保存两个dict
        # 保存到当前脚本所在目录（只保存 pickle，且同时保存 transition_dict 以便后续分析）
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        il_dir = os.path.join(cur_dir, "IL")
        os.makedirs(il_dir, exist_ok=True)
        
        il_pkl_path = os.path.join(il_dir, "il_transitions_combat_LR.pkl")
        trans_pkl_path = os.path.join(il_dir, "transition_dict_combat_LR.pkl")
        
        import pickle
        # 保存示范轨迹（IL 用）
        with open(il_pkl_path, "wb") as f:
            pickle.dump(il_transition_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        # 保存原始 transition_dict（包含 next_states, rewards, dones 等）
        with open(trans_pkl_path, "wb") as f:
            pickle.dump(transition_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Saved il_transitions_combat to: {il_pkl_path}")
        print(f"Saved transition_dict_combat to: {trans_pkl_path}")

        
    except KeyboardInterrupt:
        print("\n检测到 KeyboardInterrupt")
    finally:
        env.end_render() # 停止发送


if __name__=='__main__':
    run_rules()

