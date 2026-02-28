import os
import sys
import numpy as np
import pickle
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from TrainAndTests.MARWIL.pure_marwil_train_shoot import *
from Algorithms.PPObernouli import *

student_agent = PPO_bernouli(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                lmbda, epochs, eps, gamma, device)

def load_npz(path):
    data = np.load(path, allow_pickle=True)
    return {
        'states': list(data['states']),
        'actions': list(data['actions']),
        'returns': list(data['returns'])
    }

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def summarize(il_transition_dict):
    print("Loaded transitions:")
    for k in ('states','actions','returns'):
        v = il_transition_dict.get(k)
        print(f"  {k}: {None if v is None else len(v)} items")
    # sample few entries
    for i in range(min(3, len(il_transition_dict.get('states',[])) )):
        s = il_transition_dict['states'][i]
        a = il_transition_dict['actions'][i] if i < len(il_transition_dict['actions']) else None
        r = il_transition_dict['returns'][i] if i < len(il_transition_dict['returns']) else None
        print(f" sample {i}: state type={type(s)}, state.shape={getattr(np.asarray(s),'shape',None)}; "
              f"action type={type(a)}, returns type={type(r)}")

if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    npz_path = os.path.join(cur_dir, "il_transitions_shoot.npz")
    pkl_path = os.path.join(cur_dir, "il_transitions_shoot.pkl")

    il_transition_dict = None
    if os.path.isfile(npz_path):
        try:
            il_transition_dict = load_npz(npz_path)
            print(f"Loaded npz: {npz_path}")
        except Exception as e:
            print(f"Failed to load npz ({npz_path}): {e}")

    if il_transition_dict is None and os.path.isfile(pkl_path):
        try:
            il_transition_dict = load_pkl(pkl_path)
            print(f"Loaded pkl: {pkl_path}")
        except Exception as e:
            print(f"Failed to load pkl ({pkl_path}): {e}")

    if il_transition_dict is None:
        print("No il_transitions_shoot file found in this directory.")
        sys.exit(1)

    summarize(il_transition_dict)

    # 把 states/actions 转为 numpy array（若元素形状一致）
    try:
        states_arr = np.asarray(il_transition_dict['states'])
        actions_arr = np.asarray(il_transition_dict['actions'])
        print("Converted to numpy arrays:", states_arr.shape, actions_arr.shape)
    except Exception:
        print("States/actions could not be converted to regular numpy arrays (mixed shapes/objects).")

    # --- 新增：预处理 transitions，保证可以转换为 torch.tensor ---
    def preprocess_transitions(trans):
        """把 'states','actions','returns' 转成数值 numpy arrays（float32）。
           - 若元素形状一致则 stack。
           - 若形状不一致，不进行 pad，而是丢弃与主要形状不匹配的样本（保守策略）。
        """
        for key in ('states','actions','returns'):
            items = trans.get(key, [])
            if len(items) == 0:
                trans[key] = np.array([], dtype=np.float32)
                continue

            arr = np.asarray(items)
            # 如果不是 object（即形状一致或能直接构成 ndarray），直接转换类型
            if arr.dtype != object:
                trans[key] = arr.astype(np.float32)
                continue

            # 否则逐项检查形状
            nds = [np.asarray(x) for x in items]
            shapes = [n.shape for n in nds]

            # 如果所有形状一致，直接 stack
            if all(s == shapes[0] for s in shapes):
                trans[key] = np.stack([n.astype(np.float32) for n in nds])
                continue

            # 形状不一致：保守策略 -> 只保留与第一种形状相同的样本（不做 pad）
            target_shape = shapes[0]
            kept = [n.astype(np.float32) for n, s in zip(nds, shapes) if s == target_shape]

            if len(kept) == 0:
                # 如果没有任何匹配的样本，设为空数组
                trans[key] = np.array([], dtype=np.float32)
            else:
                trans[key] = np.stack(kept)

        return trans

    il_transition_dict = preprocess_transitions(il_transition_dict)
    print("Preprocessed shapes:", {k: np.asarray(il_transition_dict[k]).shape for k in ('states','actions','returns')})

    # 第4步：有监督学习teacher动作
    logs_dir = os.path.join(project_root, "logs/shoot")
    mission_name = 'MARWIL'
    log_dir = os.path.join(logs_dir, f"{mission_name}-run-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    logger = TensorBoardLogger(log_root=log_dir, host="127.0.0.1", port=6006, use_log_root=True, auto_show=False)
    for epoch in range(80):
        avg_actor_loss, avg_critic_loss, c = student_agent.MARWIL_update(il_transition_dict, beta=1.0)
        # 记录损失函数
        if epoch % 1 == 0:
            logger.add("il_train/avg_actor_loss", avg_actor_loss, epoch)
            logger.add("il_train/avg_critic_loss", avg_critic_loss, epoch)

            print("epoch", epoch)



    # 第5步：测试有监督学习的效果
    t_bias = 0
    env = ShootTrainEnv(args, tacview_show=1) # Battle(args, tacview_show=1)
    
    rl_steps = 0
    return_list = []
    win_list = []
    # with tqdm(total=int(num_episodes*(1-pre_train_rate)), desc='Iteration') as pbar:  # 进度条
    # for i_episode in range(int(num_episodes*(1-pre_train_rate))):
    for i_episode in range(4):  # 10
        test_run = 1
        episode_return = 0
        transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}

        # 飞机出生状态指定
        red_E = 45e3 # random.uniform(30e3, 40e3)  # 20, 60 特意训练一个近的，测试一个远的
        blue_E= -red_E
        red_height = random.uniform(3e3, 12e3)
        blue_height = random.uniform(3e3, 12e3)

        DEFAULT_RED_BIRTH_STATE = {'position': np.array([0.0, red_height, red_E]),
                                    'psi': -pi/2
                                    }
        DEFAULT_BLUE_BIRTH_STATE = {'position': np.array([0.0, blue_height, blue_E]),
                                    'psi': pi/2
                                    }
        env.reset(red_birth_state=DEFAULT_RED_BIRTH_STATE, blue_birth_state=DEFAULT_BLUE_BIRTH_STATE,
                    red_init_ammo=0, blue_init_ammo=6)
        RUAV_set_speed = random.uniform(0.7, 1.5) * 340
        RUAV_set_height = red_height
        BUAV_set_speed = random.uniform(0.7, 1.5) * 340
        
        last_launch_time = - np.inf
        random_theta_plus = 0
        random_psi_plus = 0

        done = False

        actor_grad_list = []
        critc_grad_list = []
        actor_loss_list = []
        critic_loss_list = []
        entropy_list = []
        ratio_list = []

        r_action_list = []
        b_action_list = []

        episode_start_time = time.time()

        # 环境运行一轮的情况
        for count in range(round(args.max_episode_len / dt_maneuver)):
            # print(f"time: {env.t}")  # 打印当前的 count 值
            # 回合结束判断
            # print(env.running)
            current_t = count * dt_maneuver
            if env.running == False or done:  # count == round(args.max_episode_len / dt_maneuver) - 1:
                # print('回合结束，时间为：', env.t, 's')
                break
            # 获取观测信息
            r_obs_n, r_obs_check = env.attack_obs('r')
            b_obs_n, b_obs_check = env.attack_obs('b')

            # 在这里将观测信息压入记忆
            env.RUAV.obs_memory = r_obs_check.copy()
            env.BUAV.obs_memory = b_obs_check.copy()

            b_obs = np.squeeze(b_obs_n)

            cos_delta_psi = b_obs_check["target_information"][0]
            sin_delta_psi = b_obs_check["target_information"][1]
            delta_psi = atan2(sin_delta_psi, cos_delta_psi)

            distance = b_obs_check["target_information"][3]*10e3
            alpha = b_obs_check["target_information"][4]
            AA_hor = b_obs_check["target_information"][6]
            launch_interval = b_obs_check["weapon"]*120
            missile_in_mid_term = b_obs_check["missile_in_mid_term"]

            
            # 发射导弹判决
            u, _ = student_agent.take_action(b_obs_n, explore=0) # 0 1
            ut = u[0]
            at = ut

            # Shield
            at, _ = shoot_action_shield(at, distance, alpha, AA_hor, launch_interval)

            if at == 1:
                last_launch_time = env.t
                launch_missile_immediately(env, side='b')                  


            # 机动决策
            r_action_n = decision_rule(ego_pos_=env.RUAV.pos_, ego_psi=env.RUAV.psi,
                                        enm_pos_=env.BUAV.pos_, distance=distance,
                                        ally_missiles=env.Rmissiles, enm_missiles=env.Bmissiles,
                                        o00=o00, R_cage=env.R_cage, wander=0,
                                        set_height=RUAV_set_height, set_speed=RUAV_set_speed
                                        )

            random_theta_plus = generate_ar1_value(random_theta_plus, 0.9, 0.1)
            random_psi_plus = generate_ar1_value(random_psi_plus, 0.9, 0.1)
            

            b_action_n = np.array([env.RUAV.alt-env.BUAV.alt + 1000 * random_theta_plus, 
                                    delta_psi + pi/4 * random_psi_plus, 
                                    BUAV_set_speed])

            r_action_list.append(r_action_n)
            b_action_list.append(b_action_n)

            env.step(r_action_n, b_action_n)  # 2、环境更新并反馈
            done, b_reward, _ = env.get_terminate_and_reward('b', u)
            next_b_obs, _ = env.attack_obs('b')  # 子策略的训练不要用get_obs
            env.BUAV.act_memory = b_action_n.copy()  # 存储上一步动作

            # state = next_state
            episode_return += b_reward * env.dt_maneuver

            '''显示运行轨迹'''
            # 可视化
            env.render(t_bias=t_bias)

        

        if env.lose:
            print("回合失败")
        if env.win:
            print("回合胜利")

        return_list.append(episode_return)
        win_list.append(1 - env.lose)

        # print(t_bias)
        env.clear_render(t_bias=t_bias)
        t_bias += env.t

