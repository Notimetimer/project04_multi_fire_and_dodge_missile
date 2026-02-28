import os
import sys
import numpy as np
import pickle
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from TrainAndTests.MARWIL.pure_marwil_train_attack import *
from Algorithms.PPOcontinues_std_no_state import *
from Algorithms.PPOcontinues_std_no_state import PPOContinuous

student_agent = PPOContinuous(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
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
    npz_path = os.path.join(cur_dir, "il_transitions.npz")
    pkl_path = os.path.join(cur_dir, "il_transitions_chase.pkl")

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
        print("No il_transitions_chase file found in this directory.")
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
    logs_dir = os.path.join(project_root, "logs/attack")
    mission_name = 'MARWIL'
    log_dir = os.path.join(logs_dir, f"{mission_name}-run-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    logger = TensorBoardLogger(log_root=log_dir, host="127.0.0.1", port=6006, use_log_root=True, auto_show=False)
    for epoch in range(20): # 5
        avg_actor_loss, avg_critic_loss, c = student_agent.MARWIL_update(il_transition_dict, beta=1.0)
        # 记录损失函数
        if epoch % 1 == 0:
            logger.add("il_train/avg_actor_loss", avg_actor_loss, epoch)
            logger.add("il_train/avg_critic_loss", avg_critic_loss, epoch)

            print("epoch", epoch)



    # # 第5步：测试有监督学习的效果
    # t_bias = 0
    # env = AttackTrainEnv(args, tacview_show=1) # Battle(args, tacview_show=1)
    # for i_episode in range(3):  # 10
    #     r_action_list=[]
    #     b_action_list=[]
    #     # 飞机出生状态指定
    #     init_distance = 90e3
    #     red_R_ = init_distance/2 # random.uniform(20e3, 60e3)
    #     blue_R_ = init_distance/2
    #     red_beta = pi # random.uniform(0, 2*pi)
    #     red_psi = 0 # random.uniform(0, 2*pi)
    #     red_height = 8e3 # random.uniform(3e3, 10e3)
    #     red_N = red_R_*cos(red_beta)
    #     red_E = red_R_*sin(red_beta)
    #     blue_height = 8e3 # random.uniform(3e3, 10e3)

    #     DEFAULT_RED_BIRTH_STATE = {'position': np.array([red_N, red_height, red_E]),
    #                             'psi': red_psi
    #                             }
    #     DEFAULT_BLUE_BIRTH_STATE = {'position': np.array([blue_R_, blue_height, 0.0]),
    #                                 'psi': np.random.choice([pi/2, -pi/2])
    #                                 }
    #     env.reset(red_birth_state=DEFAULT_RED_BIRTH_STATE, blue_birth_state=DEFAULT_BLUE_BIRTH_STATE,
    #             red_init_ammo=6, blue_init_ammo=6)
    #     env.dt_maneuver = dt_maneuver
    #     step = 0
    #     done = False
    #     hist_b_action = np.zeros(3)

    #     while not done:
    #         # print(env.t)
    #         r_obs_n, r_obs_check = env.attack_obs('r')
    #         b_obs_n, b_obs_check = env.attack_obs('b')
    #         # 在这里将观测信息压入记忆
    #         env.RUAV.obs_memory = r_obs_check.copy()
    #         env.BUAV.obs_memory = b_obs_check.copy()
    #         state = np.squeeze(b_obs_n)
    #         distance = norm(env.RUAV.pos_ - env.BUAV.pos_)
    #         r_action_n = decision_rule(ego_pos_=env.RUAV.pos_, ego_psi=env.RUAV.psi,
    #                                 enm_pos_=env.BUAV.pos_, distance=distance,
    #                                 ally_missiles=env.Rmissiles, enm_missiles=env.Bmissiles,
    #                                 o00=o00, R_cage=env.R_cage, wander=1
    #                                 )
    #         # r_action_n, u_r = student_agent.take_action(r_obs_n, action_bounds=action_bound, explore=False)
    #         b_action_n, u = student_agent.take_action(b_obs_n, action_bounds=action_bound, explore=False)
            
    #         # # 动作平滑（实验性）
    #         # b_action_n = action_eps*hist_b_action+(1-action_eps)*b_action_n
    #         # hist_b_action = b_action_n

    #         r_action_list.append(r_action_n)
    #         b_action_list.append(b_action_n)

    #         # ### 发射导弹，这部分不受10step约束
    #         # distance = norm(env.RUAV.pos_ - env.BUAV.pos_)
    #         # # 发射导弹判决
    #         # if distance <= 80e3 and distance >= 5e3:  # 在合适的距离范围内每0.2s判决一次导弹发射
    #         #     launch_time_count = 0
    #         #     launch_missile_with_basic_rules(env, side='r')
    #         #     launch_missile_with_basic_rules(env, side='b')

    #         env.step(r_action_n, b_action_n)  # 2、环境更新并反馈
    #         done, b_reward, _ = env.get_terminate_and_reward('b')

    #         step += 1
    #         env.render(t_bias=t_bias)
    #         time.sleep(0.01)
        
    #     env.clear_render(t_bias=t_bias)
    #     t_bias += env.t
