import os
import sys
import numpy as np
import pickle
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from datetime import datetime, timedelta
from TrainAndTests.MARWIL.pure_marwil_train_attack import *
from Algorithms.PPOcontinues_std_no_state import *
from Math_calculates.ScaleLearningRate import scale_learning_rate
from Visualize.tensorboard_visualize import TensorBoardLogger

import torch


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
    mission_name = 'AttackMixed'
    log_dir = os.path.join(logs_dir, f"{mission_name}-run-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    logger = TensorBoardLogger(log_root=log_dir, host="127.0.0.1", port=6006, use_log_root=True, auto_show=False)
    for epoch in range(5):
        avg_actor_loss, avg_critic_loss, c = student_agent.MARWIL_update(il_transition_dict, beta=1.0)
        # 记录损失函数
        if epoch % 1 == 0:
            logger.add("il_train/avg_actor_loss", avg_actor_loss, epoch)
            logger.add("il_train/avg_critic_loss", avg_critic_loss, epoch)

            print("epoch", epoch)

    # 第5步：混合强化学习训练
    
    max_steps = 20e4

    # --- 仅保存一次网络形状（meta json），如果已存在则跳过
    # log_dir = "./logs"
    

    actor_meta_path = os.path.join(log_dir, "actor.meta.json")
    critic_meta_path = os.path.join(log_dir, "critic.meta.json")

    save_meta_once(actor_meta_path, student_agent.actor.state_dict())
    save_meta_once(critic_meta_path, student_agent.critic.state_dict())
    
    # 根据参数数量缩放学习率
    actor_lr = scale_learning_rate(actor_lr, student_agent.actor)
    critic_lr = scale_learning_rate(critic_lr, student_agent.critic)
    student_agent.set_learning_rate(actor_lr=actor_lr, critic_lr=critic_lr)

    out_range_count = 0
    return_list = []
    win_list = []
    # steps_count = 0
    total_steps = 0
    i_episode = 0

    logger = TensorBoardLogger(log_root=log_dir, host="127.0.0.1", port=6006, use_log_root=True, auto_show=False)
    training_start_time = time.time()
    
    t_bias = 0

    try:
        env = AttackTrainEnv(args, tacview_show=use_tacview)
        # 强化学习训练
        rl_steps = 0
        return_list = []
        win_list = []
        # with tqdm(total=int(num_episodes*(1-pre_train_rate)), desc='Iteration') as pbar:  # 进度条
        # for i_episode in range(int(num_episodes*(1-pre_train_rate))):
        while total_steps < int(max_steps * (1 - pre_train_rate)):
            i_episode += 1
            episode_return = 0
            transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [],
                               'action_bounds': []}

            # 飞机出生状态指定
            red_R_ = random.uniform(30e3, 40e3)  # 20, 60 特意训练一个近的，测试一个远的
            red_beta = random.uniform(0, 2 * pi)
            red_psi = random.uniform(0, 2 * pi)
            red_height = random.uniform(3e3, 10e3)
            red_N = red_R_ * cos(red_beta)
            red_E = red_R_ * sin(red_beta)
            blue_height = random.uniform(3e3, 10e3)

            DEFAULT_RED_BIRTH_STATE = {'position': np.array([red_N, red_height, red_E]),
                                       'psi': red_psi
                                       }
            DEFAULT_BLUE_BIRTH_STATE = {'position': np.array([0.0, blue_height, 0.0]),
                                        'psi': pi
                                        }
            env.reset(red_birth_state=DEFAULT_RED_BIRTH_STATE, blue_birth_state=DEFAULT_BLUE_BIRTH_STATE,
                      red_init_ammo=0, blue_init_ammo=0)

            # a1 = env.BUAV.pos_  # 58000,7750,20000
            # a2 = env.RUAV.pos_  # 2000,7750,20000
            # b1 = env.UAVs[0].pos_
            # b2 = env.UAVs[1].pos_
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

                distance = norm(env.RUAV.pos_ - env.BUAV.pos_)
                # 发射导弹判决
                if distance <= 40e3 and distance >= 5e3 and count % 1 == 0:  # 在合适的距离范围内每0.2s判决一次导弹发射
                    launch_time_count = 0
                    launch_missile_if_possible(env, side='r')
                    launch_missile_if_possible(env, side='b')

                # 机动决策
                r_action_n = decision_rule(ego_pos_=env.RUAV.pos_, ego_psi=env.RUAV.psi,
                                           enm_pos_=env.BUAV.pos_, distance=distance,
                                           ally_missiles=env.Rmissiles, enm_missiles=env.Bmissiles,
                                           o00=o00, R_cage=env.R_cage, wander=1
                                           )
                b_action_n, u = student_agent.take_action(b_obs, action_bounds=action_bound, explore=True)

                # b_action_n = decision_rule(ego_pos_=env.BUAV.pos_, ego_psi=env.BUAV.psi,
                #                         enm_pos_=env.RUAV.pos_, distance=distance,
                #                         ally_missiles=env.Bmissiles, enm_missiles=env.Rmissiles,
                #                         o00=o00, R_cage=env.R_cage, wander=0
                #                         )

                r_action_list.append(r_action_n)
                b_action_list.append(b_action_n)

                env.step(r_action_n, b_action_n)  # 2、环境更新并反馈
                done, b_reward, _ = env.get_terminate_and_reward('b')
                next_b_obs, _ = env.attack_obs('b')  # 子策略的训练不要用get_obs
                env.BUAV.act_memory = b_action_n.copy()  # 存储上一步动作
                total_steps += 1

                transition_dict['states'].append(b_obs)
                transition_dict['actions'].append(u)
                transition_dict['next_states'].append(next_b_obs)
                transition_dict['rewards'].append(b_reward)
                transition_dict['dones'].append(done)
                transition_dict['action_bounds'].append(action_bound)
                # state = next_state
                episode_return += b_reward * env.dt_maneuver

                '''显示运行轨迹'''
                # 可视化
                env.render(t_bias=t_bias)

            episode_end_time = time.time()  # 记录结束时间
            # print(f"回合时长: {episode_end_time - episode_start_time} 秒")

            if env.lose == 1:
                out_range_count += 1
            return_list.append(episode_return)
            win_list.append(1 - env.lose)
            student_agent.update(transition_dict)

            # print(t_bias)
            env.clear_render(t_bias=t_bias)
            t_bias += env.t
            r_action_list = np.array(r_action_list)
            b_action_list = np.array(b_action_list)

            # --- 保存模型（强化学习阶段：actor_rein + i_episode，critic 每次覆盖）
            
            # critic overwrite
            critic_path = os.path.join(log_dir, "critic.pt")
            torch.save(student_agent.critic.state_dict(), critic_path)
            # actor RL snapshot
            if i_episode % 10 == 0:
                actor_name = f"actor_rein{i_episode}.pt"
                actor_path = os.path.join(log_dir, actor_name)
                torch.save(student_agent.actor.state_dict(), actor_path)

            # # tqdm 训练进度显示
            # if (i_episode + 1) >= 10:
            #     pbar.set_postfix({'episode': '%d' % (i_episode + 1),
            #                     'return': '%.3f' % np.mean(return_list[-10:])})
            # pbar.update(1)
            
            # 训练进度显示
            if (i_episode) >= 10:
                print(
                    f"episode {i_episode}, 进度: {total_steps / max_steps:.3f}, return: {np.mean(return_list[-10:]):.3f}")
            else:
                print(f"episode {i_episode}, total_steps {total_steps}")

            # tensorboard 训练进度显示
            logger.add("train/1 episode_return", episode_return, total_steps)
            logger.add("train/2 win", env.win, total_steps)

            actor_grad_norm = student_agent.actor_grad
            actor_pre_clip_grad = student_agent.pre_clip_actor_grad
            critic_grad_norm = student_agent.critic_grad
            critic_pre_clip_grad = student_agent.pre_clip_critic_grad
            # 梯度监控
            logger.add("train/3 actor_grad_norm", actor_grad_norm, total_steps)
            logger.add("train/5 actor_pre_clip_grad", actor_pre_clip_grad, total_steps)
            logger.add("train/4 critic_grad_norm", critic_grad_norm, total_steps)
            logger.add("train/6 critic_pre_clip_grad", critic_pre_clip_grad, total_steps)
            # 损失函数监控
            logger.add("train/7 actor_loss", student_agent.actor_loss, total_steps)
            logger.add("train/8 critic_loss", student_agent.critic_loss, total_steps)
            # 强化学习actor特殊项监控
            logger.add("train/9 entropy", student_agent.entropy_mean, total_steps)
            logger.add("train/10 ratio", student_agent.ratio_mean, total_steps)

        training_end_time = time.time()  # 记录结束时间
        elapsed = training_end_time - training_start_time
        from datetime import timedelta

        td = timedelta(seconds=elapsed)
        d = td.days
        h, rem = divmod(td.seconds, 3600)
        m, s = divmod(rem, 60)
        print(f"总训练时长: {d}天 {h}小时 {m}分钟 {s}秒")


    except KeyboardInterrupt:
        print("\n检测到 KeyboardInterrupt，正在关闭 logger ...")
    finally:
        logger.close()
        print(f"日志已保存到：{logger.run_dir}")