import sys
import os
sys.path.append(project_root)
from functools import partial
from TrainAndTests.NotInUse.PPOLCrankTraining_parel import *
import re

dt_maneuver= 0.2 # 0.2 
action_eps = 0 # 动作平滑度

# 作为参考的规则动作
def crank_behavior(delta_psi, delta_height):
    """
    crank 行为：返回 (heading_cmd, speed_cmd)
    """
    delta_psi_angle = delta_psi*180/pi
    temp = delta_psi_angle - 55
    heading_cmd = temp*pi/180
    speed_cmd = 1.1 * 340
    return delta_height, heading_cmd, speed_cmd

# 找出日期最大的目录
def get_latest_log_dir(pre_log_dir, mission_name=None):
    # 匹配 run-YYYYMMDD-HHMMSS 目录
    # pattern = re.compile(r"run-(\d{8})-(\d{6})")
    if mission_name:
        pattern = re.compile(rf"{re.escape(mission_name)}-run-(\d{{8}})-(\d{{6}})")
    else:
        pattern = re.compile(r"run-(\d{8})-(\d{6})")
    max_dt = None
    latest_dir = None
    for d in os.listdir(pre_log_dir):
        m = pattern.match(d)
        if m:
            dt_str = m.group(1) + m.group(2)  # 'YYYYMMDDHHMMSS'
            if max_dt is None or dt_str > max_dt:
                max_dt = dt_str
                latest_dir = d
    if latest_dir:
        return os.path.join(pre_log_dir, latest_dir)
    else:
        return None

# pre_log_dir = os.path.join("./logs")
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pre_log_dir = os.path.join(project_root, "logs")
log_dir = get_latest_log_dir(pre_log_dir, mission_name=mission_name)

log_dir = os.path.join(pre_log_dir, "LCrank-run-20251010-114608")

# 测试训练效果
agent = PPOContinuous(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                lmbda, epochs, eps, gamma, device)

rein_list = sorted(glob.glob(os.path.join(log_dir, "actor_rein*.pt")))
sup_list = sorted(glob.glob(os.path.join(log_dir, "actor_sup*.pt")))
latest_actor_path = rein_list[-1] if rein_list else (sup_list[-1] if sup_list else None)
if latest_actor_path:
    # 直接加载权重到现有的 agent
    sd = th.load(latest_actor_path, map_location=device, weights_only=True)
    agent.actor.load_state_dict(sd) # , strict=False)  # 忽略缺失的键
    print(f"Loaded actor for test from: {latest_actor_path}")

t_bias = 0

def run_single_test(seed, fake_input1, fake_input2):
    """单次测试的函数"""
    fake = fake_input1+fake_input2

    np.random.seed(seed)
    random.seed(seed)
    
    # 加载智能体
    agent = PPOContinuous(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                    lmbda, epochs, eps, gamma, device)
    
    rein_list = sorted(glob.glob(os.path.join(log_dir, "actor_rein*.pt")))
    sup_list = sorted(glob.glob(os.path.join(log_dir, "actor_sup*.pt")))
    latest_actor_path = rein_list[-1] if rein_list else (sup_list[-1] if sup_list else None)
    if latest_actor_path:
        sd = th.load(latest_actor_path, map_location=device, weights_only=True)
        agent.actor.load_state_dict(sd)
    
    env = CrankTrainEnv(args, tacview_show=0)  # 并行测试时关闭可视化
    far_edge = 0
    times = 0
    lose = 0
    
    try:
        r_action_list=[]
        b_action_list=[]
        
        # 飞机出生状态指定
        init_case = np.random.randint(initial_states.shape[0])
        red_R = random.uniform(initial_states[init_case][2], min(50e3, initial_states[init_case][3]))
        
        blue_height = initial_states[init_case][0]
        red_height = initial_states[init_case][1]

        blue_psi = random.uniform(-pi, pi)
        red_psi = sub_of_radian(blue_psi, pi)
        red_beta = blue_psi
        red_N = red_R*cos(red_beta)
        red_E = red_R*sin(red_beta)

        DEFAULT_RED_BIRTH_STATE = {'position': np.array([red_N, red_height, red_E]),
                                'psi': red_psi}
        DEFAULT_BLUE_BIRTH_STATE = {'position': np.array([0.0, blue_height, 0.0]),
                                    'psi': blue_psi}
                                    
        env.reset(red_birth_state=DEFAULT_RED_BIRTH_STATE, 
                    blue_birth_state=DEFAULT_BLUE_BIRTH_STATE,
                    red_init_ammo=0, blue_init_ammo=1)

        done = False
        env.dt_maneuver = dt_maneuver
        hist_b_action = np.zeros(3)

        while not done:
            r_obs_n, r_obs_check = env.crank_obs('r')
            b_obs_n, b_obs_check = env.crank_obs('b')
            
            # 反向转回字典方便排查
            b_check_obs = copy.deepcopy(env.state_init)
            arr = np.atleast_1d(np.asarray(b_obs_n)).reshape(-1)
            idx = 0
            for k in env.key_order:
                if k not in b_check_obs:
                    raise KeyError(f"key '{k}' not in state_init")
                v0 = b_check_obs[k]
                if isinstance(v0, (list, tuple, np.ndarray)):
                    length = len(v0)
                    slice_v = arr[idx: idx + length]
                    if isinstance(v0, np.ndarray):
                        b_check_obs[k] = slice_v.copy()
                    else:
                        b_check_obs[k] = slice_v.tolist()
                    idx += length
                else:
                    b_check_obs[k] = float(arr[idx])
                    idx += 1

            env.RUAV.obs_memory = r_obs_check.copy()
            env.BUAV.obs_memory = b_obs_check.copy()
            state = np.squeeze(b_obs_n)
            distance = norm(env.RUAV.pos_ - env.BUAV.pos_)
            
            if env.BUAV.ammo>0:
                new_missile = env.BUAV.launch_missile(env.RUAV, env.t, missile_class)
                env.BUAV.ammo -= 1
                new_missile.side = 'blue'
                env.Bmissiles.append(new_missile)
                env.missiles = env.Rmissiles + env.Bmissiles

            r_action_n = decision_rule(ego_pos_=env.RUAV.pos_, ego_psi=env.RUAV.psi,
                                    enm_pos_=env.BUAV.pos_, distance=distance,
                                    ally_missiles=env.Rmissiles, enm_missiles=env.Bmissiles,
                                    o00=o00, R_cage=env.R_cage, wander=1)

            # # 规则动作
            # delta_psi = b_check_obs["target_information"][1]
            # delta_height = b_check_obs["target_information"][0]
            # b_action_n = crank_behavior(delta_psi, delta_height*5000-2000)

            b_action_n, u = agent.take_action(state, action_bounds=action_bound, explore=False)
            
            b_action_n = action_eps*hist_b_action+(1-action_eps)*b_action_n
            hist_b_action = b_action_n

            r_action_list.append(r_action_n)
            b_action_list.append(b_action_n)

            env.step(r_action_n, b_action_n)
            done, b_reward, _ = env.left_crank_terminate_and_reward('b')
            done = done

        if env.lose:
            lose = 1
        else:
            lose = 0
                
    except Exception as e:
        print(f"Process {seed} failed:", e)
        return None
        
    return lose, 1  # 返回这个进程中的平均lose率

if __name__ == '__main__':
    import multiprocessing as mp
    from multiprocessing import Pool
    import time
    
    num_processes = 20
    t0 = time.time()

    fake_input1 = 42  # 示例值
    fake_input2 = 0.5  # 示例值

    # 多输入函数并行写法1，输入参数可变
    args_list = [(i, fake_input1, fake_input2) for i in range(num_processes)]

    # 多输入函数并行写法2，输入参数固定
    # 创建偏函数，固定后两个参数
    test_func = partial(run_single_test, 
                       fake_input1=fake_input1,
                       fake_input2=fake_input2)

    with Pool(num_processes) as p:
        # results = p.map(run_single_test, range(num_processes)) # 单输入函数并行写法

        # 多输入函数并行写法1，输入参数可变，结果统一返回
        results = p.starmap(run_single_test, args_list)      

        # # 多输入函数并行写法1.5，输入参数可变，且每个进程结果立即返回而不是统一返回
        # async_result = p.starmap_async(run_single_test, args_list)  # 立即返回
        # # 可以做其他事
        # results = async_result.get()  # 需要结果时再等待

        # 多输入函数并行写法2，输入参数固定
        # results = p.map(test_func, range(num_processes))
        # 使用 zip 解包成两个列表

    print(results)
    loses, episode_returns = zip(*results)
    
    # 转换为 numpy array 以便计算统计量
    loses = np.array(loses)
    episode_returns = np.array(episode_returns)
    print(loses)
    print(episode_returns)        
    
    t1 = time.time()
    
    # 过滤掉None结果（失败的进程）
    valid_results = [r for r in loses if r is not None]
    if valid_results:
        mean_lose_rate = np.mean(valid_results)
        print(f"Average lose rate across {len(valid_results)} processes: {mean_lose_rate:.3f}")
    else:
        print("All processes failed")
    
    print(f"并行程序运行时长: {t1-t0:.2f} 秒")