import multiprocessing as mp
import numpy as np
import pickle
import cloudpickle

# --- Worker 函数：运行在独立的进程中 ---
def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    # 实例化环境
    env = env_fn_wrapper.x()
    try:
        while True:
            cmd, data = remote.recv()
            
            if cmd == 'step':
                # data 是一个字典 {'r': ..., 'b': ...}
                
                # 处理红方动作：支持 (action, fire_flag) 格式
                r_input = data.get('r')
                if r_input is not None:
                    # 检查是否为 (maneuver_action, fire_flag) 结构
                    if isinstance(r_input, (tuple, list)) and len(r_input) == 2 and isinstance(r_input[1], (bool, int, np.bool_)):
                        r_action, r_fire = r_input
                        # if r_fire:
                        #     launch_missile_if_possible(env, 'r')
                    else:
                        r_action = r_input
                else:
                    r_action = np.array([0, 0, 300]) # 默认fallback

                # 处理蓝方动作
                b_input = data.get('b')
                if b_input is not None:
                     b_action = b_input
                else:
                     b_action = np.array([0, 0, 300])

                # 执行物理步进
                env.step(r_action, b_action)
                
                # 2. 手动调用特定任务的奖励和终止函数
                is_done, b_reward, _ = env.get_terminate_and_reward('b')
                
                # 3. 获取观测
                r_obs, _ = env.attack_obs('r')
                b_obs, _ = env.attack_obs('b')
                
                # ================= [新增核心逻辑 START] =================
                
                # 1. 获取 State (目前暂用 Obs 代替，为未来留接口)
                r_state = r_obs  # 以后如果环境有了 env.get_global_state() 改这里即可
                b_state = b_obs
                
                # 2. 计算 Masks (1=存活, 0=死亡)
                # 假设环境中的对象有 dead 属性 (需要在 AttackManeuverEnv 确认有此属性)
                # 如果没有 dead 属性，可以用 health > 0 判断
                b_active_mask = 1.0 if not getattr(env.BUAV, 'dead', False) else 0.0
                r_action_mask = 1.0 if not getattr(env.RUAV, 'dead', False) else 0.0
                
                # 3. 计算 Truncation (截断)
                # 通常是超时截断，非胜负平导致的结束
                # 假设 is_done 包含超时，这里简单处理：如果是超时导致的 done，则 trunc=True
                # 如果你的 env.t 是当前时间，game_time_limit 是最大时间
                trunc = False 
                # ==============================================================
                
                # [新增] 提取红方规则所需的 Raw State
                red_raw_info = {
                    'pos': env.RUAV.pos_,
                    'psi': env.RUAV.psi,
                    'vel': env.RUAV.vel_,
                    'ammo': env.RUAV.ammo
                }

                info = {
                    'win': env.win, 
                    'lose': env.lose,
                    'red_raw_info': red_raw_info # 发送给主进程计算规则
                }

                if is_done:
                    # 重置环境并获取初始观测
                    env.reset()
                    r_obs, _ = env.attack_obs('r')
                    r_obs, _ = env.attack_obs('r') 
                    # Reset 后也要更新 state
                    r_state = r_obs
                    b_state = b_obs
                    # Mask 重置为 1
                    b_active_mask = 1.0
                    r_action_mask = 1.0
                    # 重置后更新 info 中的 raw info
                    info['red_raw_info'] = {
                        'pos': env.RUAV.pos_,
                        'psi': env.RUAV.psi,
                        'vel': env.RUAV.vel_,
                        'ammo': env.RUAV.ammo
                    }

                remote.send({
                    'r_obs': r_obs,
                    'b_obs': b_obs,
                    'r_state': r_state,           # [新增]
                    'b_state': b_state,           # [新增]
                    'b_active_masks': b_active_mask, # [新增]
                    'r_action_masks': r_action_mask, # [新增]
                    'truncs': trunc,              # [新增]
                    'r_reward': 0,          
                    'b_reward': b_reward,   
                    'dones': is_done,       
                    'infos': info
                })

            elif cmd == 'reset':
                # 1. 执行环境重置 (现在它不返回 obs 了)
                env.reset() 
                # 2. [新增] 在 Worker 内部显式调用观测函数
                #    这样符合你“reset后单独调用obs”的逻辑，
                #    同时只通过管道发送一次数据回主进程
                r_obs, _ = env.attack_obs('r')
                b_obs, _ = env.attack_obs('b')
                
                # 3. 重新提取 Raw Info
                # Reset 时也需要发送 raw info
                red_raw_info = {
                    'pos': env.RUAV.pos_,
                    'psi': env.RUAV.psi,
                    'vel': env.RUAV.vel_,
                    'ammo': env.RUAV.ammo
                }
                
                # 4. 发送回主进程
                remote.send({
                    'r_obs': r_obs,
                    'b_obs': b_obs,
                    'r_state': r_obs, # init state
                    'b_state': b_obs, # init state
                    'b_active_masks': 1.0,
                    'r_action_masks': 1.0,
                    'truncs': False,
                    'infos': {
                        'win': env.win, 
                        'lose': env.lose,
                        'red_raw_info': red_raw_info # 这是一个字典，包含 'pos', 'psi' 等
                    }
                })

            elif cmd == 'close':
                remote.close()
                break
                
            else:
                print(f"Unknown command: {cmd}")
                
    except KeyboardInterrupt:
        print('Worker KeyboardInterrupt')
    finally:
        remote.close()

# 用于序列化 lambda 函数或局部函数
class CloudpickleWrapper(object):
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        self.x = pickle.loads(ob)

# --- 主类：并行环境管理器 ---
class ParallelPettingZooEnv:
    def __init__(self, env_fns):
        """
        env_fns: 一个列表，包含创建环境的函数 [fn, fn, ...]
        """
        self.closed = False # [新增] 初始化关闭状态标记
        self.n_envs = len(env_fns)
        # 创建管道
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.n_envs)])
        
        # 启动进程
        self.ps = [
            mp.Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)
        ]
        
        for p in self.ps:
            p.daemon = True # 主进程死后子进程自动退出
            p.start()
            
        for remote in self.work_remotes:
            remote.close()

    def step(self, actions_dict):
        """
        actions_dict: {'r': [action1, action2...], 'b': [action1, action2...]}
        支持 None，表示该 Agent 由 Worker 内部托管
        """
        # 1. 分发动作
        for i, remote in enumerate(self.remotes):
            worker_action = {
                'r': actions_dict['r'][i] if actions_dict['r'] is not None else None,
                'b': actions_dict['b'][i] if actions_dict['b'] is not None else None
            }
            remote.send(('step', worker_action))

        # 2. 收集结果
        results = [remote.recv() for remote in self.remotes]
        
        # 3. 堆叠数据 (Stacking)
        # 将结果列表转换为 numpy 数组，方便神经网络批处理
        # return: {'r_obs': (N, D), 'b_obs': (N, D), ...}
        
        return {
            'r_obs': np.stack([r['r_obs'] for r in results]),
            'b_obs': np.stack([r['b_obs'] for r in results]),
            # [新增] 堆叠新变量
            'r_state': np.stack([r['r_state'] for r in results]),
            'b_state': np.stack([r['b_state'] for r in results]),
            'b_active_masks': np.stack([r['b_active_masks'] for r in results]).reshape(-1, 1), # 变成 (N, 1)
            'r_action_masks': np.stack([r['r_action_masks'] for r in results]).reshape(-1, 1),
            'truncs': np.stack([r['truncs'] for r in results]).reshape(-1, 1),
            # 原有
            'r_reward': np.stack([r['r_reward'] for r in results]),
            'b_reward': np.stack([r['b_reward'] for r in results]),
            'dones': np.stack([r['dones'] for r in results]),
            'infos': [r['infos'] for r in results]
        }

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        
        results = [remote.recv() for remote in self.remotes]
        
        return {
            'r_obs': np.stack([r['r_obs'] for r in results]),
            'b_obs': np.stack([r['b_obs'] for r in results]),
            # ================= [新增] =================
            'r_state': np.stack([r['r_state'] for r in results]), # 必须把初始 state 接住
            'b_state': np.stack([r['b_state'] for r in results]), 
            # =========================================
            'infos': [r.get('infos', {}) for r in results] # 确保 reset 也返回 infos
        }

    def close(self):
        # 1. 尝试发送关闭指令
        if self.closed:  # [新增] 避免重复关闭
            return

        for remote in self.remotes:
            try:
                # [新增] 检查连接状态再发送，或者捕获发送失败的异常
                remote.send(('close', None))
            except (BrokenPipeError, EOFError, OSError):
                # 如果管道已经断开，说明子进程可能已经退出了，直接忽略
                pass
        
        # 2. 等待子进程退出
        for p in self.ps:
            p.join()
            
        self.closed = True # [新增] 标记为已关闭