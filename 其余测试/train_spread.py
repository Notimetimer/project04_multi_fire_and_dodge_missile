import torch
import numpy as np
import time
from collections import defaultdict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 从您的文件中导入 MAPPO 算法
from Algorithms.MAPPOdiscrete import MAPPO 

# 导入 PettingZoo 环境
from pettingzoo.mpe import simple_spread_v3


# import torch
# import numpy as np
# import time
# from collections import defaultdict

# # 从您的文件中导入 MAPPO 算法
# from mappo_discrete import MAPPO 

# # 导入 PettingZoo 环境
# from pettingzoo.mpe import simple_spread_v3

def main():
    # ================================================================= #
    #                         1. 环境和参数设置                            #
    # ================================================================= #
    
    # -- 核心参数 --
    # 训练时 render_mode 设置为 None，测试时会单独创建带 human 模式的环境
    train_render_mode = None 
    test_render_mode = "human" # 测试时开启可视化
    num_train_episodes = 500   # 训练的总回合数
    max_steps_per_episode = 100 # 每回合最大步数
    test_interval = 20         # 每训练多少个回合进行一次测试
    num_test_episodes_per_interval = 1 # 每次测试运行的回合数
    
    # -- MAPPO 算法超参数 --
    actor_lr = 1e-4
    critic_lr = 1e-3
    gamma = 0.99
    lmbda = 0.97
    epochs = 10
    eps = 0.2
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print(f"Using device: {device}")

    # -- 初始化训练环境 --
    # 使用 parallel_env 接口，它更适合与算法进行数据交互
    train_env = simple_spread_v3.parallel_env(N=2, local_ratio=0.5, max_cycles=max_steps_per_episode, render_mode=train_render_mode)
    
    # -- 获取环境信息 --
    train_env.reset() # 提前reset一次以获取agent信息
    num_agents = train_env.num_agents
    agent_name_to_int = {agent: i for i, agent in enumerate(train_env.agents)}
    
    first_agent = train_env.agents[0]
    obs_dim = train_env.observation_space(first_agent).shape[0]
    action_dim = train_env.action_space(first_agent).n
    
    print(f"Number of agents: {num_agents}")
    print(f"Observation dimension: {obs_dim}")
    print(f"Action dimension: {action_dim}")

    # -- 实例化 MAPPO 智能体 --
    agent = MAPPO(
        obs_dim=obs_dim,
        action_dim=action_dim,
        num_agents=num_agents,
        actor_hidden_dims=[64, 64],
        critic_hidden_dims_back=[128, 128],
        critic_hidden_dims_front=[64, 64],
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        lmbda=lmbda,
        epochs=epochs,
        eps=eps,
        gamma=gamma,
        device=device
    )

    # ================================================================= #
    #                             2. 训练循环                             #
    # ================================================================= #
    
    print("\n--- Starting Training ---")
    
    train_episode_count = 0 
    
    while train_episode_count < num_train_episodes:
        # # -- 2.1. 测试阶段 --
        # # 如果是需要进行测试的回合，跳过训练，直接进入测试
        # if (train_episode_count > 0) and (train_episode_count % test_interval == 0):
        #     print(f"\n--- Testing at {train_episode_count} training episodes ---")
        #     # 每次测试都创建一个新的可视化环境，确保渲染正常关闭和开启
        #     test_env = simple_spread_v3.parallel_env(N=2, local_ratio=0.5, max_cycles=max_steps_per_episode, render_mode=test_render_mode)
        #     test_rewards_list = []
        #     try:
        #         for i_test_episode in range(num_test_episodes_per_interval):
        #             observations, _ = test_env.reset()
        #             episode_reward = 0
        #             for step in range(max_steps_per_episode):
        #                 test_env.render()
        #                 # time.sleep(0.05)  # 减慢速度以便观察

        #                 actions_dict = {}
        #                 for agent_name in test_env.agents:
        #                     obs = observations[agent_name]
        #                     _, action = agent.take_action(obs, explore=False)
        #                     actions_dict[agent_name] = action

        #                 next_observations, rewards, terminations, truncations, _ = test_env.step(actions_dict)
        #                 observations = next_observations
        #                 episode_reward += sum(rewards.values())

        #                 if any(terminations.values()) or any(truncations.values()):
        #                     break
        #             test_rewards_list.append(episode_reward)
        #             print(f"    Test Episode {i_test_episode + 1}/{num_test_episodes_per_interval}, Reward: {episode_reward:.2f}")
        #     finally:
        #         test_env.close()
        #     print(f"--- Average Test Reward over {num_test_episodes_per_interval} episodes: {np.mean(test_rewards_list):.2f} ---\n")
            
        # -- 2.2. 训练阶段 --
        # 如果训练回合数已满，退出循环
        if train_episode_count >= num_train_episodes:
            break

        transition_dict = defaultdict(list)
        
        observations, _ = train_env.reset()
        # 如果 reset 返回空字典，再次 reset（防护）
        if not observations:
            observations, _ = train_env.reset()

        episode_reward = 0

        for step in range(max_steps_per_episode):
            # 训练时不渲染
            # train_env.render() 
            # time.sleep(0.02) 

            actions_dict = {}
            for agent_name in train_env.agents:
                obs = observations[agent_name]
                _, action = agent.take_action(obs, explore=True)
                actions_dict[agent_name] = action

            next_observations, rewards, terminations, truncations, _ = train_env.step(actions_dict)

            # 安全地构建全局状态（跳过缺失的观测）
            obs_list = [observations.get(a) for a in train_env.agents if observations.get(a) is not None]
            next_obs_list = [next_observations.get(a) for a in train_env.agents if next_observations.get(a) is not None]

            if len(obs_list) == 0 or len(next_obs_list) == 0:
                # 出现异常情况，重置本回合（或选择跳出/继续）
                observations, _ = train_env.reset()
                break

            global_state = np.concatenate(obs_list).ravel()
            next_global_state = np.concatenate(next_obs_list).ravel()
            
            for agent_name in train_env.agents:
                agent_id = agent_name_to_int[agent_name]
                obs = observations[agent_name]
                action = actions_dict[agent_name]
                reward = rewards[agent_name]
                done = terminations[agent_name] or truncations[agent_name]

                transition_dict['obs'].append(obs)
                transition_dict['global_states'].append(global_state)
                transition_dict['actions'].append(action)
                transition_dict['rewards'].append(reward)
                transition_dict['next_global_states'].append(next_global_state)
                transition_dict['dones'].append(done)
                transition_dict['agent_ids'].append(agent_id)

            observations = next_observations
            
            episode_reward += sum(rewards.values())
            
            if any(terminations.values()) or any(truncations.values()):
                break
        
        agent.update(transition_dict, adv_normed=True)
        
        train_episode_count += 1
        print(f"Training Episode {train_episode_count}/{num_train_episodes}, Total Reward: {episode_reward:.2f}, Actor Loss: {agent.actor_loss:.4f}, Critic Loss: {agent.critic_loss:.4f}")

    print("--- Training Finished ---\n")
    
    # ================================================================= #
    #                       3. 最终效果测试 (确定性策略)                   #
    # ================================================================= #

    # print("--- Starting Final Test ---")
    # final_test_env = simple_spread_v3.parallel_env(N=2, local_ratio=0.5, max_cycles=max_steps_per_episode, render_mode=test_render_mode)
    # try:
    #     for i_test in range(num_test_episodes_per_interval):
    #         observations, _ = final_test_env.reset()
    #         episode_reward = 0
    #         for _ in range(max_steps_per_episode):
    #             final_test_env.render()
    #             actions_dict = {}
    #             for agent_name in final_test_env.agents:
    #                 obs = observations[agent_name]
    #                 _, action = agent.take_action(obs, explore=False)
    #                 actions_dict[agent_name] = action

    #             next_observations, rewards, terminations, truncations, _ = final_test_env.step(actions_dict)
    #             observations = next_observations
    #             episode_reward += sum(rewards.values())

    #             if any(terminations.values()) or any(truncations.values()):
    #                 break

    #         print(f"Final Test Episode {i_test + 1}, Reward: {episode_reward:.2f}")
    # finally:
    #     final_test_env.close()
    #     train_env.close()

if __name__ == '__main__':
    main()