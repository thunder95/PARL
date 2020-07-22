# coding: utf-8
import os
import numpy as np

import parl
from parl import layers
from paddle import fluid
from parl.utils import logger
from parl.utils import action_mapping # 将神经网络输出映射到对应的 实际动作取值范围 内
from parl.utils import ReplayMemory # 经验回放
from rlschool import make_env  # 使用 RLSchool 创建飞行器环境
from model import QuadrotorModel
from agent import QuadrotorAgent
from parl.algorithms import DDPG

GAMMA = 0.99  # reward 的衰减因子，一般取 0.9 到 0.999 不等
TAU = 0.001  # target_model 跟 model 同步参数 的 软更新参数
ACTOR_LR = 0.0002  # Actor网络更新的 learning rate
CRITIC_LR = 0.001  # Critic网络更新的 learning rate
MEMORY_SIZE = 1e6  # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 1e4  # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
REWARD_SCALE = 0.01  # reward 的缩放因子
BATCH_SIZE = 256  # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
TRAIN_TOTAL_STEPS = 2e6  # 总训练步数
TEST_EVERY_STEPS = 1e4  # 每个N步评估一下算法效果，每次评估5个episode求平均reward
#CHANGE
EXPL_NOISE = 0.1 #高斯噪声

def run_episode(env, agent, rpm):
    obs = env.reset()
    total_reward, steps = 0, 0
    
    while True:
        steps += 1
        batch_obs = np.expand_dims(obs, axis=0)
        pred_action = agent.predict(batch_obs.astype('float32'))            
        pred_action = np.squeeze(pred_action)
        env_action = pred_action[0] + 0.2*pred_action[1:] 
        
        #CHANGE                           
        # 给输出动作增加探索扰动，输出限制在 [-1.0, 1.0] 范围内
        # env_action = np.clip(np.random.normal(env_action, 1.0), -1.0, 1.0)
        
        #ps: 这里使用max_action会有问题，暂时先用 -1.0, 1.0
        # env_action = np.clip(
        #         np.random.normal(pred_action, EXPL_NOISE * max_action), -max_action,
        #         max_action)     
        
        env_action = np.clip(
                np.random.normal(pred_action, EXPL_NOISE * max_action), -1.0,
                1.0)      
      
        # 动作映射到对应的 实际动作取值范围 内, action_mapping是从parl.utils那里import进来的函数            
        env_action = action_mapping(env_action, env.action_space.low[0],env.action_space.high[0])       

        next_obs, reward, done, info = env.step(env_action)
        rpm.append(obs, pred_action, REWARD_SCALE * reward, next_obs, done)      

        if rpm.size() > MEMORY_WARMUP_SIZE:
            batch_obs, batch_action, batch_reward, batch_next_obs,                     batch_terminal = rpm.sample_batch(BATCH_SIZE)
            critic_cost = agent.learn(batch_obs, batch_action, batch_reward,
                                      batch_next_obs, batch_terminal)

        obs = next_obs
        total_reward += reward

        if done:
            break
    return total_reward, steps

# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(env, agent):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        total_reward, steps = 0, 0
        while True:
            batch_obs = np.expand_dims(obs, axis=0)
            pred_action = agent.predict(batch_obs.astype('float32'))            
            pred_action = np.squeeze(pred_action)                                                    
            env_action = pred_action[0] + 0.2*pred_action[1:] 
            #CHANGE
            #env_action = np.clip(env_action, -1.0, 1.0)  
            env_action = np.clip(
                np.random.normal(pred_action, EXPL_NOISE * max_action), -1.0,
                1.0)                              
            env_action = action_mapping(env_action, env.action_space.low[0],env.action_space.high[0]) 
            next_obs, reward, done, info = env.step(env_action)
            obs = next_obs
            total_reward += reward
            steps += 1

            if done:
                break
        eval_reward.append(total_reward)
    return np.mean(eval_reward)


# 创建飞行器环境
env = make_env("Quadrotor", task="no_collision", seed=1)
env.reset()
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0] + 1

#CHANGE
max_action = float(env.action_space.high[0])
print("max action: ", max_action)

#model = QuadrotorModel(act_dim)
#algorithm = DDPG(model, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)

#CHANGE
model = QuadrotorModel(act_dim, max_action)
algorithm = parl.algorithms.TD3(
        model,
        max_action=max_action,
        gamma=GAMMA,
        tau=TAU,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR)
agent = QuadrotorAgent(algorithm, obs_dim, act_dim)
rpm = ReplayMemory(int(MEMORY_SIZE), obs_dim, act_dim)

# 启动训练
test_flag = 0
total_steps = 0
best_reward = -float('inf')
while total_steps < TRAIN_TOTAL_STEPS:
    train_reward, steps = run_episode(env, agent, rpm)
    total_steps += steps
    #logger.info('Steps: {} Reward: {}'.format(total_steps, train_reward)) # 打印训练reward
    if total_steps // TEST_EVERY_STEPS >= test_flag: # 每隔一定step数，评估一次模型
        while total_steps // TEST_EVERY_STEPS >= test_flag:
            test_flag += 1
        evaluate_reward = evaluate(env, agent)
        logger.info('Steps {}, Test reward: {}'.format(
            total_steps, evaluate_reward)) # 打印评估的reward
        if evaluate_reward >= best_reward:
            best_reward = evaluate_reward
            #ckpt = 'model_dir/steps_{}_reward_{}.ckpt'.format(total_steps,round(best_reward,2))
            #agent.save(ckpt)
            
            #CHANGE
            agent.save_actor('model_dir/actor.ckpt')
            agent.save_critic('model_dir/critic.ckpt')
            
               
     
#ckpt = 'model_dir/steps_{}_reward_{}.ckpt'.format(total_steps,round(best_reward,2))
# agent.save(ckpt)

print("done")