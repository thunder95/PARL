# coding: utf-8
import numpy as np
from parl.utils import logger
from parl.utils import action_mapping  # 将神经网络输出映射到对应的 实际动作取值范围 内
from parl.utils import ReplayMemory  # 经验回放
from rlschool import make_env  # 使用 RLSchool 创建飞行器环境
from model import QuadrotorModel
from agent import QuadrotorAgent
from parl.algorithms import DDPG
import parl

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


def run_evaluate_episode(env, agent, max_action, is_render=False):
    obs = env.reset()
    total_reward = 0
    while True:
        batch_obs = np.expand_dims(obs, axis=0)
        pred_action = agent.predict(batch_obs.astype('float32'))            
        pred_action = np.squeeze(pred_action)                                                    
        env_action = pred_action[0] + 0.2*pred_action[1:] 
            
        env_action = np.clip(
                np.random.normal(env_action, EXPL_NOISE * max_action), -1.0,
                1.0)           
        action = action_mapping(env_action, env.action_space.low[0],
                                env.action_space.high[0])

        next_obs, reward, done, info = env.step(action)

        obs = next_obs
        total_reward += reward
        if is_render:
            env.render()

        if done:
            break
    return total_reward


def main():
    # 创建飞行器环境
    env = make_env("Quadrotor", task="no_collision", seed=1)
    env.reset()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0] + 1
    max_action = float(env.action_space.high[0])
    
    
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
    
    agent.restore_critic('model_dir/critic.ckpt')
    agent.restore_actor('model_dir/actor.ckpt')
    for epics in range(1, 5):
        evaluate_reward = run_evaluate_episode(env, agent, max_action, is_render=True)
        print("evaluate_reward: ", evaluate_reward)


    

if __name__ == '__main__':
    main()
