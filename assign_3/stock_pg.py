import os
import gym
import numpy as np
import sys
import codecs

import paddle.fluid as fluid
import parl
from parl import layers
from parl.utils import logger
from parl.algorithms import PolicyGradient
from market_env import MarketEnv
import matplotlib.pyplot as plt

LEARNING_RATE = 1e-3

class Model(parl.Model):
    def __init__(self, act_dim):
        act_dim = act_dim
        hid1_size = act_dim * 20
        self.fc1 = layers.fc(size=hid1_size, act='tanh')
        self.fc2 = layers.fc(size=hid1_size, act='tanh')
        self.fc3 = layers.fc(size=act_dim, act='softmax')

    def forward(self, obs):
        out = self.fc1(obs)
        out = self.fc2(obs)
        out = self.fc3(out)
        return out

class Agent(parl.Agent):
    def __init__(self, algorithm, obs_dim, act_dim):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(Agent, self).__init__(algorithm)

    def build_program(self):
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program):
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            self.act_prob = self.alg.predict(obs)

        with fluid.program_guard(
                self.learn_program):
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            act = layers.data(name='act', shape=[1], dtype='int64')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            self.cost = self.alg.learn(obs, act, reward)

    def sample(self, obs):
        obs = np.expand_dims(obs, axis=0)
        act_prob = self.fluid_executor.run(
            self.pred_program,
            feed={'obs': obs.astype('float32')},
            fetch_list=[self.act_prob])[0]
        act_prob = np.squeeze(act_prob, axis=0)
        act = np.random.choice(range(self.act_dim), p=act_prob)
        return act

    def predict(self, obs):
        obs = np.expand_dims(obs, axis=0)
        act_prob = self.fluid_executor.run(
            self.pred_program,
            feed={'obs': obs.astype('float32')},
            fetch_list=[self.act_prob])[0]
        act_prob = np.squeeze(act_prob, axis=0)
        act = np.argmax(act_prob)  # 根据动作概率选择概率最高的动作
        return act

    def learn(self, obs, act, reward):
        act = np.expand_dims(act, axis=-1)
        feed = {
            'obs': obs.astype('float32'),
            'act': act.astype('int64'),
            'reward': reward.astype('float32')
        }
        cost = self.fluid_executor.run(
            self.learn_program, feed=feed, fetch_list=[self.cost])[0]
        return cost


def run_episode(env, agent):
    obs_list, action_list, reward_list = [], [], []
    res = env._reset()
    while True:
        # print("----->", res[1][0].shape, res[0].shape)
        # print(res[0])
        # print(res[1][0][1])
        # exit()

        # tmp_obs =  res[1][0][action].ravel()
        tmp_obs =  res[1][0].ravel()
        obs_list.append(tmp_obs)

        action = agent.sample(tmp_obs)
        # action = np.where(res[0]==0)[0][0]
        action_list.append(action)

        obs, reward, done, info = env._step(action)
        reward_list.append(reward)

        if done:
            break
    return obs_list, action_list, reward_list


# 评估 agent, 跑 5 个episode，求平均
def evaluate(env, agent, render=False):
    eval_reward = []
    for i in range(1):
        res = env._reset()
        episode_reward = 0
        while True:
            action = agent.predict(res[1][0].ravel()) # 选取最优动作
            obs, reward, isOver, _ = env._step(action)
            
            episode_reward += reward
            if reward > 0:
                 print(env.currentTargetIndex, reward, episode_reward)
           
            # print(reward, len(obs), env.currentTargetIndex)
            if render:
                tmp_state = env._render()
                # print(state[1].shape)
                x1 = list(tmp_state[1][0][0].ravel())
                x2 = list(tmp_state[1][0][1].ravel())
                y = [i+1 for i in range(env.scope)]
                # plt.clf()
                plt.plot(y, x1, color='red',linewidth=2.0,linestyle='--')
                plt.plot(y, x2, color='blue',linewidth=3.0,linestyle='-.')
                plt.title(str(env.currentTargetIndex) + "=> action: " + env.actions[action]+ ", reward: " + str(reward) + ", total_reward: " + str(episode_reward))
                plt.savefig('/home/aistudio/data/' + str(env.currentTargetIndex) + ".png")
                plt.close('all')
            if isOver:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)

# 根据一个episode的每个step的reward列表，计算每一个Step的Gt
def calc_reward_to_go(reward_list, gamma=0.99):
    """calculate discounted reward"""
    reward_arr = np.array(reward_list)
    for i in range(len(reward_arr) - 2, -1, -1):
        # G_t = r_t + γ·r_t+1 + ... = r_t + γ·G_t+1
        reward_arr[i] += gamma * reward_arr[i + 1]
    # normalize episode rewards
    reward_arr -= np.mean(reward_arr)
    reward_arr /= np.std(reward_arr)
    return reward_arr


# 创建环境
codeMap = {}
codeListFilename = "./work/kospi_10.csv"
f = codecs.open(codeListFilename, "r", "utf-8")
for line in f:
	if line.strip() != "":
		tokens = line.strip().split(",") if not "\t" in line else line.strip().split("\t")
		codeMap[tokens[0]] = tokens[1]

f.close()

env = MarketEnv(dir_path = "./work/sample_data/", target_codes = list(codeMap.keys()), input_codes = [], start_date = "2013-08-26", end_date = "2015-08-25", sudden_death = -1.0)

obs_dim = 60 * 2
act_dim = 2
print(env.actions)
logger.info('obs_dim {}, act_dim {}'.format(obs_dim, act_dim))

model = Model(act_dim=act_dim)
alg = PolicyGradient(model, lr=LEARNING_RATE)
agent = Agent(alg, obs_dim=obs_dim, act_dim=act_dim)


# 加载模型
# if os.path.exists('./model.ckpt'):
#     agent.restore('./model.ckpt')

# for i in range(2000):
#     obs_list, action_list, reward_list = run_episode(env, agent)
#     # if i % 10 == 0:
#     #     logger.info("Train Episode {}, Reward Sum {}.".format(i,
#     #                                         sum(reward_list)))

#     batch_obs = np.array(obs_list)
#     batch_action = np.array(action_list)
#     batch_reward = calc_reward_to_go(reward_list)

#     agent.learn(batch_obs, batch_action, batch_reward)
#     if (i + 1) % 50 == 0:
#         total_reward = evaluate(env, agent, render=False)
#         logger.info('Episode {}, Test reward: {}'.format(i + 1,
#                                             total_reward))


# # save the parameters to ./model.ckpt
# agent.save('./pg_model.ckpt')

#运行预测
ckpt = 'pg_model.ckpt'
agent.restore(ckpt)
evaluate_reward = evaluate(env, agent,  render=True)
logger.info('Evaluate reward: {}'.format(evaluate_reward)) # 打印评估的reward


"""
import imageio,os
images = []
filenames=sorted((fn for fn in os.listdir('./data/') if fn.endswith('.png')))
for filename in filenames:
    images.append(imageio.imread('./data/' + filename))
imageio.mimsave('gif.gif', images,duration=0.25)
"""
