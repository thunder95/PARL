import os
import numpy as np

import parl
from parl import layers
from paddle import fluid
from parl.utils import logger
from parl.utils import action_mapping # 将神经网络输出映射到对应的 实际动作取值范围 内
from parl.utils import ReplayMemory # 经验回放
import gym
import rlbench.gym
import logging
import imageio
import os
import time

MAX_EPISODES = 5000
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
GAMMA = 0.99
TAU = 0.005
MEMORY_SIZE = int(1e6)
WARMUP_SIZE = 1e3
BATCH_SIZE = 256
ENV_SEED = 1
EXPL_NOISE = 0.1  # Std of Gaussian exploration noise
MAX_STEPS_PER_EPISODES = 200

class ActorModel(parl.Model):
    def __init__(self, act_dim, max_action):
        hid1_size = 512
        hid2_size = 256

        self.fc1 = layers.fc(size=hid1_size, act='relu')
        self.fc2 = layers.fc(size=hid2_size, act='relu')
        self.fc3 = layers.fc(size=act_dim, act='tanh')

        self.max_action = max_action

    def policy(self, obs):
        hid1 = self.fc1(obs)
        hid2 = self.fc2(hid1)
        means = self.fc3(hid2)
        # means = means * self.max_action
        return means


class CriticModel(parl.Model):
    def __init__(self):
        hid1_size = 512
        hid2_size = 256

        self.fc1 = layers.fc(size=hid1_size, act='relu')
        self.fc2 = layers.fc(size=hid2_size, act='relu')
        self.fc3 = layers.fc(size=1, act=None)

        self.fc4 = layers.fc(size=hid1_size, act='relu')
        self.fc5 = layers.fc(size=hid2_size, act='relu')
        self.fc6 = layers.fc(size=1, act=None)

    #注意此处返回了两个Q
    def value(self, obs, act):
        hid1 = self.fc1(obs)
        concat1 = layers.concat([hid1, act], axis=1)
        Q1 = self.fc2(concat1)
        Q1 = self.fc3(Q1)
        Q1 = layers.squeeze(Q1, axes=[1])

        hid2 = self.fc4(obs)
        concat2 = layers.concat([hid2, act], axis=1)
        Q2 = self.fc5(concat2)
        Q2 = self.fc6(Q2)
        Q2 = layers.squeeze(Q2, axes=[1])
        return Q1, Q2

    def Q1(self, obs, act):
        hid1 = self.fc1(obs)
        concat1 = layers.concat([hid1, act], axis=1)
        Q1 = self.fc2(concat1)
        Q1 = self.fc3(Q1)
        Q1 = layers.squeeze(Q1, axes=[1])

        return Q1

class RLBenchModel(parl.Model):
    def __init__(self, act_dim, max_action):
        self.actor_model = ActorModel(act_dim, max_action)
        self.critic_model = CriticModel()

    def policy(self, obs):
        return self.actor_model.policy(obs)

    def value(self, obs, act):
        return self.critic_model.value(obs, act)

    def Q1(self, obs, act):
        return self.critic_model.Q1(obs, act)

    def get_actor_params(self):
        return self.actor_model.parameters()


class RLBenchAgent(parl.Agent):
    def __init__(self, algorithm, obs_dim, act_dim):
        assert isinstance(obs_dim, int)
        assert isinstance(act_dim, int)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(RLBenchAgent, self).__init__(algorithm)

        # Attention: In the beginning, sync target model totally.
        self.alg.sync_target(decay=0)
        self.learn_it = 0
        self.policy_freq = self.alg.policy_freq

    def build_program(self):
        self.pred_program = fluid.Program()
        self.actor_learn_program = fluid.Program()
        self.critic_learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program):
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            self.pred_act = self.alg.predict(obs)

        with fluid.program_guard(self.actor_learn_program):
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            self.actor_cost = self.alg.actor_learn(obs)

        with fluid.program_guard(self.critic_learn_program):
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            act = layers.data(
                name='act', shape=[self.act_dim], dtype='float32')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            next_obs = layers.data(
                name='next_obs', shape=[self.obs_dim], dtype='float32')
            terminal = layers.data(name='terminal', shape=[], dtype='bool')
            self.critic_cost = self.alg.critic_learn(obs, act, reward,
                                                     next_obs, terminal)

    def predict(self, obs):
        obs = np.expand_dims(obs, axis=0)
        act = self.fluid_executor.run(
            self.pred_program, feed={'obs': obs},
            fetch_list=[self.pred_act])[0]
        return act

    def learn(self, obs, act, reward, next_obs, terminal):
        self.learn_it += 1
        feed = {
            'obs': obs,
            'act': act,
            'reward': reward,
            'next_obs': next_obs,
            'terminal': terminal
        }
        critic_cost = self.fluid_executor.run(
            self.critic_learn_program,
            feed=feed,
            fetch_list=[self.critic_cost])[0]

        actor_cost = None
        if self.learn_it % self.policy_freq == 0:
            actor_cost = self.fluid_executor.run(
                self.actor_learn_program,
                feed={'obs': obs},
                fetch_list=[self.actor_cost])[0]
            self.alg.sync_target()
        return actor_cost, critic_cost

    def save_actor(self, save_path):
        program = self.actor_learn_program
        dirname = os.sep.join(save_path.split(os.sep)[:-1])
        filename = save_path.split(os.sep)[-1]
        fluid.io.save_params(
            executor=self.fluid_executor,
            dirname=dirname,
            main_program=program,
            filename=filename)

    def save_critic(self, save_path):
        program = self.critic_learn_program
        dirname = os.sep.join(save_path.split(os.sep)[:-1])
        filename = save_path.split(os.sep)[-1]
        fluid.io.save_params(
            executor=self.fluid_executor,
            dirname=dirname,
            main_program=program,
            filename=filename)

    def restore_actor(self, save_path):
        program = self.actor_learn_program
        if type(program) is fluid.compiler.CompiledProgram:
            program = program._init_program
        dirname = os.sep.join(save_path.split(os.sep)[:-1])
        filename = save_path.split(os.sep)[-1]
        fluid.io.load_params(
            executor=self.fluid_executor,
            dirname=dirname,
            main_program=program,
            filename=filename)

    def restore_critic(self, save_path):
        program = self.critic_learn_program
        if type(program) is fluid.compiler.CompiledProgram:
            program = program._init_program
        dirname = os.sep.join(save_path.split(os.sep)[:-1])
        filename = save_path.split(os.sep)[-1]
        fluid.io.load_params(
            executor=self.fluid_executor,
            dirname=dirname,
            main_program=program,
            filename=filename)

class ImageLogger(object):
    def __init__(self, path):
        self.path = path
        self.image_dict = []

    def __call__(self, _frame):
        self.image_dict.append(_frame)

    def save(self):
        imageio.mimsave(self.path, self.image_dict, 'GIF')



import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '--env', help='Fetch environment name', default='reach_target-state-v0')
parser.add_argument(
    '--train_total_episodes',
    type=int,
    default=int(3e5),
    help='maximum training episodes')
parser.add_argument(
    '--test_every_episodes',
    type=int,
    default=int(8e2),
    help='the step interval between two consecutive evaluations')
parser.add_argument(
    '--store_every_episodes',
    type=int,
    default=int(4e3),
    help='the step interval for model store')

args = parser.parse_args()

class LoggingInstance(object):
    def __init__(self, logfile):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        self.fh = logging.FileHandler(logfile, mode='a')
        self.fh.setLevel(logging.DEBUG)  # 用于写到file的等级开关
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.fh.setFormatter(formatter)
        self.logger.addHandler(self.fh)

    def logging_string(self, string_msg):
        self.logger.info(string_msg)

    def decorator(self):
        self.logger.removeHandler(self.fh)


def run_train_episode(env, agent, rpm):
    obs_list = []
    action_list = []
    reward_list = []
    terminal_info = []
    obs = env.reset()
    obs_list.append(obs)
    total_reward = 0
    steps = 0
    max_action = float(env.action_space.high[0])
    min_action = float(env.action_space.low[0])
    episode_goal = np.expand_dims(obs[-3:], axis=0)
    while MAX_STEPS_PER_EPISODES-steps:
        steps += 1
        batch_obs = np.expand_dims(obs[8:15], axis=0)
        batch_obs_with_goal = np.concatenate((batch_obs, episode_goal), axis=1)
        if rpm.size() < WARMUP_SIZE:
            action = env.action_space.sample()
        else:
            action = agent.predict(batch_obs_with_goal.astype('float32'))
            action = np.squeeze(action)

            # Add exploration noise, and clip to [-max_action, max_action]
            # action += noise()
            action = np.random.normal(action, EXPL_NOISE * max_action)
            action = np.clip(action, min_action, max_action)

        next_obs, reward, done, info = env.step(action)
        obs_list.append(next_obs)
        action_list.append(action)
        reward_list.append(reward)
        terminal_info.append(done)

        obs = next_obs
        total_reward += reward
        # print(total_reward)
        if done:
            break

    for idx in range(steps):
        obs = obs_list[idx]
        next_obs = obs_list[idx + 1]
        obs_desired_goal = np.concatenate((obs[8:15], obs[-3:]))
        next_obs_desired_goal = np.concatenate((next_obs[8:15], next_obs[-3:]))
        action = action_list[idx]
        reward = reward_list[idx]
        done = terminal_info[idx]
        obs_achieved_goal = np.concatenate((obs[8:15], obs[22:25]))
        next_obs_achieved_goal = np.concatenate((next_obs[8:15], next_obs[22:25]))
        rpm.append(obs_desired_goal, action, reward, next_obs_desired_goal, done)
        rpm.append(obs_achieved_goal, action, 1, next_obs_achieved_goal, True)

    if rpm.size() > WARMUP_SIZE:
        batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal = rpm.sample_batch(
            BATCH_SIZE)
        agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs,
                    batch_terminal)

    return total_reward


def run_evaluate_episode(env, agent, image_recoder):
    obs = env.reset()
    total_reward = 0
    episode_goal = np.expand_dims(obs[-3:], axis=0)
    steps = 0
    while MAX_STEPS_PER_EPISODES - steps:
        steps += 1
        image_recoder(env.render(mode='rgb_array'))
        batch_obs = np.expand_dims(obs[8:15], axis=0)
        batch_obs_with_goal = np.concatenate((batch_obs, episode_goal), axis=1)
        action = agent.predict(batch_obs_with_goal.astype('float32'))
        action = np.squeeze(action)
        action = action_mapping(action, env.action_space.low[0],
                                env.action_space.high[0])

        next_obs, reward, done, info = env.step(action)
        # time.sleep(0.1)
        # print(reward)

        obs = next_obs
        total_reward += reward

        if done:
            break
    return total_reward

is_train=False

if is_train:
    env = gym.make(args.env)
    logger = LoggingInstance('train.txt')
else:
    env = gym.make(args.env, render_mode='rgb_array')
    logger = LoggingInstance('eval.txt')

env.reset()
obs_dim = 7
goal_dim = 3

act_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

model = RLBenchModel(act_dim, max_action)
algorithm = parl.algorithms.TD3(
        model,
        max_action=max_action,
        gamma=GAMMA,
        tau=TAU,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR)

agent = RLBenchAgent(algorithm, obs_dim + goal_dim, act_dim)


if is_train:
    rpm = ReplayMemory(MEMORY_SIZE, obs_dim + goal_dim, act_dim)

    test_flag = 0
    store_flag = 0
    total_episodes = 16000
    while total_episodes < args.train_total_episodes:
        train_reward = run_train_episode(env, agent, rpm)
        total_episodes += 1
        logger.logging_string('Episodes: {} Reward: {}'.format(total_episodes, train_reward))
        #tensorboard.add_scalar('train/episode_reward', train_reward,total_episodes)

        if total_episodes // args.test_every_episodes >= test_flag:
            while total_episodes // args.test_every_episodes >= test_flag:
                test_flag += 1
            evaluate_reward = run_evaluate_episode(env, agent, render=False)
            logger.logging_string('Episodes {}, Evaluate reward: {}'.format(
                total_episodes, evaluate_reward))

            #tensorboard.add_scalar('eval/episode_reward', evaluate_reward,total_episodes)

        if total_episodes // args.store_every_episodes >= store_flag:
            while total_episodes // args.store_every_episodes >= store_flag:
                store_flag += 1
                agent.save_actor('RLBench/train_model/actor_' + str(total_episodes) + '.ckpt')
                agent.save_critic('RLBench/train_model/critic_' + str(total_episodes) + '.ckpt')
else:
    model_idx = 160000
    recode_path = 'RLBench/records/' + str(model_idx)
    if not os.path.exists(recode_path):
        os.makedirs(recode_path)

    agent.restore_critic('RLBench/train_model/critic_' + str(model_idx) + '.ckpt')
    agent.restore_actor('RLBench/train_model/actor_' + str(model_idx) + '.ckpt')

    for epics in range(1, 5):
        image_recoder = ImageLogger('RLBench/records/' + str(model_idx) + '/video_' + str(epics) + '.gif')
        evaluate_reward = run_evaluate_episode(env, agent, image_recoder)
        logger.logging_string('Episodes {}, Evaluate reward: {}'.format(
            epics, evaluate_reward))
        time.sleep(0.5)
        image_recoder.save()


env.close()
logger.decorator()
