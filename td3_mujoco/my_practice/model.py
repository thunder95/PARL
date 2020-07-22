# coding: utf-8
import numpy as np
import parl
from parl import layers
from paddle import fluid

class ActorModel(parl.Model):
    #def __init__(self, act_dim):
    #CHANGE
    def __init__(self, act_dim, max_action):
        hidden_dim_1, hidden_dim_2 = 64, 64
        self.fc1 = layers.fc(size=hidden_dim_1, act='tanh')
        self.fc2 = layers.fc(size=hidden_dim_2, act='tanh')
        self.fc3 = layers.fc(size=act_dim, act='tanh')
        
        self.max_action = max_action
        
    #CHANGE
    def policy(self, obs):
        x = self.fc1(obs)
        x = self.fc2(x)
        means = self.fc3(x)
        means = means * self.max_action
        return means


class CriticModel(parl.Model):
    def __init__(self):
        hidden_dim_1, hidden_dim_2 = 64, 64
        self.fc1 = layers.fc(size=hidden_dim_1, act='tanh')
        self.fc2 = layers.fc(size=hidden_dim_2, act='tanh')
        self.fc3 = layers.fc(size=1, act=None)
        
        #CHANGE
        self.fc4 = layers.fc(size=hidden_dim_1, act='tanh')
        self.fc5 = layers.fc(size=hidden_dim_2, act='tanh')
        self.fc6 = layers.fc(size=1, act=None)

    def value(self, obs, act):
        x = self.fc1(obs)
        concat = layers.concat([x, act], axis=1)
        x = self.fc2(concat)
        Q1 = self.fc3(x)
        Q1 = layers.squeeze(Q1, axes=[1])
        
        y = self.fc4(obs)
        concat2 = layers.concat([y, act], axis=1)
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


class QuadrotorModel(parl.Model):
    #CHANGE
    # def __init__(self, act_dim):
    #     self.actor_model = ActorModel(act_dim)
    def __init__(self, act_dim, max_action):
        self.actor_model = ActorModel(act_dim, max_action)
        self.critic_model = CriticModel()

    def policy(self, obs):
        return self.actor_model.policy(obs)

    def value(self, obs, act):
        return self.critic_model.value(obs, act)

    def get_actor_params(self):
        return self.actor_model.parameters()

    #CHANGE
    def Q1(self, obs, act):
        return self.critic_model.Q1(obs, act)
        
        
        