# -*- coding: utf-8 -*-

import numpy as np
import gym


class PongEmulator():
    def __init__(self):
        self.totalReward = 0
        self.env = gym.make('Pong-v0')
        self.env.reset()
    def start(self):
        # In train mode: life_lost = True but game is not over, don't restart the game
        self.env.reset()
    
    def next(self, action): # index of action int legalActions
        observation, reward_, done, info = self.env.step(action)
        self.totalReward += reward_
        state = observation.mean(axis=2)
        return state,reward_,done
    
    def randomStart(self, s_t):
        channels = s_t.shape[-1]
        self.start()
        for i in range(np.random.randint(channels, 30) + 1):
            s_t_plus_1, r_t, isTerminal = self.next(0)
            s_t[:,:, 0:channels-1] = s_t[:,:, 1:channels]
            s_t[:,:, -1] = s_t_plus_1
            if isTerminal:
                self.start()