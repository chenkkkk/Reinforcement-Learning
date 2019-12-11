# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 00:16:33 2019

@author: chen
"""

import numpy as np
import gym


class AtariEmulator():
    def __init__(self):
        self.life_lost = False
        self.totalReward = 0
        self.env = gym.make('Breakout-v0')
        self.env.reset()
        self.env.frameskip = 1
        self.ale = self.env.ale
    def start(self):
        # In train mode: life_lost = True but game is not over, don't restart the game
        if not self.life_lost or self.ale.game_over():
            self.ale.reset_game()
        self.life_lost = False
        
    def isTerminal(self):
        return self.ale.game_over() or self.life_lost
    
    def next(self, action): # index of action int legalActions
        lives = self.ale.lives() # the remaining lives
        reward = 0
        for i in range(4): # action repeat
            observation, reward_, done, info = self.env.step(action)
            reward += reward_
            self.life_lost = (lives != self.ale.lives())  # after action, judge life lost
            if self.life_lost:
                reward -= 1
            if done or self.isTerminal():
                break
        self.totalReward += reward
        state = observation.mean(axis=2)
        return state,reward,self.isTerminal()
    
    def randomStart(self, s_t):
        channels = s_t.shape[-1]
        self.start()
        self.life_lost = False
        for i in range(np.random.randint(channels, 30) + 1):
            s_t_plus_1, r_t, isTerminal = self.next(0)
            s_t[:,:, 0:channels-1] = s_t[:,:, 1:channels]
            s_t[:,:, -1] = s_t_plus_1
            if isTerminal:
                self.start()
                self.life_lost = False