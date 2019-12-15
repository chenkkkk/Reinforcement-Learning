# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 19:40:39 2019

@author: chen
"""
import random
from .game import PongGame

class gameAgent():
	def __init__(self):
		self.game = PongGame()
		self.actions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
	'''next frame'''
	def nextFrame(self, action=None):
		if action is None:
			action = random.choice(self.actions)
		return self.game.nextFrame(action)