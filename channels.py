import torch
import torch.nn as nn
import numpy as np

class Channel():
	#------------------------------------------------------
	def __init__(self, SX = config):

		self.config = config
		self._state = torch.zeros((self.SX, self.SY)).to(self.device)
	#------------------------------------------------------
	def __getattr__(self, attr):
		return self.config[attr]
	@property
	def state(self):
		return self._state
	@state.setter
	def state(self, state):
		assert state.shape == (self.SX, self.SY)
	#------------------------------------------------------
	def __getattr__(self, attr):
		return self.config[attr]
	#------------------------------------------------------
	def update(self, dX):

		self.state = torch.clip(self.state + (1.0 / self.T) * DX, 
			min = 0., max = 1.)