import torch
import torch.nn as nn
import numpy as np

class Channel():
	#------------------------------------------------------
	def __init__(self, config):

		self.config = config
		self._state = torch.zeros((self.SX, self.SY)).to(self.device)
	#------------------------------------------------------
	@property
	def state(self):
		return self._state
	@state.setter
	def state(self, state):
		assert state.shape == (self.SX, self.SY)
		self._state = state
	#------------------------------------------------------
	@property
	def SX(self):
		return self.config["SX"]
	@property
	def SY(self):
		return self.config["SY"]
	@property
	def T(self):
		return self.config["T"]
	@property
	def R(self):
		return self.config["R"]
	@property
	def device(self):
		return self.config["device"]
	#------------------------------------------------------

	def __getattr__(self, attr):
		return self.config[attr]
	#------------------------------------------------------
	def update(self, dX):

		self.state = torch.clip(self.state + (1.0 / self.T) * dX, 
			min = 0., max = 1.)