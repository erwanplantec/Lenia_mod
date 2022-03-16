import torch
import torch.nn as nn
import numpy as np

class Lenia_C(nn.Module):
	#------------------------------------------------------
	def __init__(self, config):

		super().__init__()

		self.channels = []
		self.kernels = []

		self.config = config

		self.to(self.device)
	#------------------------------------------------------
	#-----------------------PROPERTIES---------------------
	#------------------------------------------------------
	def __getattr__(self, attr):
		return self.config[attr]
	#------------------------------------------------------
	def __getitem__(self, i):
		return self.channels[i].state
	#------------------------------------------------------
	@property
	def state(self):
		return torch.cat([c.state for c in self.channels])
	@state.setter
	def state(self, state):
		assert state.shape[0] == self.C
		for i, c in enumerate(self.channels):
			c.state = state[i]
	#------------------------------------------------------
	@property
	def C(self):
		return len(self.channels)
	#------------------------------------------------------
	#------------------------BUILDING----------------------
	#------------------------------------------------------
	def add_kernel(self, kernel):
		self.add_module(kernel)
		self.kernels.append(kernel)
	#------------------------------------------------------
	def add_channel(self, channel):
		self.channels.append(channel)
	#------------------------------------------------------
	#-------------------------RUNNING----------------------
	#------------------------------------------------------
	def step(self):
		X_fft = [torch.rfft(self.state[:,:,:,i], signal_ndim=2, onesided=False) 
			for i in range(self.C)]
		dX = torch.zeros((self.C, self.SX, self.SY))
		dXn = torch.zeros((self.C, self.SX, self.SY))
		for kernel in self.kernels:
			field = kernel(X_fft)
			dX[kernel.channel] = dX[kernel.channel] + kernel.h * field
			dXn[kernel.channel] = dXn[kernel.channel] + kernel.h

		for i, c in enumerate(self.channels):
			c.update(dX[i] / dXn[i])
	#------------------------------------------------------
	def run(self, T, record = True):
		if record :
			orb = torch.zeros((T, self.C, self.SX, self.SY))
		for t in range(T):
			self.step()
			if record:
				orb[t] = self.state
		return orb if record else self.state
