import torch
import torch.nn as nn
import numpy as np

class Lenia_C(nn.Module):
	#------------------------------------------------------
	def __init__(self):

		super().__init__()

		self.channels = []
		self.kernels = []
	#------------------------------------------------------
	#-----------------------PROPERTIES---------------------
	#------------------------------------------------------
	def __getitem__(self, i):
		return self.channels[i].state
	@property
	def state(self):
		return torch.cat([c.state for c in self.channels])
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
		pass
	#------------------------------------------------------
	def run(self, T):
		pass
