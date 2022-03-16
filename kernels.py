import torch
import torch.nn as nn
import numpy as np

import typing as t

from utils import complex_mult_torch, roll_n

#==============================================================================

ker_c= lambda x, a, w, b : (b*torch.exp(-((x.unsqueeze(-1)-r)/w)**2 / 2)).sum(-1)

class Kernel(nn.Module):
	#------------------------------------------------------
	def __init__(self, channel, a, b, w, r, config):

		super().__init__()

		self.config = config

		self.channel = channel

		# Center of bumps
		self.register_parameter("a", nn.parameter(a))
		# Height of bumps
		self.register_parameter("b", nn.parameter(b))
		# Width of bumps
		self.register_parameter("w", nn.parameter(w))
		# Relative kernel radius
		self.register_parameter("r", nn.parameter(r))
	#------------------------------------------------------
	def __getattr__(self, attr):
		return self.config[attr]
	#------------------------------------------------------
	def compute_kernel(self):
		
		x = torch.arange(self.config.SX).to(self.device)
		y = torch.arange(self.config.SY).to(self.device)
		xx = x.view(-1, 1).repeat(1, self.system.SY)
		yy = y.repeat(self.config.SX, 1)
		X = (xx - int(self.config.SX / 2)).float() 
		Y = (yy - int(self.config.SY / 2)).float() 

      	D = torch.sqrt(X**2 + Y**2) / ((self.config.R + 15) * self.r)

      	kernel = torch.sigmoid(-(D - 1) * 10) * ker_c(D, 
      		self.a, self.w, self.b)

      	kernel = (kernel / torch.sum(kernel)).unsqueeze(0).unsqueeze(0)

      	self.kernel = torch.rfft(kernel, signal_ndim = 2, onesided = False).to self.device
	#------------------------------------------------------
	def forward(self, X_fft):
		pot_fft = complex_mult_torch(self.kernel, X_fft[self.channel])

		pot = torch.irfft(pot_fft, signal_ndim = 2, onseide = False)
		pot = roll_n(potential, 2, potential.size(2) // 2)
		pot = roll_n(potential, 1, potential.size(2) // 2)

		return pot