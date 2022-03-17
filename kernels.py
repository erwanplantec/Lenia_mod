import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

import typing as t

from utils import complex_mult_torch, roll_n

from spaces import *

#==============================================================================

ker_c= lambda x, a, w, b : (b*torch.exp(-((x.unsqueeze(-1)-a)/w)**2 / 2)).sum(-1)

class Kernel(nn.Module):
	#------------------------------------------------------
	def __init__(self, a, b, w, r, h, config):

		super().__init__()

		self.config = config

		# Center of bumps
		self.register_parameter("a", nn.Parameter(a))
		# Height of bumps
		self.register_parameter("b", nn.Parameter(b))
		# Width of bumps
		self.register_parameter("w", nn.Parameter(w))
		# Relative kernel radius
		self.register_parameter("r", nn.Parameter(r))
		# Relative weight
		self.register_parameter("h", nn.Parameter(h))

		self.compute_kernel()
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
	def compute_kernel(self):
		
		x = torch.arange(self.SX).to(self.device)
		y = torch.arange(self.SY).to(self.device)
		xx = x.view(-1, 1).repeat(1, self.SY)
		yy = y.repeat(self.SX, 1)
		X = (xx - int(self.SX / 2)).float() 
		Y = (yy - int(self.SY / 2)).float() 

		D = torch.sqrt(X**2 + Y**2) / ((self.R + 15) * self.r)

		kernel = torch.sigmoid(-(D - 1) * 10) * ker_c(D, 
			self.a, self.w, self.b)

		kernel = (kernel / torch.sum(kernel)).unsqueeze(0).unsqueeze(0)

		self.kernel = torch.rfft(kernel, signal_ndim = 2, onesided = False).to(self.device)
	#------------------------------------------------------
	def forward(self, X_fft):
		pot_fft = complex_mult_torch(self.kernel, X_fft)

		pot = torch.irfft(pot_fft, signal_ndim = 2, onesided = False)
		pot = roll_n(pot, 2, pot.size(2) // 2)
		pot = roll_n(pot, 1, pot.size(2) // 2).squeeze(0)

		return pot
	#------------------------------------------------------
	def show(self):
		plt.imshow(torch.irfft(self.kernel, signal_ndim=2, onesided=False).detach().view(256, 256).numpy())
		plt.show()

#==============================================================================
#===============================PARAMETER SPACE================================
#==============================================================================


class Kernel_param_space(DictSpace):
	#------------------------------------------------------
	def __init__(self, nb_k):
		spaces = Dict(
			a = BoxSpace(low=0, high=1, shape=(nb_k,3), mutation_mean=torch.zeros((nb_k,3)), 
				mutation_std=0.2*torch.ones((nb_k,3)), indpb=1, dtype=torch.float32),
			b = BoxSpace(low=0.0, high=1.0, shape=(nb_k,3), mutation_mean=torch.zeros((nb_k,3)), 
				mutation_std=0.2*torch.ones((nb_k,3)), indpb=1, dtype=torch.float32),
			w = BoxSpace(low=0.01, high=0.5, shape=(nb_k,3), mutation_mean=torch.zeros((nb_k,3)), 
				mutation_std=0.2*torch.ones((nb_k,3)), indpb=1, dtype=torch.float32),
			r = BoxSpace(low=0.2, high=1.0, shape=(nb_k,), mutation_mean=torch.zeros((nb_k,)), 
				mutation_std=0.2*torch.ones((nb_k,)), indpb=1, dtype=torch.float32),
			h = BoxSpace(low=0, high=1.0, shape=(nb_k,), mutation_mean=torch.zeros((nb_k,)), 
				mutation_std=0.2*torch.ones((nb_k,)), indpb=0.1, dtype=torch.float32)
		)
		super().__init__(spaces = spaces)

if __name__ == '__main__':
	ps = Kernel_param_space(1)
	print(ps.sample())


