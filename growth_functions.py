import torch
import torch.nn as nn
import numpy as np
from spaces import *

g_funcs = {
    0: lambda n, m, s: torch.max(torch.zeros_like(n), 1 - (n - m) ** 2 / (9 * s ** 2)) ** 4 * 2 - 1, # polynomial (quad4)
    1: lambda n, m, s: torch.exp(- (n - m) ** 2 / (2 * s ** 2)-1e-3) * 2 - 1,  # exponential / gaussian (gaus)
    2: lambda n, m, s: (torch.abs(n - m) <= s).float() * 2 - 1 , # step (stpz)
}

class Polynomial_GF(nn.Module):
	#------------------------------------------------------
	def __init__(self, m, s, config):

		super().__init__()

		self.config = config

		self.register_parameter("m", nn.Parameter(m))
		self.register_parameter("s", nn.Parameter(s))
		self.to(self.config["device"])
	#------------------------------------------------------
	def forward(self, X):
		return g_funcs[0](X, self.m, self.s)

class Exponential_GF(nn.Module):
	#------------------------------------------------------
	def __init__(self, m, s, config):

		super().__init__()

		self.config = config

		self.register_parameter("m", nn.Parameter(m))
		self.register_parameter("s", nn.Parameter(s))
		self.to(self.config["device"])
	#------------------------------------------------------
	def forward(self, X):
		return g_funcs[1](X, self.m, self.s)

class Step_GF(nn.Module):
	#------------------------------------------------------
	def __init__(self, m, s, config):

		super().__init__()

		self.config = config

		self.register_parameter("m", nn.Parameter(m))
		self.register_parameter("s", nn.Parameter(s))
		self.to(self.config["device"])
	#------------------------------------------------------
	def forward(self, X):
		return g_funcs[2](X, self.m, self.s)

class Custom_GF(nn.Module):
	#------------------------------------------------------
	def __init__(self, m, s, func, config):

		super().__init__()

		self.config = config
		self.func = func

		self.register_parameter("m", nn.Parameter(m))
		self.register_parameter("s", nn.Parameter(s))
		self.to(self.config["device"])
	#------------------------------------------------------
	def forward(self, X):
		return self.func(X, self.m, self.s)



#==============================================================================
#===============================PARAMETER SPACE================================
#==============================================================================


class GF_param_space(DictSpace):
	#------------------------------------------------------
	def __init__(self, nb_k):
		spaces = Dict(
			m = BoxSpace(low=0.05, high=0.5, shape=(nb_k,), mutation_mean=torch.zeros((nb_k,)), 
				mutation_std=0.2*torch.ones((nb_k,)), indpb=1, dtype=torch.float32),
            s = BoxSpace(low=0.001, high=0.18, shape=(nb_k,), mutation_mean=torch.zeros((nb_k,)), 
            	mutation_std=0.01**torch.ones((nb_k,)), indpb=0.1, dtype=torch.float32),
		)
		super().__init__(spaces = spaces)