import torch
import torch.nn as nn
import numpy as np
from spaces import *

import matplotlib.pyplot as plt

g_funcs = {
    0: lambda n, m, s: torch.max(torch.zeros_like(n), 1 - (n - m) ** 2 / (9 * s ** 2)) ** 4 * 2 - 1, # polynomial (quad4)
    1: lambda n, m, s: torch.exp(- (n - m) ** 2 / (2 * s ** 2)-1e-3) * 2 - 1,  # exponential / gaussian (gaus)
    2: lambda n, m, s: (torch.abs(n - m) <= s).float() * 2 - 1 , # step (stpz)
}

class Abstract_GF(nn.Module):
	#------------------------------------------------------
	def __init__(self, config):
		super().__init__()
		self.config = config
		self.to(self.config.device)
	#------------------------------------------------------
	def __str__(self):
		return self.__class__.__name__
	#------------------------------------------------------
	def forward(self, X):
		raise NotImplementedError()
	#------------------------------------------------------
	def show(self):
		x = torch.from_numpy(np.linspace(0, 1))
		y = self(x)

		plt.plot(x, y.detach().cpu().numpy())
		plt.title(self.__str__())
		plt.show()

class Polynomial_GF(Abstract_GF):
	#------------------------------------------------------
	def __init__(self, m, s, config):

		super().__init__(config)
		self.register_parameter("m", nn.Parameter(m))
		self.register_parameter("s", nn.Parameter(s))
	#------------------------------------------------------
	def __str__(self):
		return f"Polynomial : m = {self.m.data}, s = {self.s.data}"
	#------------------------------------------------------
	def forward(self, X):
		return g_funcs[0](X, self.m, self.s)

class Exponential_GF(Abstract_GF):
	#------------------------------------------------------
	def __init__(self, m, s, config):

		super().__init__(config)

		self.register_parameter("m", nn.Parameter(m))
		self.register_parameter("s", nn.Parameter(s))
	#------------------------------------------------------
	def __str__(self):
		return f"Exponential : m = {self.m.data}, s = {self.s.data}"
	#------------------------------------------------------
	def forward(self, X):
		return g_funcs[1](X, self.m, self.s)

class Step_GF(Abstract_GF):
	#------------------------------------------------------
	def __init__(self, m, s, config):

		super().__init__(config)

		self.register_parameter("m", nn.Parameter(m))
		self.register_parameter("s", nn.Parameter(s))
	#------------------------------------------------------
	def __str__(self):
		return f"Step : m = {self.m.data}, s = {self.s.data}"
	#------------------------------------------------------
	def forward(self, X):
		return g_funcs[2](X, self.m, self.s)

class Wall_GF(Abstract_GF):
	#------------------------------------------------------
	def __init__(self, config, m = .0001, s = 10):
		super().__init__(config)
		self.m = m
		self.s = s
	#------------------------------------------------------
	def __str__(self):
		return f"Wall : m = {self.m}, s = {self.s}"
	#------------------------------------------------------
	def forward(self, X):
		return - torch.clamp(X - self.m, 0, 1) * self.s


class Custom_GF(Abstract_GF):
	#------------------------------------------------------
	def __init__(self, m, s, func, config):

		super().__init__(config)

		self.func = func

		self.register_parameter("m", nn.Parameter(m))
		self.register_parameter("s", nn.Parameter(s))

		self.to(self.config["device"])
	#------------------------------------------------------
	def forward(self, X):
		return self.func(X, self.m.data, self.s.data)



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

def gf_params_generator(nb_k):
	space = GF_param_space(nb_k)
	return space.sample()