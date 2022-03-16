import torch
import torch.nn as nn
import numpy as np

import typing as t

from utils import complex_mult_torch, roll_n

class Interaction(nn.Module):
	#------------------------------------------------------
	def __init__(self, kernel, g_func, h, config):

		super().__init__()

		self.config = config

		self.add_module(kernel)
		self.add_module(g_func)

		self.register_parameter("h", nn.parameter(h))

		self.kernel, self.g_func = kernel, g_func
		self.to(self.device)
	#------------------------------------------------------
	def __getattr__(self, attr):
		return self.config[attr]
	#------------------------------------------------------
	def forward(self, X):
		return self.g_func(self.kernel(X)) * self.h