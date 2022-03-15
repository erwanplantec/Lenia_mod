import torch
import torch.nn as nn
import numpy as np

import typing as t

class Kernel(nn.Module):
	#------------------------------------------------------
	def __init__(self, growth_function : t.Callable):

		super().__init__()

		self.growth_function = growth_function
	#------------------------------------------------------
	def compute_kernel(self):
		pass
	#------------------------------------------------------
	def forward(self, X):
		return dX