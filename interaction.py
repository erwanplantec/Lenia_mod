import torch
import torch.nn as nn
import numpy as np

import typing as t

from utils import complex_mult_torch, roll_n
from kernels import Kernel, Kernel_wall
from growth_functions import Wall_GF

import matplotlib.pyplot as plt

class Interaction(nn.Module):
	#------------------------------------------------------
	def __init__(self, srce, trget, kernels, g_funcs, config):

		assert len(kernels) == len(g_funcs)

		super().__init__()

		self.config = config

		self.kernels = nn.ModuleList(kernels)
		self.g_funcs = nn.ModuleList(g_funcs)

		self.srce, self.trget = srce, trget

		self.to(self.config["device"])
	#------------------------------------------------------
	def forward(self, X_fft):
		dX = torch.zeros((len(X_fft), self.config.SX, self.config.SY))
		dXn = torch.zeros((len(X_fft)))
		for k, g in zip(self.kernels, self.g_funcs):
			dX[self.trget] = dX[self.trget] + g(k(X_fft[self.srce])) * k.h
			dXn[self.trget] = dXn[self.trget] + k.h
		return dX, dXn
	#------------------------------------------------------
	def compute_kernel(self):
		for k in self.kernels :
			k.compute_kernel()
	#------------------------------------------------------
	def add(self, kernel, g_func):
		self.kernels.append(kernel)
		self.g_funcs.append(g_func)
	#------------------------------------------------------
	@staticmethod
	def build_random(srce, trget, nb_k, g_func, k_param_space, 
		gf_param_space, config):
		g_funcs = []
		kernels = []
		k_params = k_param_space(nb_k).sample()
		gf_params = gf_param_space(nb_k).sample()
		for i in range(nb_k):
			kernels.append(Kernel(k_params.a[i], k_params.b[i], 
				k_params.w[i], k_params.r[i], k_params.h[i], config))
			g_funcs.append(g_func(gf_params.m[i], gf_params.s[i], config))

		return Interaction(srce, trget, kernels, g_funcs, config)



class Wall_Interaction(nn.Module):
	#------------------------------------------------------
	def __init__(self, srce, trget, config):
		super().__init__()

		self.config = config
		self.srce, self.trget = srce, trget

		self.kernel = Kernel_wall(config)
		self.g_func = Wall_GF(config)
	#------------------------------------------------------
	def forward(self, X_fft):
		
		with torch.no_grad():
			dX = torch.zeros((len(X_fft), self.config.SX, self.config.SY))
			dXn = torch.zeros((len(X_fft)))
			dX[self.trget] = dX[self.trget] + self.g_func(self.kernel(X_fft[self.srce]))
			dXn[self.trget] = dXn[self.trget] + 1

		return dX, dXn

