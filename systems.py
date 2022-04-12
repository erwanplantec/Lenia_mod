import torch
import torch.nn as nn
import numpy as np

from channels import Channel
from kernels import Kernel
from growth_functions import Exponential_GF
from interaction import Interaction

class Lenia_C(nn.Module):
	#------------------------------------------------------
	def __init__(self, config):

		super().__init__()

		self.channels = []
		self.kernels = nn.ModuleList([])

		self.config = config

		self.to(self.device)
	#------------------------------------------------------
	#-----------------------PROPERTIES---------------------
	#------------------------------------------------------
	def __getitem__(self, i):
		return self.channels[i].state
	#------------------------------------------------------
	@property
	def state(self):
		return torch.cat([c.state.unsqueeze(0) for c in self.channels])
	@state.setter
	def state(self, state):
		assert state.shape[0] == self.C
		for i, c in enumerate(self.channels):
			c.state = state[i, ...]
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
	@property
	def C(self):
		return len(self.channels)
	#------------------------------------------------------
	#------------------------BUILDING----------------------
	#------------------------------------------------------
	def add_kernel(self, kernel):
		self.kernels.append(kernel)
	#------------------------------------------------------
	def add_channel(self, channel):
		self.channels.append(channel)
	#------------------------------------------------------
	#-------------------------RUNNING----------------------
	#------------------------------------------------------
	def step(self, norm = True):
		X_fft = [torch.rfft(self.state[i,:,:], signal_ndim=2, onesided=False) 
			for i in range(self.C)]
		dX = torch.zeros((self.C, self.SX, self.SY)).to(self.device)
		dXn = torch.zeros((self.C)).to(self.device)
		for kernel in self.kernels:
			field, field_norm = kernel(X_fft)
			dX = dX + field
			dXn = dXn + field_norm

		for i, c in enumerate(self.channels):
			c.update((dX[i] / dXn[i]) if norm else dX[i])
	#------------------------------------------------------
	def run(self, T, record = True, norm = True):
		if record :
			orb = torch.zeros((T, self.C, self.SX, self.SY)).to(self.device)
		for t in range(T):
			self.step(norm)
			if record:
				orb[t] = self.state.detach()
		return orb.cpu() if record else self.state.cpu()
	#------------------------------------------------------
	def init_state(self, state = None):
		if state is None :
			if hasattr(self.config, "init"):
				state = self.config.init
			else :
				raise ValueError("No init in config")
		self.state = state
	#------------------------------------------------------
	def update(self):
		for kernel in self.kernels:
			kernel.compute_kernel()
	#------------------------------------------------------
	@staticmethod
	def from_matrix(matrix, config):
		sys = Lenia_C(config)
		C = matrix.shape[0]
		[sys.add_channel(Channel(config)) for _ in range(C)]
		for s in range(C):
			for t in range(C):
				if matrix[s, t]:
					sys.add_kernel(
						Interaction.build_random(s, t, 
							int(matrix[s, t]), Exponential_GF, 
							config.kernel_params, config.gf_params, 
							config)
					)
		return sys
	#------------------------------------------------------
	@staticmethod
	def from_file(file, config):
		
		params = torch.load(file, map_location = torch.device(config.device))

		return Lenia_C.from_params(params, config)
	#------------------------------------------------------
	@staticmethod
	def from_params(params, config, channel = Channel):

		system = Lenia_C(config)

		config.T = params["T"]
		config.R = params["R"]

		C = max(params["c0"].max(), params["c1"].max()) + 1

		for c in range(C):
			system.add_channel(channel(config))

		interactions = {}

		for i, (s, t) in enumerate(zip(params["c0"], params["c1"])):

			interactions[(s.item(), t.item())] = interactions.get((s.item(), t.item()), 
				Interaction(s.item(), t.item(), [], [], config))
			
			k = Kernel(params["rk"][i], params["b"][i], params["w"][i],
				params["r"][i], params["h"][i], config)				
			g = Exponential_GF(params["m"][i], params["s"][i], config)

			interactions[(s.item(), t.item())].add(k, g)

		for i in interactions.values():
			
			system.add_kernel(i)

		return system









#=====================================================================
#==============================NEAT LENIA=============================
#=====================================================================

class NEAT_Lenia(Lenia_C):
	#------------------------------------------------------
	def __init__(self, config, neat_config):

		super().__init__(config)
		
		self.neat_config = neat_config

		self.add_channel(Channel(self.config))
		self.add_kernel(Interaction.build_random(0, 0, 1, Exponential_GF, 
			self.config.kernel_params, self.config.gf_params))
	#------------------------------------------------------
	def mutate(self):
		pass
	#------------------------------------------------------
	def cross(self, system):
		pass
	#------------------------------------------------------
	def mut_add_channel(self):
		pass
	def mut_add_kernel(self):
		pass


