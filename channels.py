import torch
import torch.nn as nn
import numpy as np

class Channel(nn.Module):
	#------------------------------------------------------
	def __init__(self):

		super().__init__()

		self.state = None
	#------------------------------------------------------
	def update(self, dX):

		self.state = torch.clip(self.state + dt * DX, 0, 1)