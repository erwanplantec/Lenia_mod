import torch
import numpy as np
from torch.optim import Adam
import matplotlib.pyplot as plt
from addict import Dict
from utils import gen_vid_mpl

from systems import Lenia_C
from kernels import Kernel, Kernel_param_space, Kernel_param_space_inh, Kernel_wall
from interaction import Interaction, Wall_Interaction
from channels import Channel, Asymptotic_Channel
from growth_functions import Exponential_GF, GF_param_space, GF_param_space_inh, Wall_GF

from functools import partial


params = torch.load("init.pickle", map_location = torch.device("cpu"))

config = Dict(
  device = 'cpu',
  SX = 256,
  SY = 256,
  R = 10, 
  T = 100,
  kernel_params = Kernel_param_space,
  gf_params = GF_param_space,
  init = torch.rand((1, 256, 256))
)

system = Lenia_C.from_params(params, config)
state = torch.zeros(system.state.shape)
state[0, 98 : 158, 98 : 158] = torch.rand((60, 60))
system.state = state

traj1 = system.run(100, True)

print(traj1.max())

gen_vid_mpl(traj1)
