import torch
import numpy as np
from torch.optim import Adam
import matplotlib.pyplot as plt
from addict import Dict
from utils import gen_vid_mpl

from systems import Lenia_C
from kernels import Kernel, Kernel_param_space, Kernel_param_space_inh, Kernel_wall
from interaction import Interaction, Wall_Interaction
from channels import Channel
from growth_functions import Exponential_GF, GF_param_space, GF_param_space_inh, Wall_GF

from functools import partial


params = torch.load("crea.pickle", map_location = torch.device("cpu"))
init = params["policy_parameters"]["initialization"]["init"]
rule = params["policy_parameters"]["update_rule"]

config = Dict(
  device = 'cuda',
  SX = 256,
  SY = 256,
  R = rule["R"], 
  T = rule["T"],
  kernel_params = Kernel_param_space,
  gf_params = GF_param_space,
  init = torch.rand((1, 256, 256))
)

system = Lenia_C.from_params(rule, config)