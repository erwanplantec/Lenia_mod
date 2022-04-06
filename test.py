import torch
import numpy as np
from torch.optim import Adam
import matplotlib.pyplot as plt
from addict import Dict
from utils import gen_vid_mpl
import json
import PIL.Image, PIL.ImageDraw
import io
import matplotlib.cm as cm
import matplotlib.animation as animation

from systems import Lenia_C
from kernels import Kernel, Kernel_param_space, Kernel_param_space_inh, Kernel_wall
from interaction import Interaction, Wall_Interaction
from channels import Channel
from growth_functions import Exponential_GF, GF_param_space, GF_param_space_inh, Wall_GF

from functools import partial

config = Dict(
      device = 'cpu',
      SX = 256,
      SY = 256,
      R = 15, 
      T = 10.,
      kernel_params = Kernel_param_space,
      gf_params = GF_param_space,
      init = torch.rand((1, 256, 256))
)

Lenia_C.from_file("init.pickle", config)
