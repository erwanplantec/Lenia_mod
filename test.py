import torch
import numpy as np
from torch.optim import Adam
import matplotlib.pyplot as plt
from addict import Dict
from utils import generate_video, gen_vid_mpl
import json
import requests
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

C = 4
nb_k = 5

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

system = Lenia_C(config)

for c in range(C):
  system.add_channel(Channel(config))

# Crea -> crea kernels
system.add_kernel(
    Interaction.build_random(0, 0, nb_k, Exponential_GF, 
      config.kernel_params, config.gf_params, config)
)

# Wall -> Crea kernel
system.add_kernel(
    Wall_Interaction(1, 0, config)
)

# # Cue -> Crea
# system.add_kernel(
#     Interaction.build_random(2, 0, 1, Exponential_GF, 
#       config.kernel_params, config.gf_params, config)
# )
# system.add_kernel(
#     Interaction.build_random(3, 0, 1, Exponential_GF, 
#       config.kernel_params, config.gf_params, config)
# )

state = torch.zeros_like(system.state)

#==================Walls=================
walls = torch.zeros((config.SX, config.SY))
wall_pts = [
            ((93, 25), (163, 30)),
            ((163, 25), (168, 160)),
            ((163, 160), (250, 165)),
            ((250, 160), (255, 230)),
            ((1, 230), (255, 235)),
            ((1, 160), (6, 230)),
            ((1, 160), (93, 165)),
            ((93, 25), (98, 165))
]

for (x1, y1), (x2, y2) in wall_pts:
  walls[y1:y2, x1:x2] = 1.

#=================Crea======================
crea = torch.zeros_like(walls)
crea[35:85, 103:153] = torch.rand((50, 50))

#==================Cue======================

cue = torch.zeros_like(crea)
cue[130:150, 118:138] = 1.


state[0] = crea
state[1] = walls
state[2] = cue

system.state = state

imgs = system.run(150, True).detach().numpy()

gen_vid_mpl(imgs)



