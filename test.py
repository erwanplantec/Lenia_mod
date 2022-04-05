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
from interaction import Interaction
from channels import Channel
from growth_functions import Exponential_GF, GF_param_space, GF_param_space_inh, Wall_GF

from functools import partial

C = 1
run_steps = 100
seeds = 10

# init = torch.rand((40, 40))
# init_state[0, 108:148, 108:148] = init

s_ranges = [.18, .25, .35, .45, .55]
results = np.zeros((len(s_ranges), seeds))


for i, sp in enumerate(s_ranges):

	for seed in range(seeds):

		print(sp, seed)

		config = Dict(
			device = 'cpu',
			SX = 256,
			SY = 256,
			R = 15, 
			T = 10.,
			kernel_params = Kernel_param_space_inh,
			gf_params = partial(GF_param_space_inh, s_range = (.001, sp)),
			init = torch.rand((1, 256, 256))
		)

		system = Lenia_C(config)

		for c in range(C):
			system.add_channel(Channel(config))

		system.add_kernel(Interaction.build_random(0, 0, 10, Exponential_GF, 
			config.kernel_params, config.gf_params, config))

		system.init_state()



		N = 200

		with torch.no_grad():
			state = system.run(N, False)	

		results[i, seed] = state[0, ...].detach().numpy().sum()
	# imgs = traj.detach().numpy()


plt.plot(s_ranges, results.mean(axis = 1))
plt.xlabel("sigma +")
plt.ylabel("|X|")
plt.show()

# gen_vid_mpl(imgs)



