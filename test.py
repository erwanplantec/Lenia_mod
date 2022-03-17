import torch
import matplotlib.pyplot as plt
from addict import Dict

from systems import Lenia_C
from kernels import Kernel, Kernel_param_space
from interaction import Interaction
from channels import Channel
from growth_functions import Exponential_GF, GF_param_space


config = Dict(
	device = 'cpu',
	SX = 256,
	SY = 256,
	R = 15, 
	T = 100
)

system = Lenia_C(config)

system.add_channel(Channel(config))

inte = Interaction.build_random(0, 0, 1, Exponential_GF,
	Kernel_param_space(1), GF_param_space(1), config)


system.add_kernel(inte)

system.state = torch.rand((1, 256, 256))


fig, ax = plt.subplots(1, 2)
ax[0].imshow(system.state[0].detach().cpu().numpy())

for _ in range(100):
	system.step()

ax[1].imshow(system.state[0].detach().cpu().numpy())

plt.show()
