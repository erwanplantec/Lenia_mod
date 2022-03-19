import torch
import numpy as np
from torch.optim import Adam
import matplotlib.pyplot as plt
from addict import Dict
from utils import generate_video
import json
import requests
import PIL.Image, PIL.ImageDraw
import io
import matplotlib.cm as cm

from systems import Lenia_C
from kernels import Kernel, Kernel_param_space, Kernel_wall
from interaction import Interaction
from channels import Channel
from growth_functions import Exponential_GF, GF_param_space, Wall_GF

C = 1
run_steps = 100

init_state = torch.zeros((1, 256, 256))
init = torch.rand((40, 40))
init_state[0, 108:148, 108:148] = init

config = Dict(
	device = 'cpu',
	SX = 256,
	SY = 256,
	R = 15, 
	T = 10.,
	kernel_params = Kernel_param_space,
	gf_params = GF_param_space,
	init = init_state
)


system = Lenia_C(config)

system.add_channel(Channel(config))

system.add_kernel(Interaction.build_random(0, 0, 15, Exponential_GF, 
	config.kernel_params, config.gf_params, config))

system.init_state()

epochs = 600
f_loss = torch.nn.functional.mse_loss

TARGET_EMOJI = "🦎"

def load_image(url, max_size=256):
	r = requests.get(url)
	img = PIL.Image.open(io.BytesIO(r.content))
	img.thumbnail((max_size, max_size), PIL.Image.ANTIALIAS)
	img = np.float32(img)/255.0
	# premultiply RGB by Alpha
	img[..., :3] *= img[..., 3:]
	return img

def load_emoji(emoji):
	code = hex(ord(emoji))[2:].lower()
	url = 'https://raw.githubusercontent.com/googlefonts/noto-emoji/main/png/128/emoji_u%s.png'%code
	print(url)
	return load_image(url)

def to_alpha(x):
	return np.clip(x[..., 3:4], 0.0, 1.0)
def to_rgb(x):
	# assume rgb premultiplied by alpha
	rgb, a = x[..., :3], to_alpha(x)
	return 1.0-a+rgb
def zoom(img, scale=4):
	img = np.repeat(img, scale, 0)
	img = np.repeat(img, scale, 1)
	return img


def train(epochs, system, init, verbose = True):
	
	opt_init = Adam([init])
	opt_sys = Adam(system.parameters())
	
	for epoch in range(epochs):
		
		system.init_state()
		system.run(run_steps)
		loss = f_loss(system.state, target)

		opt_init.zero_grad()
		opt_sys.zero_grad()
		loss.backward()
		opt_init.step()
		opt_sys.step()

		system.update()

		if verbose :
			print(f"Epoch {epoch + 1}/{epochs} : loss = {loss}")


img = load_emoji((TARGET_EMOJI))
gray_img = gray_target_img=np.dot(zoom(to_rgb(img),2), [0.2989, 0.5870, 0.1140])
target = torch.from_numpy(gray_img).unsqueeze(0).float()

print(system.state.shape)

train(20, system, init)



# wgf = Polynomial_GF(torch.tensor([.5]), torch.tensor([.2]), config)
# wgf.show()

