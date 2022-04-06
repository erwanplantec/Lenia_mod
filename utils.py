import torch
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation

def complex_mult_torch(X, Y):
    """ Computes the complex multiplication in Pytorch when the tensor last dimension is 2: 0 is the real component and 1 the imaginary one"""
    assert X.shape[-1] == 2 and Y.shape[-1] == 2, 'Last dimension must be 2'
    return torch.stack(
            (X[..., 0] * Y[..., 0] - X[..., 1] * Y[..., 1],
             X[..., 0] * Y[..., 1] + X[..., 1] * Y[..., 0]),
            dim = -1)


def roll_n(X, axis, n):
    """ Rolls a tensor with a shift n on the specified axis"""
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None)
                    for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None)
                    for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)


def gen_vid_mpl(imgs, save = False, filename = None):
    frames = [] # for storing the generated images
    fig = plt.figure()
    for i in range(imgs.shape[0]):
        frames.append([plt.imshow(imgs[i, 0, ...], animated=True)])

    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,
                                    repeat_delay=1000)
    if save : 
        ani.save(filename)
    plt.show()
  


