import os
import torch
import numpy as np
import matplotlib.pyplot as plt


# TODO: Not finished yet
# 2d x-t graph
def plot_xt(x_range, t_range, x_step, t_step, net, u, save_path = "../save_path/"):
    ''' Params:
    x_range: range of x values for example [0, 1]
    t_range: range of t values for example [0, 1]
    x_step: step of x values
    t_step: step of t values
    net: neural network
    u: analytical solution
    '''
    os.makedirs(save_path, exist_ok=True)

    x = np.arange(x_range[0], x_range[1], x_step)
    t = np.arange(t_range[0], t_range[1], t_step)

    # meshgird graph
    ms_x, ms_t = np.meshgrid(x, t)
    # only torch.tensor can be calculated by neural network
    x = torch.tensor(np.ravel(ms_x).reshape(-1, 1))
    t = torch.tensor(np.ravel(ms_t).reshape(-1, 1))

    u_NN = net(x, t).numpy()
    ms_u_NN = u_NN.reshape(ms_x.shape)

    u_real = u(x, t).numpy()
    ms_u_real = u_real.reshape(ms_x.shape)

    ms_u_error = abs((ms_u_real-ms_u_NN)/ms_u_real)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    