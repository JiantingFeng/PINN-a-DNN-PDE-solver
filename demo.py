import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.autograd import Variable
from torchsummary import summary
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import os
import argparse
import logging


# demo for solving du/dx - 2du/dt - u
# PINN implementation



# def get_logger(filename, verbosity=1, name=None):
#     level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
#     formatter = logging.Formatter(
#         "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
#     )
#     logger = logging.getLogger(name)
#     logger.setLevel(level_dict[verbosity])

#     fh = logging.FileHandler(filename, "w")
#     fh.setFormatter(formatter)
#     logger.addHandler(fh)

#     sh = logging.StreamHandler()
#     sh.setFormatter(formatter)
#     logger.addHandler(sh)

#     return logger

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden_layer1 = nn.Linear(2,5)
        self.hidden_layer2 = nn.Linear(5,5)
        self.hidden_layer3 = nn.Linear(5,5)
        self.hidden_layer4 = nn.Linear(5,5)
        self.hidden_layer5 = nn.Linear(5,5)
        self.output_layer = nn.Linear(5,1)

    def forward(self, x, t):
        # TODO: support for any dimension
        inputs = torch.cat([x,t],axis=1)
        layer1_out = torch.sigmoid(self.hidden_layer1(inputs))
        layer2_out = torch.sigmoid(self.hidden_layer2(layer1_out))
        layer3_out = torch.sigmoid(self.hidden_layer3(layer2_out))
        layer4_out = torch.sigmoid(self.hidden_layer4(layer3_out))
        layer5_out = torch.sigmoid(self.hidden_layer5(layer4_out))
        output = self.output_layer(layer5_out)
        return output

def f(x, t, net):
    u = net(x, t)
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    pde = u_x - 2 * u_t - u
    return pde



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='hyperparameter for the model.')
    parser.add_argument('epoch', type=int, default=2000)
    parser.add_argument('num_points', type=int, default=500)
    # parser.add_argument('logger_path', type=str,default='./results/demo/demo.log')
    # parser.add_argument('pretrained', type=bool, default=False)
    args = parser.parse_args()
    # logging.basicConfig(filename=args.logger_path, encoding='utf-8', level=logging.DEBUG)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Current device:", device)
    # Boundary conditions
    x_bc = np.random.uniform(low=0.0, high=2.0, size=(args.num_points, 1))
    t_bc = np.zeros((args.num_points, 1))
    u_bc = 6 * np.exp(-3 * x_bc)
    # logger = get_logger(args.logger_path)
    save_path = './save_model/{}.pth'.format('demo')
    # logger.info(str(args))

    print("--------------------Initializing--------------------")
    os.makedirs('./results/', exist_ok=True)
    os.makedirs('./save_model/', exist_ok=True)
    net = Net()
    net = net.to(device)
    mse_cost_function = torch.nn.MSELoss() # Mean squared error, L2 error
    optimizer = torch.optim.Adam(net.parameters())
    # logger.info(str(args))
    print("------------------Training Starts-------------------")
    for epoch in range(args.epoch):
        # Set grad to zero
        optimizer.zero_grad()

        # boundary points for training
        pt_x_bc = Variable(torch.from_numpy(x_bc).float(), requires_grad=False).to(device)
        pt_t_bc = Variable(torch.from_numpy(t_bc).float(), requires_grad=False).to(device)
        pt_u_bc = Variable(torch.from_numpy(u_bc).float(), requires_grad=False).to(device)

        net_bc_out = net(pt_x_bc, pt_t_bc)
        mse_u = mse_cost_function(net_bc_out, pt_u_bc)  # loss of boundary

        # all points for training
        x_collocation = np.random.uniform(low=0.0, high=2.0, size=(args.num_points, 1))
        t_collocation = np.random.uniform(low=0.0, high=1.0, size=(args.num_points, 1))
        all_zeros = np.zeros((args.num_points, 1))
        pt_x_collocation = Variable(torch.from_numpy(x_collocation).float(), requires_grad=True).to(device)
        pt_t_collocation = Variable(torch.from_numpy(t_collocation).float(), requires_grad=True).to(device)
        pt_all_zeros = Variable(torch.from_numpy(all_zeros).float(), requires_grad=False).to(device)
        # f output
        f_out = f(pt_x_collocation, pt_t_collocation, net)
        mse_f = mse_cost_function(f_out, pt_all_zeros)  # loss of pde

        loss = mse_u + mse_f

        # backward proporgation
        loss.backward()
        optimizer.step()

        loss_opt = -float('inf')
        # evaluation
        with torch.autograd.no_grad():
            # logger.info('Epoch:{}, loss:{:.4f}, acc:{:.4f}'.format(epoch, loss.data, 1-loss.data))
            if epoch % 100 == 0:
    	        print(epoch,"Traning Loss: ",loss.data)
    
    
    print("------------------Training Finish-------------------")
    # summary
    # logger.info(summary(net, [(1, 2), (1, 2)]))


    fig = plt.figure()
    ax = fig.gca(projection='3d')

    x = np.arange(0,2,0.02)
    t = np.arange(0,1,0.02)
    ms_x, ms_t = np.meshgrid(x, t)
    ## Just because meshgrid is used, we need to do the following adjustment
    x = np.ravel(ms_x).reshape(-1,1)
    t = np.ravel(ms_t).reshape(-1,1)

    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
    pt_t = Variable(torch.from_numpy(t).float(), requires_grad=True).to(device)
    pt_u = net(pt_x,pt_t)
    u=pt_u.data.cpu().numpy()
    ms_u = u.reshape(ms_x.shape)

    surf = ax.plot_surface(ms_x,ms_t,ms_u, cmap=cm.coolwarm,linewidth=0, antialiased=False)
                
                

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.savefig(os.path.join('results', 'demo.svg'), transparent=True, dpi=600, )
    print("DNN solution of PDE is saved in {}".format('./results/demo.svg'))

    torch.save(net.state_dict(), save_path)
    print("DNN is saved in {}".format(str(save_path)))
