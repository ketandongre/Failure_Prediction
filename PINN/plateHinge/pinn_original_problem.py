### Initialize minLoss wrt loaded w&b
import numpy as np
import time
import sys
import os
from pyDOE import lhs
import matplotlib
import platform
if platform.system()=='Linux':
    matplotlib.use('Agg')
if platform.system()=='Windows':
    from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import shutil
import pickle
import math
import scipy.io
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


#### Creating distance Modelwith Pytorch
# torch.cuda.empty_cache()

class ANN_DistModel(nn.Module):
    def __init__(self, N_INPUT=2, N_OUTPUT=5, N_HIDDEN=20, N_LAYERS=4):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
        self.apply(self._init_weights)
    def _init_weights(self, module):
      if isinstance(module, nn.Linear):
          nn.init.xavier_normal_(module.weight)
          if module.bias is not None:
              module.bias.data.zero_()
    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x


#### Creating part Modelwith Pytorch

class ANN_PartModel(nn.Module):
    def __init__(self, N_INPUT=2, N_OUTPUT=5, N_HIDDEN=20, N_LAYERS=4):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
        self.apply(self._init_weights)
    def _init_weights(self, module):
      if isinstance(module, nn.Linear):
          nn.init.xavier_normal_(module.weight)
          if module.bias is not None:
              module.bias.data.zero_()
        
    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x


#### Creating Modelwith Pytorch

class ANN_UvModel(nn.Module):
    def __init__(self, N_INPUT=2, N_OUTPUT=5, N_HIDDEN=70, N_LAYERS=8):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
        self.apply(self._init_weights)
    def _init_weights(self, module):
      if isinstance(module, nn.Linear):
          nn.init.xavier_normal_(module.weight)
          if module.bias is not None:
              module.bias.data.zero_()
                
    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x

def GenDistPt(xmin, xmax, ymin, ymax, xc, yc, r, num_surf_pt, num):
    # num: number per edge
    # num_t: number time step
    x = np.linspace(xmin, xmax, num=num)
    y = np.linspace(ymin, ymax, num=num)
    x, y = np.meshgrid(x, y)
    # Delete point in hole
    # dst = ((x - xc) ** 2 + (y - yc) ** 2) ** 0.5
    # x = x[dst >= r]
    # y = y[dst >= r]
    x = x.flatten()[:, None]
    y = y.flatten()[:, None]
    # Refinement point near hole surface
    # theta = np.linspace(0.0, np.pi / 2.0, num_surf_pt)
    # x_surf = np.multiply(r, np.cos(theta)) + xc
    # y_surf = np.multiply(r, np.sin(theta)) + yc
    # x_surf = x_surf.flatten()[:, None]
    # y_surf = y_surf.flatten()[:, None]
    # x = np.concatenate((x, x_surf), 0)
    # y = np.concatenate((y, y_surf), 0)
    return x,y

def GenDist(XY_dist):
    dist_u = np.zeros_like(XY_dist[:, 0:1])
    dist_v = np.zeros_like(XY_dist[:, 0:1])
    dist_s11 = np.zeros_like(XY_dist[:, 0:1])
    dist_s22 = np.zeros_like(XY_dist[:, 0:1])
    dist_s12 = np.zeros_like(XY_dist[:, 0:1])
    for i in range(len(XY_dist)):
        dist_u[i, 0] = XY_dist[i][0]  # min(t, x-(-0.5))
        dist_v[i, 0] =  XY_dist[i][0]  # min(t, sqrt((x+0.5)^2+(y+0.5)^2))
        dist_s11[i, 0] = 0.5 - XY_dist[i][0]
        dist_s22[i, 0] = min(0.5 - XY_dist[i][1],XY_dist[i][1])
        dist_s12[i, 0] = min(XY_dist[i][1], 0.5 - XY_dist[i][1], 0.5 - XY_dist[i][0])
    DIST = np.concatenate(( dist_u, dist_v, dist_s11, dist_s22, dist_s12), 1)
    return XY_dist , DIST

# def DelHolePT(XY_c, xc=0, yc=0, r=0.1):
#     # Delete points within hole
#     dst = np.array([((xy[0] - xc) ** 2 + (xy[1] - yc) ** 2) ** 0.5 for xy in XY_c])
#     return XY_c[dst > r, :]

# def GenHoleSurfPT(xc, yc, r, N_PT):
#     # Generate
#     theta = np.linspace(0.0, np.pi / 2.0, N_PT)
#     xx = np.multiply(r, np.cos(theta)) + xc
#     yy = np.multiply(r, np.sin(theta)) + yc
#     xx = xx.flatten()[:, None]
#     yy = yy.flatten()[:, None]
#     return xx, yy
 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Current cuda device: ',torch.cuda.get_device_name(0))


E = 20.0
mu = 0.25
rho = 1.0
# hole_r = 0.1
#### Note: The detailed description for this case can be found in paper:
#### Physics informed deep learning for computational elastodynamicswithout labeled data.
#### https://arxiv.org/abs/2006.08472
#### But network configuration might be slightly different from what is described in paper.
PI = math.pi

# Domain bounds for x, y and t
lb = np.array([0, 0])
ub = np.array([0.5, 0.5])

# Network configuration
uv_layers   = [2] + 8 * [70] + [5]
dist_layers = [2] + 4 * [20] + [5]
part_layers = [2] + 4 * [20] + [5]

# Generate distance function for spatio-temporal space
x_dist, y_dist = GenDistPt(xmin=0, xmax=0.5, ymin=0, ymax=0.5, xc=0, yc=0, r=0.1,
                                    num_surf_pt=40, num=100)
XY_dist = np.concatenate((x_dist, y_dist), 1)
XY_dist,DIST = GenDist(XY_dist)

# Collocation point for equation residual
XY_c = lb + (ub - lb) * lhs(2, 30000)
# XY_c_ref = lb + np.array([0.15, 0.15]) * lhs(2, 30000)  # Refinement for stress concentration
# XY_c = np.concatenate((XY_c, XY_c_ref), 0)
# XY_c = DelHolePT(XY_c, xc=0, yc=0, r=0.1)

# xx, yy = GenHoleSurfPT(xc=0, yc=0, r=0.1, N_PT=83)

# HOLE = np.concatenate((xx, yy), 1)

LW = np.array([0.0, 0.0]) + np.array([0.5, 0.0]) * lhs(2, 2000)
UP = np.array([0.0, 0.5]) + np.array([0.5, 0.0]) * lhs(2, 2000)
LF = np.array([0.0, 0.0]) + np.array([0.0, 0.5]) * lhs(2, 2000)
RT = np.array([0.5, 0.0]) + np.array([0.0, 0.5]) * lhs(2, 5000)

# t_RT = RT[:, 2:3]
# period = 5  # two period in 10s
# s11_RT = 0.5 * np.sin((2 * PI / period) * t_RT + 3 * PI / 2) + 0.5
s11_RT=np.ones(RT[:,0:1].shape)
RT = np.concatenate((RT, s11_RT), 1)

# Add some boundary points into the collocation point set
XY_c = np.concatenate((XY_c, LF[::5, :], RT[::5, 0:2], UP[::5, :], LW[::5, :]), 0)

XY_dist=torch.FloatTensor(XY_dist).requires_grad_(True)
DIST=torch.FloatTensor(DIST).requires_grad_(True)
# HOLE=torch.FloatTensor(HOLE).requires_grad_(True)
LW=torch.FloatTensor(LW).requires_grad_(True)
LF=torch.FloatTensor(LF).requires_grad_(True)
UP=torch.FloatTensor(UP).requires_grad_(True)
RT=torch.FloatTensor(RT).requires_grad_(True)
XY=torch.FloatTensor(XY_c).requires_grad_(True)


# train standard neural network to fit distance training data
torch.manual_seed(123)
model_dist = ANN_DistModel()
model_dist.to(device)
if os.path.isfile('./distance.pth'):
    print("Loading...")
    model_dist.load_state_dict(torch.load('./distance.pth'))
optimizer_dist = torch.optim.Adam(model_dist.parameters(),lr=1e-3)
# minLoss=sys.maxsize
minLoss=1000*torch.mean((model_dist(XY_dist)-DIST)**2)
bestModelAt=0
torch.save(model_dist.state_dict(), "./distance.pth")
for i in range(30000):
    optimizer_dist.zero_grad()
    XY_dist=XY_dist.to(device)
    DIST=DIST.to(device)
    # compute the "data loss"
    yh = model_dist(XY_dist)

    loss1 = torch.mean((yh-DIST)**2)# use mean squared error
  
    
    # backpropagate joint loss
    # loss=loss1
    loss = 1000*(loss1)# add two loss terms together
    if i%100==0: 
      print("Epoch number: {} and the loss : {}".format(i,loss.item()))
      if loss < minLoss:
          bestModelAt=i
          minLoss=loss
          torch.save(model_dist.state_dict(), "./distance.pth")

    loss.backward()
    optimizer_dist.step()
    
    
    # # plot the result as training progresses
    # if (i+1) % 150 == 0: 
        
    #     yh = model_dist(x).detach()
    #     xp = x_physics.detach()
        
    #     plot_result(x,y,x_data,y_data,yh,xp)
        
    #     # file = "plots/pinn_%.8i.png"%(i+1)
    #     # plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
    #     # files.append(file)
        
    #     if (i+1) % 6000 == 0: plt.show()
    #     else: plt.close("all")


# train standard neural network to fit distance training data
print(f"Best model at {bestModelAt} with loss: {minLoss}")


torch.manual_seed(123)
model_part = ANN_PartModel()
model_part.to(device)
if os.path.isfile('./part.pth'):
    print("Loading...")
    model_part.load_state_dict(torch.load('./part.pth'))
optimizer_part = torch.optim.Adam(model_part.parameters(),lr=1e-3)
# minLoss=sys.maxsize
minLoss=  torch.mean((model_part(LF)[:,0])**2)+torch.mean((model_part(LF)[:,1])**2)\
         +torch.mean((model_part(UP)[:,3])**2)+torch.mean((model_part(UP)[:,4])**2)\
         +torch.mean((model_part(LW)[:,3])**2)+torch.mean((model_part(LW)[:,4])**2)\
         +torch.mean((model_part(RT[:,0:2])[:,4])**2)\
         +torch.mean((model_part(RT[:,0:2])[:,2]-RT[:,2])**2)
minLoss=1000*minLoss
bestModelAt=0
torch.save(model_part.state_dict(), "./part.pth")
for i in range(10000):
    optimizer_part.zero_grad()

    #LF
    LF=LF.to(device)
    yh_lf = model_part(LF)
    loss1=torch.mean((yh_lf[:,0])**2)+torch.mean((yh_lf[:,1])**2)
    #+torch.mean((yh_lf[:,4])**2)


    #UP
    UP=UP.to(device)
    yh_up = model_part(UP)
    loss1=loss1+torch.mean((yh_up[:,3])**2)+torch.mean((yh_up[:,4])**2);
    #LW
    LW=LW.to(device);
    yh_lw = model_part(LW)
    loss1=loss1+torch.mean((yh_lw[:,3])**2)+torch.mean((yh_lw[:,4])**2);

    #RT
    RT=RT.to(device);
    yh_rt = model_part(RT[:,0:2])
    loss1=loss1+torch.mean((yh_rt[:,4])**2);
    loss1=loss1+torch.mean((yh_rt[:,2]-RT[:,2])**2);
    
    # backpropagate joint loss
    # loss=loss1
    loss = 1000*(loss1)# add two loss terms together
    if i%100==0: 
        print("Epoch number: {} and the loss : {}".format(i,loss.item()))
        if loss < minLoss:
            bestModelAt=i
            minLoss=loss
            torch.save(model_part.state_dict(), "./part.pth")

    loss.backward()
    optimizer_part.step()
    
    
    # # plot the result as training progresses
    # if (i+1) % 150 == 0: 
        
    #     yh = model_dist(x).detach()
    #     xp = x_physics.detach()
        
    #     plot_result(x,y,x_data,y_data,yh,xp)
        
    #     # file = "plots/pinn_%.8i.png"%(i+1)
    #     # plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
    #     # files.append(file)
        
    #     if (i+1) % 6000 == 0: plt.show()
    #     else: plt.close("all")


print(f"Best model at {bestModelAt} with loss: {minLoss}")
# train standard neural network to fit distance training data

torch.manual_seed(123)
model_uv = ANN_UvModel()
model_uv.to(device)
if os.path.isfile('./uv.pth'):
    print("Loading...")
    model_uv.load_state_dict(torch.load('./uv.pth'))
    
####INITIAL LOSS####
yh = model_uv(XY)
yh=yh*model_dist(XY)+model_part(XY)
e11=torch.autograd.grad(yh[:,0], XY,torch.ones_like(yh[:,0]), create_graph=True)[0][:,0]# computes dy/dx
e12=torch.autograd.grad(yh[:,0], XY,torch.ones_like(yh[:,0]), create_graph=True)[0][:,1]
e22=torch.autograd.grad(yh[:,1], XY,torch.ones_like(yh[:,1]), create_graph=True)[0][:,1]# computes dy/dx
e12=e12+torch.autograd.grad(yh[:,1], XY,torch.ones_like(yh[:,1]), create_graph=True)[0][:,0]
sp11 = (E * e11)/ (1 - mu * mu) + (E * mu * e22) / (1 - mu * mu)
sp22 = (E * mu* e11) / (1 - mu * mu)  + (E * e22) / (1 - mu * mu) 
sp12 = (E* e12) / (2 * (1 + mu)) 
f_s11 = yh[:,2] - sp11
f_s12 = yh[:,4] - sp12
f_s22 = yh[:,3] - sp22
s11_1=torch.autograd.grad(yh[:,2], XY,torch.ones_like(yh[:,2]), create_graph=True)[0][:,0]
s12_2=torch.autograd.grad(yh[:,4], XY,torch.ones_like(yh[:,4]), create_graph=True)[0][:,1]
s22_2=torch.autograd.grad(yh[:,3], XY,torch.ones_like(yh[:,3]), create_graph=True)[0][:,1]
s12_1=torch.autograd.grad(yh[:,4], XY,torch.ones_like(yh[:,4]), create_graph=True)[0][:,0]
f_u = s11_1 + s12_2 
f_v = s22_2 + s12_1 
minLoss=torch.mean(f_s11**2)+torch.mean(f_s22**2)+torch.mean(f_s12**2)+torch.mean(f_u**2)+torch.mean(f_v**2)
########

bestModelAt=0
torch.save(model_uv.state_dict(), "./uv.pth")
optimizer_uv = torch.optim.Adam(model_uv.parameters(),lr=5e-4)
for i in range(5000):
    optimizer_uv.zero_grad()
    
    # compute the "data loss"
    XY=XY.to(device)
    yh = model_uv(XY)
    yh=yh*model_dist(XY)+model_part(XY)
    # for i in range(5):
    #   yh[:,i]=yh[:,i]*model_dist(XYT)[:,i]+model_part(XYT)[:,i]
    e11=torch.autograd.grad(yh[:,0], XY,torch.ones_like(yh[:,0]), create_graph=True)[0][:,0]# computes dy/dx
    e12=torch.autograd.grad(yh[:,0], XY,torch.ones_like(yh[:,0]), create_graph=True)[0][:,1]
    e22=torch.autograd.grad(yh[:,1], XY,torch.ones_like(yh[:,1]), create_graph=True)[0][:,1]# computes dy/dx
    e12=e12+torch.autograd.grad(yh[:,1], XY,torch.ones_like(yh[:,1]), create_graph=True)[0][:,0]
    sp11 = (E * e11)/ (1 - mu * mu) + (E * mu * e22) / (1 - mu * mu)
    sp22 = (E * mu* e11) / (1 - mu * mu)  + (E * e22) / (1 - mu * mu) 
    sp12 = (E* e12) / (2 * (1 + mu)) 

    f_s11 = yh[:,2] - sp11
    f_s12 = yh[:,4] - sp12
    f_s22 = yh[:,3] - sp22

    s11_1=torch.autograd.grad(yh[:,2], XY,torch.ones_like(yh[:,2]), create_graph=True)[0][:,0]
    s12_2=torch.autograd.grad(yh[:,4], XY,torch.ones_like(yh[:,4]), create_graph=True)[0][:,1]

    s22_2=torch.autograd.grad(yh[:,3], XY,torch.ones_like(yh[:,3]), create_graph=True)[0][:,1]
    s12_1=torch.autograd.grad(yh[:,4], XY,torch.ones_like(yh[:,4]), create_graph=True)[0][:,0]



    f_u = s11_1 + s12_2 
    f_v = s22_2 + s12_1 


    loss1 = torch.mean(f_s11**2)+torch.mean(f_s22**2)+torch.mean(f_s12**2)+torch.mean(f_u**2)+torch.mean(f_v**2)# use mean squared error
    
    ## For hole 

    # HOLE=HOLE.to(device)
    # r = hole_r
    # nx = -HOLE[:,0] / r
    # ny = -HOLE[:,1] / r
    # yh_hole = model_uv(HOLE)
    # yh_hole=yh_hole*model_dist(HOLE)+model_part(HOLE)
    # tx=torch.mul(yh_hole[:,2],nx)+torch.mul(yh_hole[:,4],ny)
    # ty=torch.mul(yh_hole[:,4],nx)+torch.mul(yh_hole[:,3],ny)
    # # tx=yh[:,2]*nx+yh[:,4]*ny
    # # ty=yh[:,4]*nx+yh[:,3]*ny
    # loss2=torch.mean(tx**2)+torch.mean(ty**2)
    
    # backpropagate joint loss
    # loss=loss1
    loss2=0
    loss = loss1 + loss2# add two loss terms together 
    loss=10*loss
    if i%100==0:
        print("Epoch number: {} and the loss : {}".format(i,loss.item()))
        if loss < minLoss:
            bestModelAt=i
            minLoss=loss
            torch.save(model_uv.state_dict(), "./uv.pth")
    loss.backward()
    optimizer_uv.step()
    
    
    # # plot the result as training progresses
    # if (i+1) % 150 == 0: 
        
    #     yh = model_dist(x).detach()
    #     xp = x_physics.detach()
        
    #     plot_result(x,y,x_data,y_data,yh,xp)
        
    #     # file = "plots/pinn_%.8i.png"%(i+1)
    #     # plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
    #     # files.append(file)
        
    #     if (i+1) % 6000 == 0: plt.show()
    #     else: plt.close("all")
# torch.save(model_uv.state_dict(), "./uv.pth")
print(f"Best model at {bestModelAt} with loss: {minLoss}")


def postProcessDef(xmin, xmax, ymin, ymax, field, s=5, num=0, scale=1):
    ''' Plot deformed plate (set scale=0 want to plot undeformed contours)
    '''
    # [x_pred, y_pred, u_pred, v_pred, s11_pred, s22_pred, s12_pred] = field
    x_pred = field[:, 0]
    y_pred = field[:, 1]
    u_pred = field[:, 2]
    v_pred = field[:, 3]
    s11_pred = field[:, 4]
    s22_pred = field[:, 5]
    s12_pred = field[:, 6]
    print(v_pred)
    #
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(9, 9))
    fig.subplots_adjust(hspace=0.2, wspace=0.3)
    cf = ax[0].scatter(x_pred + u_pred * scale, y_pred + v_pred * scale, c=u_pred, alpha=0.7, edgecolors='none',
                       cmap='rainbow', marker='o', s=s)
    ax[0].axis('square')
    for key, spine in ax[0].spines.items():
        if key == 'right' or key == 'top' or key == 'left' or key == 'bottom':
            spine.set_visible(False)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_xlim([xmin, xmax])
    ax[0].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[0].set_title(r'$u$-PINN', fontsize=16)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[0])
    cbar.ax.tick_params(labelsize=14)
    #
    cf = ax[1].scatter(x_pred + u_pred * scale, y_pred + v_pred * scale, c=v_pred, alpha=0.7, edgecolors='none',
                       cmap='rainbow', marker='o', s=s)
    for key, spine in ax[1].spines.items():
        if key in ['right', 'top', 'left', 'bottom']:
            spine.set_visible(False)
    ax[1].axis('square')
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_xlim([xmin, xmax])
    ax[1].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[1].set_title(r'$v$-PINN', fontsize=16)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[1])
    cbar.ax.tick_params(labelsize=14)
    #
    plt.savefig('./output/uv_comparison' + str(num) + '.png', dpi=200)
    plt.close('all')
    #
    # Plot predicted stress
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 9))
    fig.subplots_adjust(hspace=0.15, wspace=0.3)
    #
    cf = ax[0].scatter(x_pred + u_pred * scale, y_pred + v_pred * scale, c=s11_pred, alpha=0.7, edgecolors='none',
                       marker='s', cmap='rainbow', s=s)
    ax[0].axis('square')
    for key, spine in ax[0].spines.items():
        if key in ['right', 'top', 'left', 'bottom']:
            spine.set_visible(False)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_xlim([xmin, xmax])
    ax[0].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[0].set_title(r'$\sigma_{11}$-PINN', fontsize=16)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[0])
    cbar.ax.tick_params(labelsize=14)

    #
    cf = ax[1].scatter(x_pred + u_pred * scale, y_pred + v_pred * scale, c=s22_pred, alpha=0.7, edgecolors='none',
                       marker='s', cmap='rainbow', s=s)
    ax[1].axis('square')
    for key, spine in ax[1].spines.items():
        if key in ['right', 'top', 'left', 'bottom']:
            spine.set_visible(False)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_xlim([xmin, xmax])
    ax[1].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[1].set_title(r'$\sigma_{22}$-PINN', fontsize=16)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[1])
    cbar.ax.tick_params(labelsize=14)
    #
    cf = ax[2].scatter(x_pred + u_pred * scale, y_pred + v_pred * scale, c=s12_pred, alpha=0.7, edgecolors='none',
                       marker='s', cmap='rainbow', s=s)
    ax[2].axis('square')
    for key, spine in ax[2].spines.items():
        if key in ['right', 'top', 'left', 'bottom']:
            spine.set_visible(False)
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    ax[2].set_xlim([xmin, xmax])
    ax[2].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[2].set_title(r'$\sigma_{12}$-PINN', fontsize=16)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[2])
    cbar.ax.tick_params(labelsize=14)
    #
    plt.savefig('./output/stress_comparison' + str(num) + '.png', dpi=200)
    plt.close('all')

def newPostProcess(field, s11_pred, s22_pred, s12_pred, scale=1, s=5, xmin=0.0 , xmax=0.5 , ymin=0.0 , ymax=0.5):
    x_pred=field[:,0:1]
    y_pred=field[:,1:2]
    u_pred=field[:,2:3]
    v_pred=field[:,3:4]
    # s11_pred=field[:,4:5]
    # s22_pred=field[:,5:6]
    # s12_pred=field[:,6:7]
    # v_pred=abs(v_pred)
    s11_pred=s11_pred.cpu().detach().numpy()
    s22_pred=s22_pred.cpu().detach().numpy()
    s12_pred=s12_pred.cpu().detach().numpy()

    print(x_pred.shape, y_pred.shape,  u_pred.shape, s11_pred.shape)
    # yh=torch.tensor(np.concatenate((u_pred, v_pred),1)).requires_grad_(True)
    # XY=torch.tensor(np.concatenate((x_pred, y_pred),1)).requires_grad_(True)
    # e11=torch.autograd.grad(yh[:,0], XY,torch.ones_like(yh[:,0]), create_graph=True)[0][:,0]# computes dy/dx
    # e12=torch.autograd.grad(yh[:,0], XY,torch.ones_like(yh[:,0]), create_graph=True)[0][:,1]
    # e22=torch.autograd.grad(yh[:,1], XY,torch.ones_like(yh[:,1]), create_graph=True)[0][:,1]# computes dy/dx
    # e12=e12+torch.autograd.grad(yh[:,1], XY,torch.ones_like(yh[:,1]), create_graph=True)[0][:,0]
    # s11_pred = (E * e11)/ (1 - mu * mu) + (E * mu * e22) / (1 - mu * mu)
    # s22_pred = (E * mu* e11) / (1 - mu * mu)  + (E * e22) / (1 - mu * mu) 
    # s12_pred = (E* e12) / (2 * (1 + mu)) 

        # Plot predicted stress
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 9))
    fig.subplots_adjust(hspace=0.15, wspace=0.3)
    #
    cf = ax[0, 0].scatter(x_pred + u_pred * scale, y_pred + v_pred * scale, c=s11_pred, alpha=0.7, edgecolors='none',
                          marker='s', cmap='rainbow', s=s)
    ax[0, 0].axis('square')
    for key, spine in ax[0, 0].spines.items():
        if key in ['right', 'top', 'left', 'bottom']:
            spine.set_visible(False)
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    ax[0, 0].set_xlim([xmin, xmax])
    ax[0, 0].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[0, 0].set_title(r'$\sigma_{11}$-an', fontsize=16)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[0, 0])
    cbar.ax.tick_params(labelsize=14)
    
    #
    cf = ax[0, 1].scatter(x_pred + u_pred * scale, y_pred + v_pred * scale, c=s22_pred, alpha=0.7, edgecolors='none',
                          marker='s', cmap='rainbow', s=s)
    ax[0, 1].axis('square')
    for key, spine in ax[0, 1].spines.items():
        if key in ['right', 'top', 'left', 'bottom']:
            spine.set_visible(False)
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])
    ax[0, 1].set_xlim([xmin, xmax])
    ax[0, 1].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[0, 1].set_title(r'$\sigma_{22}$-an', fontsize=16)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[0, 1])
    cbar.ax.tick_params(labelsize=14)
    #
    cf = ax[0, 2].scatter(x_pred + u_pred * scale, y_pred + v_pred * scale, c=s12_pred, alpha=0.7, edgecolors='none',
                          marker='s', cmap='rainbow', s=s)
    ax[0, 2].axis('square')
    for key, spine in ax[0, 2].spines.items():
        if key in ['right', 'top', 'left', 'bottom']:
            spine.set_visible(False)
    ax[0, 2].set_xticks([])
    ax[0, 2].set_yticks([])
    ax[0, 2].set_xlim([xmin, xmax])
    ax[0, 2].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[0, 2].set_title(r'$\sigma_{12}$-an', fontsize=16)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[0, 2])
    cbar.ax.tick_params(labelsize=14)
    plt.savefig('./output/stress_comparison_an_' + str(1) + '.png', dpi=200)
    plt.close('all')

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
x_star = np.linspace(0, 0.5, 251)
y_star = np.linspace(0, 0.5, 251)
x_star, y_star = np.meshgrid(x_star, y_star)
x_star = x_star.flatten()[:, None]
y_star = y_star.flatten()[:, None]
# dst = ((x_star - 0) ** 2 + (y_star - 0) ** 2) ** 0.5
# x_star = x_star[dst >= 0.1]
# y_star = y_star[dst >= 0.1]
# x_star = x_star.flatten()[:, None]
# y_star = y_star.flatten()[:, None]
shutil.rmtree('./output', ignore_errors=True)
os.makedirs('./output')

xy=np.concatenate((x_star,y_star),1)
xy=torch.FloatTensor(xy).requires_grad_(True)
xy=xy.to(device)
y=model_uv(xy)
yd=model_dist(xy)
yp=model_part(xy)
y=y*yd+yp
print(y)
# y=torch.tensor(y)
e11=torch.autograd.grad(y[:,0], xy,torch.ones_like(y[:,0]), create_graph=True)[0][:,0]# computes dy/dx
e12=torch.autograd.grad(y[:,0], xy,torch.ones_like(y[:,0]), create_graph=True)[0][:,1]
e22=torch.autograd.grad(y[:,1], xy,torch.ones_like(y[:,1]), create_graph=True)[0][:,1]# computes dy/dx
e12=e12+torch.autograd.grad(y[:,1], xy,torch.ones_like(y[:,1]), create_graph=True)[0][:,0]
s11_pred = (E * e11)/ (1 - mu * mu) + (E * mu * e22) / (1 - mu * mu)
s22_pred = (E * mu* e11) / (1 - mu * mu)  + (E * e22) / (1 - mu * mu) 
s12_pred = (E* e12) / (2 * (1 + mu)) 
xy=xy.cpu().detach().numpy()
y=y.cpu().detach().numpy()
field=np.concatenate((xy,y),1)
print(field.shape)
print(xy.shape, y.shape, s11_pred.shape)
# print(field[0])
postProcessDef(xmin=0, xmax=0.50, ymin=0, ymax=0.50, s=4, scale=0, field=field)
newPostProcess(field, s11_pred, s22_pred, s12_pred)
# torch.save('./output/uv.csv',torch.tensor(y))