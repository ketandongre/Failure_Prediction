import numpy as np
import os
import time
import sys
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
import math

import matplotlib.pyplot as plt
import csv
from torch.autograd import Variable

from torch.optim import Adam, LBFGS
from torch.utils.data import Dataset, DataLoader


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.interpolate import make_interp_spline
path_fem='./platehole_data.csv'
matplotlib.rc('font', size=8)


E = 20.0
mu = 0.25
rho = 1.0
hole_r = 0.1
PI = math.pi

#### Creating distance Modelwith Pytorch

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


def GenDistPt(xmin, xmax, ymin, ymax, xc, yc, r, num_surf_pt, num):
    # num: number per edge
    # num_t: number time step
    x = np.linspace(xmin, xmax, num=num)
    y = np.linspace(ymin, ymax, num=num)
    x, y = np.meshgrid(x, y)
    # Delete point in hole
    dst = ((x - xc) ** 2 + (y - yc) ** 2) ** 0.5
    x = x[dst >= r]
    y = y[dst >= r]
    x = x.flatten()[:, None]
    y = y.flatten()[:, None]
    # Refinement point near hole surface
    theta = np.linspace(0.0, np.pi / 2.0, num_surf_pt)
    x_surf = np.multiply(r, np.cos(theta)) + xc
    y_surf = np.multiply(r, np.sin(theta)) + yc
    x_surf = x_surf.flatten()[:, None]
    y_surf = y_surf.flatten()[:, None]
    x = np.concatenate((x, x_surf), 0)
    y = np.concatenate((y, y_surf), 0)
    return x,y

def GenDist(XY_dist):
    dist_u = np.zeros_like(XY_dist[:, 0:1])
    dist_v = np.zeros_like(XY_dist[:, 0:1])
    dist_s11 = np.zeros_like(XY_dist[:, 0:1])
    dist_s22 = np.zeros_like(XY_dist[:, 0:1])
    dist_s12 = np.zeros_like(XY_dist[:, 0:1])
    for i in range(len(XY_dist)):
        dist_u[i, 0] = XY_dist[i][0]  # min(t, x-(-0.5))
        dist_v[i, 0] =  XY_dist[i][1]  # min(t, sqrt((x+0.5)^2+(y+0.5)^2))
        dist_s11[i, 0] = 0.5 - XY_dist[i][0]
        dist_s22[i, 0] = 0.5 - XY_dist[i][1]
        dist_s12[i, 0] = min(XY_dist[i][1], 0.5 - XY_dist[i][1], XY_dist[i][0], 0.5 - XY_dist[i][0])
    DIST = np.concatenate(( dist_u, dist_v, dist_s11, dist_s22, dist_s12), 1)
    return XY_dist , DIST

def DelHolePT(XY_c, xc=0, yc=0, r=0.1):
    # Delete points within hole
    dst = np.array([((xy[0] - xc) ** 2 + (xy[1] - yc) ** 2) ** 0.5 for xy in XY_c])
    return XY_c[dst > r, :]

def GenHoleSurfPT(xc, yc, r, N_PT):
    # Generate
    theta = np.linspace(0.0, np.pi / 2.0, N_PT)
    xx = np.multiply(r, np.cos(theta)) + xc
    yy = np.multiply(r, np.sin(theta)) + yc
    xx = xx.flatten()[:, None]
    yy = yy.flatten()[:, None]
    return xx, yy
 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Current cuda device: ',torch.cuda.get_device_name(0))




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
XY_c = lb + (ub - lb) * lhs(2, 50000)
XY_c_ref = lb + np.array([0.2, 0.2]) * lhs(2, 25000)  # Refinement for stress concentration
XY_c = np.concatenate((XY_c, XY_c_ref), 0)
XY_c = DelHolePT(XY_c, xc=0, yc=0, r=0.1)

xx, yy = GenHoleSurfPT(xc=0, yc=0, r=0.1, N_PT=8000)

HOLE = np.concatenate((xx, yy), 1)
LW = np.array([0.1, 0.0]) + np.array([0.4, 0.0]) * lhs(2, 5000)
UP = np.array([0.0, 0.5]) + np.array([0.5, 0.0]) * lhs(2, 5000)
LF = np.array([0.0, 0.1]) + np.array([0.0, 0.4]) * lhs(2, 5000)
RT = np.array([0.5, 0.0]) + np.array([0.0, 0.5]) * lhs(2, 8000)


s11_RT=np.ones(RT[:,0:1].shape)
RT = np.concatenate((RT, s11_RT), 1)

# Add some boundary points into the collocation point set

XY_c = np.concatenate((XY_c, HOLE[::1, :], LF[::2, :], RT[::2, 0:2], UP[::2, :], LW[::2, :]), 0)

xx, yy = GenHoleSurfPT(xc=0, yc=0, r=0.1, N_PT=XY_c.shape[0])

HOLE = np.concatenate((xx, yy), 1)
XY_dist=torch.FloatTensor(XY_dist).requires_grad_(True)
DIST=torch.FloatTensor(DIST).requires_grad_(True)
HOLE=torch.FloatTensor(HOLE).requires_grad_(True)
LW=torch.FloatTensor(LW).requires_grad_(True)
LF=torch.FloatTensor(LF).requires_grad_(True)
UP=torch.FloatTensor(UP).requires_grad_(True)
RT=torch.FloatTensor(RT).requires_grad_(True)
XY=torch.FloatTensor(XY_c).requires_grad_(True)


model_acc_dist=100000
model_acc_part=10000
model_acc_uv=100000

# train standard neural network to fit distance training data

torch.manual_seed(123)
model_dist = ANN_DistModel()
model_dist.to(device)
os.path.realpath('.')
if os.path.isfile('./distance.pth'):
    print("Loading Dist...")
    model_dist.load_state_dict(torch.load('./distance.pth'))
optimizer_dist = torch.optim.Adam(model_dist.parameters(),lr=1e-3)
minLoss=model_acc_dist
bestModelAt=0
for i in range(5000):
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
    
print(f"Best model at {bestModelAt} with loss: {minLoss}")

#########
# model_dist = ANN_DistModel()
# model_dist.load_state_dict(torch.load('./distance_paper.pth'))
# model_dist.to(device)
#########

# torch.save(model_dist.state_dict(), "./distance_paper.pth")
# train standard neural network to fit distance training data

torch.manual_seed(123)
model_part = ANN_PartModel()
if os.path.isfile('./part.pth'):
    print("Loading Part...")
    model_part.load_state_dict(torch.load('./part.pth'))
model_part.to(device)
optimizer_part = torch.optim.Adam(model_part.parameters(),lr=1e-3)
minLoss=model_acc_part
bestModelAt=0
for i in range(5000):
    optimizer_part.zero_grad()

    #LF
    LF=LF.to(device);
    yh_lf = model_part(LF)
    loss1=torch.mean((yh_lf[:,[0,4]])**2);

    #UP
    UP=UP.to(device);
    yh_up = model_part(UP)
    loss1=loss1+torch.mean((yh_up[:,[3,4]])**2);
    #LW
    LW=LW.to(device);
    yh_lw = model_part(LW)
    loss1=loss1+torch.mean((yh_lw[:,[1,4]])**2);

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


print(f"Best model at {bestModelAt} with loss: {minLoss}")
########
# model_part = ANN_PartModel()
# model_part.load_state_dict(torch.load('./part_paper.pth'))
# model_part.to(device)
########

# torch.save(model_part.state_dict(), "./part_paper.pth")
# train standard neural network to fit distance training data

class ANN_UvModel(nn.Module):
    def __init__(self, N_INPUT=2, N_OUTPUT=5, N_HIDDEN=70, N_LAYERS=8):
        super().__init__()
        activation = nn.Tanh
        self.optim = None
        
        self.fcs = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
        self.apply(self._init_weights)
    
    def lossFct(self,xy,hole,yh):
        e11=torch.autograd.grad(yh[:,0], xy,torch.ones_like(yh[:,0]), create_graph=True)[0][:,0]# computes dy/dx
        e12=torch.autograd.grad(yh[:,0], xy,torch.ones_like(yh[:,0]), create_graph=True)[0][:,1]
        e22=torch.autograd.grad(yh[:,1], xy,torch.ones_like(yh[:,1]), create_graph=True)[0][:,1]# computes dy/dx
        e12=e12+torch.autograd.grad(yh[:,1], xy,torch.ones_like(yh[:,1]), create_graph=True)[0][:,0]
        sp11 = (E * e11)/ (1 - mu * mu) + (E * mu * e22) / (1 - mu * mu)
        sp22 = (E * mu* e11) / (1 - mu * mu)  + (E * e22) / (1 - mu * mu) 
        sp12 = (E* e12) / (2 * (1 + mu)) 

        f_s11 = yh[:,2] - sp11
        f_s12 = yh[:,4] - sp12
        f_s22 = yh[:,3] - sp22

        s11_1=torch.autograd.grad(yh[:,2], xy,torch.ones_like(yh[:,2]), create_graph=True)[0][:,0]
        s12_2=torch.autograd.grad(yh[:,4], xy,torch.ones_like(yh[:,4]), create_graph=True)[0][:,1]

        s22_2=torch.autograd.grad(yh[:,3], xy,torch.ones_like(yh[:,3]), create_graph=True)[0][:,1]
        s12_1=torch.autograd.grad(yh[:,4], xy,torch.ones_like(yh[:,4]), create_graph=True)[0][:,0]
 


        f_u = s11_1 + s12_2 
        f_v = s22_2 + s12_1 


        loss1 = torch.mean(f_s11**2)+torch.mean(f_s22**2)+torch.mean(f_s12**2)# use mean squared error
        loss3=torch.mean(f_u**2)+torch.mean(f_v**2)
    ## For hole 

        # hole=hole.to(device)
        r = hole_r
        nx = -hole[:,0] / r
        ny = -hole[:,1] / r
        yh_hole = self(hole)
        yh_hole=yh_hole*model_dist(hole)+model_part(hole)
        tx=torch.mul(yh_hole[:,2],nx)+torch.mul(yh_hole[:,4],ny)
        ty=torch.mul(yh_hole[:,4],nx)+torch.mul(yh_hole[:,3],ny)
        loss2=torch.mean(tx**2)+torch.mean(ty**2)
        loss = loss1+loss2+loss3# add two loss terms together 
        return loss
    

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
    
    def train(self, data_loader, epochs, validation_data=None):
        for epoch in range(epochs):
            running_loss,batches = self._train_iteration(data_loader)
            val_loss = None
            
            if validation_data is not None:
                y_hat = self(validation_data['X'])
                val_loss = self.lossFct(input=y_hat, target=validation_data['y']).detach().numpy()
                print('[%d] loss: %.3f | validation loss: %.3f' %
                (epoch + 1, running_loss, val_loss))
            else:
                torch.save(self.state_dict(), "./uv.pth")
                print('[%d] loss: %.6f' %
                (epoch + 1, running_loss/batches))
            
            
                
    def _train_iteration(self,data_loader):
        running_loss = 0.0
        batches=0
        for i, (X,hole) in enumerate(data_loader):
            batches=batches+1
            X=X.to(device)
            hole=hole.to(device)
            X = X.float()
            hole = hole.float()
            X_ = Variable(X, requires_grad=True)
            hole=Variable(hole, requires_grad=True)

            def closure():
                if torch.is_grad_enabled():
                    self.optim.zero_grad()
                output = self(X_)*model_dist(X_)+model_part(X_)
                loss = self.lossFct(X_, hole,output)
                if loss.requires_grad:
                    # print('loss backward')
                    loss.backward()
                return loss
            
            self.optim.step(closure)
            
            # calculate the loss again for monitoring
            output = self(X_)*model_dist(X_)+model_part(X_)
            loss = closure()
            running_loss += loss.item()
            # if(i%100==99):
            #     print('loss(i%100): ', loss.item())
        return running_loss,batches
    
    # I like to include a sklearn like predict method for convenience
    def predict(self, X):
        X_ = torch.Tensor(X)
        return (self(X_)*model_dist(X_)+model_part(X_)).detach().numpy().squeeze()


class Build_Data(Dataset):
    # Constructor
    def __init__(self):
        self.x = XY
        self.y = HOLE
        self.len = self.x.shape[0]
    # Getting the data
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    # Getting length of the data
    def __len__(self):
        return self.len

#Adam 1.0
model_uv = ANN_UvModel()
if os.path.isfile('./uv.pth'):
    print("Loading UV...")
    model_uv.load_state_dict(torch.load('./uv.pth'))
model_uv.to(device)
train = Build_Data()
train_loader = DataLoader(train, batch_size=128,shuffle=True)
model_uv.optim = Adam(model_uv.parameters(), lr=1e-3)
EPOCHS=100
model_uv.train(train_loader,EPOCHS)

#Adam 2.0
model_uv = ANN_UvModel()
if os.path.isfile('./uv.pth'):
    print("Loading UV...")
    model_uv.load_state_dict(torch.load('./uv.pth'))
model_uv.to(device)
train = Build_Data()
train_loader = DataLoader(train, batch_size=128,shuffle=True)
model_uv.optim = Adam(model_uv.parameters(), lr=5e-4)
EPOCHS=20
model_uv.train(train_loader,EPOCHS)

# LFBGS
train = Build_Data()
train_loader = DataLoader(train, batch_size=XY.shape[0],shuffle=True)

model_uv = ANN_UvModel()
if os.path.isfile('./uv.pth'):
    print("Loading UV...")
    model_uv.load_state_dict(torch.load('./uv.pth'))

model_uv.to(device)
model_uv.optim = LBFGS(model_uv.parameters(), history_size=20, max_iter=1000,lr=0.1)
EPOCHS=1000
model_uv.train(train_loader,EPOCHS)
# torch.save(model_uv.state_dict(), "./uv_paper.pth")
def newPostProcess(field):
    x_pred, y_pred, u_pred, v_pred, s11_pred, s22_pred, s12_pred = field
    yh=np.concatenate((u_pred,v_pred),axis=1)
    e11=torch.autograd.grad(yh[:,0], XY,torch.ones_like(yh[:,0]), create_graph=True)[0][:,0]# computes dy/dx
    e12=torch.autograd.grad(yh[:,0], XY,torch.ones_like(yh[:,0]), create_graph=True)[0][:,1]
    e22=torch.autograd.grad(yh[:,1], XY,torch.ones_like(yh[:,1]), create_graph=True)[0][:,1]# computes dy/dx
    e12=e12+torch.autograd.grad(yh[:,1], XY,torch.ones_like(yh[:,1]), create_graph=True)[0][:,0]

def FEMcomparisionUV(xy, y_predicted, uv_fem, scale=1, s=5):
    x_pred=xy[:,0]
    y_pred=xy[:,1]
    u_pred=y_predicted[:,0]
    v_pred=y_predicted[:,1]
    u_fem=uv_fem[:,0]
    v_fem=uv_fem[:,1]
    u_error=u_pred-u_fem
    v_error=v_pred-v_fem
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(9, 9))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    x_pred= x_pred.cpu().detach().numpy()
    y_pred= y_pred.cpu().detach().numpy()
    u_pred= u_pred.cpu().detach().numpy()
    v_pred= v_pred.cpu().detach().numpy()
    u_fem= u_fem.cpu().detach().numpy()
    v_fem= v_fem.cpu().detach().numpy()
    u_error= u_error.cpu().detach().numpy()
    v_error= v_error.cpu().detach().numpy()

    cf= ax[0,0].scatter(x_pred , y_pred, c=u_pred, alpha=0.7, edgecolors='none',
                          cmap='rainbow', marker='o', s=s)
    ax[0, 0].set_title(r'$u$-PINN', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[0, 0])
    
    cf=ax[1,0].scatter(x_pred, y_pred, c=u_fem, alpha=0.7, edgecolors='none',
                        cmap='rainbow', marker='o', s=s)
    ax[1, 0].set_title(r'$u$-FEM', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[1, 0])
    
    cf=ax[2,0].scatter(x_pred, y_pred, c=u_error, alpha=0.7, edgecolors='none',
                    cmap='rainbow', marker='o', s=s)
    
    ax[2, 0].set_title(r'$u$-error', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[2, 0])

    cf= ax[0,1].scatter(x_pred, y_pred, c=v_pred, alpha=0.7, edgecolors='none',
                          cmap='rainbow', marker='o', s=s)
    ax[0, 1].set_title(r'$v$-PINN', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[0, 1])
    
    cf=ax[1,1].scatter(x_pred, y_pred, c=v_fem, alpha=0.7, edgecolors='none',
                        cmap='rainbow', marker='o', s=s)
    ax[1, 1].set_title(r'$v$-FEM', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[1, 1])
    
    cf=ax[2,1].scatter(x_pred, y_pred, c=v_error, alpha=0.7, edgecolors='none',
                    cmap='rainbow', marker='o', s=s)
    
    ax[2, 1].set_title(r'$v$-error', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[2, 1])
    plt.savefig('./output/uv_comparison'+ '.png', dpi=200)
    plt.close('all')    

def FEMcomparisionStr(xy, s_predicted, s_fem, scale=1, s=5):
    x_pred=xy[:,0]
    y_pred=xy[:,1]
    s11_pred=s_predicted[:,0]
    s12_pred=s_predicted[:,2]
    s22_pred=s_predicted[:,1]
    s11_fem=s_fem[:,0]
    s12_fem=s_fem[:,1]
    s22_fem=s_fem[:,2]
    s11_error=s11_pred-s11_fem
    s12_error=s12_pred-s12_fem
    s22_error=s22_pred-s22_fem

    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(9, 9))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    fig.tight_layout(pad=4.0)
    x_pred= x_pred.cpu().detach().numpy()
    y_pred= y_pred.cpu().detach().numpy()
    s11_pred= s11_pred.cpu().detach().numpy()
    s12_pred= s12_pred.cpu().detach().numpy()
    s22_pred= s22_pred.cpu().detach().numpy()
    s11_fem= s11_fem.cpu().detach().numpy()
    s12_fem= s12_fem.cpu().detach().numpy()
    s22_fem= s22_fem.cpu().detach().numpy()
    s11_error= s11_error.cpu().detach().numpy()
    s12_error= s12_error.cpu().detach().numpy()
    s22_error= s22_error.cpu().detach().numpy()

    lblsz=8

    cf= ax[0,0].scatter(x_pred , y_pred , c=s11_pred, alpha=0.7, edgecolors='none',
                          cmap='rainbow', marker='o', s=s)
    ax[0,0].axis('square')
    ax[0, 0].set_title(r'$s11$-PINN', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[0, 0])
    cbar.ax.tick_params(labelsize=lblsz)
    
    cf=ax[1,0].scatter(x_pred, y_pred, c=s11_fem, alpha=0.7, edgecolors='none',
                        cmap='rainbow', marker='o', s=s)
    ax[1,0].axis('square')
    ax[1, 0].set_title(r'$s11$-FEM', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[1, 0])
    cbar.ax.tick_params(labelsize=lblsz)
    
    cf=ax[2,0].scatter(x_pred , y_pred, c=s11_error, alpha=0.7, edgecolors='none',
                    cmap='rainbow', marker='o', s=s)
    ax[2,0].axis('square')
    
    ax[2, 0].set_title(r'$s11$-error', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[2, 0])
    cbar.ax.tick_params(labelsize=lblsz)


    cf= ax[0,1].scatter(x_pred , y_pred , c=s12_pred, alpha=0.7, edgecolors='none',
                          cmap='rainbow', marker='o', s=s)
    ax[0,1].axis('square')
    ax[0, 1].set_title(r'$s12$-PINN', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[0, 1])
    cbar.ax.tick_params(labelsize=lblsz)
    
    cf=ax[1,1].scatter(x_pred, y_pred, c=s12_fem, alpha=0.7, edgecolors='none',
                        cmap='rainbow', marker='o', s=s)
    ax[1,1].axis('square')
    ax[1, 1].set_title(r'$s12$-FEM', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[1, 1])
    cbar.ax.tick_params(labelsize=lblsz)
    
    cf=ax[2,1].scatter(x_pred , y_pred, c=s12_error, alpha=0.7, edgecolors='none',
                    cmap='rainbow', marker='o', s=s)
    ax[2,1].axis('square')
    
    ax[2, 1].set_title(r'$s12$-error', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[2, 1])
    cbar.ax.tick_params(labelsize=lblsz)



    cf= ax[0,2].scatter(x_pred , y_pred , c=s22_pred, alpha=0.7, edgecolors='none',
                          cmap='rainbow', marker='o', s=s)
    ax[0,2].axis('square')
    ax[0, 2].set_title(r'$s22$-PINN', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[0, 2])
    cbar.ax.tick_params(labelsize=lblsz)
    
    cf=ax[1,2].scatter(x_pred, y_pred, c=s22_fem, alpha=0.7, edgecolors='none',
                        cmap='rainbow', marker='o', s=s)
    ax[1,2].axis('square')
    ax[1, 2].set_title(r'$s22$-FEM', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[1, 2])
    cbar.ax.tick_params(labelsize=lblsz)
    
    cf=ax[2,2].scatter(x_pred , y_pred, c=s22_error, alpha=0.7, edgecolors='none',
                    cmap='rainbow', marker='o')
    ax[2,2].axis('square')
    ax[2, 2].set_title(r'$s22$-error', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[2, 2])
    cbar.ax.tick_params(labelsize=lblsz)

    plt.savefig('./output/stress_comparison'+ '.png', dpi=200)
    plt.close('all')

def FEMcomparisionUV1(xy, y_predicted, uv_fem, scale=1, s=5):
    x_pred=xy[:,0]
    y_pred=xy[:,1]
    u_pred=y_predicted[:,0]
    v_pred=y_predicted[:,1]
    u_fem=uv_fem[:,0]
    v_fem=uv_fem[:,1]
    u_error=u_pred-u_fem
    v_error=v_pred-v_fem
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(9, 6))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    fig.tight_layout(pad=4.0)
    x_pred= x_pred.cpu().detach().numpy()
    y_pred= y_pred.cpu().detach().numpy()
    u_pred= u_pred.cpu().detach().numpy()
    v_pred= v_pred.cpu().detach().numpy()
    u_fem= u_fem.cpu().detach().numpy()
    v_fem= v_fem.cpu().detach().numpy()
    u_error= u_error.cpu().detach().numpy()
    v_error= v_error.cpu().detach().numpy()

    cf= ax[0,0].scatter(x_pred , y_pred, c=u_pred, alpha=0.7, edgecolors='none',
                          cmap='rainbow', marker='o', s=s)
    ax[0,0].axis('square')
    ax[0, 0].set_title(r'$u$-PINN', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[0, 0])
    cbar.ax.set_title(r'$(m)$')
    cf=ax[0,1].scatter(x_pred, y_pred, c=u_fem, alpha=0.7, edgecolors='none',
                        cmap='rainbow', marker='o', s=s)
    ax[0,1].axis('square')
    ax[0, 1].set_title(r'$u$-FEM', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[0, 1])
    cbar.ax.set_title(r'$(m)$')
    cf=ax[0, 2].scatter(x_pred, y_pred, c=u_error, alpha=0.7, edgecolors='none',
                    cmap='rainbow', marker='o', s=s,vmin=u_fem.min(),vmax=u_fem.max())
    ax[0,2].axis('square')
    ax[0, 2].set_title(r'$u$-error', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[0, 2])
    cbar.ax.set_title(r'$(m)$')
    cf= ax[1, 0].scatter(x_pred, y_pred, c=v_pred, alpha=0.7, edgecolors='none',
                          cmap='rainbow', marker='o', s=s)
    ax[1,0].axis('square')
    ax[1, 0].set_title(r'$v$-PINN', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[1, 0])
    cbar.ax.set_title(r'$(m)$')
    cf=ax[1, 1].scatter(x_pred, y_pred, c=v_fem, alpha=0.7, edgecolors='none',
                        cmap='rainbow', marker='o', s=s)
    ax[1,1].axis('square')
    ax[1, 1].set_title(r'$v$-FEM', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[1, 1])
    cbar.ax.set_title(r'$(m)$')
    cf=ax[1, 2].scatter(x_pred, y_pred, c=v_error, alpha=0.7, edgecolors='none',
                    cmap='rainbow', marker='o', s=s,vmin=v_fem.min(),vmax=v_fem.max())
    ax[1,2].axis('square')
    
    ax[1, 2].set_title(r'$v$-error', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[1, 2])
    cbar.ax.set_title(r'$(m)$')
    plt.savefig('./output/uv_comparison_lat'+ '.png', dpi=200)
    plt.close('all')    

def FEMcomparisionStr1(xy, s_predicted, s_fem, scale=1, s=5):
    x_pred=xy[:,0]
    y_pred=xy[:,1]
    s11_pred=s_predicted[:,0]
    s12_pred=s_predicted[:,2]
    s22_pred=s_predicted[:,1]
    s11_fem=s_fem[:,0]
    s12_fem=s_fem[:,1]
    s22_fem=s_fem[:,2]
    s11_error=s11_pred-s11_fem
    s12_error=s12_pred-s12_fem
    s22_error=s22_pred-s22_fem

    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(9, 9))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    fig.tight_layout(pad=4.0)
    x_pred= x_pred.cpu().detach().numpy()
    y_pred= y_pred.cpu().detach().numpy()
    s11_pred= s11_pred.cpu().detach().numpy()
    s12_pred= s12_pred.cpu().detach().numpy()
    s22_pred= s22_pred.cpu().detach().numpy()
    s11_fem= s11_fem.cpu().detach().numpy()
    s12_fem= s12_fem.cpu().detach().numpy()
    s22_fem= s22_fem.cpu().detach().numpy()
    s11_error= s11_error.cpu().detach().numpy()
    s12_error= s12_error.cpu().detach().numpy()
    s22_error= s22_error.cpu().detach().numpy()

    lblsz=8

    cf= ax[0,0].scatter(x_pred , y_pred , c=s11_pred, alpha=0.7, edgecolors='none',
                          cmap='rainbow', marker='o', s=s)
    ax[0,0].axis('square')
    ax[0,0].set_title(r'$s11$-PINN', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[0,0])
    cbar.ax.tick_params(labelsize=lblsz)
    cbar.ax.set_title(r'$(M Pa)$')
    cf=ax[0,1].scatter(x_pred, y_pred, c=s11_fem, alpha=0.7, edgecolors='none',
                        cmap='rainbow', marker='o', s=s)
    ax[0,1].axis('square')
    ax[0,1].set_title(r'$s11$-FEM', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[0,1])
    cbar.ax.tick_params(labelsize=lblsz)
    cbar.ax.set_title(r'$(M Pa)$')
    cf=ax[0,2].scatter(x_pred , y_pred, c=s11_error, alpha=0.7, edgecolors='none',
                    cmap='rainbow', marker='o', s=s,vmin=s11_fem.min(),vmax=s11_fem.max())
    ax[0,2].axis('square')
    
    ax[0,2].set_title(r'$s11$-error', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[0,2])
    cbar.ax.tick_params(labelsize=lblsz)
    cbar.ax.set_title(r'$(M Pa)$')

    cf= ax[1,0].scatter(x_pred , y_pred , c=s12_pred, alpha=0.7, edgecolors='none',
                          cmap='rainbow', marker='o', s=s)
    ax[1,0].axis('square')
    ax[1,0].set_title(r'$s12$-PINN', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[1,0])
    cbar.ax.tick_params(labelsize=lblsz)
    cbar.ax.set_title(r'$(M Pa)$')
    cf=ax[1,1].scatter(x_pred, y_pred, c=s12_fem, alpha=0.7, edgecolors='none',
                        cmap='rainbow', marker='o', s=s)
    ax[1,1].axis('square')
    ax[1,1].set_title(r'$s12$-FEM', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[1,1])
    cbar.ax.tick_params(labelsize=lblsz)
    cbar.ax.set_title(r'$(M Pa)$')
    cf=ax[1,2].scatter(x_pred , y_pred, c=s12_error, alpha=0.7, edgecolors='none',
                    cmap='rainbow', marker='o', s=s,vmin=s12_fem.min(),vmax=s12_fem.max())
    ax[1,2].axis('square')
    
    ax[1,2].set_title(r'$s12$-error', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[1,2])
    cbar.ax.tick_params(labelsize=lblsz)
    cbar.ax.set_title(r'$(M Pa)$')


    cf= ax[2,0].scatter(x_pred , y_pred , c=s22_pred, alpha=0.7, edgecolors='none',
                          cmap='rainbow', marker='o', s=s)
    ax[2,0].axis('square')
    ax[2,0].set_title(r'$s22$-PINN', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[2,0])
    cbar.ax.tick_params(labelsize=lblsz)
    cbar.ax.set_title(r'$(M Pa)$')
    cf=ax[2,1].scatter(x_pred, y_pred, c=s22_fem, alpha=0.7, edgecolors='none',
                        cmap='rainbow', marker='o', s=s)
    ax[2,1].axis('square')
    ax[2,1].set_title(r'$s22$-FEM', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[2,1])
    cbar.ax.tick_params(labelsize=lblsz)
    cbar.ax.set_title(r'$(M Pa)$')
    cf=ax[2,2].scatter(x_pred , y_pred, c=s22_error, alpha=0.7, edgecolors='none',
                    cmap='rainbow', marker='o',vmin=s22_fem.min(),vmax=s22_fem.max())
    ax[2,2].axis('square')
    ax[2,2].set_title(r'$s22$-error', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[2,2])
    cbar.ax.tick_params(labelsize=lblsz)
    cbar.ax.set_title(r'$(M Pa)$')
    plt.savefig('./output/stress_comparison_lat'+ '.png', dpi=200)
    plt.close('all')


def postProcessDef(xmin, xmax, ymin, ymax, field, s=5, num=0, scale=1):
    ''' Plot deformed plate (set scale=0 want to plot undeformed contours)
    '''
    # [x_pred, y_pred, u_pred, v_pred, s11_pred, s22_pred, s12_pred] = field
    x_pred=field[:,0]
    y_pred=field[:,1]
    u_pred=field[:,2]
    v_pred=field[:,3]
    s11_pred=field[:,4]
    s22_pred=field[:,5]
    s12_pred=field[:,6]
    print(v_pred)
    #
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(9, 9))
    fig.subplots_adjust(hspace=0.2, wspace=0.3)
    cf = ax[0].scatter(x_pred, y_pred, c=u_pred, alpha=0.7, edgecolors='none',
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
    cf = ax[1].scatter(x_pred, y_pred, c=v_pred, alpha=0.7, edgecolors='none',
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
    plt.savefig('./output/uv' + '.png', dpi=200)
    plt.close('all')
    #
    # Plot predicted stress
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 9))
    fig.subplots_adjust(hspace=0.15, wspace=0.3)
    #
    cf = ax[0].scatter(x_pred, y_pred, c=s11_pred, alpha=0.7, edgecolors='none',
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
    cf = ax[1].scatter(x_pred, y_pred, c=s22_pred, alpha=0.7, edgecolors='none',
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
    cf = ax[2].scatter(x_pred , y_pred, c=s12_pred, alpha=0.7, edgecolors='none',
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
    plt.savefig('./output/stress' + '.png', dpi=200)
    plt.close('all')

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
x_star = np.linspace(0, 0.5, 251)
y_star = np.linspace(0, 0.5, 251)
x_star, y_star = np.meshgrid(x_star, y_star)
x_star = x_star.flatten()[:, None]
y_star = y_star.flatten()[:, None]
dst = ((x_star - 0) ** 2 + (y_star - 0) ** 2) ** 0.5
x_star = x_star[dst >= 0.1]
y_star = y_star[dst >= 0.1]
x_star = x_star.flatten()[:, None]
y_star = y_star.flatten()[:, None]
shutil.rmtree('./output', ignore_errors=True)
if os.path.isdir('./output')==False:
    os.makedirs('./output')

xy=np.concatenate((x_star,y_star),1)
xy=torch.FloatTensor(xy).requires_grad_(True)
xy=xy.to(device)
y=model_uv(xy)
yd=model_dist(xy)
yp=model_part(xy)
y=y*yd+yp
xy=xy.cpu().detach().numpy()
y=y.cpu().detach().numpy()
field=np.concatenate((xy,y),1)

postProcessDef(xmin=0, xmax=0.50, ymin=0, ymax=0.50, s=4, scale=0, field=field)

# newPostProcess(field)
my_data = np.genfromtxt(path_fem, delimiter=',')
csv_data=my_data[1:,:]
x_col, y_col = 2,3
u_col, v_col = 0,1
csv_data[:,[x_col,y_col]]+=0.25
s11_col, s12_col, s22_col=5,6,8

xy=torch.FloatTensor(csv_data[:,[x_col,y_col]]).requires_grad_(True)
# xy=xy+0.25
xy=xy.to(device)

### UV comparision ###
uv_fem=torch.FloatTensor(csv_data[:,[u_col,v_col]]).requires_grad_(True)
uv_fem=uv_fem.to(device)
y=model_uv(xy)
y_dist=model_dist(xy)
y_part=model_part(xy)
y_predicted=y*y_dist+y_part
y_predicted=y_predicted[:,[0,1]]
FEMcomparisionUV1(xy, y_predicted, uv_fem)

s_fem=torch.FloatTensor(csv_data[:,[s11_col,s12_col,s22_col]]).requires_grad_(True)
s_fem=s_fem.to(device)
y=model_uv(xy)
y_dist=model_dist(xy)
y_part=model_part(xy)
s_predicted=y*y_dist+y_part
s_predicted=s_predicted[:,[2,3,4]]
FEMcomparisionStr1(xy, s_predicted, s_fem)

n_pts=len(csv_data)
pts_on_hole = np.zeros((1,0), dtype= int)
for i in range(n_pts) :
    x_pt = csv_data[i][x_col]
    y_pt = csv_data[i][y_col]
    if x_pt**2+y_pt**2-0.01>=0.0020 and x_pt**2+y_pt**2-0.01<=0.0025:
        print(x_pt**2+y_pt**2)
        pts_on_hole=np.append(pts_on_hole,int(i))

out_names = ["$u/(m)$","$v/(m)$",r"$\sigma_{11}/(M Pa)$",r"$\sigma_{22}/(M Pa)$",r"$\sigma_{12}/(M Pa)$"]
png_names = ["u","v",r"s11",r"s22",r"s12"]
def sort_list(list1, list2):
    zipped_pairs = zip(list2, list1)
    z = [x for _, x in sorted(zipped_pairs)]
    return np.array(z)

for i in range(5):
    fem_hole = csv_data[pts_on_hole,:]
    xy_hole =  fem_hole[:,[x_col,y_col]]
    xy_hole = torch.FloatTensor(xy_hole)
    xy_hole = xy_hole.to(device)
    y_fem_hole = torch.FloatTensor(fem_hole[:,[u_col,v_col,s11_col,s22_col,s12_col]]).to(device)
    y_pred_hole = model_part(xy_hole).to(device) + model_dist(xy_hole) * model_uv(xy_hole)
    # print(y_pred_hole - y_fem_hole)
    xy_hole=xy_hole.cpu().detach().numpy()
    theta = np.flip(np.arctan(xy_hole[:,1]/xy_hole[:,0]),0)
    theta=theta*180/PI
    vals = np.flip(y_pred_hole.cpu().detach().numpy(),0)
    fem_vals = np.flip(y_fem_hole.cpu().detach().numpy(),0)
    vals[:,i]=sort_list(vals[:,i],theta)
    fem_vals[:,i]=sort_list(fem_vals[:,i],theta)
    theta=np.sort(theta)
    X_Y_Spline = make_interp_spline(theta, vals[:,i])

    X_ = np.linspace(0, 90, 500)
    Y_ = X_Y_Spline(X_)

    # Plotting the Graph
    # plt.plot(X_, Y_, label =f'PINN')
    plt.scatter(theta , vals[:,i], label = f'PINN')
    # plt.scatter(theta, vals[:,i], label = f't = {t}, PINN')
    plt.scatter(theta , fem_vals[:,i], label = f'FEM')
    plt.legend()
    plt.ylabel(out_names[i],fontsize=18)
    plt.xlabel(r"$\theta/degree$",fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=9)
    plt.savefig(f'./output/hole_output_{png_names[i]}' + '.png', dpi=200)
    plt.close('all')