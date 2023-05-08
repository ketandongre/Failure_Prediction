import numpy as np
import os
from pyDOE import lhs
import matplotlib
import platform
from scipy.interpolate import make_interp_spline
if platform.system()=='Linux':
    matplotlib.use('Agg')
if platform.system()=='Windows':
    from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import imageio
import pandas as pd
import shutil
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable

from torch.optim import Adam, LBFGS
from scipy.interpolate import make_interp_spline

class ANN_DistModel(nn.Module):
    def __init__(self, N_INPUT=3, N_OUTPUT=5, N_HIDDEN=20, N_LAYERS=4):
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



def GenDistPt(xmin, xmax, ymin, ymax, tmin, tmax, xc, yc, r, num_surf_pt,num, num_t):
    # num: number per edge
    # num_t: number time step
    x = np.linspace(xmin, xmax, num=num)
    y = np.linspace(ymin, ymax, num=num)
    x, y = np.meshgrid(x, y)
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
    # Cartisian product with time points
    t = np.linspace(tmin, tmax, num=num_t)
    xxx, ttt = np.meshgrid(x, t)
    yyy,   _ = np.meshgrid(y, t)
    xxx = xxx.flatten()[:, None]
    yyy = yyy.flatten()[:, None]
    ttt = ttt.flatten()[:, None]
    return xxx, yyy, ttt

def GenDist(XYT_dist):
    dist_u = np.zeros_like(XYT_dist[:, 0:1])
    dist_v = np.zeros_like(XYT_dist[:, 0:1])
    dist_s11 = np.zeros_like(XYT_dist[:, 0:1])
    dist_s22 = np.zeros_like(XYT_dist[:, 0:1])
    dist_s12 = np.zeros_like(XYT_dist[:, 0:1])
    for i in range(len(XYT_dist)):
        dist_u[i, 0] = min(XYT_dist[i][2], XYT_dist[i][0])  # min(t, x-(-0.5))
        dist_v[i, 0] = min(XYT_dist[i][2], XYT_dist[i][1])  # min(t, sqrt((x+0.5)^2+(y+0.5)^2))
        dist_s11[i, 0] = min(XYT_dist[i][2], 0.5 - XYT_dist[i][0])
        dist_s22[i, 0] = min(XYT_dist[i][2], 0.5 - XYT_dist[i][1])  
        dist_s12[i, 0] = min(XYT_dist[i][2], XYT_dist[i][1], 0.5 - XYT_dist[i][1], XYT_dist[i][0], 0.5 - XYT_dist[i][0])
    DIST = np.concatenate(( dist_u, dist_v, dist_s11, dist_s22, dist_s12), 1)
    return XYT_dist , DIST

def DelHolePT(XY_c, xc=0, yc=0, r=0.1):
    # Delete points within hole
    dst = np.array([((xy[0] - xc) ** 2 + (xy[1] - yc) ** 2) ** 0.5 for xy in XY_c])
    return XY_c[dst > r, :]

def GenHoleSurfPT(xc, yc, r, N_PT):
    # Generate
    theta = np.linspace(0.0, np.pi / 2.0, N_PT)
    theta=(np.pi/2)*lhs(1,N_PT)
    xx = np.multiply(r, np.cos(theta)) + xc
    yy = np.multiply(r, np.sin(theta)) + yc
    xx = xx.flatten()[:, None]
    yy = yy.flatten()[:, None]
    return xx, yy

#installing GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Current cuda device: ',torch.cuda.get_device_name(0))


E = 20.0
mu = 0.25
rho = 1.0
hole_r = 0.1
#### Note: The detailed description for this case can be found in paper:
#### Physics informed deep learning for computational elastodynamicswithout labeled data.
#### https://arxiv.org/abs/2006.08472
#### But network configuration might be slightly different from what is described in paper.
PI = math.pi
MAX_T = 10.0

# Domain bounds for x, y and t
lb = np.array([0, 0, 0.0])
ub = np.array([0.5, 0.5, 10.0])

# Network configuration
uv_layers   = [3] + 8 * [70] + [5]
dist_layers = [3] + 4 * [20] + [5]
part_layers = [3] + 4 * [20] + [5]

# Number of frames for postprocessing
N_t = int(MAX_T * 8 + 1)

# Generate distance function for spatio-temporal space
x_dist, y_dist, t_dist = GenDistPt(xmin=0, xmax=0.5, ymin=0, ymax=0.5, tmin=0, tmax=10, xc=0, yc=0, r=0.1,
                                     num_surf_pt=40,num=30, num_t=30)
XYT_dist = np.concatenate((x_dist, y_dist, t_dist), 1)
XYT_dist,DIST = GenDist(XYT_dist)
IC = lb + np.array([0.5, 0.5, 0.0]) * lhs(3, 5000)
IC = DelHolePT(IC, xc=0, yc=0, r=0.1)

XYT_c = lb + (ub - lb) * lhs(3, 45000)
XYT_c_ref = lb + np.array([0.15, 0.15,10]) * lhs(3, 20000)  # Refinement for stress concentration
XYT_c = np.concatenate((XYT_c, XYT_c_ref), 0)
XYT_c = DelHolePT(XYT_c, xc=0, yc=0, r=0.1)

xx, yy = GenHoleSurfPT(xc=0, yc=0, r=0.1, N_PT=83)
tt = np.linspace(0, 10, 83)
tt = tt[1:]
x_ho, t_ho = np.meshgrid(xx, tt)
y_ho, _ = np.meshgrid(yy, tt)
x_ho = x_ho.flatten()[:, None]
y_ho = y_ho.flatten()[:, None]
t_ho = t_ho.flatten()[:, None]
HOLE = np.concatenate((x_ho, y_ho, t_ho), 1)
LW = np.array([0.1, 0.0, 0.0]) + np.array([0.4, 0.0, 10]) * lhs(3, 4000)
UP = np.array([0.0, 0.5, 0.0]) + np.array([0.5, 0.0, 10]) * lhs(3, 4000)
LF = np.array([0.0, 0.1, 0.0]) + np.array([0.0, 0.4, 10]) * lhs(3, 4000)
RT = np.array([0.5, 0.0, 0.0]) + np.array([0.0, 0.5, 10]) * lhs(3, 7000)

t_RT = RT[:, 2:3]
period = 5  # two period in 10s
s11_RT = 0.5 * np.sin((2 * PI / period) * t_RT + 3 * PI / 2) + 0.5
RT = np.concatenate((RT, s11_RT), 1)



# Add some boundary points into the collocation point set
XYT_c = np.concatenate((XYT_c, HOLE[::4, :], LF[::4, :], RT[::4, 0:3], UP[::4, :], LW[::4, :]), 0)
# HOLE_uv=np.array([np.pi/2.0, 10.0]) * lhs(2, XYT_c.shape[0])
x_hole,y_hole=GenHoleSurfPT(0,0,0.1,XYT_c.shape[0])

HOLE_uv=np.concatenate((x_hole,y_hole,10*lhs(1, XYT_c.shape[0])),1)

IC_dist=lb + np.array([0.5, 0.5, 0.0]) * lhs(3, XYT_dist.shape[0])

HOLE_uv=torch.FloatTensor(HOLE_uv).requires_grad_(True)
XYT_dist=torch.FloatTensor(XYT_dist).requires_grad_(True)
DIST=torch.FloatTensor(DIST).requires_grad_(True)
IC=torch.FloatTensor(IC).requires_grad_(True)
LW=torch.FloatTensor(LW).requires_grad_(True)
LF=torch.FloatTensor(LF).requires_grad_(True)
UP=torch.FloatTensor(UP).requires_grad_(True)
RT=torch.FloatTensor(RT).requires_grad_(True)
XYT=torch.FloatTensor(XYT_c).requires_grad_(True)
IC_dist=torch.FloatTensor(IC_dist).requires_grad_(True)

def sigma11(time):
    return 0.5 * torch.sin((2 * PI / period) * time + 3 * PI / 2) + 0.5

def model_part(xyt):
    res=torch.zeros(list(xyt.size())[0],5)
    res[:,2]=sigma11(xyt[:,2])
    # return torch.FloatTensor(res).requires_grad_(True)
    return torch.FloatTensor(res).requires_grad_(True).to(device)

class Build_Data_dist(Dataset):
    # Constructor
    def __init__(self):
        self.x = XYT_dist
        self.y = DIST
        self.z=IC_dist
        self.len = self.x.shape[0]
    # Getting the data
    def __getitem__(self, index):
        return self.x[index], self.y[index],self.z[index]
    # Getting length of the data
    def __len__(self):
        return self.len

train_dist = Build_Data_dist()
train_loader_dist = DataLoader(train_dist, batch_size=32,shuffle=True)
torch.manual_seed(123)
model_dist = ANN_DistModel()
if os.path.isfile('./distance_time.pth'):
    model_dist.load_state_dict(torch.load('./distance_time.pth'))
    print("loaded distance model successfully...")
model_dist.to(device)
optimizer_dist = torch.optim.Adam(model_dist.parameters(),lr=5e-4)

for i in range(0):
    running_loss=0.0
    for _, [xyt_dist,dist,ic] in enumerate(train_loader_dist):
        optimizer_dist.zero_grad()
        
        # compute the "data loss"
        xyt_dist=xyt_dist.to(device)
        dist=dist.to(device)
        yh = model_dist(xyt_dist)
        loss1 = torch.mean((yh-dist)**2)# use mean squared error
        
        # # compute the "physics loss"
        ic=ic.to(device)
        dist_ic = model_dist(ic)

        D_u=dist_ic[:,0:1]
        D_v=dist_ic[:,1:2]
        du_dt  = torch.autograd.grad(D_u, ic,torch.ones_like(D_u), create_graph=True)[0][:,2]# computes dy/dx
        dv_dt  = torch.autograd.grad(D_v, ic,torch.ones_like(D_v), create_graph=True)[0][:,2]
        loss2 =torch.mean(dv_dt**2) + torch.mean(du_dt**2)
        
        yh_ic=model_dist(ic)
        loss3=torch.mean(yh_ic**2)
        # backpropagate joint loss
        # loss=loss1
        loss = 1000*(loss1 + loss2+loss3)# add two loss terms together
        loss.backward()
        optimizer_dist.step()

        running_loss+=loss.item()
        if _%100==99: 
            print('[%d, %5d] loss: %.4f' %
                (i + 1, _ + 1, running_loss / 100))
            running_loss = 0.0
            torch.save(model_dist.state_dict(), "./distance_time.pth")

class ANN_UvModel(nn.Module):
    def __init__(self, N_INPUT=3, N_OUTPUT=5, N_HIDDEN=70, N_LAYERS=8):
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
    
    def lossFct(self,xyt,hole,yh):

        e11=torch.autograd.grad(yh[:,0], xyt,torch.ones_like(yh[:,0]), create_graph=True)[0][:,0]# computes dy/dx
        e12=torch.autograd.grad(yh[:,0], xyt,torch.ones_like(yh[:,0]), create_graph=True)[0][:,1]
        e22=torch.autograd.grad(yh[:,1], xyt,torch.ones_like(yh[:,1]), create_graph=True)[0][:,1]# computes dy/dx
        e12=e12+torch.autograd.grad(yh[:,1], xyt,torch.ones_like(yh[:,1]), create_graph=True)[0][:,0]
        sp11 = (E * e11)/ (1 - mu * mu)  + (E * mu * e22)/ (1 - mu * mu) 
        sp22 = (E * mu* e11) / (1 - mu * mu)  + (E* e22) / (1 - mu * mu) 
        sp12 = (E* e12) / (2 * (1 + mu)) 

        f_s11 = yh[:,2] - sp11
        f_s12 = yh[:,4] - sp12
        f_s22 = yh[:,3] - sp22

        s11_1=torch.autograd.grad(yh[:,2], xyt,torch.ones_like(yh[:,2]), create_graph=True)[0][:,0]
        s12_2=torch.autograd.grad(yh[:,4], xyt,torch.ones_like(yh[:,4]), create_graph=True)[0][:,1]
        u_t=torch.autograd.grad(yh[:,0], xyt,torch.ones_like(yh[:,0]), create_graph=True)[0][:,2]
        u_tt=torch.autograd.grad(u_t, xyt,torch.ones_like(u_t), create_graph=True)[0][:,2]

        s22_2=torch.autograd.grad(yh[:,3], xyt,torch.ones_like(yh[:,3]), create_graph=True)[0][:,1]
        s12_1=torch.autograd.grad(yh[:,4], xyt,torch.ones_like(yh[:,4]), create_graph=True)[0][:,0]
        v_t=torch.autograd.grad(yh[:,1], xyt,torch.ones_like(yh[:,1]), create_graph=True)[0][:,2]
        v_tt=torch.autograd.grad(v_t, xyt,torch.ones_like(u_t), create_graph=True)[0][:,2]


        f_u = s11_1 + s12_2 - rho * u_tt
        f_v = s22_2 + s12_1 - rho * v_tt


        loss1 = torch.mean(f_s11**2)+torch.mean(f_s22**2)+torch.mean(f_s12**2)# use mean squared error
        loss2=torch.mean(f_u**2)+torch.mean(f_v**2)
        r = hole_r
        nx = -hole[:,0] / r
        ny = -hole[:,1] / r
        yh_hole = self(hole)
        yh_hole=yh_hole*model_dist(hole)+model_part(hole)
        tx=torch.mul(yh_hole[:,2],nx)+torch.mul(yh_hole[:,4],ny)
        ty=torch.mul(yh_hole[:,4],nx)+torch.mul(yh_hole[:,3],ny)

        loss3=torch.mean(tx**2)+torch.mean(ty**2)
        
        loss = loss1 +loss2 + loss3# add two loss terms together 
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
            X_=torch.cat((X_,hole[::20,:]),0)

            def closure():
                if torch.is_grad_enabled():
                    self.optim.zero_grad()
                output = self(X_)*model_dist(X_)+model_part(X_)
                loss = self.lossFct(X_, hole,output)
                if loss.requires_grad:
                    loss.backward()
                return loss
            
            self.optim.step(closure)
            
            # calculate the loss again for monitoring
            output = self(X_)*model_dist(X_)+model_part(X_)
            loss = closure()
            running_loss += loss.item()
        return running_loss,batches
    
    # I like to include a sklearn like predict method for convenience
    def predict(self, X):
        X_ = torch.Tensor(X)
        return (self(X_)*model_dist(X_)+model_part(X_)).detach().numpy().squeeze()


class Build_Data(Dataset):
    # Constructor
    def __init__(self):
        self.x = XYT
        self.y = HOLE_uv
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
EPOCHS=20
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
EPOCHS=100
model_uv.train(train_loader,EPOCHS)

# LFBGS
train = Build_Data()
train_loader = DataLoader(train, batch_size=XYT.shape[0],shuffle=True)

model_uv = ANN_UvModel()
if os.path.isfile('./uv.pth'):
    print("Loading UV...")
    model_uv.load_state_dict(torch.load('./uv.pth'))

model_uv.to(device)
EPOCHS=1000
model_uv.optim = LBFGS(model_uv.parameters(), history_size=10,lr=0.1,max_iter=EPOCHS)

model_uv.train(train_loader,EPOCHS)



def postProcessDef(xmin, xmax, ymin, ymax, field, s=5, num=0, scale=1):
    ''' Plot deformed plate (set scale=0 want to plot undeformed contours)
    '''
    # [x_pred, y_pred, _, u_pred, v_pred, s11_pred, s22_pred, s12_pred] = field
    x_pred=field[:,0]
    y_pred=field[:,1]
    u_pred=field[:,3]
    v_pred=field[:,4]
    s11_pred=field[:,5]
    s22_pred=field[:,6]
    s12_pred=field[:,7]
    #
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(9, 9))
    fig.subplots_adjust(hspace=0.2, wspace=0.3)
    cf = ax[0].scatter(x_pred + u_pred * scale, y_pred + v_pred * scale, c=u_pred, alpha=0.7, edgecolors='none',
                          cmap='rainbow', marker='o', s=s,vmin=0,vmax=0.03)
    ax[0].axis('square')
    for key, spine in ax[ 0].spines.items():
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
                          cmap='rainbow', marker='o', s=s,vmin=-0.01,vmax=0)
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
    # plt.draw()
    plt.savefig('./output/uv_comparison_time' + str(num) + '.png', dpi=200)
    plt.close('all')
    #
    # Plot predicted stress
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 9))
    fig.subplots_adjust(hspace=0.15, wspace=0.3)
    #
    cf = ax[0].scatter(x_pred + u_pred * scale, y_pred + v_pred * scale, c=s11_pred, alpha=0.7, edgecolors='none',
                          marker='s', cmap='rainbow', s=s,vmin=0,vmax=2.6)
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
                          marker='s', cmap='rainbow', s=s,vmin=-1,vmax=0.5)
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
                          marker='s', cmap='rainbow', s=s,vmin=-1,vmax=0.1)
    ax[2].axis('square')
    for key, spine in ax[2].spines.items():
        if key in ['right', 'top', 'left', 'bottom']:
            spine.set_visible(False)
    ax[ 2].set_xticks([])
    ax[2].set_yticks([])
    ax[2].set_xlim([xmin, xmax])
    ax[2].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[2].set_title(r'$\sigma_{12}$-PINN', fontsize=16)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[2])
    cbar.ax.tick_params(labelsize=14)
    #
    plt.savefig('./output/stress_comparison_time' + str(num) + '.png', dpi=200)
    plt.close('all')

def FEMcomparisionUV(xyt, y_predicted, uv_fem, MAX_T, N_t, scale=1, s=5):
    x_pred=xyt[:,0]
    y_pred=xyt[:,1]
    t_pred=xyt[:,2]
    u_pred=y_predicted[:,0]
    v_pred=y_predicted[:,1]
    u_fem=uv_fem[:,0]
    v_fem=uv_fem[:,1]
    u_error=u_pred-u_fem
    v_error=v_pred-v_fem
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(9, 9))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    fig.suptitle(f"t={t_pred[0].item():0.3f}")
    # x_pred= x_pred.cpu().detach().numpy()
    # y_pred= y_pred.cpu().detach().numpy()
    # u_pred= u_pred.cpu().detach().numpy()
    # v_pred= v_pred.cpu().detach().numpy()
    # u_fem= u_fem.cpu().detach().numpy()
    # v_fem= v_fem.cpu().detach().numpy()
    # u_error= u_error.cpu().detach().numpy()
    # v_error= v_error.cpu().detach().numpy()

    cf= ax[0,0].scatter(x_pred , y_pred, c=u_pred, alpha=0.7, edgecolors='none',
                        cmap='rainbow', marker='o', s=s,vmin=0, vmax=0.03)
    ax[0, 0].set_title(r'$u$-PINN', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[0, 0])
    
    cf=ax[1,0].scatter(x_pred, y_pred, c=u_fem, alpha=0.7, edgecolors='none',
                        cmap='rainbow', marker='o', s=s,vmin=0, vmax=0.03)
    ax[1, 0].set_title(r'$u$-FEM', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[1, 0])
    
    cf=ax[2,0].scatter(x_pred, y_pred, c=u_error, alpha=0.7, edgecolors='none',
                    cmap='rainbow', marker='o', s=s,vmin=0, vmax=0.03)
    
    ax[2, 0].set_title(r'$u$-error', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[2, 0])

    cf= ax[0,1].scatter(x_pred, y_pred, c=v_pred, alpha=0.7, edgecolors='none',
                        cmap='rainbow', marker='o', s=s, vmin=-0.01, vmax=0)
    ax[0, 1].set_title(r'$v$-PINN', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[0, 1])
    
    cf=ax[1,1].scatter(x_pred, y_pred, c=v_fem, alpha=0.7, edgecolors='none',
                        cmap='rainbow', marker='o', s=s, vmin=-0.01, vmax=0)
    ax[1, 1].set_title(r'$v$-FEM', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[1, 1])
    
    cf=ax[2,1].scatter(x_pred, y_pred, c=v_error, alpha=0.7, edgecolors='none',
                    cmap='rainbow', marker='o', s=s,vmin=-0.01, vmax=0)
    
    ax[2, 1].set_title(r'$v$-error', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[2, 1])
    plt.savefig('./output/uv_error_'+f"t={t_pred[0].item():0.3f}"+ '.png', dpi=200)
    plt.close('all')    

def FEMcomparisionStr(xyt, s_predicted, s_fem, scale=1, s=5):
    x_pred=xyt[:,0]
    y_pred=xyt[:,1]
    t_pred=xyt[:,2]
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
    fig.suptitle(f"t={t_pred[0].item():0.3f}")
    # x_pred= x_pred.cpu().detach().numpy()
    # y_pred= y_pred.cpu().detach().numpy()
    # s11_pred= s11_pred.cpu().detach().numpy()
    # s12_pred= s12_pred.cpu().detach().numpy()
    # s22_pred= s22_pred.cpu().detach().numpy()
    # s11_fem= s11_fem.cpu().detach().numpy()
    # s12_fem= s12_fem.cpu().detach().numpy()
    # s22_fem= s22_fem.cpu().detach().numpy()
    # s11_error= s11_error.cpu().detach().numpy()
    # s12_error= s12_error.cpu().detach().numpy()
    # s22_error= s22_error.cpu().detach().numpy()

    lblsz=8

    cf= ax[0,0].scatter(x_pred , y_pred , c=s11_pred, alpha=0.7, edgecolors='none',
                          cmap='rainbow', marker='o', s=s, vmin=0, vmax=2.6)
    ax[0,0].axis('square')
    ax[0, 0].set_title(r'$s11$-PINN', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[0, 0])
    cbar.ax.tick_params(labelsize=lblsz)
    
    cf=ax[1,0].scatter(x_pred, y_pred, c=s11_fem, alpha=0.7, edgecolors='none',
                        cmap='rainbow', marker='o', s=s, vmin=0, vmax=2.6)
    ax[1,0].axis('square')
    ax[1, 0].set_title(r'$s11$-FEM', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[1, 0])
    cbar.ax.tick_params(labelsize=lblsz)
    
    cf=ax[2,0].scatter(x_pred , y_pred, c=s11_error, alpha=0.7, edgecolors='none',
                    cmap='rainbow', marker='o', s=s, vmin=0, vmax=2.6)
    ax[2,0].axis('square')
    
    ax[2, 0].set_title(r'$s11$-error', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[2, 0])
    cbar.ax.tick_params(labelsize=lblsz)


    cf= ax[0,1].scatter(x_pred , y_pred , c=s12_pred, alpha=0.7, edgecolors='none',
                          cmap='rainbow', marker='o', s=s, vmin=-0.1, vmax=0.1)
    ax[0,1].axis('square')
    ax[0, 1].set_title(r'$s12$-PINN', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[0, 1])
    cbar.ax.tick_params(labelsize=lblsz)
    
    cf=ax[1,1].scatter(x_pred, y_pred, c=s12_fem, alpha=0.7, edgecolors='none',
                        cmap='rainbow', marker='o', s=s, vmin=-0.1, vmax=0.1)
    ax[1,1].axis('square')
    ax[1, 1].set_title(r'$s12$-FEM', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[1, 1])
    cbar.ax.tick_params(labelsize=lblsz)
    
    cf=ax[2,1].scatter(x_pred , y_pred, c=s12_error, alpha=0.7, edgecolors='none',
                    cmap='rainbow', marker='o', s=s, vmin=-0.1, vmax=0.1)
    ax[2,1].axis('square')
    
    ax[2, 1].set_title(r'$s12$-error', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[2, 1])
    cbar.ax.tick_params(labelsize=lblsz)



    cf= ax[0,2].scatter(x_pred , y_pred , c=s22_pred, alpha=0.7, edgecolors='none',
                          cmap='rainbow', marker='o', s=s,vmin=-1, vmax=0.5)
    ax[0,2].axis('square')
    ax[0, 2].set_title(r'$s22$-PINN', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[0, 2])
    cbar.ax.tick_params(labelsize=lblsz)
    
    cf=ax[1,2].scatter(x_pred, y_pred, c=s22_fem, alpha=0.7, edgecolors='none',
                        cmap='rainbow', marker='o', s=s,vmin=-1, vmax=0.5)
    ax[1,2].axis('square')
    ax[1, 2].set_title(r'$s22$-FEM', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[1, 2])
    cbar.ax.tick_params(labelsize=lblsz)
    
    cf=ax[2,2].scatter(x_pred , y_pred, c=s22_error, alpha=0.7, edgecolors='none',
                    cmap='rainbow', marker='o',vmin=-1, vmax=0.5)
    ax[2,2].axis('square')
    ax[2, 2].set_title(r'$s22$-error', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[2, 2])
    cbar.ax.tick_params(labelsize=lblsz)

    plt.savefig('./output/stress_error_'+f"t={t_pred[0].item():0.3f}"+ '.png', dpi=200)
    plt.close('all')

def create_gif(type):
    frames_stress = []
    frames_uv=[]
    for t in range(N_t):
        time=(t)*0.125
        if type is "plot":
            stress_file_name=f'./output/stress_comparison_time{t}.png'
            uv_file_name=f'./output/uv_comparison_time{t}.png'
        else:
            stress_file_name=f'./output/stress_error_t={time:0.3f}.png'
            uv_file_name=f'./output/uv_error_t={time:0.3f}.png'
        image_stress = imageio.v2.imread(stress_file_name)
        image_uv = imageio.v2.imread(uv_file_name)
        frames_stress.append(image_stress)
        frames_uv.append(image_uv)
    stress_gif_name=""
    uv_gif_name=""
    if type is "plot":
        stress_gif_name='./stress_pinn.gif'
        uv_gif_name='./uv_pinn.gif'
    else:
        stress_gif_name='./stress_error.gif'
        uv_gif_name='./uv_error.gif'
    imageio.mimsave(stress_gif_name, 
            frames_stress, 
            fps = 5, 
            loop = 1)

    imageio.mimsave(uv_gif_name, 
            frames_uv, 
            fps = 5, 
            loop = 1)


import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# x_star = np.linspace(0, 0.5, 251)
# y_star = np.linspace(0, 0.5, 251)
# x_star, y_star = np.meshgrid(x_star, y_star)
# dst = ((x_star - 0) ** 2 + (y_star - 0) ** 2) ** 0.5
# x_star = x_star[dst >= 0.1]
# y_star = y_star[dst >= 0.1]
# x_star = x_star.flatten()[:, None]
# y_star = y_star.flatten()[:, None]

# shutil.rmtree('./output', ignore_errors=True)
shutil.rmtree('./output', ignore_errors=True)
if not os.path.isdir('./output'):
    os.makedirs('./output')

path_fem = './timevaryingHole_01.csv'

my_data = np.genfromtxt(path_fem, delimiter=',')
csv_data=my_data[1:,:]
x_col, y_col, t_col = 4, 5, 1
u_col, v_col = 2,3
s11_col, s12_col, s22_col=7,8,10
xyt=torch.FloatTensor(csv_data[:,[x_col,y_col,t_col]]).requires_grad_(True)
xyt=xyt+0.25
xyt[:,2]=xyt[:,2]-0.25
xyt=xyt.to(device)
uv_fem=torch.FloatTensor(csv_data[:,[u_col,v_col]]).requires_grad_(True)
s_fem=torch.FloatTensor(csv_data[:,[s11_col,s12_col,s22_col]]).requires_grad_(True)
n_pts=int(len(uv_fem)/N_t)
model_dist=model_dist.to(device)
modev_uv=model_uv.to(device)

for i in range(N_t):
    start=i*n_pts
    end=(i+1)*n_pts
    print(start, end)
    xyt=xyt.to(device)
    uv_fem=uv_fem.to(device)
    ip=xyt[start:end,:]
    y=model_uv(ip)
    y=y.to(device)
    yd=model_dist(ip)
    yd=yd.to(device)
    yp=model_part(ip).to(device)
    y=y*yd+yp    
    ip=ip.cpu().detach().numpy()
    y=y.cpu().detach().numpy()
    fem_data_uv=uv_fem[start:end].cpu().detach().numpy()
    field=np.concatenate((ip,y),1)
    # postProcessDef(xmin=0, xmax=0.50, ymin=0, ymax=0.50, num=i, s=4, scale=0, field=field)
    # fem_data_s=s_fem[start:end].cpu().detach().numpy()
    y_s=y[:,2:]
    # FEMcomparisionUV(ip, y, fem_data_uv, MAX_T, N_t)
    # FEMcomparisionStr(ip, y_s, fem_data_s, MAX_T, N_t)

csv_data[:,y_col] = csv_data[:,y_col] +  0.25
csv_data[:,x_col] = csv_data[:,x_col] +  0.25



pts_on_hole = np.zeros((1,0), dtype= int)
for i in range(n_pts) :
    x_pt = csv_data[i][x_col]
    y_pt = csv_data[i][y_col]
    if x_pt**2+y_pt**2-0.01<=0.0001:
        print(x_pt**2+y_pt**2)
        pts_on_hole=np.append(pts_on_hole,int(i))

pts_on_line_v = np.zeros((1,0), dtype= int)
temp = np.zeros(0)
for i in range(n_pts) :
    x_pt = csv_data[i][x_col]
    y_pt = csv_data[i][y_col]
    if np.abs(x_pt-0.15)<=0.001:
        # print(x_pt**2+y_pt**2)
        if x_pt not in temp:
            temp=np.append(temp,x_pt)
            pts_on_line_v=np.append(pts_on_line_v,int(i))

pts_on_line_h = np.zeros((1,0), dtype= int)
temp = np.zeros(0)
for i in range(n_pts) :
    x_pt = csv_data[i][x_col]
    y_pt = csv_data[i][y_col]
    temp = np.zeros(0)
    if np.abs(y_pt-0.15)<=0.001:
        # print(x_pt**2+y_pt**2)
        if y_pt not in temp:
            temp=np.append(temp,y_pt)
            pts_on_line_h=np.append(pts_on_line_h,int(i))

arr_t = np.array([2, 4, 5.5])
out_names = ["$u/(m)$","$v/(m)$",r"$\sigma_{11}/(M Pa)$",r"$\sigma_{22}/(M Pa)$",r"$\sigma_{12}/(M Pa)$"]
png_names = ["u","v",r"s11",r"s22",r"s12"]

def sort_list(list1, list2):
    zipped_pairs = zip(list2, list1)
    z = [x for _, x in sorted(zipped_pairs)]
    return np.array(z)


for i in range(5):
    for t in arr_t:
        time_step = int(t/0.125)
        start = time_step*n_pts
        end = (time_step+1)*n_pts
        fem_hole = csv_data[start+pts_on_hole,:]
        xyt_hole =  fem_hole[:,[x_col,y_col,t_col]]
        xyt_hole = torch.FloatTensor(xyt_hole)
        xyt_hole = xyt_hole.to(device)
        y_fem_hole = torch.FloatTensor(fem_hole[:,[u_col,v_col,s11_col,s22_col,s12_col]]).to(device)
        y_pred_hole = model_part(xyt_hole).to(device) + model_dist(xyt_hole) * model_uv(xyt_hole)
        # print(y_pred_hole - y_fem_hole)
        xyt_hole=xyt_hole.cpu().detach().numpy()
        theta = np.flip(np.arctan(xyt_hole[:,1]/xyt_hole[:,0]),0)
        theta=theta*180/PI
        vals = np.flip(y_pred_hole.cpu().detach().numpy(),0)
        fem_vals = np.flip(y_fem_hole.cpu().detach().numpy(),0)

        X_Y_Spline = make_interp_spline(theta, vals[:,i])

        X_ = np.linspace(0, 90, 500)
        Y_ = X_Y_Spline(X_)

        # Plotting the Graph
        plt.plot(X_, Y_, label = f't = {t}, PINN')
        plt.scatter(theta , fem_vals[:,i], label = f't = {t}, FEM')
    plt.legend()
    plt.ylabel(out_names[i],fontsize=16)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=9)
    plt.xlabel(r"$\theta/degree$")
    plt.savefig(f'./output/hole_output_{png_names[i]}' + '.png', dpi=200)
    plt.close('all')

#Horizontal Line
for i in range(5):
    for t in arr_t:
        time_step = int(t/0.125)
        start = time_step*n_pts
        end = (time_step+1)*n_pts
        fem_line = csv_data[start+pts_on_line_h,:]
        # fem_line = fem_line[:,:]
        xyt_line =  fem_line[:,[x_col,y_col,t_col]]
        xyt_line = torch.FloatTensor(xyt_line)
        xyt_line = xyt_line.to(device)
        y_fem_line = torch.FloatTensor(fem_line[:,[u_col,v_col,s11_col,s22_col,s12_col]]).to(device)
        y_pred_line = model_part(xyt_line).to(device) + model_dist(xyt_line) * model_uv(xyt_line)
        # print(y_pred_hole - y_fem_hole)
        xyt_line=xyt_line.cpu().detach().numpy()
        # theta = np.flip(np.arctan(xyt_line[:,1]/xyt_line[:,0]),0)
        # theta=theta*180/PI
        # vals = np.flip(y_pred_line.cpu().detach().numpy(),0)
        # fem_vals = np.flip(y_fem_line.cpu().detach().numpy(),0)
        fem_vals = y_fem_line.cpu().detach().numpy()
        vals =  y_pred_line.cpu().detach().numpy()
        vals[:,i]= sort_list(vals[:,i],xyt_line[:,0])
        fem_vals = sort_list(fem_vals,  xyt_line[:,0])
        xyt_line[:,0]=sorted(xyt_line[:,0])
        X_Y_Spline = make_interp_spline(xyt_line[:,0], vals[:,i])


        X_ = np.linspace(0, 0.5, 500)
        Y_ = X_Y_Spline(X_)

        # Plotting the Graph
        plt.plot(X_, Y_, label = f't = {t}, PINN')
        plt.scatter(xyt_line[:,0] , fem_vals[::,i], label = f't = {t}, FEM')
    plt.legend()
    plt.ylabel(out_names[i],fontsize=16)
    plt.xlabel(r"$x$",fontsize=16)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=9)
    plt.savefig(f'./output/HLine_output_{png_names[i]}' + '.png', dpi=200)
    plt.close('all')

#Vertical Line
for i in range(5):
    for t in arr_t:
        time_step = int(t/0.125)
        start = time_step*n_pts
        end = (time_step+1)*n_pts
        fem_line = csv_data[start+pts_on_line_v,:]
        # fem_line = fem_line[:,:]
        xyt_line =  fem_line[:,[x_col,y_col,t_col]]
        xyt_line = torch.FloatTensor(xyt_line)
        xyt_line = xyt_line.to(device)
        y_fem_line = torch.FloatTensor(fem_line[:,[u_col,v_col,s11_col,s22_col,s12_col]]).to(device)
        y_pred_line = model_part(xyt_line).to(device) + model_dist(xyt_line) * model_uv(xyt_line)
        # print(y_pred_hole - y_fem_hole)
        xyt_line=xyt_line.cpu().detach().numpy()
        # theta = np.flip(np.arctan(xyt_line[:,1]/xyt_line[:,0]),0)
        # theta=theta*180/PI
        # vals = np.flip(y_pred_line.cpu().detach().numpy(),0)
        # fem_vals = np.flip(y_fem_line.cpu().detach().numpy(),0)
        fem_vals = y_fem_line.cpu().detach().numpy()
        vals =  y_pred_line.cpu().detach().numpy()
        vals[:,i]= sort_list(vals[:,i],xyt_line[:,1])
        fem_vals = sort_list(fem_vals,  xyt_line[:,1])
        xyt_line[:,1]=sorted(xyt_line[:,1])
        X_Y_Spline = make_interp_spline(xyt_line[:,1], vals[:,i])


        X_ = np.linspace(0, 0.5, 500)
        Y_ = X_Y_Spline(X_)

        # Plotting the Graph
        plt.plot(X_, Y_, label = f't = {t}, PINN')
        plt.scatter(xyt_line[:,1] , fem_vals[::,i], label = f't = {t}, FEM')
    plt.legend()
    plt.ylabel(out_names[i],fontsize=16)
    plt.xlabel(r"$y$",fontsize=16)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=9)
    plt.savefig(f'./output/VLine_output_{png_names[i]}' + '.png', dpi=200)
    plt.close('all')
# print(csv_data[end_inds])

# create_gif("plot")
# create_gif("error")

# y=model_part(xyt)+model_dist(xyt)*model_uv(xyt)
# uv_fem=uv_fem.cpu().detach().numpy()
# y=y.cpu().detach().numpy()
# s_fem = s_fem.cpu().detach().numpy()
# rmse_u=mean_squared_error(uv_fem[:,0],y[:,0],squared=False)
# rmse_v=mean_squared_error(uv_fem[:,1],y[:,1],squared=False)
# rmse_s11=mean_squared_error(s_fem[:,0],y[:,2],squared=False)
# rmse_s22=mean_squared_error(s_fem[:,2],y[:,3],squared=False)
# rmse_s12=mean_squared_error(s_fem[:,1],y[:,4],squared=False)