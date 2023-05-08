import numpy as np
import os
from pyDOE import lhs
import matplotlib
import platform
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


#### Creating part Modelwith Pytorch

class ANN_PartModel(nn.Module):
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


#### Creating Modelwith Pytorch

class ANN_UvModel(nn.Module):
    def __init__(self, N_INPUT=3, N_OUTPUT=5, N_HIDDEN=70, N_LAYERS=8):
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

def GenDistPt(xmin, xmax, ymin, ymax, tmin, tmax, xc, yc, r, num, num_t):
    # num: number per edge
    # num_t: number time step
    x = np.linspace(xmin, xmax, num=num)
    y = np.linspace(ymin, ymax, num=num)
    x, y = np.meshgrid(x, y)
    x = x.flatten()[:, None]
    y = y.flatten()[:, None]
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



#installing GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Current cuda device: ',torch.cuda.get_device_name(0))


E = 20.0
mu = 0.25
rho = 1.0
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
                                     num=30, num_t=30)
XYT_dist = np.concatenate((x_dist, y_dist, t_dist), 1)
XYT_dist,DIST = GenDist(XYT_dist)
print(XYT_dist.shape)
IC = lb + np.array([0.5, 0.5, 0.0]) * lhs(3, 8000)

XYT_c = lb + (ub - lb) * lhs(3, 40000)


LW = np.array([0.1, 0.0, 0.0]) + np.array([0.4, 0.0, 10]) * lhs(3, 8000)
UP = np.array([0.0, 0.5, 0.0]) + np.array([0.5, 0.0, 10]) * lhs(3, 8000)
LF = np.array([0.0, 0.1, 0.0]) + np.array([0.0, 0.4, 10]) * lhs(3, 8000)
RT = np.array([0.5, 0.0, 0.0]) + np.array([0.0, 0.5, 10]) * lhs(3, 8000)

t_RT = RT[:, 2:3]
period = 5  # two period in 10s
s11_RT = 0.5 * np.sin((2 * PI / period) * t_RT + 3 * PI / 2) + 0.5
# s11_RT=np.ones(RT[:,2:3].shape)
RT = np.concatenate((RT, s11_RT), 1)

def sigma11(time):
    return 0.5 * torch.sin((2 * PI / period) * time + 3 * PI / 2) + 0.5

def model_part(xyt):
    res=torch.zeros(list(xyt.size())[0],5)
    res[:,2]=sigma11(xyt[:,2])
    return torch.FloatTensor(res).requires_grad_(True)

# model_part.to(device)
# Add some boundary points into the collocation point set
XYT_c = np.concatenate((XYT_c, LF[::2, :], RT[::2, 0:3], UP[::2, :], LW[::2, :]), 0)
IC_dist=lb + np.array([0.5, 0.5, 0.0]) * lhs(3, XYT_dist.shape[0])
# print(XYT_c.shape)
XYT_dist=torch.FloatTensor(XYT_dist).requires_grad_(True)
DIST=torch.FloatTensor(DIST).requires_grad_(True)
IC=torch.FloatTensor(IC).requires_grad_(True)
LW=torch.FloatTensor(LW).requires_grad_(True)
LF=torch.FloatTensor(LF).requires_grad_(True)
UP=torch.FloatTensor(UP).requires_grad_(True)
RT=torch.FloatTensor(RT).requires_grad_(True)
XYT=torch.FloatTensor(XYT_c).requires_grad_(True)
IC_dist=torch.FloatTensor(IC_dist).requires_grad_(True)

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
optimizer_dist = torch.optim.Adam(model_dist.parameters(),lr=8e-4)

for i in range(500):
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

# class Build_Data_part(Dataset):
#     # Constructor
#     def __init__(self):
#         self.x = RT
#         self.y = LW
#         self.z=LF
#         self.a=UP
#         self.b=IC
#         self.len = self.x.shape[0]
#     # Getting the data
#     def __getitem__(self, index):
#         return self.x[index], self.y[index],self.z[index], self.a[index],self.b[index]
#     # Getting length of the data
#     def __len__(self):
#         return self.len

# train_part = Build_Data_part()
# train_loader_part = DataLoader(train_part, batch_size=32,shuffle=True)

# torch.manual_seed(123)
# model_part = ANN_PartModel()
# if os.path.isfile('./part_time.pth'):
#     model_part.load_state_dict(torch.load('./part_time.pth'))
#     print("loaded part model successfully...")

# model_part.to(device)
# optimizer_part = torch.optim.Adam(model_part.parameters(),lr=1e-3)

# for i in range(100):
#     running_loss=0.0
#     for _, [rt,lw,lf,up,ic] in enumerate(train_loader_part):
#         optimizer_part.zero_grad()
        
#         # compute the "data loss"
#         #IC
#         ic=ic.to(device)
#         yh_ic = model_part(ic)
#         loss1 = torch.mean((yh_ic[:,0])**2)+torch.mean((yh_ic[:,1])**2)+torch.mean((yh_ic[:,2])**2)+torch.mean((yh_ic[:,3])**2)+torch.mean((yh_ic[:,4])**2)# use mean squared error

#         #LF
#         lf=lf.to(device)
#         yh_lf = model_part(lf)
#         loss1=loss1+torch.mean((yh_lf[:,0])**2)+torch.mean((yh_lf[:,4])**2);

#         #UP
#         up=up.to(device)
#         yh_up = model_part(up)
#         loss1=loss1+torch.mean((yh_up[:,4])**2)+torch.mean((yh_up[:,3])**2);
#         #LW
#         lw=lw.to(device)
#         yh_lw = model_part(lw)
#         loss1=loss1+torch.mean((yh_lw[:,1])**2)+torch.mean((yh_lw[:,4])**2)

#         #RT
#         rt=rt.to(device)
#         yh_rt = model_part(rt[:,0:3])
#         loss1=loss1+torch.mean((yh_rt[:,4])**2)
#         loss1=loss1+torch.mean((yh_rt[:,2]-rt[:,3])**2)

        
#         # # compute the "physics loss"
#         yhp = model_part(ic)
#         du_dt  = torch.autograd.grad(yhp[:,0], ic,torch.ones_like(yhp[:,0]), create_graph=True)[0][:,2]# computes dy/dx
#         dv_dt  = torch.autograd.grad(yhp[:,1], ic,torch.ones_like(yhp[:,1]), create_graph=True)[0][:,2]
#         loss2 =torch.mean(dv_dt**2) + torch.mean(du_dt**2) 

#         loss = 1000*(loss1 + loss2)# add two loss terms together
#         loss.backward()
#         optimizer_part.step()

#         running_loss+=loss.item()
#         if _%100==99: 
#             print('[%d, %5d] loss: %.4f' %
#                 (i + 1, _ + 1, running_loss / 100))
#             running_loss = 0.0
#             torch.save(model_part.state_dict(), "./part_time.pth")



torch.manual_seed(123)
model_uv = ANN_UvModel()
if os.path.isfile('./uv_time.pth'):
    model_uv.load_state_dict(torch.load('./uv_time.pth'))
    print("loaded uv model successfully...")

model_uv.to(device)
train_loader = torch.utils.data.DataLoader(XYT, batch_size=32,shuffle=True)
optimizer_uv = torch.optim.Adam(model_uv.parameters(),lr=1e-4)

for i in range(500):
    running_loss=0.0
    loss_plate_stress=0.0 
    loss_plate_uv=0.0
    for _, xyt in enumerate(train_loader):

        optimizer_uv.zero_grad()
        xyt=xyt.to(device)
        # compute the "data loss"
        yh = model_uv(xyt)
        yh=yh*model_dist(xyt)+model_part(xyt).to(device)

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
        
        # backpropagate joint loss
        # loss=loss1
        
        loss = loss1 +loss2# add two loss terms together 
        
        loss=10*loss
        loss.backward()
        optimizer_uv.step()


        running_loss+=loss.item()
        loss_plate_stress+=loss1.item()
        loss_plate_uv+=loss2.item()


        if _%100==99: 
            print("palte loss stress: ",loss_plate_stress/100,"palte loss uv: ",loss_plate_uv/100)
            torch.save(model_uv.state_dict(), "./uv_time.pth")
            print('[%d, %5d] loss: %.4f' %
                (i + 1, _ + 1, running_loss / 100))
            running_loss = 0.0
            loss_plate_stress = 0.0
            loss_plate_uv=0.0


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
                          cmap='rainbow', marker='o', s=s,vmin=0,vmax=0.025)
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
                          cmap='rainbow', marker='o', s=s,vmin=-0.007,vmax=0)
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
                          marker='s', cmap='rainbow', s=s,vmin=0,vmax=1)
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

def create_gif():
    frames_stress = []
    frames_uv=[]
    for t in range(N_t):
        image_stress = imageio.v2.imread(f'./output/stress_comparison_time{t}.png')
        image_uv = imageio.v2.imread(f'./output/uv_comparison_time{t}.png')
        frames_stress.append(image_stress)
        frames_uv.append(image_uv)
    imageio.mimsave('./stress_pinn.gif', 
            frames_stress, 
            fps = 5, 
            loop = 1)

    imageio.mimsave('./uv_pinn.gif', 
            frames_uv, 
            fps = 5, 
            loop = 1)


import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
x_star = np.linspace(0, 0.5, 251)
y_star = np.linspace(0, 0.5, 251)
x_star, y_star = np.meshgrid(x_star, y_star)
x_star = x_star.flatten()[:, None]
y_star = y_star.flatten()[:, None]

shutil.rmtree('./output', ignore_errors=True)
if not os.path.isdir('./output'):
    os.makedirs('./output')


for i in range(N_t):
    t_star = np.zeros((x_star.size, 1))
    t_star.fill(i * MAX_T / (N_t - 1))
    xyt=np.concatenate((x_star,y_star,t_star),1)
    xyt=torch.FloatTensor(xyt).requires_grad_(True)
    xyt=xyt.to(device)
    y=model_uv(xyt)
    yd=model_dist(xyt)
    yp=model_part(xyt).to(device)
    y=y*yd+yp
    xyt=xyt.cpu().detach().numpy()
    y=y.cpu().detach().numpy()
    field=np.concatenate((xyt,y),1)
    postProcessDef(xmin=0, xmax=0.50, ymin=0, ymax=0.50, num=i, s=4, scale=0, field=field)

create_gif()