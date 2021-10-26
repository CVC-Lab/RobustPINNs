import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader 
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from Utils import *
import numpy as np


class PINN(nn.Module):
    def __init__(self, LBs, UBs, Layers, N0, Nb,
                 InError = 0., Activation = nn.Tanh(), device = 'cpu', do_smoothing = False):
        super(PINN, self).__init__()
        self.LBs = torch.tensor(LBs, dtype=torch.float32).to(device)
        self.UBs = torch.tensor(UBs, dtype=torch.float32).to(device)
        self.Layers = Layers
        self.in_dim  = Layers[0]
        self.out_dim = Layers[-1]
        self.N0 = N0
        self.Nb = Nb
        self.InError = InError
        self.Activation = Activation
        self.do_smoothing = do_smoothing
        self.XT0, self.u0_true, self.v0_true, self.u0, self.v0  = InitialCondition(self.N0, LBs[0], UBs[0], InError)
        if do_smoothing:
            self.u0, self.v0, self.GP_U, self.GP_V = self.Smoothing(self.XT0, self.u0, self.v0)
        else:
            self.GP_U, self.GP_V = None, None
        self.XT0 = self.XT0.to(device)
        self.u0 = self.u0.to(device) 
        self.v0 = self.v0.to(device)
        self.XTbL, self.XTbU = BoundaryCondition(self.Nb, self.LBs[0], self.UBs[0])
        self.XTbL = self.XTbL.to(device) 
        self.XTbU = self.XTbU.to(device)
        #self.XT_Grid = MeshGrid(self.LBs, self.UBs, self.Nf)
        self.device = device
        #print(self.device)
        self._nn = self.build_model()
        self._nn.to(self.device)
        self.Loss = torch.nn.MSELoss(reduction='mean')
    
    def build_model(self):
        Seq = nn.Sequential()
        for ii in range(len(self.Layers)-1):
            this_module = nn.Linear(self.Layers[ii], self.Layers[ii+1])
            nn.init.xavier_normal_(this_module.weight)
            Seq.add_module("Linear" + str(ii), this_module)
            if not ii == len(self.Layers)-2:
                Seq.add_module("Activation" + str(ii), self.Activation)
        return Seq

    
    def Smoothing(self, XT0, u0, v0):
        X = XT0[:, 0].reshape(-1, 1)
        kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) + \
                 WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+2))
        GP_U = GPR(kernel = kernel, alpha = 0.0).fit(X, u0)
        print(GP_U.kernel_)
        GP_V = GPR(kernel = kernel, alpha = 0.0).fit(X, v0)
        print(GP_V.kernel_)
        U0 = torch.tensor(GP_U.predict(X), dtype = torch.float32).reshape(-1)
        V0 = torch.tensor(GP_V.predict(X), dtype = torch.float32).reshape(-1)
        return U0, V0, GP_U, GP_V
        
    
    def forward(self, x):
        #print(x.device)
        x = x.reshape((-1,self.in_dim))  
        x = 2*(x - self.LBs)/(self.UBs - self.LBs) - 1.0
        x = x.to(self.device)
        #print(x.device)
        return torch.reshape(self._nn.forward(x), (-1, self.out_dim))

    def ICLoss(self):
        if self.do_smoothing and (not self.GP_U == None) and (not self.GP_V == None):
            XT0 = torch.cat( (torch.rand(self.N0,1).uniform_(self.LBs[0], self.UBs[0]),
                              self.LBs[1]*torch.ones((self.N0,1))), 
                             dim = 1 
                           ).to(self.device)
            u0 = torch.tensor(self.GP_U.predict(XT0[:,0].reshape(-1,1)), dtype = torch.float32 ).reshape(-1).to(self.device)
            v0 = torch.tensor(self.GP_V.predict(XT0[:,0].reshape(-1,1)), dtype = torch.float32 ).reshape(-1).to(self.device)
        else:
            XT0 = self.XT0
            u0  = self.u0
            v0 =  self.v0                
        UV0_pred = self.forward(XT0)
        u0_pred = UV0_pred[:,0].reshape(-1)
        v0_pred = UV0_pred[:,1].reshape(-1)
        return self.Loss(u0_pred, u0) + self.Loss(v0_pred, v0)
                    #)torch.mean((u0_pred - u0)**2) + torch.mean((v0_pred - v0)**2)

    def BCLoss(self):
        UVb_L, UVb_U = self.forward(self.XTbL), self.forward(self.XTbU)
        ub_l, vb_l = UVb_L[:, 0], UVb_L[:, 1]
        ub_u, vb_u = UVb_U[:, 0], UVb_U[:, 1]
        ub_l_x = torch.autograd.grad(outputs=ub_l.to(self.device), 
                                     inputs=self.XTbL, 
                                     grad_outputs=torch.ones(ub_l.shape).to(self.device), 
                                     create_graph = True,
                                     allow_unused=True)[0][:,0]
    
        vb_l_x = torch.autograd.grad(outputs=vb_l.to(self.device), 
                                     inputs=self.XTbL, 
                                     grad_outputs=torch.ones(vb_l.shape).to(self.device),
                                     create_graph = True,
                                     allow_unused=True)[0][:,0]
    
        ub_u_x = torch.autograd.grad(outputs=ub_u.to(self.device), 
                                     inputs=self.XTbU, 
                                     grad_outputs=torch.ones(ub_u.shape).to(self.device), 
                                     create_graph = True,
                                     allow_unused=True)[0][:,0]
    
        vb_u_x = torch.autograd.grad(outputs=vb_u.to(self.device), 
                                     inputs=self.XTbU, 
                                     grad_outputs=torch.ones(vb_u.shape).to(self.device), 
                                     create_graph = True,
                                     allow_unused=True)[0][:,0]
        return self.Loss(ub_l, ub_u) + self.Loss(vb_l, vb_u) + \
               self.Loss(ub_l_x, ub_u_x) + self.Loss(vb_l_x, vb_u_x)

    def PhysicsLoss(self, XTGrid):
        XTGrid = XTGrid.to(self.device)
        UVf = self.forward(XTGrid)
        uf, vf = UVf[:, 0], UVf[:, 1]
        uf_t = torch.autograd.grad(outputs=uf.to(self.device), 
                                   inputs=XTGrid, 
                                   grad_outputs=torch.ones(uf.shape).to(self.device), 
                                   create_graph = True,
                                   allow_unused=True)[0][:,1]
        vf_t = torch.autograd.grad(outputs=vf.to(self.device), 
                                   inputs=XTGrid, 
                                   grad_outputs=torch.ones(uf.shape).to(self.device),
                                   create_graph = True,
                                   allow_unused=True)[0][:,1]
        uf_x = torch.autograd.grad(outputs=uf.to(self.device), 
                                   inputs=XTGrid, 
                                   grad_outputs=torch.ones(uf.shape).to(self.device),
                                   create_graph = True,
                                   allow_unused=True)[0][:,0]
        uf_xx = torch.autograd.grad(outputs=uf_x.to(self.device), 
                                   inputs=XTGrid, 
                                   grad_outputs=torch.ones(uf_x.shape).to(self.device),
                                   create_graph = True,
                                   allow_unused=True)[0][:,0]
        vf_x = torch.autograd.grad(outputs=vf.to(self.device), 
                                   inputs=XTGrid, 
                                   grad_outputs=torch.ones(vf.shape).to(self.device),
                                   create_graph = True,
                                   allow_unused=True)[0][:,0]
        vf_xx = torch.autograd.grad(outputs=vf_x.to(self.device), 
                                    inputs=XTGrid, 
                                    grad_outputs=torch.ones(vf_x.shape).to(self.device),
                                    create_graph = True,
                                    allow_unused=True)[0][:,0]
        lossf =  torch.mean((0.5*uf_xx - vf_t + (uf**2 + vf**2)*uf)**2 )  + \
                 torch.mean((0.5*vf_xx + uf_t + (uf**2 + vf**2)*vf)**2 )
    
        return lossf
