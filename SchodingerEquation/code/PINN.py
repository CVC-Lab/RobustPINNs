import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader 
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import numpy as np
from scipy import io
import argparse
import os
import pickle

data = io.loadmat("../dataset/NLS.mat")
t = torch.tensor(data['tt'], dtype = torch.float32).reshape(-1)
x = torch.tensor(data['x'], dtype = torch.float32).reshape(-1)
Exact = data['uu']
Exact_u = torch.tensor(np.real(Exact), dtype = torch.float32)
Exact_v = torch.tensor(np.imag(Exact), dtype = torch.float32)
Exact_h = torch.sqrt(Exact_u**2 + Exact_v**2)
X, T = np.meshgrid(x,t)
X_star = torch.tensor(np.hstack((X.flatten()[:,None], T.flatten()[:,None])), dtype = torch.float32)
u_star = torch.flatten(torch.transpose(Exact_u,0,1))
v_star = torch.flatten(torch.transpose(Exact_v,0,1))
h_star = torch.flatten(torch.transpose(Exact_h,0,1))

def InitialCondition(N0, LB, UB, InError = 0.):
    indices = (X_star[:,0] >= LB) & (X_star[:,0] < UB) & (X_star[:,1] == 0.)
    XT0 = X_star[indices]
    u0 = u_star[indices]
    v0 = v_star[indices]
    indices = np.random.choice(XT0.shape[0], N0, replace=False)
    XT0 = XT0[indices]
    u0 = u0[indices]
    v0 = v0[indices]
    return XT0, u0, v0, u0 + InError*torch.randn_like(u0), v0 + InError*torch.randn_like(v0)

def BoundaryCondition(Nb, LB, UB, device):
    ## Choose Nb time instances on x = LB and x = UB
    tb_indices = np.random.choice(t.shape[0], Nb, replace=False)
    tb = t[tb_indices].to(device)
    XTL = torch.cat(( LB*torch.ones((Nb,1),device=device,dtype=torch.float32), tb.reshape(-1,1)), dim = 1)
    XTL.requires_grad_()
    XTU = torch.cat(( UB*torch.ones((Nb,1),device=device,dtype=torch.float32), tb.reshape(-1,1)), dim = 1)
    XTU.requires_grad_()
    return  XTL, XTU

def MeshGrid(LBs, UBs, Nf):
    n = round(Nf**0.5)
    XGrid, TGrid = np.meshgrid(np.arange(float(LBs[0]), float(UBs[0]), float((UBs[0] - LBs[0])/n)), 
                               np.arange(float(LBs[1]), float(UBs[1]), float((UBs[1] - LBs[1])/n)))
    XTGrid = np.append(XGrid.reshape(1,-1), TGrid.reshape(1,-1), axis = 0).T
    #xt_grid_indices = np.random.choice(XTGrid.shape[0], Nf, replace=False)
    # xt_f = torch.tensor(XTGrid[xt_grid_indices], dtype = torch.float32, requires_grad=True)
    xt_f = torch.tensor(XTGrid, dtype = torch.float32, requires_grad=True)
    return xt_f

def SewingBoundary(x_value, LB, UB, Ni):
    TGrid = np.arange(LB, UB, (UB - LB)/(Ni*10))
    boundary_indices = np.random.choice(TGrid.shape[0], Ni, replace = False)
    XTGrid = np.append(float(x_value)*np.ones((1, TGrid.shape[0])), TGrid.reshape(1,-1), axis = 0).T
    return torch.tensor(XTGrid[boundary_indices], dtype = torch.float32, requires_grad=True)



class PINN(nn.Module):
    def __init__(self, LBs, UBs, Layers, N0, Nb, Nf, Nt, 
                 InError = 0., Activation = nn.Tanh(), 
                 model_name = "PINN.model", device = 'cpu',
                 do_smoothing = False, N0pool = 0,
                 threshold = 0.9, display_freq = 100, do_consLaw = False):
        super(PINN, self).__init__()
        self.LBs = torch.tensor(LBs, dtype=torch.float32).to(device)
        self.UBs = torch.tensor(UBs, dtype=torch.float32).to(device)
        self.device = device
        #print(self.LBs)
        #print(self.UBs)
        self.Layers = Layers
        self.in_dim  = Layers[0]
        self.out_dim = Layers[-1]
        self.N0 = N0
        self.Nb = Nb
        self.Nf = Nf
        self.Nt = Nt
        self.InError = InError
        self.Activation = Activation
        self.do_smoothing = do_smoothing
        self.do_consLaw = do_consLaw
        self.N0pool = np.maximum(N0pool, N0)
        self.threshold = threshold
        self.XT0, self.u0_true, self.v0_true, self.u0, self.v0  = InitialCondition(self.N0pool, LBs[0], UBs[0], InError)
        if do_smoothing:
            self.u0, self.v0, self.GP_U, self.GP_V, self.u_selections, self.v_selections, self.IP_U_indices, self.IP_V_indices = self.Smoothing(self.XT0, self.u0, self.v0)
        else:
            self.GP_U, self.GP_V, self.u_selections, self.v_selection, self.IP_U_indices, self.IP_V_indices = None, None, self.N0, self.N0, None, None
        self.XT0 = self.XT0.to(device)
        self.u0 = self.u0.to(device) 
        self.v0 = self.v0.to(device)
        self.XTbL, self.XTbU = BoundaryCondition(self.Nb, self.LBs[0], self.UBs[0], self.device)
        self.XTbL = self.XTbL.to(device) 
        self.XTbU = self.XTbU.to(device)
        #self.XT_Grid = MeshGrid(self.LBs, self.UBs, self.Nf)
        #print(self.device)
        self._nn = self.build_model()
        self._nn.to(self.device)
        self.Loss = torch.nn.MSELoss(reduction='mean')
        self.model_name = model_name
        self.display_freq = display_freq
    
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

        n0 = int(self.N0/2)
        
        GP_U = GPR(kernel = kernel, alpha = 0.0).fit(X[0:n0], u0[0:n0])
        print(GP_U.kernel_)
        GP_V = GPR(kernel = kernel, alpha = 0.0).fit(X[0:n0], v0[0:n0])
        print(GP_V.kernel_)

        u_selections = n0
        v_selections = n0

        IP_U_indices = list(range(n0))
        IP_V_indices = list(range(n0))

        u_kernel = GP_U.kernel_.get_params()["k1__k2"]
        v_kernel = GP_V.kernel_.get_params()["k1__k2"]
        
        for i in range(n0+1, self.N0pool):
            x = np.array([X[i].tolist()])
            K_U = u_kernel.__call__(x, X[IP_U_indices])
            K_V = v_kernel.__call__(x, X[IP_V_indices])
            if (np.max(K_U) < self.threshold or self.N0pool == self.N0) and u_selections <= self.N0:
                IP_U_indices.append(i)
                u_selections = u_selections + 1
            if (np.max(K_V) < self.threshold or self.N0pool == self.N0) and v_selections <= self.N0:
                IP_V_indices.append(i)
                v_selections = v_selections + 1

        print("Number of IPs chosen for u: {}".format(len(IP_U_indices)))
        print("Number of IPs chosen for v: {}".format(len(IP_V_indices)))
        GP_U = GP_U.fit(X[IP_U_indices], u0[IP_U_indices])
        GP_V = GP_V.fit(X[IP_V_indices], v0[IP_V_indices])               

        U0 = torch.tensor(GP_U.predict(X), dtype = torch.float32).reshape(-1)
        V0 = torch.tensor(GP_V.predict(X), dtype = torch.float32).reshape(-1)
        
        return U0, V0, GP_U, GP_V, u_selections, v_selections, IP_U_indices, IP_V_indices
        
    
    def forward(self, x):
        #print(x.device)
        #print(x.dtype)
        x = x.to(self.device)
        x = x.reshape((-1,self.in_dim))  
        x = 2*(x - self.LBs)/(self.UBs - self.LBs) - 1.0
        #print(x.device)
        out = torch.reshape(self._nn.forward(x), (-1, self.out_dim))
        #print(out.dtype)
        return out

    def ICLoss(self):
        if self.do_smoothing and (not self.GP_U == None) and (not self.GP_V == None):
            XT0 = torch.cat( (torch.rand(self.N0,1).uniform_(self.LBs[0], self.UBs[0]),
                              self.LBs[1].cpu()*torch.ones((self.N0,1))), 
                             dim = 1 ).to(torch.device('cpu'))
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
        lossf =  torch.mean((0.5*uf_xx - vf_t + (uf**2 + vf**2)*uf)**2 + (0.5*vf_xx + uf_t + (uf**2 + vf**2)*vf)**2 )

        if self.do_consLaw:
            # n = round(XTGrid.shape[0]**0.5)
            # if len(np.unique(XTGrid[:,1].cpu().reshape(-1).numpy()) == n:
            #        reshape_size = n
            # elif len(np.unique(XTGrid[:,1].cpu().reshape(-1).numpy()) == n+1:
            #        reshape_size = n+1
            reshape_size = len(np.unique(XTGrid[:,1].cpu().reshape(-1).detach().numpy()))
            law = torch.mean((uf**2+vf**2).reshape(reshape_size,-1), dim=1)
            #law1_initial = law1[0]
            lossc = torch.mean((law)**2)
            return lossf + lossc
        else:
            return lossf

class cPINN:
    def __init__(self, boundaries, t_domain, Layers, N0, Nb, Nf, Ni, Nt, #optimizer,
                 InError = 0., Activation = nn.Tanh(),
                 model_name = "cPINN.model", device = 'cpu',
                 do_smoothing = False, do_consLaw = False, N0pool = 0,
                 threshold = 0.9, display_freq = 100,
                 do_regularize = False, regmode = 'L2', regparam = 1.e-4):
        self.boundaries = torch.tensor(boundaries).to(device)
        self.tLow  = t_domain[0]
        self.tHigh = t_domain[1]
        self.Layers = Layers
        self.in_dim  = Layers[0]
        self.out_dim = Layers[-1]
        self.N0 = N0
        self.Nb = Nb
        self.Nf = Nf
        self.Ni = Ni
        self.Nt = Nt        
        self.N0pool = np.maximum(N0pool, N0)
        self.display_freq = display_freq
        self.threshold = threshold
        self.do_smoothing = do_smoothing
        self.do_consLaw = do_consLaw
        self.do_regularize = do_regularize
        self.regmode = regmode
        self.regparam = regparam
        self.InError = InError
        self.Activation = Activation
        self.device = device
        self.model_name = model_name
        #print("SPINN = ", self.device)
        self.Loss = torch.nn.MSELoss(reduction='mean')
        self.PINNs = self.build_model(boundaries, Layers, N0, Nb, Nf, Nt)
        
    def build_model(self, boundaries, Layers, N0, Nb, Nf, Nt):
        list_PINNs = []
        for ii in range(len(boundaries) - 1):
            LBs, UBs = [boundaries[ii], self.tLow ], [boundaries[ii + 1], self.tHigh]
            list_PINNs.append(PINN(LBs, UBs, Layers, N0, Nb, Nf, Nt, self.InError, self.Activation, self.model_name, self.device, self.do_smoothing, self.N0pool, self.threshold, self.display_freq, self.do_consLaw))
        return list_PINNs
    
    def parameters(self):
        list_params = []
        for pinn in self.PINNs:
            list_params += list(pinn.parameters())
        return list_params

    def regLoss(self):
        loss = torch.tensor(0.0, dtype = torch.float32, device=self.device, requires_grad = True)
        list_params = self.parameters()
        for param in list_params:
            if self.regmode == 'L1':
                loss = loss + param.abs().sum()
            elif self.regmode == 'L2':
                loss = loss + (param**2).sum()
        return loss
    
    def BoundaryLoss(self):
        XTbL, XTbU = BoundaryCondition(self.Nb, self.boundaries[0], self.boundaries[-1], self.device)
        UVb_L, UVb_U = self.PINNs[0].forward(XTbL), self.PINNs[-1].forward(XTbU)
        ub_l, vb_l = UVb_L[:, 0], UVb_L[:, 1]
        ub_u, vb_u = UVb_U[:, 0], UVb_U[:, 1]
        ub_l_x = torch.autograd.grad(outputs=ub_l.to(self.device), 
                                     inputs=XTbL, 
                                     grad_outputs=torch.ones(ub_l.shape).to(self.device), 
                                     create_graph = True,
                                     allow_unused=True)[0][:,0]
    
        vb_l_x = torch.autograd.grad(outputs=vb_l.to(self.device), 
                                     inputs=XTbL, 
                                     grad_outputs=torch.ones(vb_l.shape).to(self.device),
                                     create_graph = True,
                                     allow_unused=True)[0][:,0]
    
        ub_u_x = torch.autograd.grad(outputs=ub_u.to(self.device), 
                                     inputs=XTbU, 
                                     grad_outputs=torch.ones(ub_u.shape).to(self.device), 
                                     create_graph = True,
                                     allow_unused=True)[0][:,0]
    
        vb_u_x = torch.autograd.grad(outputs=vb_u.to(self.device), 
                                     inputs=XTbU, 
                                     grad_outputs=torch.ones(vb_u.shape).to(self.device), 
                                     create_graph = True,
                                     allow_unused=True)[0][:,0]
        return self.Loss(ub_l, ub_u) + self.Loss(vb_l, vb_u) + \
               self.Loss(ub_l_x, ub_u_x) + self.Loss(vb_l_x, vb_u_x)
    
    
    def InterfaceLoss(self):
        loss = torch.tensor(0.0, dtype = torch.float32, device=self.device, requires_grad = True)
        if len(self.boundaries) == 2:
            return loss
        else:
            for ii in range(len(self.boundaries)-2):
                XTI = SewingBoundary(self.boundaries[ii+1], self.tLow, self.tHigh, self.Ni)
                UV_0 = self.PINNs[ii].forward(XTI)
                UV_1 = self.PINNs[ii+1].forward(XTI)
                ui_0, vi_0 = UV_0[:, 0], UV_0[:, 1]
                ui_1, vi_1 = UV_1[:, 0], UV_1[:, 1]
                
                ui0_x = torch.autograd.grad(outputs=ui_0.to(self.device), 
                                            inputs=XTI, 
                                            grad_outputs=torch.ones(ui_0.shape).to(self.device), 
                                            create_graph = True,
                                            allow_unused=True)[0][:,0]
                vi0_x = torch.autograd.grad(outputs=vi_0.to(self.device), 
                                            inputs=XTI, 
                                            grad_outputs=torch.ones(vi_0.shape).to(self.device), 
                                            create_graph = True,
                                            allow_unused=True)[0][:,0]
                ui1_x = torch.autograd.grad(outputs=ui_1.to(self.device), 
                                            inputs=XTI, 
                                            grad_outputs=torch.ones(ui_1.shape).to(self.device), 
                                            create_graph = True,
                                            allow_unused=True)[0][:,0]
                vi1_x = torch.autograd.grad(outputs=vi_1.to(self.device), 
                                            inputs=XTI, 
                                            grad_outputs=torch.ones(vi_1.shape).to(self.device), 
                                            create_graph = True,
                                            allow_unused=True)[0][:,0]
                loss = loss + self.Loss(ui_0, ui_1)   + self.Loss(vi_0, vi_1) + \
                        self.Loss(ui0_x, ui1_x) + self.Loss(vi0_x, vi1_x)
                
            return loss
    
    def Eval(self, xt):
        xt = xt.to(self.device)
        to_return = torch.zeros(xt.shape[0], 2).to(self.device)
        for ii in range(len(self.boundaries) - 1):
            indices = (xt[:,0] >= self.boundaries[ii]) & (xt[:,0] < self.boundaries[ii+1])
            this_xt = xt[indices]
            this_uv = self.PINNs[ii].forward(this_xt)
            to_return[indices] += this_uv
        return to_return
            

    def Train(self, n_iters, weights=(1.0,1.0,1.0,1.0)):
        params = list(self.parameters())
        optimizer = optim.Adam(params, lr=1e-3)
        min_loss = 999999.0
        #message_print_count = min(n_iters, 100)
        Training_Losses = []
        Test_Losses = []
        #LBs, UBs = [self.boundaries[0], self.tLow], [self.boundaries[1], self.tHigh]
        #XTGrid = MeshGrid(LBs, UBs, self.Nf)
        for jj in range(n_iters):
            #Total_Loss = 0.0 #torch.tensor(0.0, dtype = torch.float32, device=self.device, requires_grad = True)
            Total_ICLoss = torch.tensor(0.0, dtype = torch.float32, device=self.device, requires_grad = True)
            Total_BCLoss = torch.tensor(0.0, dtype = torch.float32, device=self.device, requires_grad = True)
            Total_PhysicsLoss = torch.tensor(0.0, dtype = torch.float32, device=self.device, requires_grad = True)
            for ii in range(len(self.boundaries) - 1):
                  LBs, UBs = [self.boundaries[ii], self.tLow], [self.boundaries[ii + 1], self.tHigh]
                  XTGrid = MeshGrid(LBs, UBs, self.Nf)
                  Total_ICLoss = Total_ICLoss + self.PINNs[ii].ICLoss()
                  #Total_BCLoss = Total_BCLoss + self.PINNs[ii].BCLoss() 
                  Total_PhysicsLoss = Total_PhysicsLoss + self.PINNs[ii].PhysicsLoss(XTGrid)
            #Total_ICLoss = self.PINNs[0].ICLoss()
            Total_BCLoss = self.BoundaryLoss()
            InterfaceLoss = self.InterfaceLoss()
            Total_Loss = weights[0]*Total_ICLoss + weights[1]*Total_BCLoss\
                       + weights[2]*Total_PhysicsLoss + weights[3]*InterfaceLoss
            if self.do_regularize:
                RegLoss =  self.regparam * self.regLoss()
                Total_Loss = Total_Loss + RegLoss
            optimizer.zero_grad()
            Total_Loss.backward()
            optimizer.step()
            Training_Losses.append(float(Total_Loss))
            if jj > int(n_iters/4):
                if Total_Loss < min_loss:
                    torch.save(self, self.model_name)
                    min_loss = float(Total_Loss)
            indices = np.random.choice(X_star.shape[0], self.Nt, replace=False)
            Test_XT = X_star[indices]
            Test_UV = self.Eval(Test_XT)
            #print(u_star[indices].reshape(-1))
            #print(Test_UV[:,0].reshape(-1))
            #print(v_star[indices].reshape(-1))
            #print(Test_UV[:,1].reshape(-1))
            u_exact = u_star[indices].reshape(-1).to(self.device)
            v_exact = v_star[indices].reshape(-1).to(self.device)
            Test_Loss = torch.mean( (u_exact - Test_UV[:,0].reshape(-1))**2 + \
                                    (v_exact - Test_UV[:,1].reshape(-1))**2 
                                  )
            Test_Losses.append(float(Test_Loss))
            # Training_Losses.append(float(Total_Loss))
            if jj % self.display_freq == self.display_freq - 1 or jj == n_iters - 1 or jj == 0:
                print("Iteration Number = {}".format(jj+1))
                print("\tIC Loss = {}".format(float(Total_ICLoss)))
                print("\tBC Loss = {}".format(float(Total_BCLoss)))
                print("\tPhysics Loss = {}".format(float(Total_PhysicsLoss)))
                print("\tInterface Loss = {}".format(float(InterfaceLoss)))
                if self.do_regularize:
                    print("\tRegularization Loss = {}".format(float(RegLoss)))
                print("\tTraining Loss = {}".format(float(Total_Loss)))
                print("\tTest Loss = {}".format(float(Test_Loss)))
        return Training_Losses, Test_Losses


if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--domains', type=int, default=1, help='The number of domains when using a cPINN')
        parser.add_argument('--layers', type=int, default=4, help='The number of hidden layers in the neural network')
        parser.add_argument('--nodes', type=int, default=40, help='The number of nodes per hidden layer in the neural network')
        parser.add_argument('--N0', type=int, default=50, help='The number of points to use on the initial condition')
        parser.add_argument('--Nb', type=int, default=50, help='The number of points to use on the boundary condition')
        parser.add_argument('--Nf', type=int, default=20000, help='The number of collocation points to use')
        parser.add_argument('--Ni', type=int, default=40, help='The number of points to use on interfaces for cPINNs')
        parser.add_argument('--Nt', type=int, default=10000, help='The number of points to use to calculate the MSE loss')
        parser.add_argument('--error', type=float, default=0.0, help="The standard deviation of the noise for the initial condition")
        parser.add_argument('--smooth', default=False, action='store_true', help='Do SGP/GP smoothing')
        parser.add_argument('--consLaw', default=False, action='store_true', help='Apply conservation law regularizer')
        parser.add_argument('--N0pool', type=int, default=50, help='The pool of points to select inducing points from for GP/SGP')
        parser.add_argument('--threshold', type=float, default=1.0, help='The threshold for choosing inducing point for GP/SGP') #use 0.995 for SGP
        parser.add_argument('--epochs', type=int, default=50000, help='The number of epochs to train the neural network')
        parser.add_argument('--display-freq', type=int, default=1000, help='How often to display loss information')
        parser.add_argument('--model-name', type=str, default='PINN_model', help='File name to save the model')
        parser.add_argument('--regularize', default=False, action='store_true', help='Do regularization')
        parser.add_argument('--regmode', type=str, default='L2', help='Mode of PINN regularization: L1 or L2')
        parser.add_argument('--regparam', type=float, default=1.e-4, help='The hyperparameter for regularization')
        


        args = parser.parse_args()
        
        if not os.path.exists("../models/"):
            os.mkdir("../models/")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        NDomains = args.domains
        NHiddenLayers = args.layers
        ##LBs = [-5.0, 0.0]
        ##UBs = [5.0, np.pi/2.0]
        boundaries = (1/NDomains)*np.arange(0,NDomains+1)*10.0 - 5 
        t_domain = [0., np.pi/2]
        Layers = [2] + [args.nodes]*NHiddenLayers + [2]
        N0 = int(args.N0/NDomains)
        Nf = int(args.Nf/NDomains)
        N0pool = int(args.N0pool/NDomains)
        Activation = nn.Tanh()

        cpinn = cPINN(boundaries = boundaries,
                      t_domain = t_domain,
                      Layers = Layers,
                      N0 = N0,
                      Nb = args.Nb,
                      Nf = Nf,
                      Ni = args.Ni,
                      Nt = args.Nt,
                      InError = args.error,
                      Activation = Activation,
                      device = device,
                      model_name = "../models/" + args.model_name + ".model",
                      do_smoothing = args.smooth,
                      do_consLaw = args.consLaw,
                      N0pool = N0pool,
                      threshold = args.threshold,
                      display_freq = args.display_freq,
                      do_regularize = args.regularize,
                      regmode = args.regmode,
                      regparam = args.regparam)

        Losses = cpinn.Train(args.epochs)
        
        
            
        torch.save(Losses, "../models/" + args.model_name + ".data")
