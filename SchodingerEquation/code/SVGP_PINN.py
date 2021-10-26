import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader 
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from Utils import *
import numpy as np
import time
start_time = time.time()

class PINN(nn.Module):
    def __init__(self, LBs, UBs, Layers, N0, Nb, Nf, Nt, 
                 InError = 0., Activation = nn.Tanh(), 
                 model_name = "PINN.model", device = 'cpu',
                 do_smoothing = False, N0pool = 0, N0start = 10000,
                 threshold = 0.9, display_freq = 100):
        super(PINN, self).__init__()
        self.LBs = torch.tensor(LBs, dtype=torch.float32).to(device)
        self.UBs = torch.tensor(UBs, dtype=torch.float32).to(device)
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
        self.N0pool = np.maximum(N0pool, N0)
        self.N0start = np.minimum(N0, N0start)
        self.threshold = threshold
        self.XT0, self.u0_true, self.v0_true, self.u0_err, self.v0_err  = InitialCondition(self.N0pool, LBs[0], UBs[0], InError)
        if do_smoothing:
            self.u0, self.v0, self.GP_U, self.GP_V, self.u_selections, self.v_selections, self.IP_U_indices, self.IP_V_indices = self.Smoothing(self.XT0, self.u0_err, self.v0_err)
        else:
            self.GP_U, self.GP_V, self.u_selections, self.v_selection, self.IP_U_indices, self.IP_V_indices = None, None, self.N0, self.N0, None, None
            self.u0, self.v0 = self.u0_err, self.v0_err
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

        n0 = self.N0start
        
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
            if np.max(K_U) < self.threshold and u_selections < self.N0:
                IP_U_indices.append(i)
                u_selections = u_selections + 1
            if np.max(K_V) < self.threshold and v_selections < self.N0:
                IP_V_indices.append(i)
                v_selections = v_selections + 1

        GP_U = GP_U.fit(X[IP_U_indices], u0[IP_U_indices])
        GP_V = GP_V.fit(X[IP_V_indices], v0[IP_V_indices])               

        U0 = torch.tensor(GP_U.predict(X), dtype = torch.float32).reshape(-1)
        V0 = torch.tensor(GP_V.predict(X), dtype = torch.float32).reshape(-1)

        print("IPs for u:", len(IP_U_indices))
        print("IPs for v:", len(IP_V_indices))
        
        return U0, V0, GP_U, GP_V, u_selections, v_selections, IP_U_indices, IP_V_indices
        
    
    def forward(self, x):
        #print(x.device)
        x = x.to(self.device)
        x = x.reshape((-1,self.in_dim))  
        x = 2*(x - self.LBs)/(self.UBs - self.LBs) - 1.0
        #print(x.device)
        return torch.reshape(self._nn.forward(x), (-1, self.out_dim))

    def ICLoss(self):
        if self.do_smoothing and (not self.GP_U == None) and (not self.GP_V == None):
            XT0 = torch.cat( (torch.rand(self.N0pool,1).uniform_(self.LBs[0], self.UBs[0]),
                              self.LBs[1]*torch.ones((self.N0pool,1))), 
                             dim = 1 
                           ).to(torch.device('cpu'))
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
    
        return lossf

        
    
    def Train(self, n_iters, weights=(1.0,1.0,1.0)):
        params = list(self.parameters())
        optimizer = optim.Adam(params, lr=1e-3)
        min_loss = 999999.0
        #message_print_count = min(n_iters, 100)
        Training_Losses = []
        Test_Losses = []
        XTGrid = MeshGrid(self.LBs.cpu(), self.UBs.cpu(), self.Nf)
        for jj in range(n_iters):
            ICLoss = self.ICLoss()
            BCLoss = self.BCLoss()
            PhysicsLoss = self.PhysicsLoss(XTGrid)
            Total_Loss = weights[0]*ICLoss \
                       + weights[1]*BCLoss \
                       + weights[2]*PhysicsLoss
            optimizer.zero_grad()
            Total_Loss.backward()
            optimizer.step()
            if jj > int(n_iters/4):
                if Total_Loss < min_loss:
                    torch.save(self, self.model_name)
                    min_loss = float(Total_Loss)
            indices = np.random.choice(X_star.shape[0], self.Nt, replace=False)
            Test_XT = X_star[indices]
            Test_UV = self.forward(Test_XT).cpu()
            Test_Loss = torch.mean( (u_star[indices].reshape(-1) - Test_UV[:,0].reshape(-1))**2 + \
                                        (v_star[indices].reshape(-1) - Test_UV[:,1].reshape(-1))**2 
                                      )
            Test_Losses.append(float(Test_Loss))
            Training_Losses.append(float(Total_Loss))
            if jj == 0 or jj % self.display_freq == self.display_freq - 1 or jj == n_iters - 1:
                print(self.model_name+" Iteration Number = {}".format(jj+1))
                print("\tIC Loss = {}".format(float(ICLoss)))
                print("\tBC Loss = {}".format(float(BCLoss)))
                print("\tPhysics Loss = {}".format(float(PhysicsLoss)))
                print("\tTraining Loss = {}".format(float(Total_Loss)))
                print("\tTest Loss = {}".format(float(Test_Loss)))
                print("\tTime Taken = {}".format(float(time.time()-start_time)), flush=True)
        return Training_Losses, Test_Losses
