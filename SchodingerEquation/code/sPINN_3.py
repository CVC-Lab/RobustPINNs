import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader 
from Utils import *
from PINN import PINN
import numpy as np


class StaggeredPINN:
    def __init__(self, boundaries, t_domain, Layers, N0, Nb, Nf, Ni, #optimizer,
                 InError = 0., Activation = nn.Tanh(), device = 'cpu', 
                 model_name = "sPINN.model", do_smoothing = False):
        self.boundaries = boundaries
        self.tLow  = t_domain[0]
        self.tHigh = t_domain[1]
        self.Layers = Layers
        self.in_dim  = Layers[0]
        self.out_dim = Layers[-1]
        self.N0 = N0
        self.Nb = Nb
        self.Nf = Nf
        self.Ni = Ni
        self.do_smoothing = do_smoothing
        #self.optimizer = optimizer
        self.InError = InError
        self.Activation = Activation
        self.device = device
        self.model_name = model_name
        #print("SPINN = ", self.device)
        self.Loss = torch.nn.MSELoss(reduction='mean')
        self.PINNs = self.build_model(boundaries, Layers, N0, Nb)
                                      #Nf)
        
    def build_model(self, boundaries, Layers, N0, Nb):
        list_PINNs = []
        for ii in range(len(boundaries) - 1):
            LBs, UBs = [boundaries[ii], self.tLow ], [boundaries[ii + 1], self.tHigh]
            list_PINNs.append(PINN(LBs, UBs, Layers, N0, Nb, self.InError, self.Activation, self.device, self.do_smoothing))
        return list_PINNs
    
    def parameters(self):
        list_params = []
        for pinn in self.PINNs:
            list_params += list(pinn.parameters())
        return list_params
    
    
    def BoundaryLoss(self):
        XTbL, XTbU = BoundaryCondition(self.Nb, self.boundaries[0], self.boundaries[-1])
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
        to_return = torch.zeros(xt.shape[0], 2)
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
        for jj in range(n_iters):
            #Total_Loss = 0.0 #torch.tensor(0.0, dtype = torch.float32, device=self.device, requires_grad = True)
            Total_ICLoss = 0.0
            Total_BCLoss = 0.0
            Total_PhysicsLoss = 0.0
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
            optimizer.zero_grad()
            Total_Loss.backward()
            optimizer.step()
            if jj > int(n_iters/4):
                if Total_Loss < min_loss:
                    torch.save(self, self.model_name)
                    min_loss = float(Total_Loss)
            indices = np.random.choice(X_star.shape[0], 10000, replace=False)
            Test_XT = X_star[indices]
            Test_UV = self.Eval(Test_XT)
            Test_Loss = torch.mean( (u_star[indices].reshape(-1) - Test_UV[:,0].reshape(-1))**2 + \
                                        (v_star[indices].reshape(-1) - Test_UV[:,1].reshape(-1))**2 
                                   )
            Training_Losses.append(float(Total_Loss))
            Test_Losses.append(float(Test_Loss))
            if jj % 50 == 0 or jj == n_iters - 1:
                print("Iteration Number = {}".format(jj))
                print("\tIC Loss = {}".format(float(Total_ICLoss)))
                print("\tBC Loss = {}".format(float(Total_BCLoss)))
                print("\tPhysics Loss = {}".format(float(Total_PhysicsLoss)))
                print("\tInterface Loss = {}".format(float(InterfaceLoss)))
                print("\tTest Loss = {}".format(float(Test_Loss)))
        return Training_Losses, Test_Losses
