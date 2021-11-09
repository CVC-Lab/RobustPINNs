import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader 
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import numpy as np
from scipy import io
import matplotlib.pyplot as plt
import argparse

data = io.loadmat('../data/burgers_shock.mat')
t = torch.tensor(data['t'], dtype = torch.float32) 
x = torch.tensor(data['x'], dtype = torch.float32) 
Exact_u = torch.tensor(data['usol'], dtype = torch.float32) 
X, T = np.meshgrid(x,t)
X_star = torch.tensor(np.hstack((X.flatten()[:,None], T.flatten()[:,None])), dtype = torch.float32)
u_star = torch.flatten(torch.transpose(Exact_u,0,1))

alpha = 0.01/np.pi


def InitialCondition(N0, LB, UB, InError = 0.):
    x = torch.tensor([])
    if (type(LB) != type(x)):
      LB = torch.tensor(LB).cpu()
    else:
      LB = LB.cpu()
    if (type(UB) != type(x)):
      UB = torch.tensor(UB).cpu()
    else:
      UB = UB.cpu()
    indices = (X_star[:,0] >= LB) & (X_star[:,0] < UB) & (X_star[:,1] == 0.)
    XT0 = X_star[indices]
    u0 = u_star[indices]
    indices = np.random.choice(XT0.shape[0], N0, replace=False)
    XT0 = XT0[indices]
    u0 = u0[indices]
    return XT0, u0, u0 + InError*torch.randn_like(u0)

def BoundaryCondition(Nb, LB, UB):
    x = torch.tensor([])
    if (type(LB) != type(x)):
      LB = torch.tensor(LB).cpu()
    else:
      LB = LB.cpu()
    if (type(UB) != type(x)):
      UB = torch.tensor(UB).cpu()
    else:
      UB = UB.cpu()
    tb_indices = np.random.choice(t.shape[0], Nb, replace=False)
    tb = t[tb_indices]
    XTL = torch.cat(( LB*torch.ones((Nb,1)), tb.reshape(-1,1)), dim = 1)
    XTL.requires_grad_()
    XTU = torch.cat(( UB*torch.ones((Nb,1)), tb.reshape(-1,1)), dim = 1)
    XTU.requires_grad_()
    return  XTL, XTU

def MeshGrid(LBs, UBs, Nf):
    x = torch.tensor([])
    if (type(LBs) != type(x)):
      LBs = torch.tensor(LBs).cpu()
    else:
      LBs = LBs.cpu()
    if (type(UBs) != type(x)):
      UBs = torch.tensor(UBs).cpu()
    else:
      UBs = UBs.cpu()
    n = round(Nf**0.5)
    XGrid, TGrid = np.meshgrid(np.arange(LBs[0], UBs[0]+0.001, (UBs[0] - LBs[0])/(n-1)), 
                               np.arange(LBs[1], UBs[1]+0.001, (UBs[1] - LBs[1])/(n-1)))
    return XGrid, TGrid



def SewingBoundary(x_value, LB, UB, Ni):
    x = torch.tensor([])
    if (type(LB) != type(x)):
      LB = torch.tensor(LB).cpu()
    else:
      LB = LB.cpu()
    if (type(UB) != type(x)):
      UB = torch.tensor(UB).cpu()
    else:
      UB = UB.cpu()
    TGrid = np.arange(LB, UB, (UB - LB)/(Ni*10))
    boundary_indices = np.random.choice(TGrid.shape[0], Ni, replace = False)
    XTGrid = np.append(x_value.cpu()*torch.tensor(np.ones((1, TGrid.shape[0]))), TGrid.reshape(1,-1), axis = 0).T
    return torch.tensor(XTGrid[boundary_indices], dtype = torch.float32, requires_grad=True)

class PINN(nn.Module):
    def __init__(self, LBs, UBs, Layers, N0, Nb, Nf, Nt, 
                 InError = 0., Activation = nn.Tanh(), 
                 model_name = "PINN.model", device = 'cpu',
                 do_smoothing = False, N0pool = 0, N01 = 1000,
                 threshold = 0.9, display_freq = 100):
        super(PINN, self).__init__()
        self.LBs = torch.tensor(LBs, dtype=torch.float32).to(device)
        self.UBs = torch.tensor(UBs, dtype=torch.float32).to(device)
        self.Layers = Layers
        self.in_dim  = Layers[0]
        self.out_dim = Layers[-1]
        self.N0 = N0
        self.Nb = Nb
        self.Nf = Nf
        self.Nt = Nt
        self.N01 = np.minimum(N0,N01)
        self.InError = InError
        self.Activation = Activation
        self.do_smoothing = do_smoothing
        self.N0pool = np.maximum(N0pool, N0)
        self.threshold = threshold
        self.XT0, self.u0_true, self.u0_err  = InitialCondition(self.N0pool, LBs[0], UBs[0], InError)
        if do_smoothing:
            self.u0, self.GP_U, self.u_selections, self.IP_U_indices = self.Smoothing(self.XT0, self.u0_err)
        else:
            self.u0, self.GP_U, self.u_selections, self.IP_U_indices = self.u0_err, None, self.N0, range(self.N0)
        self.XT0 = self.XT0.to(device)
        self.u0 = self.u0.to(device) 
        self.XTbL, self.XTbU = BoundaryCondition(self.Nb, self.LBs[0], self.UBs[0])
        self.XTbL = self.XTbL.to(device) 
        self.XTbU = self.XTbU.to(device)
        self.device = device
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

    
    def Smoothing(self, XT0, u0):
        X = XT0[:, 0].reshape(-1, 1)
        kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) + \
                 WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+2))

        n0 = self.N01  
        
        GP_U = GPR(kernel = kernel, alpha = 0.0).fit(X[0:n0], u0[0:n0])
        print(GP_U.kernel_)

        u_selections = n0

        IP_U_indices = list(range(n0))

        u_kernel = GP_U.kernel_.get_params()["k1__k2"]
        for i in range(n0+1, self.N0pool): 
            x = np.array([X[i].tolist()])
            K_U = u_kernel.__call__(x, X[IP_U_indices])
            if (np.max(K_U) < self.threshold and u_selections < self.N0) or self.N0 == self.N0pool: 
                IP_U_indices.append(i)
                u_selections = u_selections + 1

        GP_U = GP_U.fit(X[IP_U_indices], u0[IP_U_indices])
        U0 = torch.tensor(GP_U.predict(X), dtype = torch.float32).reshape(-1)
        print("IPs for u:", len(IP_U_indices))        
        return U0, GP_U, u_selections, IP_U_indices
        
    
    def forward(self, x):
        x = x.to(self.device)
        x = x.reshape((-1,self.in_dim))  
        x = 2*(x - self.LBs)/(self.UBs - self.LBs) - 1.0
        return torch.reshape(self._nn.forward(x), (-1, self.out_dim))

    def ICLoss(self):
        if self.do_smoothing and (not self.GP_U == None):
            XT0 = torch.cat( (torch.rand(self.N0,1).uniform_(self.LBs[0], self.UBs[0]),
                              self.LBs[1].cpu()*torch.ones((self.N0,1))), 
                             dim = 1 
                           ).to(torch.device('cpu'))
            u0 = torch.tensor(self.GP_U.predict(XT0[:,0].reshape(-1,1)), dtype = torch.float32 ).reshape(-1).to(self.device)
        else:
            XT0 = self.XT0
            u0  = self.u0
        UV0_pred = self.forward(XT0)
        u0_pred = UV0_pred[:,0].reshape(-1)
        return self.Loss(u0_pred, u0)

    def BCLoss(self):
        ub_l, ub_u = self.forward(self.XTbL), self.forward(self.XTbU)
        return torch.mean(ub_l**2 + ub_u**2) 

    def PhysicsLoss(self, XTGrid):
        XTGrid = XTGrid.to(self.device)
        uf = self.forward(XTGrid)[:,0]
        uf_x, uf_t = torch.autograd.grad(outputs=uf.to(self.device), 
                                   inputs=XTGrid, 
                                   grad_outputs=torch.ones(uf.shape).to(self.device), 
                                   create_graph = True,
                                   allow_unused=True)[0].T
        uf_xx = torch.autograd.grad(outputs=uf_x.to(self.device), 
                                   inputs=XTGrid, 
                                   grad_outputs=torch.ones(uf_x.shape).to(self.device),
                                   create_graph = True,
                                   allow_unused=True)[0][:,0]
        lossf =  self.Loss(uf_t + uf*uf_x, alpha*uf_xx)
    
        return lossf

class cPINN:
    def __init__(self, boundaries, t_domain, Layers, N0, Nb, Nf, Ni, Nt, #optimizer,
                 InError = 0., Activation = nn.Tanh(),
                 model_name = "cPINN.model", device = 'cpu',
                 do_smoothing = False, N0pool = 0, N01 = 1000,
                 threshold = 0.9, display_freq = 100, do_colehopf=False):
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
        self.N01 = np.minimum(N0,N01)   
        self.N0pool = np.maximum(N0pool, N0)
        self.display_freq = display_freq
        self.threshold = threshold
        self.do_smoothing = do_smoothing
        self.InError = InError
        self.Activation = Activation
        self.device = device
        self.model_name = model_name
        self.Loss = torch.nn.MSELoss(reduction='mean')
        self.PINNs = self.build_model(boundaries, Layers, N0, Nb, Nf, Nt)
        self.do_colehopf = do_colehopf
        
    def build_model(self, boundaries, Layers, N0, Nb, Nf, Nt):
        list_PINNs = []
        for ii in range(len(boundaries) - 1):
            LBs, UBs = [boundaries[ii], self.tLow ], [boundaries[ii + 1], self.tHigh]
            list_PINNs.append(PINN(LBs, UBs, Layers, N0, Nb, Nf, Nt, self.InError, self.Activation, self.model_name, self.device, self.do_smoothing, self.N0pool, self.N01, self.threshold, self.display_freq))
        return list_PINNs
    
    def parameters(self):
        list_params = []
        for pinn in self.PINNs:
            list_params += list(pinn.parameters())
        return list_params
    
    
    def BoundaryLoss(self):
        XTbL, XTbU = BoundaryCondition(self.Nb, self.boundaries[0], self.boundaries[-1])
        ub_l, ub_u = self.PINNs[0].forward(XTbL).to(self.device), self.PINNs[-1].forward(XTbU).to(self.device)
        return torch.mean(ub_l**2 + ub_u**2)
    
    
    def InterfaceLoss(self):
        loss = torch.tensor(0.0, dtype = torch.float32, device=self.device, requires_grad = True)
        if len(self.boundaries) == 2:
            return loss
        else:
            for ii in range(len(self.boundaries)-2):
                XTI = SewingBoundary(self.boundaries[ii+1], self.tLow, self.tHigh, self.Ni).to(self.device)
                ui_0 = self.PINNs[ii].forward(XTI).to(self.device)
                ui_1 = self.PINNs[ii+1].forward(XTI).to(self.device)
                
                ui0_x = torch.autograd.grad(outputs=ui_0.to(self.device), 
                                            inputs=XTI, 
                                            grad_outputs=torch.ones(ui_0.shape).to(self.device), 
                                            create_graph = True,
                                            allow_unused=True)[0][:,0]
                ui1_x = torch.autograd.grad(outputs=ui_1.to(self.device), 
                                            inputs=XTI, 
                                            grad_outputs=torch.ones(ui_1.shape).to(self.device), 
                                            create_graph = True,
                                            allow_unused=True)[0][:,0]
                loss = loss + self.Loss(ui_0, ui_1) + self.Loss(ui0_x, ui1_x) 
                
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
            
    def ColeHopfLoss(self, XT, shape):
        XT = XT.to(self.device)
        U = self.Eval(XT)[:,0]
        dx = float(XT[1,0].detach() - XT[0,0].detach())
        F = dx*torch.cumsum(U.reshape(shape), dim=1).reshape(-1)/(2*alpha)
        F = torch.clamp(F, min=-10, max=10000)
        V = torch.exp(-F)
        V = V.to(self.device)
        u_x, u_t = torch.autograd.grad(outputs=U.to(self.device), 
                                   inputs=XT, 
                                   grad_outputs=torch.ones(U.shape).to(self.device), 
                                   create_graph = True,
                                   allow_unused=True)[0].T
        Ut = dx*torch.cumsum(u_t.reshape(shape), dim=1).reshape(-1)
        
        lossf = torch.sum((V*(alpha*u_x - U**2/2 - Ut))**2)
        return lossf
        
    
    def Train(self, n_iters, weights=(1.0,1.0,1.0,1.0)):
        params = list(self.parameters())
        optimizer = optim.Adam(params, lr=1e-3)
        min_loss = 999999.0
        Training_Losses = []
        Test_Losses = []
        for jj in range(n_iters):
            Total_ICLoss = torch.tensor(0.0, dtype = torch.float32, device=self.device, requires_grad = True)
            Total_BCLoss = torch.tensor(0.0, dtype = torch.float32, device=self.device, requires_grad = True)
            Total_PhysicsLoss = torch.tensor(0.0, dtype = torch.float32, device=self.device, requires_grad = True)
            for ii in range(len(self.boundaries) - 1):
                LBs, UBs = [self.boundaries[ii], self.tLow], [self.boundaries[ii + 1], self.tHigh]
                XGrid, TGrid = MeshGrid(LBs, UBs, self.Nf)
                XTGrid = torch.tensor(np.append(XGrid.reshape(1,-1), TGrid.reshape(1,-1), axis = 0).T, dtype = torch.float32, device=self.device, requires_grad=True)
                Total_ICLoss = Total_ICLoss + self.PINNs[ii].ICLoss()
                Total_PhysicsLoss = Total_PhysicsLoss + self.PINNs[ii].PhysicsLoss(XTGrid)
            if self.do_colehopf:
                LBs, UBs = [self.boundaries[0], self.tLow], [self.boundaries[-1], self.tHigh]
                XGrid, TGrid = MeshGrid(LBs, UBs, self.Nf*len(self.PINNs))
                XTGrid = torch.tensor(np.append(XGrid.reshape(1,-1), TGrid.reshape(1,-1), axis = 0).T, dtype = torch.float32, device=self.device, requires_grad=True)
                Total_CHLoss = self.ColeHopfLoss(XTGrid, XGrid.shape).to(self.device)
            else:
                Total_CHLoss = torch.tensor(0.0, dtype = torch.float32, device=self.device, requires_grad = True)
            Total_BCLoss = self.BoundaryLoss()
            InterfaceLoss = self.InterfaceLoss()
            Total_Loss = weights[0]*Total_ICLoss + weights[1]*Total_BCLoss\
                        + weights[2]*Total_PhysicsLoss + weights[3]*InterfaceLoss + 1.0*Total_CHLoss
            optimizer.zero_grad()
            Total_Loss.backward()
            optimizer.step()
            if jj > int(n_iters/4):
                if Total_Loss < min_loss:
                    torch.save(self, self.model_name)
                    min_loss = float(Total_Loss)
            indices = np.random.choice(X_star.shape[0], self.Nt, replace=False)
            Test_XT = X_star[indices]
            Test_UV = self.Eval(Test_XT)
            u_exact = u_star[indices].reshape(-1).to(self.device)
            Test_Loss = self.Loss(u_exact, Test_UV[:,0].reshape(-1))
            Test_Losses.append(float(Test_Loss))
            Training_Losses.append(float(Total_Loss))
            if jj % self.display_freq == self.display_freq - 1 or jj == n_iters - 1 or jj == 0:
                print("Iteration Number = {}".format(jj+1))
                print("\tIC Loss = {}".format(float(Total_ICLoss)))
                print("\tBC Loss = {}".format(float(Total_BCLoss)))
                print("\tPhysics Loss = {}".format(float(Total_PhysicsLoss)))
                print("\tCH Loss = {}".format(float(Total_CHLoss)))
                print("\tInterface Loss = {}".format(float(InterfaceLoss)))
                print("\tTraining Loss = {}".format(float(Total_Loss)))
                print("\tTest Loss = {}".format(float(Test_Loss)))
        return Training_Losses, Test_Losses


if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument('--domains', type=int, default=1, help='The number of domains when using a cPINN')
        parser.add_argument('--layers', type=int, default=4, help='The number of hidden layers in the neural network')
        parser.add_argument('--nodes', type=int, default=40, help='The number of nodes per hidden layer in the neural network')
        parser.add_argument('--N0', type=int, default=50, help='The number of points to use on the initial condition')
        parser.add_argument('--Nb', type=int, default=50, help='The number of points to use on the boundary condition')
        parser.add_argument('--Nf', type=int, default=10000, help='The number of collocation points to use')
        parser.add_argument('--Ni', type=int, default=50, help='The number of points to use on interfaces for cPINNs')
        parser.add_argument('--Nt', type=int, default=1000, help='The number of points to use to calculate the MSE loss')
        parser.add_argument('--error', type=float, default=0.0, help="The standard deviation of the noise for the initial condition")
        parser.add_argument('--smooth', dest='smooth', action='store_true', help='Do SGP/GP smoothing')
        parser.add_argument('--N0pool', type=int, default=50, help='The pool of points to select inducing points from for SGP')
        parser.add_argument('--threshold', type=float, default=1.0, help='The threshold for selecting inducing points for SGP')
        parser.add_argument('--epochs', type=int, default=50000, help='The number of epochs to train the neural network')
        parser.add_argument('--model-name', type=str, default='PINN_model', help='File name to save the model')
        parser.add_argument('--display-freq', type=int, default=1000, help='How often to display loss information')
        parser.add_argument('--do-colehopf', dest='do_colehopf', action='store_true', help='Do Cole-Hopf transform constrain')
        


        args = parser.parse_args()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)
        NDomains = args.domains
        NHiddenLayers = args.layers
        boundaries = (1/NDomains)*np.arange(0,NDomains+1)*2 - 1
        t_domain = [0., 1.]
        Layers = [2] + [args.nodes]*NHiddenLayers + [1]
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
                      N0pool = N0pool,
                      threshold = args.threshold,
                      do_colehopf = args.do_colehopf,
                      display_freq = args.display_freq )

        Losses = cpinn.Train(args.epochs)

        torch.save(Losses, "../models/" + args.model_name + ".data")
