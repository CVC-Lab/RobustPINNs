import torch
import numpy as np
from scipy import io


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

def BoundaryCondition(Nb, LB, UB):
    ## Choose Nb time instances on x = LB and x = UB
    tb_indices = np.random.choice(t.shape[0], Nb, replace=False)
    tb = t[tb_indices]
    XTL = torch.cat(( LB*torch.ones((Nb,1)), tb.reshape(-1,1)), dim = 1)
    XTL.requires_grad_()
    XTU = torch.cat(( UB*torch.ones((Nb,1)), tb.reshape(-1,1)), dim = 1)
    XTU.requires_grad_()
    return  XTL, XTU

def MeshGrid(LBs, UBs, Nf):
    n = round(Nf**0.5)*2
    #print(n)
    XGrid, TGrid = np.meshgrid(np.arange(LBs[0], UBs[0], (UBs[0] - LBs[0])/n), 
                               np.arange(LBs[1], UBs[1], (UBs[1] - LBs[1])/n))
    XTGrid = np.append(XGrid.reshape(1,-1), TGrid.reshape(1,-1), axis = 0).T
    xt_grid_indices = np.random.choice(XTGrid.shape[0], Nf, replace=False)
    xt_f = torch.tensor(XTGrid[xt_grid_indices], dtype = torch.float32, requires_grad=True)
    return xt_f

def SewingBoundary(x_value, LB, UB, Ni):
    TGrid = np.arange(LB, UB, (UB - LB)/(Ni*10))
    boundary_indices = np.random.choice(TGrid.shape[0], Ni, replace = False)
    XTGrid = np.append(x_value*np.ones((1, TGrid.shape[0])), TGrid.reshape(1,-1), axis = 0).T
    return torch.tensor(XTGrid[boundary_indices], dtype = torch.float32, requires_grad=True)
