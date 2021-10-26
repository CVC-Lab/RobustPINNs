import torch
import numpy as np
import matplotlib.pyplot as plt
from Utils import *
from PINN import PINN
from sPINN_3 import StaggeredPINN as cPINN

modelname = "sPINN_Domains1_Layers6_Nodes70_Error010_smoothing_trial2"
modelname = "sPINN_1domain_01error_70nodes_6hlayers"
##modelname = "PINN_test (2)"
#modelname = "PINN with conservation law (1)"

prefix = "../Models/"
#prefix = "../../../../../Downloads/"

pinn = torch.load(prefix+modelname+".model",map_location="cpu").PINNs[0]
pinn.device="cpu"
##for i in range(len(pinn.PINNs)):
##    pinn.PINNs[i].device = "cpu"
##for tt in t:
##    xt = torch.cat((x.reshape(256,1),tt*torch.ones(256,1)),dim=1)
##    uv = pinn.Eval(xt)
##    u = uv[:,0]
##    v = uv[:,1]

plt.rc("font",size=20)




X_star.requires_grad_()

UVf = pinn.forward(X_star)
uf, vf = UVf[:, 0], UVf[:, 1]
uf_x = torch.autograd.grad(outputs=uf, 
                           inputs=X_star, 
                           grad_outputs=torch.ones(uf.shape), 
                           create_graph = True,
                           allow_unused=True)[0][:,0]
vf_x = torch.autograd.grad(outputs=vf, 
                           inputs=X_star, 
                           grad_outputs=torch.ones(uf.shape),
                           create_graph = True,
                           allow_unused=True)[0][:,0]
##integrad = 2*uf*uf_t + 2*vf*vf_t

##uf = Exact_u.transpose(0,1)
##vf = Exact_v.transpose(0,1)

plt.rc("font",size=20)
plt.rc("legend",fontsize=16)

law1 = uf**2 + vf**2
law1 = torch.mean(law1.reshape(201,256),dim=1)
plt.plot(t,law1.detach())
plt.title(r"Law 1: $\int u^2 + v^2 dx$")
plt.xlabel("time")
plt.tight_layout()
plt.show()

law2 = uf*vf_x + vf*uf_x
law2 = torch.mean(law2.reshape(201,256),dim=1)
plt.plot(t,law2.detach())
plt.title(r"Law 2: $\int u v_x + v u_x dx$")
plt.xlabel("time")
plt.tight_layout()
plt.show()

law3 = uf_x**2 + vf_x**2 - uf**4 - 2*uf**2*vf**2 - vf**4
law3 = torch.mean(law3.reshape(201,256),dim=1)
plt.plot(t,law3.detach())
plt.title(r"Law 3: $\int u_x^2 + v_x^2 - u^4 - 2 u^2 v^2 - v^4 dx$")
plt.xlabel("time")
plt.tight_layout()
plt.show()
