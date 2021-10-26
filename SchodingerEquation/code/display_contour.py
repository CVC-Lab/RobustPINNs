import torch
import numpy as np
import matplotlib.pyplot as plt
from Utils import *
from PINN import *
from sPINN_3 import StaggeredPINN as cPINN

modelname = "sPINN_Domains1_Layers6_Nodes70_Error010_smoothing_trial2"
prefix = "../Models/"

##modelname = "sPINN_1domain_0error_70nodes_6hlayers"
##prefix = "../Models/"

modelname = "SVGP30"
prefix = "../../SVGP_PINN_Schrodinger/Models/"

##prefix = "../../SVGP_PINN_Schrodinger/cPINN_3domains/"
##modelname = "cPINN_NLS_3_domainsSVGP"

pinn = torch.load(prefix+modelname+".model",map_location='cpu')

pinn.device = 'cpu'
##for i in range(len(pinn.PINNs)):
##    pinn.PINNs[i].device='cpu'
    #print(pinn.PINNs[i].u_selections,pinn.PINNs[i].v_selections)

uv_approx = pinn.forward(X_star)
h_approx = torch.sqrt(uv_approx[:,0]**2+uv_approx[:,1]**2)

levels = np.linspace(0,4,10)

plt.rc('font', size=20)
plt.rc('legend', fontsize=16)

##plt.contourf(X,T,h_approx.reshape(201,256).detach().numpy(), levels=levels)
plt.contourf(X,T,Exact_h.transpose(0,1),levels=levels)
plt.title("Exact")
plt.ylabel("t")
plt.xlabel("x")

plt.colorbar()

plt.tight_layout()

plt.show()


##plt.contour(Exact_h.transpose(0,1), levels=levels)
##plt.show()


##error = (uv_approx[:,0]-u_star)**2+(uv_approx[:,1]-v_star)**2
##
##plt.contourf(error.reshape(201,256).detach().numpy())
##plt.colorbar()
##
##plt.show()
