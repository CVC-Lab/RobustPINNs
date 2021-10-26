import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PINN import *
import itertools

models = [
            "../Models/sPINN_1domain_0error_70nodes_6hlayers",
            "../Models/sPINN_1domain_01error_70nodes_6hlayers",
            "../Models/sPINN_2domain_01error_50nodes_6hlayers_equalweights",
            "../Models/sPINN_Domains1_Layers6_Nodes70_Error010_smoothing_trial2"
    ]



cpu = torch.device('cpu')

##pinns = []
##
##for i in range(len(models)):
##    pinns.append(torch.load(models[i]+"_PINN_Burgers_error_0.5.model",map_location=cpu).PINNs[0])
##    pinns[i].device = cpu
##    


pinn = torch.load(models[2]+".model",map_location=cpu)
#pinn = torch.load("../../../../../Downloads/PINN with conservation law (1).model",map_location=cpu)
pinn.device = cpu
for i in range(len(pinn.PINNs)):
    pinn.PINNs[i].device = cpu
uv = pinn.Eval(X_star)
h = torch.sqrt(uv[:,0]**2+uv[:,1]**2)
u = uv[:,0]
v = uv[:,1]
approx = h.reshape(201,256).detach().numpy()
exact = Exact_h
#

times = range(100)


plt.rc("font",size=20)
plt.rc("legend",fontsize=16)

fig,ax = plt.subplots()
line1, = ax.plot([],[],lw=3)
line2, = ax.plot([],[],lw=3)

def init():
    ax.set_xlim(-5,5)
    ax.set_ylim(torch.min(exact[:,0])-0.2,torch.max(exact[:,0])+0.2)
    line1.set_data(x.reshape(-1),exact[:,0])
    line2.set_data(x.reshape(-1),approx[0,:])

def run(i):
    if i < 50: i = 0
    else: i = i-50 
    ax.set_ylim(torch.min(exact[:,i])-0.2,torch.max(exact[:,i])+0.2)
    line1.set_data(x.reshape(-1),exact[:,i])
    line2.set_data(x.reshape(-1),approx[i,:])
    return line1,line2,

ani = animation.FuncAnimation(fig, run, frames=251, interval=100, init_func=init)

ani.save("../../GIFs/cPINN_NLS_h.gif")

plt.show()
