import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PINN import *
from sPINN_3 import *
import itertools

modelname = "sPINN_Domains1_Layers6_Nodes70_Error010_smoothing_trial2.model"
modelname2 = "plus_std.model"
modelname3 = "minus_std.model"

cpu = torch.device('cpu')

model = torch.load(modelname,map_location=cpu).PINNs[0]
model.device = cpu
model2 = torch.load(modelname2,map_location=cpu).PINNs[0]
model2.device = cpu
model3 = torch.load(modelname3,map_location=cpu).PINNs[0]
model3.device = cpu


uv_mean = model.forward(X_star)
u_mean = uv_mean[:,0].reshape(201,256).detach()
v_mean = uv_mean[:,1].reshape(201,256).detach()
h_mean = torch.sqrt(u_mean**2 + v_mean**2)
uv_plus = model2.forward(X_star)
u_plus = uv_plus[:,0].reshape(201,256).detach().cpu()
v_plus = uv_plus[:,1].reshape(201,256).detach().cpu()
h_plus = torch.sqrt(u_plus**2 + v_plus**2)
uv_minus = model3.forward(X_star)
u_minus = uv_minus[:,0].reshape(201,256).detach().cpu()
v_minus = uv_minus[:,1].reshape(201,256).detach().cpu()
h_minus = torch.sqrt(u_minus**2 + v_minus**2)

times = [0,50,100,175]
u_min = torch.minimum(u_mean,u_minus)
u_min = torch.minimum(u_min,u_plus)
u_max = torch.maximum(u_mean,u_minus)
u_max = torch.maximum(u_max,u_plus)
v_min = torch.minimum(v_mean,v_minus)
v_min = torch.minimum(v_min,v_plus)
v_max = torch.maximum(v_mean,v_minus)
v_max = torch.maximum(v_max,v_plus)
h_min = torch.minimum(h_mean,h_minus)
h_min = torch.minimum(h_min,h_plus)
h_max = torch.maximum(h_mean,h_minus)
h_max = torch.maximum(h_max,h_plus)

exact = Exact_h
mean = h_mean
m = h_min
M = h_max

delay = 50

plt.rc("font",size=20)
plt.rc("legend",fontsize=16)

fig,ax = plt.subplots()
bounds = ax.fill_between(x.reshape(-1),m[0],M[0],color="grey",alpha=0.5)
line1, = ax.plot(x,exact[:,0],lw=3)
line2, = ax.plot(x,mean[0],lw=3)

def init():
    ax.set_xlim(-5,5)
    ax.set_ylim(torch.min(exact[:,0])-0.2,torch.max(exact[:,0])+0.2)
    bounds = ax.fill_between(x.reshape(-1),m[0],M[0],color="grey",alpha=0.5)
    line1.set_data(x.reshape(-1),exact[:,0])
    line2.set_data(x.reshape(-1),mean[0,:])

def run(i):
    if i < delay: i = 0
    else: i = i - delay
    ax.clear()
    ax.set_ylim(torch.min(exact[:,i])-0.2,torch.max(exact[:,i])+0.2)
    bounds = ax.fill_between(x.reshape(-1),m[i],M[i],color="grey",alpha=0.5)
    line1, = ax.plot(x,exact[:,i],lw=3)
    line2, = ax.plot(x,mean[i],lw=3)
    return bounds,line1,line2,

ani = animation.FuncAnimation(fig, run, frames=201+delay, interval=100)

ani.save("../GIFs/NLS_h_GP_with_bounds.gif")

plt.show()
