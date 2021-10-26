import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader 
import numpy as np
from scipy import io
import matplotlib.pyplot as plt
from sPINN_3 import StaggeredPINN as sPINN
from Utils import *

modelname = "sPINN_Domains1_Layers6_Nodes70_Error010_smoothing_trial2.model"
modelname2 = "plus_std.model"
modelname3 = "minus_std.model"

cpu = torch.device('cpu')

model = torch.load(modelname)
model2 = torch.load(modelname2,map_location=cpu)
model2.device = cpu
model2.PINNs[0].device = cpu
model3 = torch.load(modelname3,map_location=cpu)
model3.device = cpu
model3.PINNs[0].device = cpu


uv_mean = model.Eval(X_star)
u_mean = uv_mean[:,0].reshape(201,256).detach()
v_mean = uv_mean[:,1].reshape(201,256).detach()
h_mean = torch.sqrt(u_mean**2 + v_mean**2)
uv_plus = model2.Eval(X_star)
u_plus = uv_plus[:,0].reshape(201,256).detach().cpu()
v_plus = uv_plus[:,1].reshape(201,256).detach().cpu()
h_plus = torch.sqrt(u_plus**2 + v_plus**2)
uv_minus = model3.Eval(X_star)
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

x_axis = model.PINNs[0].GP_U.X_train_.reshape(-1)
u0_err = model.PINNs[0].GP_U.y_train_.reshape(-1)
v0_err = model.PINNs[0].GP_V.y_train_.reshape(-1)
h0_err = np.sqrt(u0_err**2 + v0_err**2)

##u_min = u_minus
##u_max = u_plus
##v_min = v_minus
##v_max = v_plus
##h_min = h_minus
##h_max = h_plus

plt.rc('font', size=20)
plt.rc('legend', fontsize=16)

for tt in times:
  time_str = str(np.round(tt*np.pi/400,2))
##  plt.fill_between(x,u_min[tt],u_max[tt],color='grey',alpha=0.5,label="1 StD")
##  plt.plot(x,u_mean[tt],label="mean",color="green")
##  plt.plot(x,Exact_u[:,tt], label="exact",color="red",linestyle="dashed")
##  if tt == 0:
##    plt.scatter(x_axis, u0_err, marker="x", label = "data", color = 'blue')
##  plt.legend()
##  plt.title("t="+time_str)
##  plt.ylabel("u")
##  plt.xlabel("x")
##  plt.tight_layout()
##  plt.savefig("u_t_"+time_str+".png")
##  plt.show()
##  plt.fill_between(x,v_min[tt],v_max[tt],color='grey',alpha=0.5,label="1 StD")
##  plt.plot(x,v_mean[tt],label="mean",color="green")
##  plt.plot(x,Exact_v[:,tt], label="exact",color="red",linestyle="dashed")
##  if tt == 0:
##    plt.scatter(x_axis, v0_err, marker="x", label = "data", color = 'blue')
##  plt.title("t="+time_str)
##  plt.ylabel("v")
##  plt.xlabel("x")
##  plt.legend()
##  plt.tight_layout()
##  plt.savefig("v_t_"+time_str+".png")
##  plt.show()
  plt.fill_between(x,h_min[tt],h_max[tt],color='grey',alpha=0.5,label="1 StD")
  plt.plot(x,h_mean[tt],label="mean",color="green")
  plt.plot(x,Exact_h[:,tt], label="exact",color="red",linestyle="dashed")
  if tt == 0:
    plt.scatter(x_axis, h0_err, marker="x", label = "data", color = 'blue')
  plt.title("t="+time_str)
  plt.legend()
  plt.ylabel("|h|")
  plt.xlabel("x")
  plt.tight_layout()
  plt.savefig("h_t_"+time_str+".png")
  plt.show()
