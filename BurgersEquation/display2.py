import torch
import numpy as np
import matplotlib.pyplot as plt
from PINN import *

models = []

folder = ""

savefolder = ""

cpu = torch.device("cpu")

plt.rc("font", size=20)
plt.rc("legend", fontsize=16)

levels = np.linspace(-1.35,1.35,50)

##plt.subplot(5,4,1)
plt.contourf(X,T,Exact_u.transpose(0,1), levels=levels)
plt.title("Exact")
plt.colorbar()
plt.tight_layout()
plt.savefig(savefolder+"ExactContour.png")
plt.show()


for i,model in enumerate(models):
    print(model)
    pinn = torch.load(folder + model, map_location = cpu)
    pinn.device = cpu
    for i in range(len(pinn.PINNs)):
        pinn.PINNs[i].device = cpu
    print(pinn.PINNs[0].u_selections)
    model = model.replace("V","")
    model = model.replace(".model","")
    u_approx = pinn.Eval(X_star)[:,0].reshape(100,256).cpu()
    approx = u_approx.detach().numpy()
##    plt.subplot(4,5,5*i+1)
    plt.contourf(X,T,approx, levels = levels)
    plt.title(model.replace("cPINN_Burgers_error_0.5", " domain cPINN"))
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(savefolder + model+"_PINN_Contour.png")
    plt.show()
##    plt.subplot(4,5,5*i+2)
    plt.plot(x,approx[0], label = "cPINN-evaluated", color = "green")
    plt.plot(x,Exact_u[:,0], label = "Analytical", color="red", linestyle="dashed")
    u_exact = torch.tensor(Exact_u, dtype=torch.float32)
    print(torch.mean((u_approx - u_exact.transpose(0,1))**2))
    plt.scatter((pinn.PINNs[0].XT0.cpu()[:,0]), pinn.PINNs[0].u0_err.cpu(), marker='x', label = "Training Points",color="blue")
    if model == "SVGP":
        plt.scatter(pinn.PINNs[0].XT0.cpu()[pinn.PINNs[0].IP_U_indices,0], pinn.PINNs[0].u0_err.cpu()[pinn.PINNs[0].IP_U_indices], marker = 'o', label = "IPs")
    for j in range(1,len(pinn.PINNs)):
            plt.scatter((pinn.PINNs[i].XT0.cpu()[:,0]), pinn.PINNs[i].u0_err.cpu(), marker='x',color="blue")
            if model == "SVGP":
                plt.scatter(pinn.PINNs[i].XT0.cpu()[pinn.PINNs[i].IP_U_indices,0], pinn.PINNs[i].u0_err.cpu()[pinn.PINNs[0].IP_U_indices], marker = 'o')
    plt.title("t = 0")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(savefolder + model+"_PINN_t_0.png")
    plt.show()
##    plt.subplot(4,5,5*i+3)
    plt.plot(x,approx[25], label = "cPINN-evaluated", color = "green")
    plt.plot(x,Exact_u[:,25], label = "Analytical", color="red", linestyle="dashed")
    plt.title("t = 0.25")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(savefolder + model+"_PINN_t_25.png")
    plt.show()
##    plt.subplot(4,5,5*i+4)
    plt.plot(x,approx[50], label = "cPINN-evaluated", color = "green")
    plt.plot(x,Exact_u[:,50], label = "Analytical", color="red", linestyle="dashed")
    plt.title("t = 0.5")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(savefolder + model+"_PINN_t_5.png")
    plt.show()
##    plt.subplot(4,5,5*i+5)
    plt.plot(x,approx[99], label = "cPINN-evaluated", color = "green")
    plt.plot(x,Exact_u[:,99], label = "Analytical", color="red", linestyle="dashed")
    plt.title("t = 1")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(savefolder + model+"_PINN_t_1.png")
    plt.show()
plt.show()
