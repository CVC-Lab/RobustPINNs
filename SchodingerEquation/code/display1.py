import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader 
import numpy as np
from scipy import io
import matplotlib.pyplot as plt
from PINN import *
from sPINN_4 import StaggeredPINN as cPINN
from Utils import *

modelnames = [
             'test1.model'
             ]

prefix = "..\/Models\/"

modelnames = [
             'no_error.model',
             'no_smoothing.model'

             ]

prefix = "../../SVGP_PINN_Schrodinger/cPINN_3domains/cPINN_NLS_3_domains"

cpu = torch.device('cpu')

device = cpu

def InterfaceGradients(ui_0, vi_0, ui_1, vi_1, XTI):

    ui0_x = torch.autograd.grad(outputs=ui_0.to(device), 
                                            inputs=XTI, 
                                            grad_outputs=torch.ones(ui_0.shape).to(device), 
                                            create_graph = True,
                                            allow_unused=True)[0][:,0]
    vi0_x = torch.autograd.grad(outputs=vi_0.to(device), 
                                            inputs=XTI, 
                                            grad_outputs=torch.ones(vi_0.shape).to(device), 
                                            create_graph = True,
                                            allow_unused=True)[0][:,0]
    ui1_x = torch.autograd.grad(outputs=ui_1.to(device), 
                                            inputs=XTI, 
                                            grad_outputs=torch.ones(ui_1.shape).to(device), 
                                            create_graph = True,
                                            allow_unused=True)[0][:,0]
    vi1_x = torch.autograd.grad(outputs=vi_1.to(device), 
                                            inputs=XTI, 
                                            grad_outputs=torch.ones(vi_1.shape).to(device), 
                                            create_graph = True,
                                            allow_unused=True)[0][:,0]
    return u10_x, vi0_x, ui1_x, vi1_x

def makePlots(modelname):
    model = torch.load(prefix + modelname,map_location=cpu)
    model.device = cpu
    for i in range(len(model.PINNs)):
        model.PINNs[i].device = cpu
    figprefix = modelname.replace(".model", "")
    if False:#'smoothing' in modelname:
        X_in = []
        U_in = []
        V_in = []
        u_in_true = []
        u_fit = []
        v_in_true = []
        v_fit = []
        u_sorted = []
        v_sorted = []
        u_eval = []
        v_eval = []
        for ii in range(len(model.PINNs)):
            X_in += list(model.PINNs[ii].GP_U.X_train_.reshape(-1))
            U_in += list(model.PINNs[ii].GP_U.y_train_.reshape(-1))
            u_in_true += list(model.PINNs[ii].u0_true)
            u_fit += list(model.PINNs[ii].u0)
            V_in += list(model.PINNs[ii].GP_V.y_train_.reshape(-1))
            v_in_true += list(model.PINNs[ii].v0_true)
            v_fit += list(model.PINNs[ii].v0)
                        
            uv_eval = model.Eval(model.PINNs[ii].XT0).detach().numpy()
            this_u_eval, this_v_eval = uv_eval[:,0], uv_eval[:,1]
            u_eval += list(this_u_eval)
            v_eval += list(this_v_eval)
        X_sorted = [x for x, _ in sorted(zip(X_in,u_in_true), key=lambda pair: pair[0])]
        u_sorted = np.array([u for _, u in sorted(zip(X_in,u_in_true), key=lambda pair: pair[0])])
        v_sorted = np.array([u for _, u in sorted(zip(X_in,v_in_true), key=lambda pair: pair[0])])
        h_sorted = (u_sorted**2 + v_sorted**2)**0.5
        H_in = (np.array(U_in)**2 + np.array(V_in)**2)**0.5
        h_in_true = (np.array(u_in_true)**2 + np.array(v_in_true)**2)**0.5
        h_fit = (np.array(u_fit)**2 + np.array(v_fit)**2)**0.5
        h_eval = (np.array(u_eval)**2 + np.array(v_eval)**2)**0.5
            

        fig, ax = plt.subplots()
        plt.plot(X_sorted, u_sorted, label = 'Analytical')
        plt.scatter(X_in, U_in, marker='x', label = 'Measured' )#, 'r', 'x')
        plt.scatter(X_in, u_fit, marker = 'o', label = 'GP-Fitted')#, 'g', 'o')
        plt.scatter(X_in, u_eval, marker = '+', label = 'PINN-evaluated')
        plt.xlabel('x')
        plt.ylabel('u')
        plt.legend()
        plt.savefig("../Figures3domain/" + figprefix + "_U.png")
        plt.show()

        fig, ax = plt.subplots()
        plt.plot(X_sorted, v_sorted, label = 'Analytical')
        plt.scatter(X_in, V_in, marker='x', label = 'Measured' )#, 'r', 'x')
        plt.scatter(X_in, v_fit, marker = 'o', label = 'GP-Fitted')#, 'g', 'o')
        plt.scatter(X_in, v_eval, marker = '+', label = 'PINN-evaluated')
        plt.xlabel('x')
        plt.ylabel('v')
        plt.legend()
        plt.savefig("../Figures3domain/" + figprefix + "_V.png")
        plt.show()


        fig, ax = plt.subplots()
        plt.plot(X_sorted, h_sorted, label = 'Analytical')
        plt.scatter(X_in, H_in, marker='x', label = 'Measured' )#, 'r', 'x')
        plt.scatter(X_in, h_fit, marker = 'o', label = 'GP-Fitted')#, 'g', 'o')
        plt.scatter(X_in, h_eval, marker = '+', label = 'PINN-evaluated')
        plt.xlabel('x')
        plt.ylabel('|h|')
        plt.legend()
        plt.savefig("../Figures3domain/" + figprefix + "_H.png")
        plt.show()
    else:
        X_in = []
        U_in = []
        V_in = []
        for ii in range(len(model.PINNs)):
            X_in += list(model.PINNs[ii].XT0[:,0])
            U_in += list(model.PINNs[ii].u0)
            V_in += list(model.PINNs[ii].v0) 
        H_in = (np.array(U_in)**2 + np.array(V_in)**2)**0.5

    Ts = t[np.arange(0,202, 25)]
    Ts = t[[0,50,100,175]]
    print(Ts)
    for _ttime in Ts:
        _time = int(_ttime*100)/100.0
        indices = X_star[:, 1] == _ttime
        this_XT = X_star[indices]
        this_x = this_XT[:,0].reshape(-1)
        this_u = u_star[indices]
        this_v = v_star[indices]
        this_h = h_star[indices]
        UV = model.Eval(this_XT)
        U = UV[:, 0].reshape(-1)
        V = UV[:, 1].reshape(-1)
        H = torch.sqrt(U**2 + V**2) 
        fig, ax = plt.subplots()
        plt.plot(this_x.detach().numpy(), U.detach().numpy(), 'g-', label = 'PINN-evaluated')
        plt.plot(this_x.detach().numpy(), this_u.detach().numpy(), 'r--', label = 'Analytical')
        if _time == 0.:
            plt.scatter(X_in, U_in, marker = 'x', label = 'Training Points')
        plt.xlabel('x')
        plt.ylabel('u')
        plt.title("t = {}".format(_time))
        plt.legend()
        plt.savefig('../Figures3domain/' + figprefix + '_U_t_{}.png'.format(_time))
        plt.show()

        fig, ax = plt.subplots()
        plt.plot(this_x.detach().numpy(), V.detach().numpy(), 'g-', label = 'PINN-evaluated')
        plt.plot(this_x.detach().numpy(), this_v.detach().numpy(), 'r--', label = 'Analytical')
        if _time == 0.:
            plt.scatter(X_in, V_in, marker = 'x', label = 'Training Points')
        plt.xlabel('x')
        plt.ylabel('v')
        plt.title("t = {}".format(_time))
        plt.legend()
        plt.savefig('../Figures3domain/' + figprefix + '_V_t_{}.png'.format(_time))
        plt.show()

        fig, ax = plt.subplots()
        plt.plot(this_x.detach().numpy(), H.detach().numpy(), 'g-', label = 'PINN-evaluated')
        plt.plot(this_x.detach().numpy(), this_h.detach().numpy(), 'r--', label = 'Analytical')
        if _time == 0.:
            plt.scatter(X_in, H_in, marker = 'x', label = 'Training Points')
        plt.xlabel('x')
        plt.ylabel('|h|')
        plt.title("t = {}".format(_time))
        plt.legend()
        plt.savefig('../Figures3domain/' + figprefix + '_H_t_{}.png'.format(_time))
        plt.show()
        
    boundaries = model.boundaries
    if False:#len(boundaries) > 2:
        for jj in range(1, len(boundaries)-1):
            _x = int(boundaries[jj]*100)/100.0
            indices = X_star[:, 0] == boundaries[jj]
            this_XT = X_star[indices]
            this_t = this_XT[:,1].reshape(-1)
            this_u = u_star[indices]
            this_v = v_star[indices]
            this_h = h_star[indices]
            UV0 = model.PINNs[jj-1].forward(this_XT)
            U0 = UV0[:, 0].reshape(-1)
            V0 = UV0[:, 1].reshape(-1)
            H0 = torch.sqrt(U0**2 + V0**2) 
            UV1 = model.PINNs[jj].forward(this_XT)
            U1 = UV1[:, 0].reshape(-1)
            V1 = UV1[:, 1].reshape(-1)
            H1 = torch.sqrt(U1**2 + V1**2) 
            
            this_XT.requires_grad_()
            U0x, V0x, U1x, V1x = InterfaceGradients(U0, V0, U1, V1, this_XT)
            
            fig, ax = plt.subplots()
            plt.plot(this_t.detach().numpy(), U0x.reshape(-1).detach().numpy(), 'g-', label = 'PINN0-evaluated')
            plt.plot(this_t.detach().numpy(), U1x.reshape(-1).detach().numpy(), 'b--', label = 'PINN1-evaluated')
            #plt.plot(this_t.detach().numpy(), this_u.detach().numpy(), 'r--', label = 'Analytical')
            plt.xlabel('t')
            plt.ylabel('ux')
            plt.title("x = {}".format(_x))
            plt.legend()
            plt.savefig('../Figures3domain/' + figprefix + '_Ux_x_{}.png'.format(_x))
            plt.show()
            
            fig, ax = plt.subplots()
            plt.plot(this_t.detach().numpy(), V0x.reshape(-1).detach().numpy(), 'g-', label = 'PINN0-evaluated')
            plt.plot(this_t.detach().numpy(), V1x.reshape(-1).detach().numpy(), 'b--', label = 'PINN1-evaluated')
            #plt.plot(this_t.detach().numpy(), this_u.detach().numpy(), 'r--', label = 'Analytical')
            plt.xlabel('t')
            plt.ylabel('vx')
            plt.title("x = {}".format(_x))
            plt.legend()
            plt.savefig('../Figures3domain/' + figprefix + '_Vx_x_{}.png'.format(_x))
            plt.show()
            
            
            fig, ax = plt.subplots()
            plt.plot(this_t.detach().numpy(), U0.detach().numpy(), 'g-', label = 'PINN0-evaluated')
            plt.plot(this_t.detach().numpy(), U1.detach().numpy(), 'b--', label = 'PINN1-evaluated')
            plt.plot(this_t.detach().numpy(), this_u.detach().numpy(), 'r--', label = 'Analytical')
            plt.xlabel('t')
            plt.ylabel('u')
            plt.title("x = {}".format(_x))
            plt.legend()
            plt.savefig('../Figures3domain/' + figprefix + '_U_x_{}.png'.format(_x))
            plt.show()

            fig, ax = plt.subplots()
            plt.plot(this_t.detach().numpy(), V0.detach().numpy(), 'g-', label = 'PINN0-evaluated')
            plt.plot(this_t.detach().numpy(), V1.detach().numpy(), 'b--', label = 'PINN1-evaluated')
            plt.plot(this_t.detach().numpy(), this_v.detach().numpy(), 'r--', label = 'Analytical')
            plt.xlabel('t')
            plt.ylabel('v')
            plt.title("x = {}".format(_x))
            plt.legend()
            plt.savefig('../Figures3domain/' + figprefix + '_V_x_{}.png'.format(_x))
            plt.show()

            fig, ax = plt.subplots()
            plt.plot(this_t.detach().numpy(), H0.detach().numpy(), 'g-', label = 'PINN0-evaluated')
            plt.plot(this_t.detach().numpy(), H1.detach().numpy(), 'b--', label = 'PINN1-evaluated')
            plt.plot(this_t.detach().numpy(), this_h.detach().numpy(), 'r--', label = 'Analytical')
            plt.xlabel('t')
            plt.ylabel('|h|')
            plt.title("x = {}".format(_x))
            plt.legend()
            plt.savefig('../Figures3domain/' + figprefix + '_H_x_{}.png'.format(_x))
            plt.show()

for modelname in modelnames:
    makePlots(modelname)
