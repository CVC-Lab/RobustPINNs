
import argparse
import torch 
from torch import nn
from torch import optim
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ExpSineSquared
from scipy import io
import os, sys
import time
from scipy.special import iv

SAVE_PATH = "./saved_models/"
if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)

Re = np.pi/0.01
mu = 1/Re
lmbda = 1/(4*np.pi*mu)


class timer:
    def __init__(self):
        self.cur = time.time()
    def update(self):
        print("Time Taken = ", time.time() - self.cur)
        self.cur = time.time()

def exact_soln(XYT):
    x, y, t = XYT[:,0], XYT[:,1], XYT[:,2]
    c = 0.25/(1 + torch.exp( Re * (-t -4*x + 4*y)/32))
    u  = (0.75 - c).reshape(-1,1)
    v = (0.75 + c).reshape(-1,1)

    return torch.cat((u, v), 1)


def stacked_grid(x,y,t):
    X, Y, T = torch.meshgrid(x, y, t)
    return torch.hstack((X.flatten()[:, None], Y.flatten()[:, None], T.flatten()[:,None])).float()


def InitialCondition(N0, LB, UB, InError = 0.):
    n_per_dim = int(np.round(np.sqrt(N0)))
    t_in = torch.tensor([LB[-1]]).float()
    x_in = torch.linspace(LB[0], UB[0], n_per_dim).float()
    y_in = torch.linspace(LB[1], UB[1], n_per_dim).float()
    XYT_in = stacked_grid(x_in, y_in, t_in)
    uv0 = exact_soln(XYT_in)
    return XYT_in, uv0, uv0 + InError*torch.randn_like(uv0)

def BoundaryPoints(nb, xb, yb, LBs, UBs, where = 'left'):
    n_per_dim = int(np.round(np.sqrt(nb)))
    if where in ['left', 'right']:
        Xb = torch.tensor([xb]).float()
        Yb = torch.linspace(LBs[1], UBs[1], n_per_dim).float()
    else:
        Yb = torch.tensor([yb]).float()
        Xb = torch.linspace(LBs[0], UBs[0], n_per_dim).float()
    Tb = torch.linspace(LBs[-1], UBs[-1], nb).float()
    return stacked_grid(Xb,Yb,Tb)


def BoundaryCondition(Nb, LBs, UBs):
    ## left boundary: 
    ## Choose Nb time instances on x = LB, x = UB, y = LB, and y = UB
    nb = int(np.round(Nb/4))
    XYTleft = BoundaryPoints(nb, LBs[0], LBs[1], LBs, UBs, 'left')
    XYTright = BoundaryPoints(nb, UBs[0], LBs[1], LBs, UBs, 'right')
    XYTtop = BoundaryPoints(nb, LBs[0], UBs[1], LBs, UBs, 'top')
    XYTbottom = BoundaryPoints(nb, LBs[0], LBs[1], LBs, UBs, 'bottom')
    return XYTleft, XYTright, XYTtop, XYTbottom


class PINN(nn.Module):
    def __init__(self, 
                 LBs, 
                 UBs, 
                 Layers = [3, 256, 256, 256, 256, 1], 
                 N0 = 512, 
                 Nb = 256, 
                 Nf = 50000, 
                 Nt = 50000, 
                 InError = 0., 
                 Activation = nn.Tanh(), 
                 model_name = "2DBurgersPINN.model", 
                 device = torch.device('cpu'),
                 do_smoothing = False, 
                 N0pool = 512, 
                 N01 = 512,
                 threshold = 0.9, 
                 display_freq = 100,
                ):
        super(PINN,self).__init__()
        self.LBs = torch.tensor(LBs, dtype=torch.float32)
        self.UBs = torch.tensor(UBs, dtype=torch.float32)
        self.Layers = Layers
        self.in_dim  = Layers[0]
        self.out_dim = Layers[-1]
        self.N0 = N0
        self.Nb = Nb
        self.Nf = Nf
        self.Nt = Nt
        self.InError = InError
        self.Activation = Activation
        self.do_smoothing = do_smoothing
        self.N0pool = np.maximum(N0pool, N0)
        self.N01 = np.minimum(N0, N01)
        self.threshold = threshold
        self.XT0, self.uv0_true, self.uv0 = InitialCondition(self.N0pool, self.LBs, self.UBs, InError)
        if do_smoothing:
            self.uv0, self.GP, self.selections = self.Smoothing(self.XT0, self.uv0)
        else:
            self.GP, self.selections = None, [self.N0, self.N0]
        torch.save([self.XT0, self.uv0_true, self.uv0, self.GP], SAVE_PATH + model_name + "_IC.data")

        # if self.do_smoothing and (not self.GP == None):
        #     self.h0 = torch.tensor(self.GP.predict(self.XT0[:,0:-1].cpu().numpy())).float().reshape(-1)

        self.left, self.right, self.top, self.bottom = BoundaryCondition(self.Nb, self.LBs, self.UBs)
        self.device = device
        self._nn = self.build_model().to(self.device)
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

        
    def forward(self, x):
        x = 2*(x - self.LBs.to(x.device)/((self.UBs - self.LBs).to(x.device))) - 1.0
        #print(3.5, x.shape)
        return self._nn.forward(x.to(self.device))
    

    def Smoothing(self, XT0, uv0err):
        X = XT0[:, 0:-1].cpu()
        U = uv0err[:,0]
        V = uv0err[:,1]
        kernel_u = 1.0 * RBF(length_scale=(1.0, 1.0), length_scale_bounds=(1e-2, 1e3)) + \
                   WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+2))
        kernel_v = 1.0 * RBF(length_scale=(1.0, 1.0), length_scale_bounds=(1e-2, 1e3)) + \
                   WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+2))

        n0 = self.N01
        print("Smoothing is required. Selecting {} points initially".format(n0))
        selectionsThis_u = n0
        selectionsThis_v = n0
        indicesThis_u = np.random.choice(X.shape[0], n0, replace=False).tolist() # list(range(n0))
        indicesThis_v = np.random.choice(X.shape[0], n0, replace=False).tolist() # list(range(n0))
        GP_u = GPR(kernel = kernel_u, alpha = 0.0).fit(X[indicesThis_u], U[indicesThis_u].cpu())
        GP_v = GPR(kernel = kernel_v, alpha = 0.0).fit(X[indicesThis_v], V[indicesThis_v].cpu())
        kernelThis_u = GP_u.kernel_.get_params()["k1__k2"]
        kernelThis_v = GP_v.kernel_.get_params()["k1__k2"]
        for j in range (XT0.shape[0]):
            if j in indicesThis_u:
                continue
            x = np.array([X[j].tolist()])
            K_u = kernelThis_u.__call__(x, X[indicesThis_u])
            if np.max(K_u) < self.threshold and selectionsThis_u < self.N0:
                indicesThis_u.append(j)
                selectionsThis_u += 1
            if selectionsThis_u == self.N0:
                break


        for j in range (XT0.shape[0]):
            if j in indicesThis_v:
                continue
            x = np.array([X[j].tolist()])
            K_v = kernelThis_v.__call__(x, X[indicesThis_v])
            if np.max(K_v) < self.threshold and selectionsThis_v < self.N0:
                indicesThis_v.append(j)
                selectionsThis_v += 1
            if selectionsThis_v == self.N0:
                break

        print("Selecting total {}, {} points for u and v".format(selectionsThis_u, selectionsThis_v))
        if selectionsThis_u > n0:
            GP_u = GPR(kernel = kernel_u, alpha = 0.0).fit(X[indicesThis_u], U[indicesThis_u].cpu())
        if selectionsThis_v > n0:
            GP_v = GPR(kernel = kernel_v, alpha = 0.0).fit(X[indicesThis_v], V[indicesThis_v].cpu())
        print(GP_u.kernel_)
        print(GP_v.kernel_)
        u0=torch.tensor(GP_u.predict(X)).float().reshape(-1,1)
        v0=torch.tensor(GP_v.predict(X)).float().reshape(-1,1)
        return torch.cat((u0,v0),1), [GP_u, GP_v], [selectionsThis_u, selectionsThis_v]
    

    def ICLoss(self):
        uv0_pred = self.forward(self.XT0)
        loss = self.Loss(uv0_pred, self.uv0.to(self.device))

        return loss


    def BCLoss(self):
        U_L, U_R, U_T, U_B = self.forward(self.left), self.forward(self.right), self.forward(self.top), self.forward(self.bottom)
        ULx, URx, UTx, UBx = exact_soln(self.left).to(self.device), exact_soln(self.right).to(self.device), \
                             exact_soln(self.top).to(self.device), exact_soln(self.bottom).to(self.device)
        return self.Loss(U_L, ULx) + self.Loss(U_R, URx) + \
               self.Loss(U_T, UTx) + self.Loss(U_B, UBx)



    def PhysicsLoss(self, XYTGrid):
        xyt = XYTGrid.requires_grad_(True).to(self.device)
        uv = self.forward(xyt)
        u = uv[:,0]
        v = uv[:,1]

        u_grad = torch.autograd.grad(outputs=u, inputs=xyt, grad_outputs=torch.ones(u.shape).to(self.device), create_graph=True, allow_unused=True)[0]
        ux = u_grad[:,0]
        uy = u_grad[:,1]
        ut = u_grad[:,2]
        uxx = torch.autograd.grad(outputs=ux, inputs=xyt, create_graph=True, grad_outputs=torch.ones(u.shape).to(self.device),allow_unused=True)[0][:,0]
        uyy = torch.autograd.grad(outputs=uy, inputs=xyt, create_graph=True, grad_outputs=torch.ones(u.shape).to(self.device),allow_unused=True)[0][:,1]

        v_grad = torch.autograd.grad(outputs=v, inputs=xyt, grad_outputs=torch.ones(u.shape).to(self.device), create_graph=True, allow_unused=True)[0]
        vx = v_grad[:,0]
        vy = v_grad[:,1]
        vt = v_grad[:,2]
        vxx = torch.autograd.grad(outputs=vx, inputs=xyt, create_graph=True, grad_outputs=torch.ones(u.shape).to(self.device),allow_unused=True)[0][:,0]
        vyy = torch.autograd.grad(outputs=vy, inputs=xyt, create_graph=True, grad_outputs=torch.ones(u.shape).to(self.device),allow_unused=True)[0][:,1]

        
        lossf = self.Loss(ut + u*ux + v*uy - (1/Re)*(uxx + uyy), torch.zeros_like(uxx, device=self.device).float()) + \
                self.Loss(vt + u*vx + v*vy - (1/Re)*(vxx + vyy), torch.zeros_like(vxx, device=self.device).float())

        return lossf


    def Train(self, n_iters, weights=(1.0,1.0,1.0)):
        # T = timer()
        params = list(self.parameters())
        optimizer = optim.Adam(params, lr=1e-3)
        min_loss = 999999.0
        Training_Losses = []
        Test_Losses = []
        indices = np.random.choice(XYT.shape[0], self.Nf, replace=False)
        XTGrid = XYT[indices]
        Timer = timer()
        for jj in range(n_iters):
            ICLoss = self.ICLoss()
            BCLoss = self.BCLoss()
            PhysicsLoss = self.PhysicsLoss(XTGrid)
            Total_Loss = weights[0]*ICLoss \
                       + weights[1]*BCLoss \
                       + weights[2]*PhysicsLoss
            optimizer.zero_grad()
            Total_Loss.backward()
            optimizer.step()
            if jj > int(n_iters/4):
                if Total_Loss < min_loss:
                    torch.save(self.state_dict(), SAVE_PATH + self.model_name + ".model")
                    min_loss = float(Total_Loss)
            Test_XYT = XYT
            Test_U = self.forward(Test_XYT)
            Test_Loss = torch.mean( (Exact_U.to(self.device).reshape(-1) - Test_U.reshape(-1))**2 )
            Test_Losses.append(float(Test_Loss))
            Training_Losses.append(float(Total_Loss))
            if jj % self.display_freq == self.display_freq - 1 or jj == n_iters - 1 or jj == 0:
                print("Mode = {}".format(args.mode))
                print("Iteration Number = {}".format(jj+1))
                Timer.update()
                print("\tIC Loss = {}".format(float(ICLoss)))
                print("\tBC Loss = {}".format(float(BCLoss)))
                print("\tPhysics Loss = {}".format(float(PhysicsLoss)))
                print("\tTraining Loss = {}".format(float(Total_Loss)))
                print("\tTest Loss = {}".format(float(Test_Loss)))
                Losses = [Training_Losses, Test_Losses]
                torch.save(Losses, SAVE_PATH + self.model_name + ".data")
        return Training_Losses, Test_Losses

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type = int, default = 0, help = "0: no error, 1: error but no smoothing, 2: GP-smoothing, 3: SGP-smoothing")
    parser.add_argument("--epochs", type = int, default = 20000, help = "Number of epochs/iterations")
    parser.add_argument("--layers", type = int, default = 4, help = "Number of hidden layers")
    parser.add_argument("--nodes", type = int, default = 256, help = "Number of nodes per hidden layer")
    parser.add_argument("--N0", type = int, default = 1024, help = "Number of points on initial timeslice")
    parser.add_argument("--Nb", type = int, default = 256, help = "Number of points for boundary condition")
    parser.add_argument("--Nf", type = int, default = 50000, help = "Number of collocation points for enforcing physics")
    parser.add_argument("--Nt", type = int, default = 50000, help = "Number of points to evaluate test MSE")
    parser.add_argument("--N01", type = int, default = 512, help = "Initial selection of Inducing points for SGP")
    parser.add_argument("--M", type = int, default = 768, help = "Maximum allowed inducing points for SGP")
    parser.add_argument("--in-error", type = float, default = 0.0, help = "Error-size")
    parser.add_argument("--display-freq", type = int, default = 400, help = "How often to display errors (once per x epochs)")
    parser.add_argument("--model-name", type = str, default = "" , help = "model name")

    args = parser.parse_args()
    N0pool = args.N0
    InError = args.in_error
    if args.mode > 0 and InError == 0.0:
        print("Zero error not acceptable for these modes! Adding error-size of 0.5")
        InError = 0.5

    if args.mode == 0:
        N0 = N0pool
        do_smoothing = False
        N01 = N0pool
        model_name = "PINN_Burgers_2d_no_error" + ("_" + args.model_name.strip('_') if args.model_name else "")
    elif args.mode == 1:
        N0 = N0pool
        do_smoothing = False
        N01 = N0pool
        model_name = "PINN_Burgers_2d_no_smoothing" + ("_" + args.model_name.strip('_') if args.model_name else "")
    elif args.mode == 2:
        N0 = N0pool
        do_smoothing = True
        N01 = N0pool
        model_name = "PINN_Burgers_2d_GP" + ("_" + args.model_name.strip('_') if args.model_name else "")
    elif args.mode == 3:
        N0 = args.M
        do_smoothing = True
        N01 = args.N01
        model_name = "PINN_Burgers_2d_SGP" + ("_" + args.model_name.strip('_') if args.model_name else "")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
    LBs = [0., 0., 0.0]
    UBs = [1.0, 1.0, 1.0]

    Nx, Ny, Nt = 100, 100, 20
    X = torch.linspace(LBs[0], UBs[0], Nx).float()
    Y = torch.linspace(LBs[1], UBs[1], Ny).float()
    T = torch.linspace(LBs[2], UBs[2], Nt).float()
    XYT = stacked_grid(X,Y,T)
    Exact_U = exact_soln(XYT)
    
    umax, vmax = Exact_U[:,0].max().item(), Exact_U[:,1].max().item()
    umax_idx = Exact_U[:,0] == umax
    vmax_idx = Exact_U[:,1] == vmax
          
    # print(Exact_U)


    Layers = [3] +  args.layers*[args.nodes]+ [2]
    Activation = nn.Tanh()

    pinn = PINN(LBs = LBs,
                UBs = UBs,
                Layers = Layers,
                InError = InError,
                N0 = N0,
                N0pool = N0pool,
                N01 = N01,
                Nb = args.Nb,
                Nf = args.Nf,
                Nt = args.Nt,
                Activation = Activation,
                device = device,
                model_name = model_name,
                do_smoothing = do_smoothing,
                threshold = .9955555,
                display_freq = args.display_freq )

    Losses = pinn.Train(args.epochs)
    print("MSE Loss for best model: ", Losses[1][Losses[0].index(min(Losses[0]))])
