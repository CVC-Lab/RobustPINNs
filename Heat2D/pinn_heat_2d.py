
import argparse
import torch 
from torch import nn
from torch import optim
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ExpSineSquared
from scipy import io
import os
import time

SAVE_PATH = "./saved_models/"
if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)


class timer:
    def __init__(self):
        self.cur = time.time()
    def update(self):
        print("Time Taken = ", time.time() - self.cur)
        self.cur = time.time()

def exact_soln(XYT):
    x, y, t = XYT[:,0], XYT[:,1], XYT[:,2]
    return 3*torch.sin(np.pi*x)*torch.sin(np.pi*y)*torch.exp(-2*(np.pi**2)*t) + torch.sin(3*np.pi*x)*torch.sin(np.pi*y)*torch.exp(-10*(np.pi**2)*t)

def stacked_grid(x,y,t):
    X, Y, T = torch.meshgrid(x, y, t)
    return torch.hstack((X.flatten()[:, None], Y.flatten()[:, None], T.flatten()[:,None])).float()




def InitialCondition(N0, LB, UB, InError = 0.):
    n_per_dim = int(np.round(np.sqrt(N0)))
    t_in = torch.tensor([LB[-1]]).float()
    x_in = torch.linspace(LB[0], UB[0], n_per_dim).float()
    y_in = torch.linspace(LB[1], UB[1], n_per_dim).float()
    XYT_in = stacked_grid(x_in, y_in, t_in)
    u0 = exact_soln(XYT_in)
    return XYT_in, u0, u0 + InError*torch.randn_like(u0)

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
                 model_name = "2DHeatPINN.model", 
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
        self.XT0, self.h0_true, self.h0 = InitialCondition(self.N0pool, self.LBs, self.UBs, InError)
        if do_smoothing:
            self.h0, self.GP, self.selections = self.Smoothing(self.XT0, self.h0)
        else:
            self.GP, self.selections = None, self.N0
        torch.save([self.XT0, self.h0_true, self.h0, self.GP], SAVE_PATH + model_name + "_IC.data")

        if self.do_smoothing and (not self.GP == None):
            self.h0 = torch.tensor(self.GP.predict(self.XT0[:,0:-1].cpu().numpy())).float().reshape(-1)

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
        return self._nn.forward(x.to(self.device)).reshape(-1)
    

    def Smoothing(self, XT0, h0err):
        X = XT0[:, 0:-1].cpu()
        kernel = 1.0 * RBF(length_scale=(1.0, 1.0), length_scale_bounds=(1e-2, 1e3)) + \
                 WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+2))
        n0 = self.N01
        print("Smoothing is required. Selecting {} points initially".format(n0))
        selectionsThis = n0
        indicesThis = np.random.choice(X.shape[0], n0, replace=False).tolist() # list(range(n0))
        GP = GPR(kernel = kernel, alpha = 0.0).fit(X[indicesThis], h0err[indicesThis].cpu())
        kernelThis = GP.kernel_.get_params()["k1__k2"]
        for j in range (XT0.shape[0]):
            if j in indicesThis:
                continue
            x = np.array([X[j].tolist()])
            K = kernelThis.__call__(x, X[indicesThis])
            if np.max(K) < self.threshold and selectionsThis < self.N0:
                indicesThis.append(j)
                selectionsThis += 1
            if selectionsThis == self.N0:
                break
        print("Selecting total {} points".format(selectionsThis))        
        if selectionsThis > n0:
            GP = GPR(kernel = kernel, alpha = 0.0).fit(X[indicesThis], h0err[indicesThis].cpu())
        print(GP.kernel_)
        h0=torch.tensor(GP.predict(X)).float().reshape(-1)
        return h0, GP, selectionsThis
    

    def ICLoss(self):
        h0_pred = self.forward(self.XT0)
        loss = self.Loss(h0_pred, self.h0.to(self.device))

        return loss


    def BCLoss(self):
        U_L, U_R, U_T, U_B = self.forward(self.left), self.forward(self.right), self.forward(self.top), self.forward(self.bottom)
        return self.Loss(U_L, torch.zeros_like(U_L, device=self.device).float()) + \
               self.Loss(U_R, torch.zeros_like(U_R, device=self.device).float()) + \
               self.Loss(U_T, torch.zeros_like(U_T, device=self.device).float()) + \
               self.Loss(U_B, torch.zeros_like(U_B, device=self.device).float())



    def PhysicsLoss(self, XYTGrid):
        xyt = XYTGrid.requires_grad_(True).to(self.device)
        u = self.forward(xyt)
        u_grad = torch.autograd.grad(outputs=u, inputs=xyt, grad_outputs=torch.ones(u.shape).to(self.device), create_graph=True, allow_unused=True)[0]
        ux = u_grad[:,0]
        uy = u_grad[:,1]
        ut = u_grad[:,2]
        uxx = torch.autograd.grad(outputs=ux, inputs=xyt, create_graph=True, grad_outputs=torch.ones(u.shape).to(self.device),allow_unused=True)[0][:,0]
        uyy = torch.autograd.grad(outputs=uy, inputs=xyt, create_graph=True, grad_outputs=torch.ones(u.shape).to(self.device),allow_unused=True)[0][:,1]
        
        lossf = self.Loss(uxx+uyy-ut, torch.zeros_like(uxx, device=self.device).float())

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
        print("Zero error not acceptable for these modes! Adding error-size of 1.0")
        InError = 1.0

    if args.mode == 0:
        N0 = N0pool
        InError = 0.
        do_smoothing = False
        N01 = N0pool
        model_name = "PINN_Heat_2d_no_error" + ("_" + args.model_name.strip('_') if args.model_name else "")
    elif args.mode == 1:
        N0 = N0pool
        InError = 1.0
        do_smoothing = False
        N01 = N0pool
        model_name = "PINN_Heat_2d_no_smoothing" + ("_" + args.model_name.strip('_') if args.model_name else "")
    elif args.mode == 2:
        N0 = N0pool
        InError = 1.0
        do_smoothing = True
        N01 = N0pool
        model_name = "PINN_Heat_2d_GP" + ("_" + args.model_name.strip('_') if args.model_name else "")
    elif args.mode == 3:
        N0 = args.M
        InError = 1.0
        do_smoothing = True
        N01 = args.N01
        model_name = "PINN_Heat_2d_SGP" + ("_" + args.model_name.strip('_') if args.model_name else "")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
    LBs = [0., 0., 0.0]
    UBs = [1.0, 1.0, 0.1]

    Nx, Ny, Nt = 100, 100, 20
    X = torch.linspace(LBs[0], UBs[0], Nx).float()
    Y = torch.linspace(LBs[1], UBs[1], Ny).float()
    T = torch.linspace(LBs[2], UBs[2], Nt).float()
    XYT = stacked_grid(X,Y,T)
    Exact_U = exact_soln(XYT)


    Layers = [3] + args.layers*[args.nodes] + [1]
    Activation = nn.Tanh()

    pinn = PINN(LBs = LBs,
                UBs = UBs,
                Layers = Layers,
                InError = InError,
                Nb = args.Nb,
                Nf = args.Nf,
                Nt = args.Nt,
                N0 = N0,
                N0pool = N0pool,
                N01 = N01,
                Activation = Activation,
                device = device,
                model_name = model_name,
                do_smoothing = do_smoothing,
                threshold = .9955555,
                display_freq = args.display_freq )

    Losses = pinn.Train(args.epochs)
    print("MSE Loss for best model: ", Losses[1][Losses[0].index(min(Losses[0]))])
