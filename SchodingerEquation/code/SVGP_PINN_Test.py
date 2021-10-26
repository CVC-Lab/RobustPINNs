import time
startTime = time.time()

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
from SVGP_PINN import PINN
from scipy import io
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type = int, default = 10000, help = 'The number of epochs/iterations to train')
parser.add_argument('--N0', type = int, default = 50, help = 'The number of points for the initial condition')
parser.add_argument('--Nb', type = int, default = 50, help = 'The number of points for the periodic boundary condition')
parser.add_argument('--Nf', type = int, default = 20000, help = 'The number of collocation points to enforce the physics')
parser.add_argument('--Nt', type = int, default = 10000, help = 'The number of points to choose when calculating test loss')
parser.add_argument('--error', type = float, default = 0.0, help = 'The SD of the error in the initial data')
parser.add_argument('--smooth', type = bool, default = False, help = 'Whether to do Gaussian Process smoothing of the initial data')
parser.add_argument('--model-name', type = str, default = 'PINN_test', help = 'The name of the model for storing the results')
parser.add_argument('--display-freq', type = int, default = 100, help = 'Display an update of the loss after this many epochs')
parser.add_argument('--threshold', type = float, default = 0.9, help = 'Threshold for inducing point selection for Gaussian Process')
parser.add_argument('--N0pool', type = int, default = 50, help = 'The number of points in the pool to selection inducing points from')
parser.add_argument('--N0start', type = int, default = 50, help = 'How many inducing points to start with for SVGP')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NHiddenLayers = 6
LBs = [-5.0, 0.0]
UBs = [5.0, np.pi/2.0]
Layers = [2] + [70]*NHiddenLayers + [2]
Activation = nn.Tanh()

pinn = PINN(    LBs = LBs,
		UBs = UBs,
		Layers = Layers,
		N0 = args.N0,
		Nb = args.Nb,
		Nf = args.Nf,
                Nt = args.Nt,
	        InError = args.error,
		Activation = Activation,
		device = device,
		model_name = "../Models/"+ args.model_name + ".model",
		do_smoothing = args.smooth,
                N0pool = args.N0pool,
                N0start = args.N0start,
                threshold = args.threshold,
                display_freq = args.display_freq )

Losses = pinn.Train(args.epochs)

torch.save(Losses, "../Models/"+ args.model_name + ".data")


endTime = time.time()
timeTaken = endTime - startTime
print(timeTaken)
