import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
from sPINN_4 import StaggeredPINN as sPINN
from scipy import io
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
NDomains = 1
NHiddenLayers = 6
boundaries = (1/NDomains)*np.arange(0,NDomains+1)*10.0 - 5 
t_domain = [0., np.pi/2]
Layers = [2] + [70]*NHiddenLayers + [2]
N0 = int(50/NDomains) 
Nb = int(50)
Nf = int(20000/NDomains**0.5) 
Ni = 40 
InError = 0.1
Activation = nn.Tanh()
sPINNs_noSmoothing = []
sPINNs_Smoothing = []

spinn = sPINN(boundaries = boundaries,
		t_domain = t_domain,
		Layers = Layers,
		N0 = N0,
		Nb = Nb,
		Nf = Nf,
              	Ni = Ni,
	        InError = 0.0,
		Activation = Activation,
		device = device,
		model_name = "../Models/test1.model",
		do_smoothing = True)

spinn.Train(50000)
