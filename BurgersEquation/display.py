import torch
import numpy as np
import matplotlib.pyplot as plt
from PINN import *

models = [
            "GP_PINN_Burgers_error_0.5.data",
            "SVGP_PINN_Burgers_error_0.5.data",
            "Standard_PINN_Burgers_error_0.5.data",
            "PINN_Burgers_no_error.data"
    ]


for model in models:
    Losses = torch.load(model)
    for i in range(len(Losses[1])):
        plt.scatter(np.arange(len(Losses[1])),Losses[1], label=model, s=.1)
plt.title("Training Losses")
plt.xlabel("Epochs")
plt.yscale("log")
plt.legend(labelcolor=["blue","orange","green"])
plt.show()
