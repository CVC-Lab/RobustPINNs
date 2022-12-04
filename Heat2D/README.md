The script `pinn_heat_2d.py` can be used to run the solver for Burgers' equation for a various number of scenarios. The different modes can be selected by modifying the command line arguments. The code is set up to run a single domain PINN for GP-smoothed, SGP-smoothed, or smoothing-free training. The different arguments accepted by the script are summarized below:

| Options | Description|
|---|---|
|**`--layers`**|       The number of hidden layers in the neural network|
|**`--nodes`**|        The number of nodes per hidden layer in the neural network|
|**`--N0`** |               The number of points to use on the initial condition|
|**`--N01`** |               Initial selection of Inducing points for SGP|
|**`--M`** |               Maximum allowed inducing points for SGP|
|**`--Nb`** |               The number of points to use on the boundary condition|
|**`--Nf`** |             The number of collocation points to use|
|**`--Nt`** |              The number of points to use to calculate the MSE loss|
|**`--in-error`**|         The standard deviation of the noise for the initial condition|
|**`--epochs`**|       The number of epochs to train the neural network|
|**`--display-freq`**|  How often to display loss information|
|**`--model-name`**| File name to save the model|


To run the code: </br>
`python pinn_heat_2d.py <options>`
