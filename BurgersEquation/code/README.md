The script `PINN.py` can be used to run the solver for Burgers' equation for a various number of scenarios. The different modes can be selected by modifying the command line arguments. The different arguments accepted by the script are summarized below:

| Options | Description|
|---|---|
|**`--domains`**|      The number of domains when using a cPINN|
|**`--layers`**|       The number of hidden layers in the neural network|
|**`--nodes`**|        The number of nodes per hidden layer in the neural network|
|**`--N0`** |               The number of points to use on the initial condition|
|**`--Nb`** |               The number of points to use on the boundary condition|
|**`--Nf`** |             The number of collocation points to use|
|**`--Ni`** |              The number of points to use on interfaces for cPINNs|
|**`--Nt`** |              The number of points to use to calculate the MSE loss|
|**`--error`**|         The standard deviation of the noise for the initial condition|
|**`--smooth`**|              Do SGP/GP smoothing|
|**`--N0pool`**|        The pool of points to select inducing points from for GP/SGP|
|**`--threshold`**|     The threshold for choosing inducing point for GP/SGP|
|**`--epochs`**|       The number of epochs to train the neural network|
|**`--display-freq`**|  How often to display loss information|
|**`--model-name`**| File name to save the model|
|**`--do-colehopf`**| Perform Cole-Hopf transformation constraint|

To run the code: </br>
`cd code`
`mkdir -p ../models` </br>
`python PINN.py <options>`

For running with full Gaussian process smoothing, the values for `N0` and `N0pool` should be set to equal values, same as the number of points considered on the initial time slice. For a sparse GP, choose `N0pool > N0`. When using `cPINNs`, the options `layers` and `nodes` represent the number of hidden layers and the number of nodels per hidder layer for individual MLPs in each of the subdomains.
