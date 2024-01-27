# A Data-Driven Approach for Certifying Asymptotic Stability and Cost Evaluation for Hybrid Systems

Simulation for examples in HSCC'24 paper: A Data-Driven Approach for Certifying 
Asymptotic Stability and Cost Evaluation for Hybrid Systems

Author: Carlos A. Montenegro G.
Revision: 0.0.0.2 Date: 01/26/2024
https://github.com/camonten/DataDriven_Lyap_CE

----------------------------------------------------------------------------
# `Installation`

Prerequisites:
- `conda` (download `Anaconda Distribution` [here](https://www.anaconda.com/download))
- `git` (install as described [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git))
- For Windows users:
  - `WSL` (install as described [here](https://learn.microsoft.com/en-us/windows/wsl/install))
  

### Setting up the `conda` environment.

1. Clone the repository and `cd` into it
```bash
git clone https://github.com/camonten/DataDriven_Lyap_CE.git
```

```bash
cd DataDriven_Lyap_CE
```

2. Create a conda environment
```bash
conda env create --file=environment.yml
```

3. Activate the conda environment
```bash
conda activate hy_lyap_ce
```

4. Install `JAX` using
```bash
conda install -c conda-forge jaxlib
conda install -c conda-forge jax
```

5. For Mac and Linux users, open Jupyter using
```bash
jupyter notebook
```

For Windows users on WSL run
 ```bash
jupyter notebook --no-browser
```

and follow the instructions below.

----------------------------------------------------------------------------
## 5. Case of Study: Lyapunov Function and Cost Upper Bound for Oscillator with Impacts

The notebook `HyOscillator_Train.ipynb` contains the code to obtain the figures in Section 5 of the paper. You can open the Jupyter notebook `HyOscillator_Train.ipynb` either by VS Code or Anaconda Navigator, but please make sure the kernel in use corresponds to that of the conda environment you created before.  

By setting the boolean variable `training_lyapunov = True`, the setting of \*Section 5.1. Data-Driven Lyapunov Function\* is run and the figures therein are plotted. By setting the boolean variable `training_lyapunov = False`, the setting of \*Section 5.2. Data-Driven Cost Upper Bound\* is run, and the figures therein are plotted. See details below.

The following actions are executed upon the corresponding option.

 - `training_lyapunov = True`:
     1. it will instantiate the hybrid dynamical system with its data,
     2. it will create the coverings for the flow and jump sets used for training (with $\varepsilon = 0.01$ and $\mu = 1.1\varepsilon$),
     3. it will create and train a neural network that approximates a Lyapunov function with the following hyperparameters
         - net_dims = (2, 16, 32)
         - n_epochs = 230
         - $\tau_C = 0.037$ and $\tau_D = 0.049$
     4. it will generate figures 2 and 3, and thanks to Theorem 3.11 we certify the set $\mathcal{A} = \{ 0\}$ practically
         pre-asymptotically stable.

 - `training_lyapunov = False`:
     1. it will instantiate the hybrid dynamical system with its data,
     2. it will create the coverings for the flow and jump sets used for training (with $\varepsilon = 0.01$),
     3. it will instantiate the stage cost for flows as $L_C(x) = 0.5|x|^2$ and the stage cost for jumps as $L_D(x) = 0.15|x|^2$
     4. it will create and train a neural network that approximates a Lyapunov function with the following hyperparameters
         - net_dims = (2, 16, 32)
         - n_epochs = 500
         - $\eta_C = 0.058$ and $\eta_D = 0.044$
     5. it will generate figures 4 and 5, and thanks to Theorem 4.4 we certify that the trained neural net defines
         an upper bound on the cost of solutions to the hybrid oscillator.

Given the stochastic nature of training neural networks, the weights of the network may be updated every time the training process is run. Thus, the output figures might be slightly different upon each execution of the code. A workspace with selected trained weights will be provided to replicate the plots included in the final version of the paper.