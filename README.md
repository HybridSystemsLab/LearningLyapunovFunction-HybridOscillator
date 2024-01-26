# A Data-Driven Approach for Certifying Asymptotic Stability and Cost Evaluation for Hybrid Systems

Simulation for examples in HSCC'24 paper: A Data-Driven Approach for Certifying 
Asymptotic Stability and Cost Evaluation for Hybrid Systems

Author: Carlos A. Montenegro G.
Revision: 0.0.0.2 Date: 01/26/2024
https://github.com/camonten/DataDriven_Lyap_CE

----------------------------------------------------------------------------
# `Installation`

Prerequisites:
- `conda` (install as described [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html))
- `JAX`

### Setting up the `conda` environment.

1. Clone the repository and `cd` into it
```bash
git clone https://github.com/camonten/DataDriven_Lyap_CE.git && cd DataDriven_Lyap_CE
```

2. Create a conda environment
```bash
conda env create --file=environment.yml
```

3. Activate the conda environment
```bash
conda activate hy_lyap_ce
```

4. Install `JAX` as described [here](https://jax.readthedocs.io/en/latest/installation.html)


----------------------------------------------------------------------------
## Case of Study: Lyapunov Function and Cost Upper Bound for Oscillator with Impacts

1. If you want to train from scratch the models and generate new plots, run [HyOscillator_Train.ipynb](HyOscillator_Train.ipynb) and follow the instructions therein.
   
    As a result,

     - `For practical pre-asymptotic stability`:
         1. it will instantiate the hybrid dynamical system with its data,
         2. it will create the coverings for the flow and jump sets used for training (with $\varepsilon = 0.01$ and $\mu = 1.1\varepsilon$),
         3. it will create and train a neural network that approximates a Lyapunov function with the following hyperparameters
              - net_dims = (2, 16, 32)
              - n_epochs = 500
              - $\tau_C = 0.053$ and $\tau_D = 0.049$
          4. it will generate figures 2 and 3, and thanks to Theorem 3.11 we certify the set $\mathcal{A} = \{ 0\}$ practically
             pre-asymptotically stable.

     - `For data-driven cost upper bound`:
         1. it will instantiate the hybrid dynamical system with its data,
         2. it will create the coverings for the flow and jump sets used for training (with $\varepsilon = 0.01$),
         3. it will instantiate the stage cost for flows as $L_C(x) = 0.5|x|^2$ and the stage cost for jumps as $L_D(x) = 0.15|x|^2$
         4. it will create and train a neural network that approximates a Lyapunov function with the following hyperparameters
              - net_dims = (2, 16, 32)
              - n_epochs = 500
              - $\eta_C = 0.058$ and $\eta_D = 0.044$
          5. it will generate figures 4 and 5, and thanks to Theorem 4.4 we certify that the trained neural net defines
             an upper bound on the cost of solutions to the hybrid oscillator.

2. If you want to just recreate the figures without training, run [HyOscillator_Figs.ipynb](HyOscillator_Figs.ipynb) and
   follow the instructions therein.
   
   As a result, it will load the weights of the last training and reproduce figures 2 and 3 (for practical pre-asymptotic stability) and figures 4 and 5 (for data-driven cost upper bound).








