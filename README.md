# A Data-Driven Approach for Certifying Asymptotic Stability and Cost Evaluation for Hybrid Systems

Simulation for examples in HSCC'24 paper: A Data-Driven Approach for Certifying 
Asymptotic Stability and Cost Evaluation for Hybrid Systems

Author: Carlos A. Montenegro G.
Revision: 0.0.0.3 Date: 02/26/2024
https://github.com/camonten/DataDriven_Lyap_CE

----------------------------------------------------------------------------
# `Structure of the project`

The project is structured as follows:

```
.
├── HyArc.mat
├── HyOscillator_Dynamics
│   ├── OscillatorImpacts.m
│   └── main.m
├── HyOscillator_Train.ipynb
├── README.md
├── environment.yml
├── saved_models
│   ├── trained_ce_final.npy
│   └── trained_lyap_final.npy
└── utils.py
```

Files description:

- `HyArc.mat`: Matlab's workspace file containing the hybrid arc used to get the plot in Figure 2.
- `HyOscillator_Dynamics`
    - `OscillatorImpacts.m`: Oscillator with impacts modeled as a HybridSystem subclass
    - `main.m`: Compute solutions to the oscillator with impacts
- `HyOscillator_Train.ipynb`: Jupyter notebook to run the main code to get Figures 2, 3, 4, and 5.
- `environment.yml`
- `saved_models`: saved models trained with the hyperparameters reported in Section 5.
    - `trained_ce_final.npy`
    - `trained_lyap_final.npy`
- `utils.py`: Utilities needed for the experiments and plotting, such as neural network-related functions, and formatting for results post-processing.


# `Installation`

Prerequisites:
- `git` (install as described [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git))
- `Matlab` (Developed in R2022b). Install HyEQ Toolbox 3.0.0.76, available [at this link](https://www.mathworks.com/matlabcentral/fileexchange/41372-hybrid-equations-toolbox). 
- `Docker` (install as described [here](https://www.docker.com/products/docker-desktop/))

### Setting up the docker image

1. Clone the repository and `cd` into it
```bash
git clone https://github.com/camonten/DataDriven_Lyap_CE.git
```

```bash
cd DataDriven_Lyap_CE
```

2. Open `Docker Desktop` (or make sure `Docker` is running).

3. Build the Docker image using the following command:
```bash
docker build -t dockerfile .
```

4. Once the Docker image is built, run the Docker container based on this image:
```bash
docker run -it -p 8888:8888 dockerfile
```

5. Inside the Container (i.e., in the same terminal you ran the Docker container in the previous step) run the following command:
```bash
jupyter notebook --ip 0.0.0.0 --NotebookApp.token='' --NotebookApp.password='' --no-browser --allow-root
```

6. Access the following URL:
 ```bash
localhost:8888
```

and follow the instructions below.

----------------------------------------------------------------------------
## Section 5. Case of Study: Lyapunov Function and Cost Upper Bound for Oscillator with Impacts

The notebook `HyOscillator_Train.ipynb` contains the code to obtain the figures in Section 5 of the paper, please make sure the kernel in use corresponds to that of the conda environment you created before.  

A single code is used to obtain the simulation and figures of the Lyapunov function or the cost upper bound. Either can be selected by specifying the corresponding value of the input parameter  `training_lyapunov` as follows:

By setting `training_lyapunov = True`, the scenario of \*Section 5.1. Data-Driven Lyapunov Function\* on the paper is run and the figures therein are plotted. By setting `training_lyapunov = False`, the scenario of \*Section 5.2. Data-Driven Cost Upper Bound\* on the paper is run, and the figures therein are plotted. See details below.

The following actions are executed by running every cell in the Jupyter notebook under the corresponding option.

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
     5. it will generate figures 4 and 5, and thanks to Theorem 4.4 we certify that the trained neural net defines an upper bound on the cost of solutions to the hybrid oscillator.

Given the stochastic nature of training neural networks, the weights of the network may be updated every time the training process is run. Thus, the output figures might be slightly different upon each execution of the code. A workspace with selected trained weights will be provided to replicate the plots included in the final version of the paper.