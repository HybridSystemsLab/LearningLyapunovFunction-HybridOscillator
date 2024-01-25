"""
This file defines utilities needed for the experiments and plotting, such as
neural network related functions, and formatting for results postprocessing.


Authors: Carlos A. Montenegro, Hybrid Systems Laboratory, UC Santa Cruz
         (GitHub: camonten)

         Santiago Jimenez Leudo, Hybrid Systems Laboratory, UC Santa Cruz
         (Github: sjleudo)
"""

import numpy as np

import jax.random as jrandom
import jax
import jax.nn as jnn
import jax.numpy as jnp
from jax.example_libraries import optimizers
from jax.flatten_util import ravel_pytree
from functools import partial

from math import sqrt
import matplotlib

from matplotlib.ticker import ScalarFormatter


class ScalarFormatterClass(ScalarFormatter):
    def _set_format(self):
        self.format = "%1.2f"


def latexify(fig_width=None, fig_height=None, columns=1):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    assert columns in [1, 2]

    if fig_width is None:
        fig_width = 3.39 if columns == 1 else 6.9  # width in inches

    if fig_height is None:
        golden_mean = (sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
        fig_height = fig_width * golden_mean  # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print(
            "WARNING: fig_height too large:"
            + fig_height
            + "so will reduce to"
            + MAX_HEIGHT_INCHES
            + "inches."
        )
        fig_height = MAX_HEIGHT_INCHES

    # NB (bart): default font-size in latex is 11. This should exactly match
    # the font size in the text if the figwidth is set appropriately.
    # Note that this does not hold if you put two figures next to each other using
    # minipage. You need to use subplots.
    params = {
        # "backend": "ps",
        "text.latex.preamble": [
            r"\usepackage{gensymb}",
            r"\usepackage{amsfonts}",
        ],
        "axes.labelsize": 12,  # fontsize for x and y labels (was 12 and before 10)
        "axes.titlesize": 12,
        "font.size": 12,  # was 12 and before 10
        "legend.fontsize": 12,  # was 12 and before 10
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "text.usetex": True,
        "figure.figsize": [fig_width, fig_height],
        "font.family": "serif",
    }

    matplotlib.rcParams.update(params)


# def combinations(arrays):
#     """Return a single array with combinations of parameters.

#     Parameters
#     ----------
#     arrays: list of np.array

#     Returns
#     -------
#     array - np.array
#         An array that contains all combinations of the input arrays
#     """
#     return np.array(np.meshgrid(*arrays)).T.reshape(-1, len(arrays))


""" NEURAL NET RELATED """


class NeuralNet:
    def __init__(
        self,
        net_dims,
        args,
        dyn,
        opt_name,
        act_fun,
        ell_C=None,
        ell_D=None,
        opt_kwargs={},
    ):
        """Constructor for neural network class used to approximate Lyapunov functions.

        Args:
            net_dims: dimensions of layers in neural network.
            args: NamedTuple containing the hyperparameters used for training.
            dyn: hybrid dynamcs with flow and jump maps.
            opt_name: name of the optimizer, choose between: "Adam", "Adagrad", "Nesterov", "GD", "SGD"
            act_fun: activation function for the NN. JAX function either jnp or jax.nn is accepted
            ell_C: stage cost for flows. Defaults to None.
            ell_D: stage cost for jumps. Defaults to None.
            opt_kwargs: optimizer arguments. Defaults to {}.
        """
        self.args = args
        self.opt_name = opt_name
        self._init_optimizer(opt_kwargs)
        self.net_dims = net_dims
        self.act_fun = act_fun

        self.key = jrandom.PRNGKey(5433)

        # self.forward is a function that passes a batch of data through the NN
        self.forward = jax.vmap(self.forward_indiv, in_axes=(0, None))

        # stage costs
        if ell_C is None and ell_D is None:
            print("Training for practical pAS!")
        else:
            print("Training for cost upper bound!")
        self.ell_C = ell_C
        self.ell_D = ell_D

        # hybrid dynamics
        self.dyn = dyn

        # Lagrange multipliers estimates for training
        self.lam_C, self.lam_D = args.lam_C, args.lam_D
        # penalty parameter
        self.mu = args.mu

    def __call__(self, x, params):
        """PyTorch-like calling of class-based neural network forward method."""

        return self.forward(x, params)

    @partial(jax.jit, static_argnums=0)
    def step(self, epoch, opt_state, dataset):
        """Do one step of optimization based on self.loss.

        Args:
            epoch: Currrent step of training.
            opt_state: Current state of network parameters.
            dataset: Training dataset.
        Returns:
            opt_state of NN parameters after this optimization step.
        """

        params = self.get_params(opt_state)
        grads = jax.grad(self.loss, argnums=0)(params, dataset)

        return self.opt_update(epoch, grads, opt_state)

    @partial(jax.jit, static_argnums=0)
    def loss(self, params, dataset):
        loss, _ = self.loss_and_constraints(params, dataset)
        return loss

    @partial(jax.jit, static_argnums=0)
    def constraints(self, params, dataset):
        _, constraints = self.loss_and_constraints(params, dataset)
        return constraints

    @partial(jax.jit, static_argnums=0)
    def loss_and_constraints(self, params, dataset):
        """Calculate loss for training neural network subject to Lyapunov constraints.

        Args:
            dataset: Training dataset.
            params: Trainable parameters of the neural network.
        Returns:
            Value of the loss function.
        """

        def forward_grad(x):
            return jax.vmap(self.model_grad_indiv, in_axes=(0, None))(x, params)

        def flows_constraint(x, params):
            return jax.vmap(self.flows_constraint_indiv, in_axes=(0, None))(x, params)

        def jumps_constraint(x, params):
            return jax.vmap(self.jumps_constraint_indiv, in_axes=(0, None))(x, params)

        def penalty(vect):
            return jnp.sum(jnn.relu(vect) ** 2)

        def constraint_pct(vect):
            frac_incorrect = jnp.sum(jnp.heaviside(vect, 0)) / vect.shape[0]
            return (1.0 - frac_incorrect) * 100.0

        """ Augmented Lagrangian """
        # penalize large Lipschitz constants of the NN
        param_loss = jnp.sum(jnp.linalg.norm(ravel_pytree(params)[0]))

        # enforce Lyapunov decrease condition along flows
        flows_consts = flows_constraint(dataset["x_flows"], params) + self.args.gam_C
        flows_loss = penalty(flows_consts)
        flows_const_pct = constraint_pct(flows_consts)

        # enforce Lyapunov decrease condition along jumps
        jumps_consts = jumps_constraint(dataset["x_jumps"], params) + self.args.gam_D
        jumps_loss = penalty(jumps_consts)
        jumps_const_pct = constraint_pct(jumps_consts)

        # Goal: make all derivatives small
        x_all = jnp.vstack(
            (
                dataset["x_flows"],
                dataset["x_jumps"],
            )
        )
        dV_loss = jnp.sum(jnp.linalg.norm(forward_grad(x_all)))

        # calculate slack variables
        s_flows = jnn.relu(-flows_consts - self.mu * self.lam_C)
        s_jumps = jnn.relu(-jumps_consts - self.mu * self.lam_D)

        # compute the augmented Lagrangian
        loss = (
            self.args.lam_grad * dV_loss
            + self.args.lam_param * param_loss
            - jnp.sum(self.lam_C * (-flows_consts - s_flows))
            - jnp.sum(self.lam_D * (-jumps_consts - s_jumps))
            + 1
            / (2 * self.mu)
            * (
                jnp.sum((-flows_consts - s_flows) ** 2)
                + jnp.sum((-jumps_consts - s_jumps) ** 2)
            )
        )

        all_consts = {
            "flows": flows_const_pct,
            "jumps": jumps_const_pct,
        }

        return loss, all_consts

    def flows_constraint_indiv(self, x, params):
        dV = self.model_grad_indiv(x, params)

        term1 = jnp.dot(self.dyn.f(x), dV)
        term2 = self.ell_C(x) if self.ell_C is not None else 0

        return term1 + term2

    def jumps_constraint_indiv(self, x, params):
        term1 = self.forward_indiv(
            self.dyn.g(x), params, self.act_fun
        ) - self.forward_indiv(x, params, self.act_fun)
        term2 = self.ell_D(x) if self.ell_D is not None else 0

        return term1 + term2

    def model_grad_indiv(self, x, params):
        return jax.grad(self.forward_indiv)(x, params, self.act_fun)

    @staticmethod
    def forward_indiv(x: jnp.ndarray, params: list, act_fun: jax.nn, eps: float = 0.1):
        """Forward pass through the neural network for a single instance.

        Args:
            x: Single instance to input into NN.
            params: Trainable parameters of neural network.
        Returns:
            Output from passing instance through neural network.
        """

        activations = x
        for G1, G2 in params:
            weight = jnp.hstack((G1.T @ G1 + eps * jnp.eye(G1.shape[1]), G2))
            outputs = activations.dot(weight)
            activations = act_fun(outputs)

        out = jnp.linalg.norm(activations)

        return jnp.squeeze(out)

    @staticmethod
    def random_layer_params(m: int, n: int, key):
        """Randomly intialize a weight matrix and bias vector.

        Args:
            m, n: Dimensions of weight matrix.
            key: Jax random number generator.
            scale: Scale of random initialization.
        Returns:
            weight: Randomly initialized weight matrix.
            bias: Randomly initialized bias vector.
        """

        q = int(np.ceil(0.5 * (m + 1)))

        w_key, b_key = jrandom.split(key)

        G1 = jrandom.normal(w_key, (q, m))
        G2 = jrandom.normal(b_key, (m, (n - m)))

        return G1, G2

    def init_params(self, verbose=False):
        """Intialize the parameters of the neural network with random values."""

        if not np.all(np.diff(self.net_dims) >= 0):
            raise ValueError(
                "Each layer must maintain or increase the dimension of its input!"
            )

        keys = jrandom.split(self.key, len(self.net_dims))
        dimensions = zip(self.net_dims[:-1], self.net_dims[1:], keys)
        params = [self.random_layer_params(m, n, k) for (m, n, k) in dimensions]

        if verbose is True:
            print(f"Param shapes are: {[(p1.shape, p2.shape) for (p1, p2) in params]}")

        return params

    def _init_optimizer(self, kwargs):
        if self.opt_name == "SGD":
            self.opt_init, self.opt_update, self.get_params = optimizers.sgd(**kwargs)

        elif self.opt_name == "Adam":
            self.opt_init, self.opt_update, self.get_params = optimizers.adam(**kwargs)

        elif self.opt_name == "Adagrad":
            (
                self.opt_init,
                self.opt_update,
                self.get_params,
            ) = optimizers.adagrad(**kwargs)

        elif self.opt_name == "Nesterov":
            (
                self.opt_init,
                self.opt_update,
                self.get_params,
            ) = optimizers.nesterov(**kwargs)

        else:
            opts = ["Adam", "Adagrad", "Nesterov", "GD", "SGD"]
            raise NotImplementedError(f'Supported optimizers: {" | ".join(opts)}')


class AverageMeter(object):
    def __init__(self, name, fmt=":f"):
        """Computes and stores the average and current value.

        Params:
            name: Name of variable that we are keeping track of.
            fmt: Format for printing data.
        """

        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        """Reset all of the intermediate values and averages."""

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update average value.

        Params:
            val: Value being appended.
            n: Number of items that were used to compute val.
        """

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
