import numpy as np

import jax
import jax.numpy as jnp

import jax.flatten_util
import flax.traverse_util
from flax.core import FrozenDict


def uniform_sample(low: float, high: float) -> float:
    return np.random.uniform(low, high)


def log_sample(low: float, high: float) -> float:
    return np.exp(np.random.uniform(np.log(low), np.log(high)))


def average_tree(tree):
    return jax.tree_util.tree_map(lambda x: jnp.mean(x), tree)


def add_trees(tree1, tree2):
    return jax.tree_util.tree_map(lambda x, y: x + y, tree1, tree2)


def diff_trees(tree1, tree2):
    return jax.tree_util.tree_map(lambda x, y: x - y, tree1, tree2)


def scalar_times_params(scalar, parameters):
    return jax.tree_util.tree_map(lambda p: scalar * p, parameters)


def scalars_times_params(scalars, parameters):
    return jax.tree_util.tree_map(lambda p: scalars * p, parameters)


def flatten(parameters) -> jnp.ndarray:
    return jax.flatten_util.ravel_pytree(parameters)[0]


def has_nan(parameters) -> bool:
    return any(jnp.isnan(flatten(parameters)))


def cast_float_as_pytree(f, tree):
    return jax.tree_map(lambda p: jnp.broadcast_to(f, p.shape), tree)


def replace_subtree_leaves(tree, path, newValue):
    """
    Functionally replace all leaves in the given subtree with `newValue`.

    Args:
        tree: A FrozenDict or nested dict (pytree).
        path: Tuple of keys leading to the subtree (e.g. ('params', 'slater')).
        newValue: The value to broadcast to every leaf (same shape as original leaves).

    Returns:
        A new FrozenDict with the modified subtree.
    """
    flat = flax.traverse_util.flatten_dict(tree)
    newFlat = {}

    for key, value in flat.items():
        if key[:len(path)] == path:
            if hasattr(value, "shape"):
                newLeaf = jnp.broadcast_to(newValue, value.shape)
            else:
                newLeaf = newValue
            newFlat[key] = newLeaf
        else:
            newFlat[key] = value

    return FrozenDict(flax.traverse_util.unflatten_dict(newFlat))


def smw_solve(P: jnp.ndarray, f: jnp.ndarray, lam: float) -> jnp.ndarray:
    """
    Solves S θ = f where S = λI + PᵀP using the Sherman-Morrison-Woodbury
    formula, operating in the W-dimensional space (W = P.shape[0]) rather
    than inverting the larger P-dimensional matrix.

        θ = λ⁻¹ f - λ⁻¹ Pᵀ (λI + PPᵀ)⁻¹ P f
    """
    A = lam * jnp.eye(P.shape[0], dtype=P.dtype) + P @ P.T
    y = jnp.linalg.solve(A, P @ f)
    return (f - P.T @ y) / lam


class EnergyMinimization:
    """Updates parameters using standard energy minimization (gradient descent)."""

    def __init__(self, logWavefunction, localEnergy):
        self.localEnergy = localEnergy
        self.parameterGrad = jax.grad(logWavefunction.apply, argnums=0)
        self.parameterGradBatchFlat = jax.vmap(
            self._get_parameter_gradient, in_axes=(None, 0)
        )

    def __call__(self, parameters, walkerRs, learningRate):

        if isinstance(learningRate, (float, int)):
            learningRateTree = cast_float_as_pytree(learningRate, parameters)
        else:
            learningRateTree = learningRate

        localEnergies = self.localEnergy.batch(parameters, walkerRs)
        parameterGrads = self.parameterGradBatchFlat(parameters, walkerRs)
        (flatParameters, unravel) = jax.flatten_util.ravel_pytree(parameters)
        (learningRateFlat, _) = jax.flatten_util.ravel_pytree(learningRateTree)

        exp_H = jnp.average(localEnergies)
        exp_O = jnp.average(parameterGrads, axis=0)
        exp_OH = jnp.average(localEnergies[:, None] * parameterGrads, axis=0)

        f_k = 2 * (exp_O * exp_H - exp_OH)
        parameterStep = learningRateFlat * f_k
        updatedParameters = unravel(flatParameters + parameterStep)

        return updatedParameters

    def _get_parameter_gradient(self, parameters, rs):
        localParameterGrads = self.parameterGrad(parameters, rs)
        return jax.flatten_util.ravel_pytree(localParameterGrads)[0]


class StochasticReconfiguration:
    """
    Updates parameters using Stochastic Reconfiguration (SR), optionally with
    SPRING momentum (Goldshlager et al. 2024, arXiv:2401.10190).

    Setting mu=0.0 (default) reduces to plain SR with no momentum.

    SPRING momentum formula:
        g_k = 2 (<O> <H> - <OH>)          (energy gradient)
        f_k = g_k + λ μ history            (momentum-augmented force)
        step = (S + λI)⁻¹ f_k
        new_history = scale * step
    """

    def __init__(self, logWavefunction, localEnergy, mode: str = 'normal'):
        self.localEnergy = localEnergy
        self.mode = mode
        self.parameterGrad = jax.grad(logWavefunction.apply, argnums=0)
        self.parameterGradBatchFlat = jax.vmap(
            self._get_parameter_gradient, in_axes=(None, 0)
        )

    def __call__(
        self,
        parameters,
        walkerRs: jnp.ndarray,
        learningRate: float,
        diagonalShift: float,
        mu: float = 0.0,
        history=0.0,
        parameter_mask=None,
    ):
        """
        Args:
            parameters: Current wavefunction parameters (pytree).
            walkerRs: Walker positions (W, N, dim).
            learningRate: Step size η.
            diagonalShift: SR regularization λ.
            mu: SPRING momentum coefficient (0.0 = plain SR).
            history: Previous parameter step for momentum (scalar or flat array).
            parameter_mask: Optional pytree matching parameters with 0/1 leaf
                values; leaves with 0 are held fixed. None means all active.

        Returns:
            (step_clipped, local_energies, updated_parameters, new_history)
        """
        localEnergies = self.localEnergy.batch(parameters, walkerRs)          # (W,)
        parameterGrads = self.parameterGradBatchFlat(parameters, walkerRs)    # (W, P)
        (flatParameters, unravel) = jax.flatten_util.ravel_pytree(parameters)
        numWalkers = parameterGrads.shape[0]

        if parameter_mask is not None:
            maskFlat = flatten(parameter_mask)
            parameterGrads = parameterGrads * maskFlat[None, :]

        exp_H = jnp.average(localEnergies)                                    # ()
        exp_O = jnp.average(parameterGrads, axis=0)                           # (P,)
        exp_OH = jnp.average(localEnergies[:, None] * parameterGrads, axis=0) # (P,)

        g_k = 2 * (exp_O * exp_H - exp_OH)                                    # (P,)
        f_k = g_k + diagonalShift * mu * history                              # (P,)
        U = parameterGrads - exp_O[None, :]                                   # (W, P)

        if self.mode == 'normal':
            s_jk = U.T @ U / numWalkers
            diagonalMatrix = diagonalShift * jnp.eye(s_jk.shape[0], dtype=s_jk.dtype)
            parameterStep = jnp.linalg.solve(s_jk + diagonalMatrix, f_k)

        elif self.mode == 'smw':
            P = U / jnp.sqrt(numWalkers)
            parameterStep = smw_solve(P, f_k, diagonalShift)

        else:
            raise Exception("Specified SR mode not supported.")

        fisherNorm = jnp.dot(f_k, parameterStep)
        diagonalCorrection = diagonalShift * jnp.linalg.norm(parameterStep) ** 2
        fisherNorm = jnp.sqrt(
            jnp.maximum(fisherNorm - diagonalCorrection, 1e-8)
        )
        scale = jnp.minimum(1.0, 1.0 / fisherNorm)
        scale = jnp.minimum(scale, 1.0 / jnp.linalg.norm(parameterStep))

        new_history = scale * parameterStep
        parameterStep = scale * learningRate * parameterStep

        if parameter_mask is not None:
            parameterStep = parameterStep * maskFlat

        updatedParameters = unravel(flatParameters + parameterStep)

        return (scale < 1.0, localEnergies, updatedParameters, new_history)

    def _get_parameter_gradient(self, parameters, rs):
        localParameterGrads = self.parameterGrad(parameters, rs)
        return jax.flatten_util.ravel_pytree(localParameterGrads)[0]