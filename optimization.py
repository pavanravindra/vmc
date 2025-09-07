import numpy as np

import jax
import jax.numpy as jnp

import jax.flatten_util
import flax.traverse_util
from flax.core import FrozenDict

def logSample(low, high):
    return np.exp(np.random.uniform(np.log(low), np.log(high)))

def averageTree(tree):
    return jax.tree_util.tree_map(lambda x: jnp.mean(x), tree)

def addTrees(tree1, tree2):
    return jax.tree_util.tree_map(lambda x, y: x + y, tree1, tree2)

def diffTrees(tree1, tree2):
    return jax.tree_util.tree_map(lambda x, y: x - y, tree1, tree2)

def scalarTimesParams(scalar, parameters):
    return jax.tree_util.tree_map(lambda p: scalar * p, parameters)

def scalarsTimesParams(scalars, parameters):
    return jax.tree_util.tree_map(lambda p: scalars * p, parameters)

def flatten(parameters):
    return jax.flatten_util.ravel_pytree(parameters)[0]

def hasnan(parameters):
    return any(jnp.isnan(flatten(parameters)))

def castFloatAsPytree(f, tree):
    return jax.tree_map(lambda p: jnp.broadcast_to(f, p.shape), tree)

def replaceSubtreeLeaves(tree, path, newValue):
    """
    Functionally replace all leaves in the given subtree with `new_value`.

    Args:
        tree: A FrozenDict or nested dict (pytree).
        path: Tuple of keys leading to the subtree (e.g. ('params', 'slater')).
        new_value: The value to broadcast to every leaf (same shape as original leaves).

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

class EnergyMinimization:
    """
    Updates parameters using standard energy minimization.
    """

    def __init__(self, logWavefunction, localEnergy):
        self.localEnergy = localEnergy
        self.parameterGrad = jax.grad(logWavefunction.apply, argnums=0)
        self.parameterGradBatchFlat = jax.vmap(
            self.getParameterGradient, in_axes=(None,0)
        )

    def __call__(self, parameters, walkerRs, learningRate):
        
        if isinstance(learningRate, (float, int)):
            learningRateTree = castFloatAsPytree(learningRate, parameters)
        else:
            learningRateTree = learningRate
        
        localEnergies = self.localEnergy.batch(parameters, walkerRs)
        parameterGrads = self.parameterGradBatchFlat(parameters, walkerRs)
        (flatParameters,unravel) = jax.flatten_util.ravel_pytree(parameters)
        (learningRateFlat,_) = jax.flatten_util.ravel_pytree(learningRateTree)

        exp_H = jnp.average(localEnergies)
        exp_O = jnp.average(parameterGrads, axis=0)
        exp_OH = jnp.average(localEnergies[:,None] * parameterGrads, axis=0)

        f_k = 2 * (exp_O * exp_H - exp_OH)
        parameterStep = learningRateFlat * f_k
        updatedParameters = unravel(flatParameters + parameterStep)
        
        return updatedParameters

    def getParameterGradient(self, parameters, rs):
        localParameterGrads = self.parameterGrad(parameters, rs)
        return jax.flatten_util.ravel_pytree(localParameterGrads)[0]

class StochasticReconfiguration:
    """
    Updates parameters using stochastic reconfiguration.

    NOTE: With this implementation, the `learningRate` parameter passed to
    `__call__` can either be a float that is applied evenly to every parameter,
    or a pytree that specifies parameter-specific learning rates.
    """

    def __init__(self, logWavefunction, localEnergy):
        self.localEnergy = localEnergy
        self.parameterGrad = jax.grad(logWavefunction.apply, argnums=0)
        self.parameterGradBatchFlat = jax.vmap(
            self.getParameterGradient, in_axes=(None,0)
        )

    def __call__(self, parameters, walkerRs, learningRate, diagonalShift=0., maxNorm=jnp.inf):
        
        if isinstance(learningRate, (float, int)):
            learningRateTree = castFloatAsPytree(learningRate, parameters)
        else:
            learningRateTree = learningRate

        localEnergies = self.localEnergy.batch(parameters, walkerRs)
        parameterGrads = self.parameterGradBatchFlat(parameters, walkerRs)
        (flatParameters,unravel) = jax.flatten_util.ravel_pytree(parameters)
        (learningRateFlat,_) = jax.flatten_util.ravel_pytree(learningRateTree)

        exp_H = jnp.average(localEnergies)
        exp_O = jnp.average(parameterGrads, axis=0)
        exp_OH = jnp.average(localEnergies[:,None] * parameterGrads, axis=0)

        f_k = 2 * (exp_O * exp_H - exp_OH) * learningRateFlat
        
        cov = (parameterGrads.T @ parameterGrads) / parameterGrads.shape[0]
        s_jk = cov - jnp.outer(exp_O,exp_O)
        diagonalMatrix = diagonalShift * jnp.eye(s_jk.shape[0])
        parameterStep = jnp.linalg.solve(s_jk + diagonalMatrix, f_k)
        scale = jnp.minimum(1.0, maxNorm / jnp.linalg.norm(parameterStep))
        updatedParameters = unravel(flatParameters + scale * parameterStep)

        """
        print(s_jk)
        print(f_k)
        print(scale)
        print(parameterStep)
        print(flatParameters + scale * parameterStep)

        print("-------------------------------")
        """
        
        return updatedParameters

    def getParameterGradient(self, parameters, rs):
        localParameterGrads = self.parameterGrad(parameters, rs)
        return jax.flatten_util.ravel_pytree(localParameterGrads)[0]

class DiagonalStochasticReconfiguration:
    """
    Updates parameters using stochastic reconfiguration with a diagonal Fisher
    information matrix.

    NOTE: With this implementation, the `learningRate` parameter passed to
    `__call__` can either be a float that is applied evenly to every parameter,
    or a pytree that specifies parameter-specific learning rates.
    """

    def __init__(self, logWavefunction, localEnergy):
        self.localEnergy = localEnergy
        self.parameterGrad = jax.grad(logWavefunction.apply, argnums=0)
        self.parameterGradBatchFlat = jax.vmap(
            self.getParameterGradient, in_axes=(None,0)
        )

    def __call__(self, parameters, walkerRs, learningRate, diagonalShift=0.):
        
        if isinstance(learningRate, (float, int)):
            learningRateTree = castFloatAsPytree(learningRate, parameters)
        else:
            learningRateTree = learningRate

        localEnergies = self.localEnergy.batch(parameters, walkerRs)
        parameterGrads = self.parameterGradBatchFlat(parameters, walkerRs)
        (flatParameters,unravel) = jax.flatten_util.ravel_pytree(parameters)
        (learningRateFlat,_) = jax.flatten_util.ravel_pytree(learningRateTree)
        
        exp_H = jnp.average(localEnergies)
        exp_O = jnp.average(parameterGrads, axis=0)
        exp_OH = jnp.average(localEnergies[:,None] * parameterGrads, axis=0)

        f_k = 2 * (exp_O * exp_H - exp_OH)

        s_jk = jnp.average(parameterGrads ** 2, axis=0) - (exp_O ** 2)
        parameterStep = learningRateFlat * f_k / (s_jk + diagonalShift)
        updatedParameters = unravel(flatParameters + parameterStep)
        
        return updatedParameters

    def getParameterGradient(self, parameters, rs):
        localParameterGrads = self.parameterGrad(parameters, rs)
        return jax.flatten_util.ravel_pytree(localParameterGrads)[0]

class StochasticReconfigurationMomentum:
    """
    Updates parameters using stochastic reconfiguration with momentum.

    TODO: This is deprecated since it doesn't allow for learning rates to vary
    for different types of parameters (i.e., I still need to allow for the
    learning rate to be a pytree).
    """

    def __init__(self, logWavefunction, localEnergy):
        self.localEnergy = localEnergy
        self.parameterGrad = jax.grad(logWavefunction.apply, argnums=0)
        self.parameterGradBatchFlat = jax.vmap(
            self.getParameterGradient, in_axes=(None,0)
        )

    def __call__(self, parameters, walkerRs, learningRate, diagonalShift, mu, history):

        localEnergies = self.localEnergy.batch(parameters, walkerRs)
        parameterGrads = self.parameterGradBatchFlat(parameters, walkerRs)
        (_,unravel) = jax.flatten_util.ravel_pytree(parameters)

        exp_H = jnp.average(localEnergies)
        exp_O = jnp.average(parameterGrads, axis=0)
        exp_OH = jnp.average(localEnergies[:,None] * parameterGrads, axis=0)

        g_k = 2 * (exp_O * exp_H - exp_OH)
        f_k = g_k + diagonalShift * mu * history

        jk_traj = parameterGrads.T[:,None,:] * parameterGrads.T[None,:,:]
        s_jk = jnp.average(jk_traj, axis=2) - jnp.outer(exp_O,exp_O)
        diagonalMatrix = diagonalShift * jnp.eye(s_jk.shape[0])
        parameterDirection = jnp.linalg.solve(s_jk + diagonalMatrix, f_k)
        parameterStep = unravel(learningRate * parameterDirection)
        updatedParameters = addTrees(parameters, parameterStep)
        
        return (updatedParameters,parameterDirection)

    def getParameterGradient(self, parameters, rs):
        localParameterGrads = self.parameterGrad(parameters, rs)
        return jax.flatten_util.ravel_pytree(localParameterGrads)[0]
