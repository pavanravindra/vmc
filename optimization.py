import numpy as np

import jax
import jax.numpy as jnp

import jax.flatten_util
import flax.traverse_util
from flax.core import FrozenDict

def uniformSample(low, high):
    return np.random.uniform(low, high)

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

    def __call__(self, parameters, walkerRs, learningRate, diagonalShift):
        
        localEnergies = self.localEnergy.batch(parameters, walkerRs)
        parameterGrads = self.parameterGradBatchFlat(parameters, walkerRs)
        (flatParameters,unravel) = jax.flatten_util.ravel_pytree(parameters)

        exp_H = jnp.average(localEnergies)
        exp_O = jnp.average(parameterGrads, axis=0)
        exp_OH = jnp.average(localEnergies[:,None] * parameterGrads, axis=0)

        f_k = 2 * (exp_O * exp_H - exp_OH)
        
        cov = (parameterGrads.T @ parameterGrads) / parameterGrads.shape[0]
        s_jk = cov - jnp.outer(exp_O,exp_O)
        diagonalMatrix = diagonalShift * jnp.eye(s_jk.shape[0])
        parameterStep = jnp.linalg.solve(s_jk + diagonalMatrix, f_k)
        
        fisherNorm = jnp.dot(f_k, parameterStep)
        diagonalCorrection = diagonalShift * jnp.linalg.norm(parameterStep)**2
        fisherNorm = jnp.sqrt(
            jnp.maximum(fisherNorm - diagonalCorrection, 1e-8)
        )
        scale = jnp.minimum(1.0, 1.0 / fisherNorm)
        scale = jnp.minimum(scale, 1.0 / jnp.linalg.norm(parameterStep))
        parameterStep = scale * learningRate * parameterStep
        updatedParameters = unravel(flatParameters + parameterStep)

        return ( scale < 1.0 , localEnergies , updatedParameters )

    def getParameterGradient(self, parameters, rs):
        localParameterGrads = self.parameterGrad(parameters, rs)
        return jax.flatten_util.ravel_pytree(localParameterGrads)[0]

def smw_solve(P, f, lam):
    """
    Given a system of equations:

        S theta = f     where
        S = lambda I + P.T @ P

    We are interested in computing theta by inverting S.

    If the shape of P is (W,P) with W << P, the inverse can be computed more
    efficiently using the Sherman-Morrison-Woodbury formula. It works by
    keeping things in the W-dimensional space, rather than inverting in the
    larger P-dimensional space.

    This is particularly useful in SR for NQS, where the number of parameters
    is almost always larger than the number of walkers.

    Formula:
    
        theta = lambda^{-1} f - lambda^{-1} P.T (lambda I + P P.T)^{-1} P f
    """
    term1 = f / lam
    term2 = P.T / lam @ jnp.linalg.solve(
        lam * jnp.eye(P.shape[0]) + P @ P.T, P @ f
    )
    return term1 - term2

class StochasticReconfigurationSMW:
    """
    Updates parameters using stochastic reconfiguration using the
    Sherman-Morrison-Woodbury formula (useful when the number of parameters is
    greater than the number of walkers).
    """

    def __init__(self, logWavefunction, localEnergy):
        self.localEnergy = localEnergy
        self.parameterGrad = jax.grad(logWavefunction.apply, argnums=0)
        self.parameterGradBatchFlat = jax.vmap(
            self.getParameterGradient, in_axes=(None,0)
        )

    def __call__(self, parameters, walkerRs, learningRate, diagonalShift):
        
        localEnergies = self.localEnergy.batch(parameters, walkerRs)
        parameterGrads = self.parameterGradBatchFlat(parameters, walkerRs)
        (flatParameters,unravel) = jax.flatten_util.ravel_pytree(parameters)
        numWalkers = parameterGrads.shape[0]

        exp_H = jnp.average(localEnergies)
        exp_O = jnp.average(parameterGrads, axis=0)
        exp_OH = jnp.average(localEnergies[:,None] * parameterGrads, axis=0)

        f_k = 2 * (exp_O * exp_H - exp_OH)
        U = parameterGrads - jnp.average(parameterGrads, axis=0, keepdims=True)
        P = U / jnp.sqrt(numWalkers)
        parameterStep = smw_solve(P, f_k, diagonalShift)
        
        fisherNorm = jnp.dot(f_k, parameterStep)
        diagonalCorrection = diagonalShift * jnp.linalg.norm(parameterStep)**2
        fisherNorm = jnp.sqrt(
            jnp.maximum(fisherNorm - diagonalCorrection, 1e-8)
        )
        scale = jnp.minimum(1.0, 1.0 / fisherNorm)
        scale = jnp.minimum(scale, 1.0 / jnp.linalg.norm(parameterStep))
        parameterStep = scale * learningRate * parameterStep
        updatedParameters = unravel(flatParameters + parameterStep)

        return ( scale < 1.0 , localEnergies , updatedParameters )

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
        (flatParameters,unravel) = jax.flatten_util.ravel_pytree(parameters)

        exp_H = jnp.average(localEnergies)
        exp_O = jnp.average(parameterGrads, axis=0)
        exp_OH = jnp.average(localEnergies[:,None] * parameterGrads, axis=0)

        g_k = 2 * (exp_O * exp_H - exp_OH)
        f_k = g_k + diagonalShift * mu * history

        cov = (parameterGrads.T @ parameterGrads) / parameterGrads.shape[0]
        s_jk = cov - jnp.outer(exp_O,exp_O)
        diagonalMatrix = diagonalShift * jnp.eye(s_jk.shape[0])
        parameterStep = jnp.linalg.solve(s_jk + diagonalMatrix, f_k)
        fisherNorm = jnp.dot(f_k, parameterStep)
        diagonalCorrection = diagonalShift * jnp.linalg.norm(parameterStep)**2
        fisherNorm = jnp.sqrt(
            jnp.maximum(fisherNorm - diagonalCorrection, 1e-8)
        )
        scale = jnp.minimum(1.0, 1.0 / fisherNorm)
        scale = jnp.minimum(scale, 1.0 / jnp.linalg.norm(parameterStep))
        history = scale * parameterStep
        parameterStep = scale * learningRate * parameterStep
        updatedParameters = unravel(flatParameters + parameterStep)

        return ( scale < 1.0 , localEnergies , updatedParameters , history )


    def getParameterGradient(self, parameters, rs):
        localParameterGrads = self.parameterGrad(parameters, rs)
        return jax.flatten_util.ravel_pytree(localParameterGrads)[0]
