import numpy as np

import jax
import jax.numpy as jnp

import jax.flatten_util
import flax.traverse_util
from flax.core import FrozenDict

def flatten(parameters):
    return jax.flatten_util.ravel_pytree(parameters)[0]

def hasnan(parameters):
    return any(jnp.isnan(flatten(parameters)))

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
    A = lam * jnp.eye(P.shape[0], dtype=P.dtype) + P @ P.T
    y = jnp.linalg.solve(A, P @ f)
    return (f - P.T @ y) / lam

class StochasticReconfigurationMomentumDegrade:
    """
    Updates parameters using stochastic reconfiguration with momentum.

    TODO: This is deprecated since it doesn't allow for learning rates to vary
    for different types of parameters (i.e., I still need to allow for the
    learning rate to be a pytree).
    """

    def __init__(self, logWavefunction, localEnergy, mode='normal'):
        self.localEnergy = localEnergy
        self.mode = mode
        self.parameterGrad = jax.grad(logWavefunction.apply, argnums=0)
        self.parameterGradBatchFlat = jax.vmap(
            self.getParameterGradient, in_axes=(None,0)
        )

    def __call__(self, parameters, walkerRs, learningRate, diagonalShift, mu, history):

        localEnergies = self.localEnergy.batch(parameters, walkerRs)
        parameterGrads = self.parameterGradBatchFlat(parameters, walkerRs)
        (flatParameters,unravel) = jax.flatten_util.ravel_pytree(parameters)
        numWalkers = parameterGrads.shape[0]

        exp_H = jnp.average(localEnergies)
        exp_O = jnp.average(parameterGrads, axis=0)
        exp_OH = jnp.average(localEnergies[:,None] * parameterGrads, axis=0)

        g_k = 2 * (exp_O * exp_H - exp_OH)
        f_k = g_k + diagonalShift * mu * history
        U = parameterGrads - exp_O[None,:]

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
        diagonalCorrection = diagonalShift * jnp.linalg.norm(parameterStep)**2
        fisherNorm = jnp.sqrt(
            jnp.maximum(fisherNorm - diagonalCorrection, 1e-8)
        )
        scale = jnp.minimum(1.0, 1.0 / fisherNorm)
        scale = jnp.minimum(scale, 1.0 / jnp.linalg.norm(parameterStep))
        newHistory = scale * parameterStep
        parameterStep = scale * learningRate * parameterStep
        updatedParameters = unravel(flatParameters + parameterStep)

        return ( scale < 1.0 , localEnergies , updatedParameters , newHistory )

    def getParameterGradient(self, parameters, rs):
        localParameterGrads = self.parameterGrad(parameters, rs)
        return jax.flatten_util.ravel_pytree(localParameterGrads)[0]

class StochasticReconfigurationMomentumDegradeExpH:
    """
    Updates parameters using stochastic reconfiguration with momentum.

    TODO: This is deprecated since it doesn't allow for learning rates to vary
    for different types of parameters (i.e., I still need to allow for the
    learning rate to be a pytree).
    """

    def __init__(self, logWavefunction, localEnergy, mode='normal'):
        self.localEnergy = localEnergy
        self.mode = mode

    def __call__(self, parameters, walkerRs, learningRate, diagonalShift, mu, history):

        localEnergies = self.localEnergy.batch(parameters, walkerRs)
        (flatParameters,unravel) = jax.flatten_util.ravel_pytree(parameters)

        exp_H = jnp.average(localEnergies)

        newHistory = exp_H
        updatedParameters = unravel(flatParameters + exp_H)

        return ( True , localEnergies , updatedParameters , newHistory )

class StochasticReconfigurationMomentumDegradeExpO:
    """
    Updates parameters using stochastic reconfiguration with momentum.

    TODO: This is deprecated since it doesn't allow for learning rates to vary
    for different types of parameters (i.e., I still need to allow for the
    learning rate to be a pytree).
    """

    def __init__(self, logWavefunction, localEnergy, mode='normal'):
        self.mode = mode
        self.parameterGrad = jax.grad(logWavefunction.apply, argnums=0)
        self.parameterGradBatchFlat = jax.vmap(
            self.getParameterGradient, in_axes=(None,0)
        )

    def __call__(self, parameters, walkerRs, learningRate, diagonalShift, mu, history):

        parameterGrads = self.parameterGradBatchFlat(parameters, walkerRs)
        (flatParameters,unravel) = jax.flatten_util.ravel_pytree(parameters)

        exp_O = jnp.average(parameterGrads, axis=0)

        localEnergies = 0.0
        newHistory = exp_O
        updatedParameters = unravel(flatParameters + exp_O)

        return ( True , localEnergies , updatedParameters , newHistory )

    def getParameterGradient(self, parameters, rs):
        localParameterGrads = self.parameterGrad(parameters, rs)
        return jax.flatten_util.ravel_pytree(localParameterGrads)[0]

class StochasticReconfigurationMomentumDegradeFK:
    """
    Updates parameters using stochastic reconfiguration with momentum.

    TODO: This is deprecated since it doesn't allow for learning rates to vary
    for different types of parameters (i.e., I still need to allow for the
    learning rate to be a pytree).
    """

    def __init__(self, logWavefunction, localEnergy, mode='normal'):
        self.localEnergy = localEnergy
        self.mode = mode
        self.parameterGrad = jax.grad(logWavefunction.apply, argnums=0)
        self.parameterGradBatchFlat = jax.vmap(
            self.getParameterGradient, in_axes=(None,0)
        )

    def __call__(self, parameters, walkerRs, learningRate, diagonalShift, mu, history):

        localEnergies = self.localEnergy.batch(parameters, walkerRs)
        parameterGrads = self.parameterGradBatchFlat(parameters, walkerRs)
        (flatParameters,unravel) = jax.flatten_util.ravel_pytree(parameters)
        numWalkers = parameterGrads.shape[0]

        exp_H = jnp.average(localEnergies)
        exp_O = jnp.average(parameterGrads, axis=0)
        exp_OH = jnp.average(localEnergies[:,None] * parameterGrads, axis=0)

        g_k = 2 * (exp_O * exp_H - exp_OH)
        f_k = g_k + diagonalShift * mu * history

        newHistory = f_k
        updatedParameters = unravel(flatParameters + f_k)

        return ( True , localEnergies , updatedParameters , newHistory )

    def getParameterGradient(self, parameters, rs):
        localParameterGrads = self.parameterGrad(parameters, rs)
        return jax.flatten_util.ravel_pytree(localParameterGrads)[0]

class StochasticReconfigurationMomentumDegradePS:
    """
    Updates parameters using stochastic reconfiguration with momentum.

    TODO: This is deprecated since it doesn't allow for learning rates to vary
    for different types of parameters (i.e., I still need to allow for the
    learning rate to be a pytree).
    """

    def __init__(self, logWavefunction, localEnergy, mode='normal'):
        self.localEnergy = localEnergy
        self.mode = mode
        self.parameterGrad = jax.grad(logWavefunction.apply, argnums=0)
        self.parameterGradBatchFlat = jax.vmap(
            self.getParameterGradient, in_axes=(None,0)
        )

    def __call__(self, parameters, walkerRs, learningRate, diagonalShift, mu, history):

        localEnergies = self.localEnergy.batch(parameters, walkerRs)
        parameterGrads = self.parameterGradBatchFlat(parameters, walkerRs)
        (flatParameters,unravel) = jax.flatten_util.ravel_pytree(parameters)
        numWalkers = parameterGrads.shape[0]

        exp_H = jnp.average(localEnergies)
        exp_O = jnp.average(parameterGrads, axis=0)
        exp_OH = jnp.average(localEnergies[:,None] * parameterGrads, axis=0)

        g_k = 2 * (exp_O * exp_H - exp_OH)
        f_k = g_k + diagonalShift * mu * history
        U = parameterGrads - exp_O[None,:]

        if self.mode == 'normal':
            s_jk = U.T @ U / numWalkers
            diagonalMatrix = diagonalShift * jnp.eye(s_jk.shape[0], dtype=s_jk.dtype)
            parameterStep = jnp.linalg.solve(s_jk + diagonalMatrix, f_k)

        elif self.mode == 'smw':
            P = U / jnp.sqrt(numWalkers)
            parameterStep = smw_solve(P, f_k, diagonalShift)

        else:
            raise Exception("Specified SR mode not supported.")
        
        newHistory = parameterStep
        updatedParameters = unravel(flatParameters + parameterStep)

        return ( True , localEnergies , updatedParameters , newHistory )

    def getParameterGradient(self, parameters, rs):
        localParameterGrads = self.parameterGrad(parameters, rs)
        return jax.flatten_util.ravel_pytree(localParameterGrads)[0]
