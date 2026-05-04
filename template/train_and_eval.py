import sys, os

sys.path.append("/burg-archive/ccce/users/phr2114/vmc")
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
#os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core.frozen_dict import freeze, unfreeze
import numpy as np

import wavefunctions, folx_hamiltonian, trajectory, optimization

jax.config.update("jax_enable_x64", True)

import time

print("STARTING\n\n")

start = time.time()

###############################################################################
#   Settings for whole simulation                                             #
###############################################################################

r_ws = xxxRSxxx
N = 56
tau = 1.25
dim = 2
numKpoints = 169

walkers = 1024

eqSteps      = 2000
trainSteps   = 1000
trainEqSteps = 20
evalSteps    = 1000

NUp = N // 2
NDown = N - NUp
spins = ( NUp , NDown )

lattice = wavefunctions.compute_lattice(
    N, r_ws, dim, basis_matrix=jnp.array([[7,0],[0,4*jnp.sqrt(3)]])
)
kpoints = wavefunctions.gen_k_points(numKpoints, lattice, dim)

( acceptMin , acceptMax ) = ( 0.55 , 0.65 )

T = 1000
diagonalShift = 1e-3
mu = 0.9


###############################################################################
#   Hyperparameter sampling                                                   #
###############################################################################

if not os.path.exists("hyperparameters.txt"):

    hyperparameters = np.full(1, np.nan)
    hyperparameters[0] = optimization.log_sample(xxxLRMIN0xxx,xxxLRMAX0xxx)
    np.savetxt("hyperparameters.txt", hyperparameters)

hyperparameters = jnp.array(np.loadtxt("hyperparameters.txt"), ndmin=1)

print("HYPERPARAMETERS: " + str(hyperparameters))

eta0 = hyperparameters[0]


###############################################################################
#   Creating wavefunction, walker updater, optimizer, and local energy        #
###############################################################################

rng = jax.random.PRNGKey(558)
rng, rs_rng, init_rng = jax.random.split(rng, 3)

centers = trajectory.generate_striped_afm(spins, lattice, dim, gridShape=(7,4))
(upCenters,downCenters) = centers[:spins[0]], centers[spins[0]:]
rs = centers[None,:,:] + (r_ws / 10) * jax.random.normal(rs_rng, shape=(walkers,N,dim))

alpha = 1 / (r_ws ** 2.0)
upCoeffs = wavefunctions.generate_gaussian_mp_coeffs(kpoints, upCenters, alpha)
downCoeffs = wavefunctions.generate_gaussian_mp_coeffs(kpoints, downCenters, alpha)

wavefunction = wavefunctions.LogSlaterCYJastrow(spins, dim, lattice, kpoints[:spins[0]])
mala = trajectory.MALAUpdater(wavefunction, r_ws)
if dim == 2:
    localEnergy = folx_hamiltonian.LocalEnergyUEG2D(wavefunction, lattice, sparsity_threshold=6)
elif dim == 3:
    localEnergy = folx_hamiltonian.LocalEnergyUEG3D(wavefunction, lattice, sparsity_threshold=6)
optimizer = optimization.StochasticReconfiguration(
    wavefunction, localEnergy, mode='normal'
)

updateWalkerPositions = jax.jit(mala.update_batch)
computeEnergies = jax.jit(localEnergy.batch)
updateParameters = jax.jit(optimizer)

parameters = wavefunction.init_batch(init_rng, rs)

"""
parameters = unfreeze(parameters)
tunedAs = wavefunctions.load_parameters(
    "/burg-archive/ccce/users/phr2114/vmc/gaussian-reference/CYJastrow/rs{}_best/parameters.msgpack".format(str(r_ws))
)['params']['CYJastrow']['As_same_diff']
parameters['params']['CYJastrow']['As_same_diff'] = tunedAs
parameters = freeze(parameters)
"""

print("NUM PARAMETERS: " + str(optimization.flatten(parameters).size))
print("DTYPE: " + str(rs.dtype))


###############################################################################
#   Equilibrating walkers before parameter optimization                       #
###############################################################################

print("Starting equilibration...")

rng = jax.random.PRNGKey(1126)

acceptRates = [np.nan]
acceptArrays = np.zeros(walkers)

for dt in range(eqSteps):

    if dt % 100 == 0:
        avgRate = np.average(np.array(acceptRates))
        acceptRates = []
        print("Step {:5d}   Acc Rate {:4f}   tau = {:.5f}".format(dt,avgRate,tau))

    rng, traj_rng = jax.random.split(rng, 2)
    newRs = updateWalkerPositions(parameters, rs, traj_rng, tau)

    acceptRate = trajectory.acceptance_rate(rs, newRs)

    acceptRates.append(acceptRate)
    acceptArrays += trajectory.acceptance_array(rs, newRs) / eqSteps

    if acceptRate < acceptMin:
        tau = tau * 0.9
    elif acceptRate > acceptMax:
        tau = tau * 1.1

    rs = newRs

print("Finished equilibration!\n")

print(np.sort(acceptArrays)[:10])

if np.min(acceptArrays) < 0.4:
    np.savetxt("FAILED_EQUILIBRATE", [])
    raise Exception("DEAD WALKER DURING EQUILIBRATION")


###############################################################################
#   Parameter optimization                                                    #
###############################################################################

print("Starting optimization...")

rng = jax.random.PRNGKey(151)

history = 0.0

energyTraj = np.full(trainSteps, np.nan)

checkpointEnergies = []
checkpointParameters = []

for dt in range(trainSteps):

    localLearningRate = eta0 / (1 + (dt / T))
    ( maxNorm , currentEnergies , newParameters , history ) = updateParameters(
        parameters, rs, localLearningRate, diagonalShift, mu, history
    )

    if optimization.has_nan(newParameters):
        np.savetxt("NAN_OPTIMIZATION", [])
        raise Exception("Parameters have somehow NaNed...")

    currentEnergy = jnp.average(currentEnergies) / N
    energyTraj[dt] = currentEnergy

    if (dt + 1) % 100 == 0:
        checkpointEnergies.append(currentEnergy)
        checkpointParameters.append(newParameters)

    parameters = newParameters

    acceptArrays = np.zeros(walkers)

    for _ in range(trainEqSteps):

        rng, traj_rng = jax.random.split(rng, 2)
        newRs = updateWalkerPositions(parameters, rs, traj_rng, tau)

        acceptRate = trajectory.acceptance_rate(rs, newRs)
        if acceptRate < acceptMin:
            tau = tau * 0.9
        elif acceptRate > acceptMax:
            tau = tau * 1.1

        acceptArrays += trajectory.acceptance_array(rs, newRs) / trainEqSteps

        rs = newRs

    print("Step {:5d}   Energy: {:.5f}   Norm: {:5}   Min Acc: {:.3f}   Avg Acc: {:.3f}   Param Norm: {:.5f}".format(
        dt, currentEnergy, str(maxNorm),
        np.min(acceptArrays), np.average(acceptArrays), np.linalg.norm(optimization.flatten(parameters))
    ))

print("Finished optimization!\n")

checkpointEnergies = np.array(checkpointEnergies)
parameters = checkpointParameters[np.argmin(checkpointEnergies)]

wavefunctions.save_parameters("parameters.msgpack", parameters)
np.savetxt("energy_traj.txt", energyTraj)

###############################################################################
#   Equilibrating walkers before energy evaluations                           #
###############################################################################

rng, rs_rng = jax.random.split(rng, 2)
rs = centers[None,:,:] + (r_ws / 10) * jax.random.normal(rs_rng, shape=(walkers,N,dim))

print("Starting equilibration...")

rng = jax.random.PRNGKey(386)

acceptRates = [np.nan]
acceptArrays = np.zeros(walkers)

for dt in range(eqSteps):

    if dt % 100 == 0:
        avgRate = np.average(np.array(acceptRates))
        acceptRates = []
        print("Step {:5d}   Acc Rate {:4f}   tau = {:.5f}".format(dt,avgRate,tau))

    rng, traj_rng = jax.random.split(rng, 2)
    newRs = updateWalkerPositions(parameters, rs, traj_rng, tau)

    acceptRate = trajectory.acceptance_rate(rs, newRs)

    acceptRates.append(acceptRate)
    acceptArrays += trajectory.acceptance_array(rs, newRs) / eqSteps

    if acceptRate < acceptMin:
        tau = tau * 0.9
    elif acceptRate > acceptMax:
        tau = tau * 1.1

    rs = newRs

print("Finished equilibration!\n")

print(np.sort(acceptArrays)[:10])

if np.min(acceptArrays) < 0.4:
    np.savetxt("FAILED_REEQUILIBRATE", [])
    raise Exception("DEAD WALKER DURING RE-EQUILIBRATION")


###############################################################################
#   Energy evaluation                                                         #
###############################################################################

print("Starting energy evaluation...")

rng = jax.random.PRNGKey(493)

acceptRates = [np.nan]
acceptArrays = np.zeros(walkers)
energies = np.full((evalSteps,walkers),np.nan)

for dt in range(evalSteps):

    if dt % 100 == 0:
        avgRate = np.average(np.array(acceptRates))
        acceptRates = []
        print("Step {:5d}   Acc Rate {:4f}   tau = {:.5f}".format(dt,avgRate,tau))

    rng, traj_rng = jax.random.split(rng, 2)
    energies[dt,:] = computeEnergies(parameters, rs) / N
    newRs = updateWalkerPositions(parameters, rs, traj_rng, tau)

    acceptRate = trajectory.acceptance_rate(rs, newRs)

    acceptRates.append(acceptRate)
    acceptArrays += trajectory.acceptance_array(rs, newRs) / evalSteps

    rs = newRs

averageEnergies = np.average(energies, axis=1)

print("Finished energy evaluation!\n")

print(np.sort(acceptArrays)[:10])

if np.min(acceptArrays) < 0.4:
    np.savetxt("DEADWALKER", [])
    raise Exception("DEAD WALKER DURING EVALUATION")

###############################################################################
#   Write relevant results out to files                                       #
###############################################################################

wavefunctions.save_parameters("parameters.msgpack", parameters)
np.savetxt("time.txt", [(time.time() - start) / 3600])
np.savetxt("tau.txt", [tau])
trajectory.blocking_analysis(averageEnergies, writeQ="statistics.txt")