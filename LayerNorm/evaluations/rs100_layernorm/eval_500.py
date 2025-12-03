import sys, os

sys.path.append("/burg-archive/ccce/users/phr2114/vmc")
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
#os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax
import jax.numpy as jnp
import numpy as np

import wavefunctions, hamiltonian, trajectory, optimization

jax.config.update("jax_enable_x64", False)

import time

print("STARTING\n\n")

debugFlag = True
start = time.time()

parameterTimestep = 500

###############################################################################
#   Settings for whole simulation                                             #
###############################################################################

r_ws = 100
N = 14
tau = 1.25

walkers = 1024

eqSteps      = 2000
trainSteps   = 2000
trainEqSteps = 20
evalSteps    = 1000

NUp = N // 2
NDown = N - NUp
spins = ( NUp , NDown )

L = wavefunctions.computeL(N, r_ws)

( acceptMin , acceptMax ) = ( 0.55 , 0.65 )

T = trainSteps / 2
diagonalShift = 1e-3
mu = 0.9

###############################################################################
#   Creating wavefunction, walker updater, optimizer, and local energy        #
###############################################################################

#wavefunction = wavefunctions.LogSlaterCYJastrow(spins, L)
wavefunction = wavefunctions.LogMessagePassingSJBLayerNorm(
    spins, L, 3, 32, 32, 24, 8
)
mala = trajectory.MALAUpdater(wavefunction, r_ws)
localEnergy = hamiltonian.LocalEnergyUEG(wavefunction, L, truncationLimit=5)
optimizer = optimization.StochasticReconfigurationMomentum(
    wavefunction, localEnergy
)

updateWalkerPositions = jax.jit(mala.updateBatch)
computeEnergies = jax.jit(localEnergy.batch)
updateParameters = jax.jit(optimizer)

rng = jax.random.PRNGKey(558)

rng, rs_rng, init_rng = jax.random.split(rng, 3)
rs = trajectory.wignerCrystal(spins, r_ws, L, walkers, rs_rng, dim=3)
parameters = wavefunctions.loadParameters(
    "parameters_{}.msgpack".format(parameterTimestep)
)

###############################################################################
#   Equilibrating walkers before energy evaluations                           #
###############################################################################

rng, rs_rng = jax.random.split(rng, 2)
rs = trajectory.wignerCrystal(spins, r_ws, L, walkers, rs_rng, dim=3)

print("Starting equilibration...")

rng = jax.random.PRNGKey(386)

startRs = rs
acceptRates = [np.nan]
acceptArrays = np.zeros(walkers)

for dt in range(eqSteps):

    if dt % 100 == 0:
        avgRate = np.average(np.array(acceptRates))
        acceptRates = []
        print("Step {:5d}   Acc Rate {:4f}   tau = {:.5f}".format(dt,avgRate,tau))

    rng, traj_rng = jax.random.split(rng, 2)
    newRs = updateWalkerPositions(parameters, rs, traj_rng, tau)

    acceptRate = trajectory.acceptanceRate(rs, newRs)
    
    acceptRates.append(acceptRate)
    acceptArrays += trajectory.acceptanceArray(rs, newRs) / eqSteps

    if acceptRate < acceptMin:
        tau = tau * 0.9
    elif acceptRate > acceptMax:
        tau = tau * 1.1

    rs = newRs

print("Finished equilibration!\n")

print(np.sort(acceptArrays)[:10])

if np.min(acceptArrays) < 0.4:
    np.savetxt("FAILED_REEQUILIBRATE", [])
    np.save("big_acceptArrays.npy", acceptArrays)
    np.save("big_startRs.npy", startRs)
    np.save("big_endRs.npy", rs)
    raise Exception("DEAD WALKER DURING RE-EQUILIBRATION")


###############################################################################
#   Energy evaluation                                                         #
###############################################################################


print("Starting energy evaluation...")

rng = jax.random.PRNGKey(493)

startRs = rs
acceptRates = [np.nan]
acceptArrays = np.zeros(walkers)
energies = np.full((evalSteps,walkers),np.nan)

for dt in range(evalSteps):

    if dt % 1000 == 0:
        avgRate = np.average(np.array(acceptRates))
        acceptRates = []
        print("Step {:5d}   Acc Rate {:4f}   tau = {:.5f}".format(dt,avgRate,tau))

    rng, traj_rng = jax.random.split(rng, 2)
    energies[dt,:] = computeEnergies(parameters, rs) / N
    newRs = updateWalkerPositions(parameters, rs, traj_rng, tau)

    acceptRate = trajectory.acceptanceRate(rs, newRs)
    
    acceptRates.append(acceptRate)
    acceptArrays += trajectory.acceptanceArray(rs, newRs) / evalSteps

    rs = newRs

averageEnergies = np.average(energies, axis=1)

print("Finished energy evaluation!\n")

print(np.sort(acceptArrays)[:10])

if np.min(acceptArrays) < 0.4:
    np.savetxt("DEADWALKER", [])
    np.save("big_acceptArrays.npy", acceptArrays)
    np.save("big_startRs.npy", startRs)
    np.save("big_endRs.npy", rs)
    raise Exception("DEAD WALKER DURING EVALUATION")

###############################################################################
#   Write relevant results out to files                                       #
###############################################################################

trajectory.blockingAnalysis(
    averageEnergies, writeQ="statistics_{}.txt".format(parameterTimestep)
)
