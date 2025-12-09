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

###############################################################################
#   Settings for whole simulation                                             #
###############################################################################

r_ws = 1
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
#   Hyperparameter sampling                                                   #
###############################################################################

if not os.path.exists("hyperparameters.txt"):
    
    hyperparameters = np.full(1, np.nan)
    hyperparameters[0] = optimization.logSample(5e-2,1e0)
    np.savetxt("hyperparameters.txt", hyperparameters)

hyperparameters = jnp.array(np.loadtxt("hyperparameters.txt"), ndmin=1)

print("HYPERPARAMETERS: " + str(hyperparameters))

eta0 = hyperparameters[0]


###############################################################################
#   Creating wavefunction, walker updater, optimizer, and local energy        #
###############################################################################

#wavefunction = wavefunctions.LogSlaterCYJastrow(spins, L)
wavefunction = wavefunctions.LogMessagePassingSJB(
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
parameters = wavefunction.initBatch(init_rng, rs)

###############################################################################
#   Equilibrating walkers before parameter optimization                       #
###############################################################################

print("Starting equilibration...")

rng = jax.random.PRNGKey(1126)

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
    np.savetxt("FAILED_EQUILIBRATE", [])
    np.save("big_acceptArrays.npy", acceptArrays)
    np.save("big_startRs.npy", startRs)
    np.save("big_endRs.npy", rs)
    raise Exception("DEAD WALKER DURING EQUILIBRATION")


###############################################################################
#   Parameter optimization                                                    #
###############################################################################


print("Starting optimization...")

rng = jax.random.PRNGKey(151)
history = 0

startRs = rs

for dt in range(trainSteps):

    localLearningRate = eta0 / (1 + (dt / T))
    ( maxNorm , currentEnergies , newParameters , history ) = updateParameters(
        parameters, rs, localLearningRate, diagonalShift, mu, history
    )

    if optimization.hasnan(newParameters):
        np.savetxt("FAILED_REEQUILIBRATE", [])
        wavefunctions.saveParameters(
            "parameters_failed.msgpack", parameters
        )
        np.save("big_startRs.npy", startRs)
        np.save("big_endRs.npy", rs)
        raise Exception("Parameters have somehow NaNed...")

    parameters = newParameters

    acceptArrays = np.zeros(walkers)

    for _ in range(trainEqSteps):

        rng, traj_rng = jax.random.split(rng, 2)
        newRs = updateWalkerPositions(parameters, rs, traj_rng, tau)

        acceptRate = trajectory.acceptanceRate(rs, newRs)
        if acceptRate < acceptMin:
            tau = tau * 0.9
        elif acceptRate > acceptMax:
            tau = tau * 1.1

        acceptArrays += trajectory.acceptanceArray(rs, newRs) / trainEqSteps

        rs = newRs

    print("Step {:5d}   Energy: {:.5f}   Norm: {:5}   Min Acc: {:.3f}   Avg Acc: {:.3f}   Param Norm: {:.5f}".format(
        dt, jnp.average(currentEnergies) / N, str(maxNorm),
        np.min(acceptArrays), np.average(acceptArrays), np.linalg.norm(optimization.flatten(parameters))
    ))

print("Finished optimization!\n")

wavefunctions.saveParameters("parameters.msgpack", parameters)

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

wavefunctions.saveParameters("parameters.msgpack", parameters)
np.savetxt("time.txt", [(time.time() - start) / 3600])
np.savetxt("tau.txt", [tau])
np.save("big_energies.npy", energies)
trajectory.blockingAnalysis(averageEnergies, writeQ="statistics.txt")
