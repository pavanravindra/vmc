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

jax.config.update("jax_default_matmul_precision", "float32")
jax.config.update("jax_enable_x64", False)

import time

print("STARTING\n\n")

debugFlag = True
start = time.time()

###############################################################################
#   Settings for whole simulation                                             #
###############################################################################

r_ws = 50
N = 56
tau = 1.25
dim = 2
numKpoints = 169

walkers = 1024

eqSteps      = 2000
trainSteps   = 1000
trainEqSteps = 100
evalSteps    = 1000

NUp = N // 2
NDown = N - NUp
spins = ( NUp , NDown )

lattice = wavefunctions.computeLattice(
    N, r_ws, dim, basis_matrix=jnp.array([[7,0],[0,4*jnp.sqrt(3)]])
)
kpoints = wavefunctions.genKpoints(numKpoints, lattice, dim)[:spins[0]]

( acceptMin , acceptMax ) = ( 0.55 , 0.65 )

T = 1000
diagonalShift = 1e-3
mu = 0.9


###############################################################################
#   Hyperparameter sampling                                                   #
###############################################################################

if not os.path.exists("hyperparameters.txt"):
    
    hyperparameters = np.full(1, np.nan)
    hyperparameters[0] = optimization.logSample(1e-1,1e1)
    np.savetxt("hyperparameters.txt", hyperparameters)

hyperparameters = jnp.array(np.loadtxt("hyperparameters.txt"), ndmin=1)

print("HYPERPARAMETERS: " + str(hyperparameters))

eta0 = hyperparameters[0]


###############################################################################
#   Creating wavefunction, walker updater, optimizer, and local energy        #
###############################################################################

class TestWavefunction(wavefunctions.Wavefunction):
    """
    TODO
    """
    spins : (int,int)
    dim : int
    lattice : jnp.ndarray
    kpoints : jnp.ndarray
    hiddenFeatures : int

    def setup(self):
        
        self.slaterUp = wavefunctions.LogSimpleSlater(self.spins[0], self.dim, self.kpoints)
        self.slaterDown = wavefunctions.LogSimpleSlater(self.spins[1], self.dim, self.kpoints)
        self.CYJastrow = wavefunctions.LogCYJastrow(self.spins, self.lattice)

        self.weights1 = nn.Dense(self.hiddenFeatures)
        self.weights2 = nn.Dense(1)

        self.linear1 = nn.Dense(self.hiddenFeatures)
        self.linear2 = nn.Dense(self.hiddenFeatures)
        
        self.jastrow1 = nn.Dense(self.hiddenFeatures)
        self.jastrow2 = nn.Dense(
            1, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros
        )
        
        self.backflow1 = nn.Dense(self.hiddenFeatures)
        self.backflow2 = nn.Dense(
            self.dim, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros
        )

    def __call__(self, rs):

        N = sum(self.spins)

        v_ij = wavefunctions.generateFeatures(rs, self.spins, self.lattice)   # (N,N,2*dim+2)
        g_ij = nn.sigmoid(
            self.weights2(nn.swish(self.weights1(v_ij)))
        )                                                       # (N,N,1)
        h_ij = self.linear2(nn.swish(self.linear1(v_ij)))       # (N,N,d)

        eye_mask = jnp.eye(N, dtype=bool)[:,:,None]
        h_i = jnp.sum(
            jnp.where(~eye_mask, g_ij * h_ij, 0.0), axis=1
        ) / N                                                   # (N,d)

        CYJastrow = self.CYJastrow(rs)
        neuralJastrow = jnp.sum(
            self.jastrow2(nn.swish(self.jastrow1(h_i)))
        ) / N

        backflow = self.backflow2(nn.swish(self.backflow1(h_i))) # (N,dim)
        xs = rs + backflow
        
        slaterUp = self.slaterUp(xs[:self.spins[0],:])
        slaterDown = self.slaterDown(xs[self.spins[0]:,:])
        
        return slaterUp + slaterDown + CYJastrow + neuralJastrow


wavefunction = TestWavefunction(spins, dim, lattice, kpoints, 32)
mala = trajectory.MALAUpdater(wavefunction, r_ws)
if dim == 2:
    localEnergy = folx_hamiltonian.LocalEnergyUEG2D(wavefunction, lattice, sparsity_threshold=6)
elif dim == 3:
    localEnergy = folx_hamiltonian.LocalEnergyUEG(wavefunction, lattice)
optimizer = optimization.StochasticReconfigurationMomentum(
    wavefunction, localEnergy, mode='normal'
)

updateWalkerPositions = jax.jit(mala.updateBatch)
computeEnergies = jax.jit(localEnergy.batch)
updateParameters = jax.jit(optimizer)

rng = jax.random.PRNGKey(558)

rng, rs_rng, init_rng = jax.random.split(rng, 3)
rs = trajectory.wignerCrystal(
    spins, lattice, r_ws, walkers, rs_rng, dim=dim, gridShape=(7,4)
)
parameters = wavefunction.initBatch(init_rng, rs)

parameters = unfreeze(parameters)
tunedAs = wavefunctions.loadParameters(
    "/burg-archive/ccce/users/phr2114/vmc/dynamic-test/CYJastrow/rs{}_best/parameters.msgpack".format(str(r_ws))
)['params']['CYJastrow']['As_same_diff']
parameters['params']['CYJastrow']['As_same_diff'] = tunedAs
parameters = freeze(parameters)

print("NUM PARAMETERS: " + str(optimization.flatten(parameters).size))
print("DTYPE: " + str(rs.dtype))


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

startRs = rs
history = 0.0

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
rs = trajectory.wignerCrystal(
    spins, lattice, r_ws, walkers, rs_rng, dim=dim, gridShape=(7,4)
)

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

    if dt % 100 == 0:
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

"""
if np.min(acceptArrays) < 0.4:
    np.savetxt("DEADWALKER", [])
    np.save("big_acceptArrays.npy", acceptArrays)
    np.save("big_startRs.npy", startRs)
    np.save("big_endRs.npy", rs)
    raise Exception("DEAD WALKER DURING EVALUATION")
"""

###############################################################################
#   Write relevant results out to files                                       #
###############################################################################

wavefunctions.saveParameters("parameters.msgpack", parameters)
np.savetxt("time.txt", [(time.time() - start) / 3600])
np.savetxt("tau.txt", [tau])
np.save("big_energies.npy", energies)
trajectory.blockingAnalysis(averageEnergies, writeQ="statistics.txt")
