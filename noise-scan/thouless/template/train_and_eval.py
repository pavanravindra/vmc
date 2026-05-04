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

#jax.config.update("jax_default_matmul_precision", "float32")
jax.config.update("jax_enable_x64", True)

import time

print("STARTING\n\n")

debugFlag = True
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
trainEqSteps = 25
evalSteps    = 1000

NUp = N // 2
NDown = N - NUp
spins = ( NUp , NDown )

lattice = wavefunctions.computeLattice(
    N, r_ws, dim, basis_matrix=jnp.array([[7,0],[0,4*jnp.sqrt(3)]])
)
kpoints = wavefunctions.genKpoints(numKpoints, lattice, dim)

( acceptMin , acceptMax ) = ( 0.55 , 0.65 )


###############################################################################
#   Creating wavefunction, walker updater, optimizer, and local energy        #
###############################################################################

if xxxTRIALxxx == 0:
    parameter_noise = 0
else:
    parameter_noise = optimization.logSample(xxxLRMIN0xxx,xxxLRMAX0xxx)
    np.savetxt("parameter_noise.txt", [parameter_noise])

print("PARAMETER NOISE: " + str(parameter_noise))

class TestWavefunction(wavefunctions.Wavefunction):
    """
    TODO
    """
    spins : (int,int)
    dim : int
    lattice : jnp.ndarray
    kpoints : jnp.ndarray
    upCoeffs : jnp.ndarray
    downCoeffs : jnp.ndarray

    def setup(self):
        
        self.slaterUp = wavefunctions.LogThoulessSlater(
            self.spins[0], self.dim, self.kpoints, self.upCoeffs
        )
        self.slaterDown = wavefunctions.LogThoulessSlater(
            self.spins[1], self.dim, self.kpoints, self.downCoeffs
        )
        self.CYJastrow = wavefunctions.LogCYJastrow(self.spins, self.lattice)

    def __call__(self, rs):
        
        CYJastrow = self.CYJastrow(rs)
        slaterUp = self.slaterUp(rs[:self.spins[0],:])
        slaterDown = self.slaterDown(rs[self.spins[0]:,:])
        
        return slaterUp + slaterDown + CYJastrow

rng = jax.random.PRNGKey(558)
rng, rs_rng, init_rng = jax.random.split(rng, 3)

centers = trajectory.generateStripedAFM(spins, lattice, dim, gridShape=(7,4))
(upCenters,downCenters) = centers[:spins[0]], centers[spins[0]:]
rs = centers[None,:,:] + (r_ws / 10) * jax.random.normal(rs_rng, shape=(walkers,N,dim))

coeffShape = (numKpoints-spins[0], spins[0])
upRotation = jnp.zeros(coeffShape) + parameter_noise * np.random.normal(size=coeffShape)
downRotation = jnp.zeros(coeffShape) + parameter_noise * np.random.normal(size=coeffShape)
upCoeffs = wavefunctions.occ_columns_from_theta(upRotation)
downCoeffs = wavefunctions.occ_columns_from_theta(downRotation)

wavefunction = TestWavefunction(
    spins, dim, lattice, kpoints, upCoeffs, downCoeffs
)
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

parameters = wavefunction.initBatch(init_rng, rs)

parameters = unfreeze(parameters)
tunedAs = wavefunctions.loadParameters(
    "/burg-archive/ccce/users/phr2114/vmc/thouless-test/CYJastrow/rs{}_best/parameters.msgpack".format(str(r_ws))
)['params']['CYJastrow']['As_same_diff']
parameters['params']['CYJastrow']['As_same_diff'] = tunedAs
parameters = freeze(parameters)

print("NUM PARAMETERS: " + str(optimization.flatten(parameters).size))
print("DTYPE: " + str(rs.dtype))

print(parameters)

###############################################################################
#   Equilibrating walkers before energy evaluations                           #
###############################################################################

rng, rs_rng = jax.random.split(rng, 2)
rs = centers[None,:,:] + (r_ws / 10) * jax.random.normal(rs_rng, shape=(walkers,N,dim))

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
    #np.save("big_acceptArrays.npy", acceptArrays)
    #np.save("big_startRs.npy", startRs)
    #np.save("big_endRs.npy", rs)
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

if np.min(acceptArrays) < 0.4:
    np.savetxt("DEADWALKER", [])
    #np.save("big_acceptArrays.npy", acceptArrays)
    #np.save("big_startRs.npy", startRs)
    #np.save("big_endRs.npy", rs)
    raise Exception("DEAD WALKER DURING EVALUATION")

###############################################################################
#   Write relevant results out to files                                       #
###############################################################################

wavefunctions.saveParameters("parameters.msgpack", parameters)
np.savetxt("time.txt", [(time.time() - start) / 3600])
np.savetxt("tau.txt", [tau])
trajectory.blockingAnalysis(averageEnergies, writeQ="statistics.txt")
