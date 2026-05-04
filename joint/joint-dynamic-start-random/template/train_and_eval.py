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

checkEvery   = 100
patience     = 3

NUp = N // 2
NDown = N - NUp
spins = ( NUp , NDown )

lattice = wavefunctions.computeLattice(
    N, r_ws, dim, basis_matrix=jnp.array([[7,0],[0,4*jnp.sqrt(3)]])
)
kpoints = wavefunctions.genKpoints(numKpoints, lattice, dim)

( acceptMin , acceptMax ) = ( 0.55 , 0.65 )

T = 1000
diagonalShift = 1e-3
mu = 0.9


###############################################################################
#   Hyperparameter sampling                                                   #
###############################################################################

if not os.path.exists("hyperparameters.txt"):
    
    hyperparameters = np.full(1, np.nan)
    hyperparameters[0] = optimization.logSample(xxxLRMIN0xxx,xxxLRMAX0xxx)
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
    upCoeffs : jnp.ndarray
    downCoeffs : jnp.ndarray
    hiddenFeatures : int

    def setup(self):
        
        self.slaterUp = wavefunctions.LogMPSlater(
            self.spins[0], self.dim, self.kpoints, self.upCoeffs
        )
        self.slaterDown = wavefunctions.LogMPSlater(
            self.spins[1], self.dim, self.kpoints, self.downCoeffs
        )
        self.CYJastrow = wavefunctions.LogCYJastrow(self.spins, self.lattice)

        self.weights1_nb = nn.Dense(self.hiddenFeatures)
        self.weights2_nb = nn.Dense(1)

        self.linear1_nb = nn.Dense(self.hiddenFeatures)
        self.linear2_nb = nn.Dense(self.hiddenFeatures)
        
        self.jastrow1_nb = nn.Dense(self.hiddenFeatures)
        self.jastrow2_nb = nn.Dense(
            1, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros
        )
        
        self.backflow1_nb = nn.Dense(self.hiddenFeatures)
        self.backflow2_nb = nn.Dense(
            self.dim, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros
        )

    def __call__(self, rs):

        N = sum(self.spins)

        eye_mask = jnp.eye(N, dtype=bool)[:,:,None]
        v_ij = wavefunctions.generateFeatures(rs, self.spins, self.lattice)   # (N,N,2*dim+2)

        g_ij = nn.sigmoid(
            self.weights2_nb(nn.swish(self.weights1_nb(v_ij)))
        )                                                       # (N,N,1)
        h_ij = self.linear2_nb(nn.swish(self.linear1_nb(v_ij))) # (N,N,d)

        h_i = jnp.sum(
            jnp.where(~eye_mask, g_ij * h_ij, 0.0), axis=1
        ) / N                                                   # (N,d)

        jastrow_nb = jnp.sum(
            self.jastrow2_nb(nn.swish(self.jastrow1_nb(h_i)))
        ) / N

        backflow_nb = self.backflow2_nb(nn.swish(self.backflow1_nb(h_i))) # (N,dim)

        xs = rs + backflow_nb
        
        CYJastrow = self.CYJastrow(rs)
        slaterUp = self.slaterUp(xs[:self.spins[0],:])
        slaterDown = self.slaterDown(xs[self.spins[0]:,:])
        
        return slaterUp + slaterDown + CYJastrow + jastrow_nb

rng = jax.random.PRNGKey(558)
rng, rs_rng, init_rng = jax.random.split(rng, 3)

centers = trajectory.generateStripedAFM(spins, lattice, dim, gridShape=(7,4))
(upCenters,downCenters) = centers[:spins[0]], centers[spins[0]:]
rs = centers[None,:,:] + (r_ws / 10) * jax.random.normal(rs_rng, shape=(walkers,N,dim))

upCoeffs = np.random.normal(size=(numKpoints, spins[0]))
downCoeffs = np.random.normal(size=(numKpoints, spins[1]))

wavefunction = TestWavefunction(spins, dim, lattice, kpoints, upCoeffs, downCoeffs, 32)
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
referenceParameters = wavefunctions.loadParameters(
    "/burg-archive/ccce/users/phr2114/vmc/joint/dynamic/rs{}_best/parameters.msgpack".format(str(r_ws))
)
for key in referenceParameters['params'].keys():
    parameters['params'][key] = referenceParameters['params'][key]
parameters = freeze(parameters)

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
    raise Exception("DEAD WALKER DURING EQUILIBRATION")


###############################################################################
#   Parameter optimization                                                    #
###############################################################################


print("Starting optimization...")

rng = jax.random.PRNGKey(151)

history = 0.0

energyTraj = []

checkpointEnergies = []
checkpointParameters = []

for dt in range(trainSteps):

    ( maxNorm , currentEnergies , newParameters , history ) = updateParameters(
        parameters, rs, eta0, diagonalShift, mu, history
    )

    if optimization.hasnan(newParameters):
        np.savetxt("PARAMETER_NAN", [])
        raise Exception("Parameters have somehow NaNed...")

    currentEnergy = jnp.average(currentEnergies) / N
    energyTraj.append(currentEnergy)

    if (dt + 1) % checkEvery == 0:

        windowAverage = np.average(energyTraj[-checkEvery:])
        checkpointEnergies.append(windowAverage)
        checkpointParameters.append(parameters)
        numCheckpoints = len(checkpointEnergies)

        if numCheckpoints > patience:
            bestIdx = np.argmin(checkpointEnergies)
            if bestIdx < numCheckpoints - patience:
                print("Early stopping at step {:5d} with energy {:.5f}!".format(
                    dt+1, checkpointEnergies[bestIdx]
                ))
                break;

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
        dt, currentEnergy, str(maxNorm),
        np.min(acceptArrays), np.average(acceptArrays), np.linalg.norm(optimization.flatten(parameters))
    ))

print("Finished optimization!\n")

parameters = checkpointParameters[np.argmin(checkpointEnergies)]

wavefunctions.saveParameters("parameters.msgpack", parameters)
np.savetxt("energy_traj.txt", np.array(energyTraj))

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

    acceptRate = trajectory.acceptanceRate(rs, newRs)
    
    acceptRates.append(acceptRate)
    acceptArrays += trajectory.acceptanceArray(rs, newRs) / evalSteps

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

wavefunctions.saveParameters("parameters.msgpack", parameters)
np.savetxt("time.txt", [(time.time() - start) / 3600])
np.savetxt("tau.txt", [tau])
trajectory.blockingAnalysis(averageEnergies, writeQ="statistics.txt")
