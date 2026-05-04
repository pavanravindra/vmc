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

r_ws = 5
N = 14
tau = 1.25
dim = 3
numKpoints = 169

walkers = 1024

eqSteps      = 2000
trainSteps   = 2000
trainEqSteps = 25
evalSteps    = 1000

NUp = N // 2
NDown = N - NUp
spins = ( NUp , NDown )

lattice = wavefunctions.computeLattice(N, r_ws, dim)
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
    hyperparameters[0] = optimization.logSample(1e-2,1e1)
    np.savetxt("hyperparameters.txt", hyperparameters)

hyperparameters = jnp.array(np.loadtxt("hyperparameters.txt"), ndmin=1)

print("HYPERPARAMETERS: " + str(hyperparameters))

eta0 = hyperparameters[0]


###############################################################################
#   Creating wavefunction, walker updater, optimizer, and local energy        #
###############################################################################

class TestWavefunction(wavefunctions.Wavefunction):
    """
    T-round neural-backflow wavefunction.

    For t = 1, this reduces exactly to the original one-layer architecture.

    For t >= 2, each additional round builds new pair features from:
        v_ij,
        h_i^(t-1) - h_j^(t-1),
        h_i^(t-1) * h_j^(t-1)

    Naming convention is chosen so that:
        T=1 uses the original parameter names
        T=2 matches the explicit two-layer version exactly

    In particular:
        round 1: weights1_nb, weights2_nb, ...
        round 2: weights1_nb_2, weights2_nb_2, ...
        round 3: weights1_nb_3, weights2_nb_3, ...
        etc.
    """
    spins: tuple[int, int]
    dim: int
    lattice: jnp.ndarray
    kpoints: jnp.ndarray
    hiddenFeatures: int
    T: int = 1

    def setup(self):

        if self.T < 1:
            raise ValueError(f"T must be >= 1, got T={self.T}")

        self.slaterUp = wavefunctions.LogSimpleSlater(
            self.spins[0], self.dim, self.kpoints
        )
        self.slaterDown = wavefunctions.LogSimpleSlater(
            self.spins[1], self.dim, self.kpoints
        )
        self.CYJastrow = wavefunctions.LogCYJastrow(self.spins, self.lattice)

        for t in range(1, self.T + 1):

            suffix = "" if t == 1 else f"_{t}"

            setattr(self, f"weights1_nb{suffix}", nn.Dense(self.hiddenFeatures))
            setattr(self, f"weights2_nb{suffix}", nn.Dense(1))

            setattr(self, f"linear1_nb{suffix}", nn.Dense(self.hiddenFeatures))
            setattr(self, f"linear2_nb{suffix}", nn.Dense(self.hiddenFeatures))

            setattr(self, f"jastrow1_nb{suffix}", nn.Dense(self.hiddenFeatures))
            setattr(
                self,
                f"jastrow2_nb{suffix}",
                nn.Dense(
                    1,
                    kernel_init=nn.initializers.zeros,
                    bias_init=nn.initializers.zeros,
                ),
            )

            setattr(self, f"backflow1_nb{suffix}", nn.Dense(self.hiddenFeatures))
            setattr(
                self,
                f"backflow2_nb{suffix}",
                nn.Dense(
                    self.dim,
                    kernel_init=nn.initializers.zeros,
                    bias_init=nn.initializers.zeros,
                ),
            )

    def __call__(self, rs):

        N = sum(self.spins)

        eye_mask = jnp.eye(N, dtype=bool)[:, :, None]
        v_ij = wavefunctions.generateFeatures(rs, self.spins, self.lattice)  # (N,N,2*dim+2)

        total_jastrow_nb = 0.0
        total_backflow_nb = jnp.zeros_like(rs)

        h_i_prev = None

        for t in range(1, self.T + 1):

            suffix = "" if t == 1 else f"_{t}"

            weights1 = getattr(self, f"weights1_nb{suffix}")
            weights2 = getattr(self, f"weights2_nb{suffix}")
            linear1 = getattr(self, f"linear1_nb{suffix}")
            linear2 = getattr(self, f"linear2_nb{suffix}")
            jastrow1 = getattr(self, f"jastrow1_nb{suffix}")
            jastrow2 = getattr(self, f"jastrow2_nb{suffix}")
            backflow1 = getattr(self, f"backflow1_nb{suffix}")
            backflow2 = getattr(self, f"backflow2_nb{suffix}")

            if t == 1:
                x_ij = v_ij
            else:
                h_i_row = h_i_prev[:, None, :]   # (N,1,hiddenFeatures)
                h_i_col = h_i_prev[None, :, :]   # (1,N,hiddenFeatures)

                dh_ij = h_i_row - h_i_col        # (N,N,hiddenFeatures)
                ph_ij = h_i_row * h_i_col        # (N,N,hiddenFeatures)

                x_ij = jnp.concatenate([v_ij, dh_ij, ph_ij], axis=-1)

            g_ij = nn.sigmoid(
                weights2(nn.swish(weights1(x_ij)))
            )  # (N,N,1)

            h_ij = linear2(
                nn.swish(linear1(x_ij))
            )  # (N,N,hiddenFeatures)

            h_i = jnp.sum(
                jnp.where(~eye_mask, g_ij * h_ij, 0.0), axis=1
            ) / N  # (N,hiddenFeatures)

            jastrow_nb_t = jnp.sum(
                jastrow2(nn.swish(jastrow1(h_i)))
            ) / N

            backflow_nb_t = backflow2(
                nn.swish(backflow1(h_i))
            )  # (N,dim)

            total_jastrow_nb = total_jastrow_nb + jastrow_nb_t
            total_backflow_nb = total_backflow_nb + backflow_nb_t

            h_i_prev = h_i

        xs = rs + total_backflow_nb

        CYJastrow = self.CYJastrow(rs)
        slaterUp = self.slaterUp(xs[:self.spins[0], :])
        slaterDown = self.slaterDown(xs[self.spins[0]:, :])

        return slaterUp + slaterDown + CYJastrow + total_jastrow_nb

rng = jax.random.PRNGKey(558)
rng, rs_rng, init_rng = jax.random.split(rng, 3)

centers = trajectory.generateBCC(spins, lattice, dim)
(upCenters,downCenters) = centers[:spins[0]], centers[spins[0]:]
rs = centers[None,:,:] + (r_ws / 10) * jax.random.normal(rs_rng, shape=(walkers,N,dim))

wavefunction = TestWavefunction(
    spins=spins, dim=dim, lattice=lattice, kpoints=kpoints, hiddenFeatures=32, T=2
)
mala = trajectory.MALAUpdater(wavefunction, r_ws)
if dim == 2:
    localEnergy = folx_hamiltonian.LocalEnergyUEG2D(wavefunction, lattice, sparsity_threshold=6)
elif dim == 3:
    localEnergy = folx_hamiltonian.LocalEnergyUEG3D(wavefunction, lattice, sparsity_threshold=6)
optimizer = optimization.StochasticReconfigurationMomentum(
    wavefunction, localEnergy, mode='smw'
)

updateWalkerPositions = jax.jit(mala.updateBatch)
computeEnergies = jax.jit(localEnergy.batch)
updateParameters = jax.jit(optimizer)

parameters = wavefunction.initBatch(init_rng, rs)

parameters = unfreeze(parameters)
tunedAs = wavefunctions.loadParameters(
    "/burg-archive/ccce/users/phr2114/vmc/width/CYJastrow/rs{}_best/parameters.msgpack".format(str(r_ws))
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
    #np.save("big_acceptArrays.npy", acceptArrays)
    #np.save("big_startRs.npy", startRs)
    #np.save("big_endRs.npy", rs)
    raise Exception("DEAD WALKER DURING EQUILIBRATION")


###############################################################################
#   Parameter optimization                                                    #
###############################################################################


print("Starting optimization...")

rng = jax.random.PRNGKey(151)

startRs = rs
history = 0.0

energyTraj = np.full(trainSteps, np.nan)

for dt in range(trainSteps):

    localLearningRate = eta0 / (1 + (dt / T))
    ( maxNorm , currentEnergies , newParameters , history ) = updateParameters(
        parameters, rs, localLearningRate, diagonalShift, mu, history
    )

    if optimization.hasnan(newParameters):
        np.savetxt("FAILED_REEQUILIBRATE", [])
        #wavefunctions.saveParameters(
        #    "parameters_failed.msgpack", parameters
        #)
        #np.save("big_startRs.npy", startRs)
        #np.save("big_endRs.npy", rs)
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

    currentEnergy = jnp.average(currentEnergies) / N
    energyTraj[dt] = currentEnergy

    print("Step {:5d}   Energy: {:.5f}   Norm: {:5}   Min Acc: {:.3f}   Avg Acc: {:.3f}   Param Norm: {:.5f}".format(
        dt, currentEnergy, str(maxNorm),
        np.min(acceptArrays), np.average(acceptArrays), np.linalg.norm(optimization.flatten(parameters))
    ))

print("Finished optimization!\n")

wavefunctions.saveParameters("parameters.msgpack", parameters)
np.savetxt("energy_traj.txt", energyTraj)

jax.tree_util.tree_map(lambda x: x.block_until_ready(), parameters)

wavefunctions.saveParameters("parameters.msgpack", parameters)
reloaded_parameters = wavefunctions.loadParameters("parameters.msgpack")

flat_old = np.array(optimization.flatten(parameters))
flat_new = np.array(optimization.flatten(reloaded_parameters))
print("max abs diff after reload =", np.max(np.abs(flat_old - flat_new)))

parameters = reloaded_parameters

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

np.savetxt("time.txt", [(time.time() - start) / 3600])
np.savetxt("tau.txt", [tau])
trajectory.blockingAnalysis(averageEnergies, writeQ="statistics.txt")
