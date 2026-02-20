import sys, os

sys.path.append("/burg-archive/ccce/users/phr2114/vmc")
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
#os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core.frozen_dict import freeze, unfreeze
import numpy as np

import wavefunctions, trajectory, optimization
import old_hamiltonian, full_hamiltonian, lin_hamiltonian, new_hamiltonian, chunk_hamiltonian, folx_hamiltonian

jax.config.update("jax_enable_x64", False) # Single precision
#jax.config.update("jax_enable_x64", True)  # Double Precision

import time

print("STARTING\n")

###############################################################################
#   Settings for whole simulation                                             #
###############################################################################

r_ws = 50
N = 56
tau = 65 # 1.25
dim = 2
numKpoints = 169

walkers = 8 # 1024

eqSteps      = 1000 # 2000
trainSteps   = 3 # 10 # 1000
trainEqSteps = 3 # 20
evalSteps    = 3 # 10 # 1000

NUp = N // 2
NDown = N - NUp
spins = ( NUp , NDown )

lattice = wavefunctions.computeLattice(
    N, r_ws, dim, basis_matrix=jnp.array([[7,0],[0,4*jnp.sqrt(3)]])
)

( acceptMin , acceptMax ) = ( 0.55 , 0.65 )


###############################################################################
#   Creating wavefunction, walker updater, optimizer, and local energy        #
###############################################################################

class CustomWavefunction(wavefunctions.Wavefunction):
    spins : (int, int)
    dim: int
    lattice: jnp.ndarray
    kpoints: jnp.ndarray
    hiddenFeatures : int
    upCoeffs: jnp.ndarray
    downCoeffs: jnp.ndarray

    def setup(self):
        self.slaterUp = wavefunctions.LogMPSlater(
            self.spins[0], self.dim, self.kpoints, self.upCoeffs
        )
        self.slaterDown = wavefunctions.LogMPSlater(
            self.spins[1], self.dim, self.kpoints, self.downCoeffs
        )
        self.CYJastrow = wavefunctions.LogCYJastrow(self.spins, self.lattice)
        
        self.linear1 = nn.Dense(self.hiddenFeatures)
        self.linear2 = nn.Dense(
            self.dim+1, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros
        )

    def __call__(self, rs):

        v_ij = wavefunctions.generateFeatures(rs, self.spins, self.lattice)   # (N,N,2*dim+2)
        n_ij = self.linear2(nn.swish(self.linear1(v_ij)))       # (N,N,dim+1)

        jastrowFeatures = n_ij[:,:,0]                           # (N,N)
        backflowFeatures = n_ij[:,:,1:]                         # (N,N,dim)

        backflow = jnp.average(backflowFeatures, axis=1)        # (N,dim)
        xs = rs + backflow
        
        slaterUp = self.slaterUp(xs[:self.spins[0],:])
        slaterDown = self.slaterDown(xs[self.spins[0]:,:])
        CYJastrow = self.CYJastrow(rs)
        neuralJastrow = 0.5 * jnp.sum(jastrowFeatures) / sum(self.spins)
        
        return slaterUp + slaterDown + CYJastrow + neuralJastrow

class SmallWavefunction(wavefunctions.Wavefunction):
    spins : (int,int)
    dim : int
    lattice : jnp.ndarray
    kpoints : jnp.ndarray
    hiddenFeatures : int

    def setup(self):
        
        self.slaterUp = wavefunctions.LogSimpleSlater(self.spins[0], self.dim, self.kpoints)
        self.slaterDown = wavefunctions.LogSimpleSlater(self.spins[1], self.dim, self.kpoints)
        
        self.linear1 = nn.Dense(self.hiddenFeatures)
        self.linear2 = nn.Dense(
            self.dim+1, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros
        )

    def __call__(self, rs):

        v_ij = wavefunctions.generateFeatures(rs, self.spins, self.lattice)   # (N,N,2*dim+2)
        n_ij = self.linear2(nn.swish(self.linear1(v_ij)))       # (N,N,dim+1)

        jastrowFeatures = n_ij[:,:,0]                           # (N,N)
        backflowFeatures = n_ij[:,:,1:]                         # (N,N,dim)

        backflow = jnp.average(backflowFeatures, axis=1)        # (N,dim)
        xs = rs + backflow
        
        slaterUp = self.slaterUp(xs[:self.spins[0],:])
        slaterDown = self.slaterDown(xs[self.spins[0]:,:])
        neuralJastrow = 0.5 * jnp.sum(jastrowFeatures) / sum(self.spins)
        
        return slaterUp + slaterDown + neuralJastrow

#upCoeffs = jnp.array(np.random.normal(size=(numKpoints, spins[0])))
#downCoeffs = jnp.array(np.random.normal(size=(numKpoints, spins[1])))

upCoeffs = jnp.eye(numKpoints, spins[0])
downCoeffs = jnp.eye(numKpoints, spins[1])

"""
# This is faster chunked than scanned!
# This is a MP-2SJB wavefunction
kpoints = wavefunctions.genKpoints(numKpoints, lattice, dim)
wavefunction = CustomWavefunction(
    spins, dim, lattice, kpoints, 32, upCoeffs, downCoeffs
)
"""

kpoints = wavefunctions.genKpoints(spins[0], lattice, dim)
#wavefunction = wavefunctions.LogSimpleSlaters(spins, dim, kpoints)
#wavefunction = wavefunctions.LogSlaterCYJastrow(spins, dim, lattice, kpoints)
#wavefunction = wavefunctions.LogTwoBodySJB(spins, dim, lattice, kpoints, 32)
wavefunction = SmallWavefunction(spins, dim, lattice, kpoints, 32)



mala = trajectory.MALAUpdater(wavefunction, r_ws)
updateWalkerPositions = jax.jit(mala.updateBatch)

energies = [
    (
        "Old",
        old_hamiltonian.LocalEnergyUEG2D(wavefunction, lattice).configuration
    ),
    (
        "New",
        new_hamiltonian.LocalEnergyUEG2D(wavefunction, lattice).configuration
    ),
    (
        "Folx",
        folx_hamiltonian.LocalEnergyUEG2D(wavefunction, lattice).configuration
    ),
]

energies = [(e[0],jax.jit(e[1])) for e in energies]

rng = jax.random.PRNGKey(558)

rng, rs_rng, init_rng = jax.random.split(rng, 3)
rs = lattice[1,1] * jax.random.uniform(rs_rng, shape=(walkers, N, dim))
rs = trajectory.wignerCrystal(
    spins, lattice, r_ws, walkers, rs_rng, dim=dim, gridShape=(7,4)
)
parameters = wavefunction.initBatch(init_rng, rs)

print("NUM PARAMETERS: {}\n".format(optimization.flatten(parameters).size))

print("-----------------------")



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

if np.min(acceptArrays) < 0.4:
    print(np.sort(acceptArrays)[:10])
    print(np.sort(acceptArrays)[-10:])
    raise Exception("DEAD WALKER DURING EQUILIBRATION")

print("Finished equilibration!\n")

print("-----------------------")

rs = rs[0]


###############################################################################
#   Energy evaluation                                                         #
###############################################################################

rng = jax.random.PRNGKey(493)

rsList = []
for dt in range(evalSteps):
    rng, traj_rng = jax.random.split(rng, 2)
    rsList.append(
        rs + dt * 1e-4 * jax.random.normal(traj_rng, shape=rs.shape)
    )
rsList = jnp.array(rsList)

results = []

for (name, computeEnergies) in energies:

    times = []
    energyResults = []

    for dt in range(evalSteps):

        start = time.time()

        rs = rsList[dt]
        localEnergies = computeEnergies(parameters, rs) / N
        localEnergies.block_until_ready()

        stop = time.time()

        times.append(stop - start)
        energyResults.append(jnp.mean(localEnergies))

        print("{}\t{}".format(name, dt))

    results.append((name, np.mean(np.array(energyResults)[1:]), np.mean(np.array(times)[1:])))


###############################################################################
#   Print out statistics                                                      #
###############################################################################

print("------------------------------------\n\n\n")
    
print("{}\t{}\t{}".format("NAME   ", "ENERGY   ", "TIME   "))
for (name, energy, time) in results:
    print("{}\t{:.8f}\t{:.5f}".format(name, energy, time))
