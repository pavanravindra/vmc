import sys, os

sys.path.append("/burg-archive/ccce/users/phr2114/vmc")
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
#os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax
import jax.numpy as jnp
from flax.core.frozen_dict import freeze, unfreeze
import numpy as np

import wavefunctions, trajectory, optimization
import local_hamiltonian, old_hamiltonian

jax.config.update("jax_default_matmul_precision", "float32")
jax.config.update("jax_enable_x64", False)

import time

print("STARTING\n")

###############################################################################
#   Settings for whole simulation                                             #
###############################################################################

r_ws = 50
N = 18 # 56
tau = 1.25
dim = 2
numKpoints = 169

walkers = 128 # 1024

eqSteps      = 3 # 10 # 2000
trainSteps   = 3 # 10 # 1000
trainEqSteps = 3 # 20
evalSteps    = 3 # 10 # 1000

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
history = 1e-3


###############################################################################
#   Hyperparameter sampling                                                   #
###############################################################################

hyperparameters = jnp.array([1e0])
eta0 = hyperparameters[0]


###############################################################################
#   Creating wavefunction, walker updater, optimizer, and local energy        #
###############################################################################

#upCoeffs = jnp.array(np.random.normal(size=(numKpoints, spins[0])))
#downCoeffs = jnp.array(np.random.normal(size=(numKpoints, spins[1])))

upCoeffs = jnp.eye(numKpoints, spins[0])
downCoeffs = jnp.eye(numKpoints, spins[1])

"""
wavefunction = wavefunctions.LogFlatironMP(
    spins, dim, lattice, kpoints, upCoeffs, downCoeffs, 32, 3, 32, 32
)
"""
kpoints = wavefunctions.genKpoints(spins[0], lattice, dim)
wavefunction = wavefunctions.LogSimpleSlaters(
    spins, dim, kpoints
)

energies = [
    #(
    #    "Total Energy",
    #    local_hamiltonian.LocalEnergyUEG2D(wavefunction, lattice)
    #),
    (
        "Old",
        old_hamiltonian.LocalKineticEnergy(wavefunction)
        #old_hamiltonian.LocalEnergyUEG2D(wavefunction, lattice)
    ),
    (
        "New",
        local_hamiltonian.LocalKineticEnergy(wavefunction)
        #local_hamiltonian.LocalEnergyUEG2D(wavefunction, lattice)
    ),
]

energies = [(e[0],jax.jit(e[1].batch)) for e in energies]

rng = jax.random.PRNGKey(558)

rng, rs_rng, init_rng = jax.random.split(rng, 3)
rs = trajectory.wignerCrystal(
    spins, lattice, r_ws, walkers, rs_rng, dim=dim, gridShape=(7,4)
)
parameters = wavefunction.initBatch(init_rng, rs)

print("NUM PARAMETERS: {}\n".format(optimization.flatten(parameters).size))



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
