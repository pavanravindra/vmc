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

jax.config.update("jax_enable_x64", False) # Single precision
#jax.config.update("jax_enable_x64", True)  # Double Precision

import time

print("STARTING\n")

###############################################################################
#   Settings for whole simulation                                             #
###############################################################################

r_ws = 50
N = 18 # 56
dim = 2

walkers = 128 # 1024
rsIdx = 107

NUp = N // 2
NDown = N - NUp
spins = ( NUp , NDown )

lattice = wavefunctions.computeLattice(
    N, r_ws, dim, basis_matrix=jnp.array([[7,0],[0,4*jnp.sqrt(3)]])
)


###############################################################################
#   Creating wavefunction, walker updater, optimizer, and local energy        #
###############################################################################

kpoints = wavefunctions.genKpoints(spins[0], lattice, dim)
wavefunction = wavefunctions.LogSimpleSlaters(
    spins, dim, kpoints
)

energies = [
    (
        "Old",
        old_hamiltonian.laplacian
    ),
    (
        "New",
        local_hamiltonian.laplacian
    ),
]

energies = [
    (
        "Old",
        old_hamiltonian.LocalKineticEnergy(wavefunction).configuration
    ),
    (
        "New",
        local_hamiltonian.LocalKineticEnergy(wavefunction).configuration
    ),
]

rng = jax.random.PRNGKey(558)

rng, rs_rng, init_rng = jax.random.split(rng, 3)
rs = trajectory.wignerCrystal(
    spins, lattice, r_ws, walkers, rs_rng, dim=dim, gridShape=(7,4)
)
parameters = wavefunction.initBatch(init_rng, rs)

print("NUM PARAMETERS: {}\n".format(optimization.flatten(parameters).size))

print("Analytic kinetic energy:\t{:.8f}".format(jnp.trace(kpoints @ kpoints.T)))
print("-----------------------")



###############################################################################
#   Energy evaluation                                                         #
###############################################################################

results = []

rs = rs[rsIdx]
np.save("problem_rs.npy", rs)

x = jnp.array(0.0)
rs = jnp.array(np.load("problem_rs.npy"), dtype=x.dtype)

print("HIT: {}".format(rs.shape))

for (name, computeEnergy) in energies:

    start = time.time()

    localEnergy = computeEnergy(parameters, rs) # / N
    #localEnergy = computeEnergy(wavefunction.apply, parameters, rs)
    localEnergy.block_until_ready()

    stop = time.time()

    results.append((name, localEnergy, stop-start))

    print("DONE {}".format(name))

###############################################################################
#   Print out statistics                                                      #
###############################################################################

print("------------------------------------\n\n\n")
    
print("{}\t{}\t{}".format("NAME   ", "ENERGY   ", "TIME   "))
for (name, energy, time) in results:
    print("{}\t{:.8f}\t{:.5f}".format(name, energy, time))
