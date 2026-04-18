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
import custom_wavefunctions

#jax.config.update("jax_default_matmul_precision", "float32")
jax.config.update("jax_enable_x64", True)

import time

print("STARTING")

###############################################################################
#   Settings for whole simulation                                             #
###############################################################################

numTrials = 5

walkers = 1024
r_ws = 20
tau = 1.25

diagonalShift = 1e-3
eta = 1e-1
mu = 0.9

#setting = "2D_big_with_static"
setting = "3D_small"

if setting == "2D_big_with_static":
    dim = 2
    N = 56
    numKpoints = 169
elif setting == "3D_small":
    dim = 3
    N = 14
    numKpoints = N
else:
    raise Exception("Invalid setting!")

if dim == 2:
    lattice = wavefunctions.computeLattice(
        N, r_ws, dim, basis_matrix=jnp.array([[7,0],[0,4*jnp.sqrt(3)]])
    )
elif dim == 3:
    lattice = wavefunctions.computeLattice(N, r_ws, dim)
else:
    raise Exception("Invalid dimension!")

NUp = N // 2
NDown = N - NUp
spins = ( NUp , NDown )

kpoints = wavefunctions.genKpoints(numKpoints, lattice, dim)


###############################################################################
#   Creating wavefunction, walker updater, optimizer, and local energy        #
###############################################################################

rng = jax.random.PRNGKey(558)
rng, rs_rng, init_rng = jax.random.split(rng, 3)

if dim == 2:
    centers = trajectory.generateStripedAFM(spins, lattice, dim, gridShape=(7,4))
elif dim == 3:
    centers = trajectory.generateBCC(spins, lattice, dim)
(upCenters,downCenters) = centers[:spins[0]], centers[spins[0]:]
rs = centers[None,:,:] + (r_ws / 10) * jax.random.normal(rs_rng, shape=(walkers,N,dim))

alpha = 1 / (r_ws ** 2.0)
upCoeffs = wavefunctions.generateGaussianMPCoeffs(kpoints, upCenters, alpha)
downCoeffs = wavefunctions.generateGaussianMPCoeffs(kpoints, downCenters, alpha)

wavefunction = custom_wavefunctions.DynamicRoundsWavefunction(
    spins, dim, lattice, kpoints[:spins[0]], 32, T=5
)
mala = trajectory.MALAUpdater(wavefunction, r_ws)
if dim == 2:
    localEnergy = folx_hamiltonian.LocalEnergyUEG2D(wavefunction, lattice, sparsity_threshold=6)
elif dim == 3:
    localEnergy = folx_hamiltonian.LocalEnergyUEG3D(wavefunction, lattice, sparsity_threshold=6)
optimizer = optimization.StochasticReconfigurationMomentum(
    wavefunction, localEnergy, mode='smw'
)

parameters = wavefunction.initBatch(init_rng, rs)
(flatParameters,unravel) = jax.flatten_util.ravel_pytree(parameters)
newFlatParameters = flatParameters + np.random.normal(size=flatParameters.shape) * 1e-3
parameters = unravel(newFlatParameters)
history = np.random.normal(size=flatParameters.shape) * 1e-3

print("NUM PARAMETERS: " + str(optimization.flatten(parameters).size))
print("DTYPE: " + str(rs.dtype))
print("")


###############################################################################
#   Calculate times for different operations with current wavefunction        #
###############################################################################

evaluateWavefunction = jax.jit(wavefunction.applyBatch)
updateWalkers = jax.jit(mala.updateBatch)
computeEnergies = jax.jit(localEnergy.batch)
updateParameters = jax.jit(optimizer)

operations = [
    ("Wavefunction eval", evaluateWavefunction, (parameters, rs)),
    ("Walker update    ", updateWalkers, (parameters, rs, rng, tau)),
    ("Energy evaluation", computeEnergies, (parameters, rs)),
    ("Update parameters", updateParameters, (parameters, rs, eta, diagonalShift, mu, history)),
]

for (name, operation, args) in operations:
    result = operation(*args)
    if isinstance(result, jnp.ndarray):
        result.block_until_ready()
    else:
        try:
            result[0].block_until_ready()
        except:
            raise Exception("Wrong dtype...")

print("")

for (name, operation, args) in operations:
    times = []
    for trial in range(numTrials):
        start = time.time()
        result = operation(*args)
        if isinstance(result, jnp.ndarray):
            result.block_until_ready()
        else:
            try:
                result[0].block_until_ready()
            except:
                raise Exception("Wrong dtype...")
        stop = time.time()
        times.append(stop - start)
    print(name + ":\t" + str(np.mean(times[1:])) + " seconds")
