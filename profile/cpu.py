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

import time

#jax.config.update("jax_enable_x64", False) # Single precision
jax.config.update("jax_enable_x64", True)  # Double Precision

r_ws = 50
N = 18 # 56
dim = 2

x = jnp.array(0.0)
rs = jnp.array(np.load("problem_rs.npy"), dtype=x.dtype)

print(rs.dtype)

NUp = N // 2
NDown = N - NUp
spins = ( NUp , NDown )

lattice = wavefunctions.computeLattice(
    N, r_ws, dim, basis_matrix=jnp.array([[7,0],[0,4*jnp.sqrt(3)]])
)

kpoints = wavefunctions.genKpoints(spins[0], lattice, dim)
wavefunction = wavefunctions.LogSimpleSlaters(
    spins, dim, kpoints
)

rng = jax.random.PRNGKey(558)
rng, rs_rng, init_rng = jax.random.split(rng, 3)
parameters = wavefunction.init(init_rng, rs)

print("NUM PARAMETERS: {}\n".format(optimization.flatten(parameters).size))

print("Analytic kinetic energy:\t{:.8f}".format(jnp.trace(kpoints @ kpoints.T)))

#local_hamiltonian.laplacian(wavefunction.apply, parameters, rs)
print(local_hamiltonian.LocalKineticEnergy(wavefunction).configuration(
    parameters, rs
))
