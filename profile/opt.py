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

import wavefunctions, hamiltonian, trajectory
import local_optimization

jax.config.update("jax_default_matmul_precision", "float32")
jax.config.update("jax_enable_x64", False)

import time

print("STARTING\n")

###############################################################################
#   Settings for whole simulation                                             #
###############################################################################

r_ws = 50
N = 56
tau = 1.25
dim = 2
numKpoints = 169

walkers = 1024

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

upCoeffs = jnp.array(np.random.normal(size=(numKpoints, spins[0])))
downCoeffs = jnp.array(np.random.normal(size=(numKpoints, spins[1])))

wavefunction = wavefunctions.LogFlatironMP(
    spins, dim, lattice, kpoints, upCoeffs, downCoeffs, 32, 3, 32, 32
)
mala = trajectory.MALAUpdater(wavefunction, r_ws)
if dim == 2:
    localEnergy = hamiltonian.LocalEnergyUEG2D(wavefunction, lattice)
elif dim == 3:
    localEnergy = hamiltonian.LocalEnergyUEG(wavefunction, lattice)
optimizers = [
    ( 
        "Full",
        local_optimization.StochasticReconfigurationMomentumDegrade(
            wavefunction, localEnergy, mode='smw'
        )
    ),
    ( 
        "ExpH",
        local_optimization.StochasticReconfigurationMomentumDegradeExpH(
            wavefunction, localEnergy, mode='smw'
        )
    ),
    ( 
        "ExpO",
        local_optimization.StochasticReconfigurationMomentumDegradeExpO(
            wavefunction, localEnergy, mode='smw'
        )
    ),
    #( 
    #    "FK",
    #    local_optimization.StochasticReconfigurationMomentumDegradeFK(
    #        wavefunction, localEnergy, mode='smw'
    #    )
    #),
    #( 
    #    "PS",
    #    local_optimization.StochasticReconfigurationMomentumDegradePS(
    #        wavefunction, localEnergy, mode='smw'
    #    )
    #),
]
optimizers = [(o[0],jax.jit(o[1])) for o in optimizers]

updateWalkerPositions = jax.jit(mala.updateBatch)
computeEnergies = jax.jit(localEnergy.batch)

rng = jax.random.PRNGKey(558)

rng, rs_rng, init_rng = jax.random.split(rng, 3)
rs = trajectory.wignerCrystal(
    spins, lattice, r_ws, walkers, rs_rng, dim=dim, gridShape=(7,4)
)
parameters = wavefunction.initBatch(init_rng, rs)

parameters = unfreeze(parameters)
tunedAs = wavefunctions.loadParameters(
    "/burg-archive/ccce/users/phr2114/vmc/ortho-test/CYJastrow/rs{}_best/parameters.msgpack".format(str(r_ws))
)['params']['CYJastrow']['As_same_diff']
parameters['params']['CYJastrow']['As_same_diff'] = tunedAs
parameters = freeze(parameters)

print("OPTIMIZERS:")
for (name,_) in optimizers:
    print(name)
print()

print("NUM PARAMETERS: {}\n".format(local_optimization.flatten(parameters).size))

###############################################################################
#   Parameter optimization                                                    #
###############################################################################


rng = jax.random.PRNGKey(151)
results = []

for (name, optimizer) in optimizers:

    times = []

    for dt in range(trainSteps):

        start = time.time()

        ( maxNorm , currentEnergies , newParameters , history ) = optimizer(
            parameters, rs, eta0, diagonalShift, mu, history
        )
        newParameters['params']['CYJastrow']['As_same_diff'].block_until_ready()

        stop = time.time()

        times.append(stop - start)

        rng, traj_rng = jax.random.split(rng, 2)
        rs = rs + 1e-4 * jax.random.normal(traj_rng, shape = rs.shape)

        if local_optimization.hasnan(newParameters):
            np.savetxt("FAILED_REEQUILIBRATE", [])
            wavefunctions.saveParameters(
                "parameters_failed.msgpack", parameters
            )
            np.save("big_startRs.npy", startRs)
            np.save("big_endRs.npy", rs)
            raise Exception("Parameters have somehow NaNed...")

        print("{}\t{}".format(name, dt))

    results.append((name, np.mean(np.array(times)[1:])))

###############################################################################
#   Print out statistics                                                      #
###############################################################################

print("------------------------------------\n\n\n")

for (name,time) in results:
    print("{}\t{:.5f}".format(name, time))
