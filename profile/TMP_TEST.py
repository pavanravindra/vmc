import sys, os

sys.path.append("/burg-archive/ccce/users/phr2114/vmc")
#os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"
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

walkers = 64 # 1024

eqSteps      = 100 # 2000
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

class LocalMP(wavefunctions.Wavefunction):
    """
    TODO
    """
    spins : (int,int)
    dim : int
    lattice : jnp.ndarray
    kpoints : jnp.ndarray
    d : int                  # number of features for everything
    T : int                  # number of rounds of message passing

    def setup(self):
        
        self.slaterUp = wavefunctions.LogSimpleSlater(self.spins[0], self.dim, self.kpoints)
        self.slaterDown = wavefunctions.LogSimpleSlater(self.spins[1], self.dim, self.kpoints)
        self.CYJastrow = wavefunctions.LogCYJastrow(self.spins, self.lattice)

        self.hi0  = self.param('hi0',  lambda _ : jnp.zeros(self.d))

        self.MLP_edge01 = nn.Dense(self.d)
        self.MLP_edge02 = nn.Dense(self.d)

        self.Wqs = [nn.Dense(self.d, use_bias=False) for _ in range(self.T)]
        self.Wks = [nn.Dense(self.d, use_bias=False) for _ in range(self.T)]

        self.ws = [self.param('w' + str(t), lambda _ : jnp.zeros(self.d)) for t in range(self.T)]
        self.MLP_att0 = [nn.Dense(self.d) for _ in range(self.T)]
        self.MLP_att1 = [nn.Dense(self.d) for _ in range(self.T)]

        self.MLP_msg0 = [nn.Dense(self.d) for _ in range(self.T)]
        self.MLP_msg1 = [nn.Dense(self.d) for _ in range(self.T)]
        
        self.MLP_node0 = [nn.Dense(self.d) for _ in range(self.T)]
        self.MLP_node1 = [nn.Dense(self.d) for _ in range(self.T)]
        
        self.MLP_edge0 = [nn.Dense(self.d) for _ in range(self.T)]
        self.MLP_edge1 = [nn.Dense(self.d) for _ in range(self.T)]

        self.jastrow0 = nn.Dense(self.d)
        self.jastrow1 = nn.Dense(
            1, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros
        )

        self.backflow0 = nn.Dense(self.d)
        self.backflow1 = nn.Dense(
            self.dim, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros
        )

    def __call__(self, rs):

        N = sum(self.spins)
        
        hit = jnp.broadcast_to(self.hi0, (N,self.d))             # (N,d)
        vij = wavefunctions.generateFeatures(rs, self.spins, self.lattice)      # (N,N,d)
        hijt = self.MLP_edge02(nn.swish(self.MLP_edge01(vij)))

        for t in range(self.T):

            qit = self.Wqs[t](hit)
            kit = self.Wks[t](hit)
            sijt = qit @ kit.T / jnp.sqrt(self.d) + self.MLP_att1[t](nn.swish(self.MLP_att0[t](hijt))) @ self.ws[t]
            gijt = nn.sigmoid(sijt)

            mijt = self.MLP_msg1[t](nn.swish(self.MLP_msg0[t](hijt)))
            eye_mask = jnp.eye(N, dtype=bool)[:,:,None]
            Mit = jnp.sum(
                jnp.where(~eye_mask, gijt[:,:,None] * mijt, 0.0),
                axis=1
            )

            hit = hit + self.MLP_node1[t](nn.swish(self.MLP_node0[t](
                jnp.concatenate([hit, Mit], axis=-1)
            )))

            aijt = hit[:,None,:] + hit[None,:,:]
            bijt = hit[:,None,:] * hit[None,:,:]

            hijt = hijt + self.MLP_edge1[t](nn.swish(self.MLP_edge0[t](
                jnp.concatenate([hijt, vij, aijt, bijt], axis=-1)
            )))
        
        neuralJastrow = jnp.sum(self.jastrow1(nn.swish(self.jastrow0(hit)))) / N
        backflow = self.backflow1(nn.swish(self.backflow0(hit)))

        xs = rs + backflow

        CYJastrow = self.CYJastrow(rs)
        slaterUp = self.slaterUp(xs[:self.spins[0],:])
        slaterDown = self.slaterDown(xs[self.spins[0]:,:])
        
        return slaterUp + slaterDown + CYJastrow + neuralJastrow


kpoints = wavefunctions.genKpoints(spins[0], lattice, dim)
wavefunction = LocalMP(spins, dim, lattice, kpoints, 32, 2)

mala = trajectory.MALAUpdater(wavefunction, r_ws)
updateWalkerPositions = jax.jit(mala.updateBatch)

energies = [
    (
        "Old",
        old_hamiltonian.LocalEnergyUEG2D(wavefunction, lattice).batch
    ),
    (
        "New",
        new_hamiltonian.LocalEnergyUEG2D(wavefunction, lattice).batch
    ),
    (
        "folx0",
        folx_hamiltonian.LocalEnergyUEG2D(wavefunction, lattice, 0).batch
    ),
    #(
    #    "folx2",
    #    folx_hamiltonian.LocalEnergyUEG2D(wavefunction, lattice, 2).batch
    #),
    #(
    #    "folx4",
    #    folx_hamiltonian.LocalEnergyUEG2D(wavefunction, lattice, 4).batch
    #),
    (
        "folx6",
        folx_hamiltonian.LocalEnergyUEG2D(wavefunction, lattice, 6).batch
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

    if dt % 1 == 0:
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
    est_time = time * (1024 / walkers) * 2000 / 3600
    print("{}\t{:.8f}\t{:.5f}".format(name, energy, est_time))
