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

class LocalMP(wavefunctions.Wavefunction):
    """
    TODO
    """
    spins : (int,int)
    dim : int
    lattice : jnp.ndarray
    kpoints : jnp.ndarray
    hiddenFeatures : int     # number of features for all MLPs
    T : int                  # number of rounds of message passing
    d1 : int                 # number of features for 1-electron channels
    d2 : int                 # number of features for 2-electron channels

    def setup(self):
        
        self.slaterUp = wavefunctions.LogSimpleSlater(self.spins[0], self.dim, self.kpoints)
        self.slaterDown = wavefunctions.LogSimpleSlater(self.spins[1], self.dim, self.kpoints)
        self.CYJastrow = wavefunctions.LogCYJastrow(self.spins, self.lattice)

        self.dv = 2 * self.dim + 2   # dimensionality of 2-electron features

        self.hi0  = self.param('hi0',  lambda _ : jnp.zeros(self.d1))
        self.hij0 = self.param('hij0', lambda _ : jnp.zeros(self.d2 - self.dv))

        self.Wqs = [nn.Dense(self.d2, use_bias=False) for _ in range(self.T)]
        self.Wks = [nn.Dense(self.d2, use_bias=False) for _ in range(self.T)]

        self.attentionLayers = [nn.Dense(self.d2) for _ in range(self.T)]

        self.Fmt1s = [nn.Dense(self.hiddenFeatures) for _ in range(self.T)]
        self.Fmt2s = [nn.Dense(self.d2) for _ in range(self.T)]

        self.F1t1s = [nn.Dense(self.hiddenFeatures) for _ in range(self.T)]
        self.F1t2s = [nn.Dense(self.d1) for _ in range(self.T)]
        
        self.F2t1s = [nn.Dense(self.hiddenFeatures) for _ in range(self.T)]
        self.F2t2s = [nn.Dense(self.d2 - self.dv) for _ in range(self.T)]

        """
        self.backflowLinear = nn.Dense(self.dim)
        """
        self.backflowLinear = nn.Dense(
            self.dim, kernel_init=nn.initializers.zeros, use_bias=False
        )

        self.preLinear = nn.Dense(self.d1)
        self.jastrowLinear1 = nn.Dense(self.hiddenFeatures)
        self.jastrowLinear2 = nn.Dense(self.hiddenFeatures)
        self.jastrowLinear3 = nn.Dense(self.hiddenFeatures)
        """
        self.jastrowLinear4 = nn.Dense(1)
        """
        self.jastrowLinear4 = nn.Dense(
            1, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros
        )

    def __call__(self, rs):

        N = sum(self.spins)
        
        vij = wavefunctions.generateFeatures(rs, self.spins, self.lattice)      # (N,N,dv)
        hit = jnp.broadcast_to(self.hi0, (N,self.d1))             # (N,d1)
        hijt = jnp.broadcast_to(self.hij0, (N,N,self.d2-self.dv)) # (N,N,d2-dv)

        for t in range(self.T):

            git = hit                                             # (N,d1)
            gijt = jnp.concatenate([vij, hijt], axis=-1)          # (N,N,d2)

            qijt = self.Wqs[t](gijt)                              # (N,N,d2)
            kijt = self.Wks[t](gijt)                              # (N,N,d2)

            """
            Aijt = qijt * kijt
            Aijt = self.attentionLayers[t](nn.gelu(
                jnp.einsum("ild,ljd->ijd", qijt, kijt) / jnp.sqrt(N)
            ))                                                    # (N,N,d2)
            Aijt = self.attentionLayers[t](
                nn.gelu(
                    jnp.matmul(
                        jnp.transpose(qijt, (2, 0, 1)),
                        jnp.transpose(kijt, (2, 0, 1)),
                    ).transpose(1, 2, 0) / jnp.sqrt(N)
                )
            )
            """
            Aijt = self.attentionLayers[t](nn.gelu(
                jax.lax.dot_general(
                    jnp.transpose(qijt, (2,0,1)), jnp.transpose(kijt, (2,0,1)),
                    dimension_numbers=(((2,), (1,)), ((0,), (0,)))
                ).transpose(1,2,0) / jnp.sqrt(N)
            ))
            Fmijt = self.Fmt2s[t](nn.gelu(self.Fmt1s[t](gijt)))   # (N,N,d2)
            mijt = Aijt * Fmijt                                   # (N,N,d2)

            eye_mask = jnp.eye(N, dtype=bool)[:,:,None]
            acc_mijt = jnp.sum(jnp.where(~eye_mask, mijt, 0.0), axis=1)
            hit += self.F1t2s[t](nn.gelu(self.F1t1s[t](
                jnp.concatenate([acc_mijt, git], axis=-1)
            )))                                                   # (N,N,d1)
            
            hijt += self.F2t2s[t](nn.gelu(self.F2t1s[t](
                jnp.concatenate([mijt, gijt], axis=-1)
            )))                                                   # (N,N,d2-dv)

        backflow = self.backflowLinear(hit)
        xs = rs + backflow
        
        recLattice = jnp.linalg.inv(self.lattice)
        xsWrap = ((xs @ recLattice) % 1) @ self.lattice

        jastrowInput = jnp.concatenate(
            [hit, nn.gelu(self.preLinear(xsWrap))], axis=-1
        )
        neuralJastrow = self.jastrowLinear1(jastrowInput)
        neuralJastrow += self.jastrowLinear2(nn.gelu(neuralJastrow))
        neuralJastrow += self.jastrowLinear3(nn.gelu(neuralJastrow))
        neuralJastrow = jnp.sum(self.jastrowLinear4(nn.gelu(neuralJastrow)))
        
        slaterUp = self.slaterUp(xsWrap[:self.spins[0],:])
        slaterDown = self.slaterDown(xsWrap[self.spins[0]:,:])
        CYJastrow = self.CYJastrow(rs)
        slaterUp = self.slaterUp(xs[:self.spins[0],:])
        slaterDown = self.slaterDown(xs[self.spins[0]:,:])
        
        return slaterUp + slaterDown + CYJastrow + neuralJastrow

#upCoeffs = jnp.array(np.random.normal(size=(numKpoints, spins[0])))
#downCoeffs = jnp.array(np.random.normal(size=(numKpoints, spins[1])))

upCoeffs = jnp.eye(numKpoints, spins[0])
downCoeffs = jnp.eye(numKpoints, spins[1])

"""
# This is a MP-2SJB wavefunction
kpoints = wavefunctions.genKpoints(numKpoints, lattice, dim)
wavefunction = CustomWavefunction(
    spins, dim, lattice, kpoints, 32, upCoeffs, downCoeffs
)
"""

kpoints = wavefunctions.genKpoints(spins[0], lattice, dim)
#wavefunction = wavefunctions.LogSimpleSlaters(spins, dim, kpoints)
#wavefunction = wavefunctions.LogSlaterCYJastrow(spins, dim, lattice, kpoints)
wavefunction = wavefunctions.LogTwoBodySJB(spins, dim, lattice, kpoints, 32)
#wavefunction = wavefunctions.LogFlatironNOMP(spins, dim, lattice, kpoints, 16, 1, 16, 16)
#wavefunction = LocalMP(spins, dim, lattice, kpoints, 32, 3, 32, 32)

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
