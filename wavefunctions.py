import jax
import jax.numpy as jnp

from flax import linen as nn
from flax.serialization import to_bytes, from_bytes
from flax.core import FrozenDict

from pyscf import gto, scf, ao2mo
import qc

import numpy as np

import itertools

def computeL(N, r_ws):
    n = 3. / (4. * jnp.pi * (r_ws ** 3.))
    V = N / n
    L = V ** (1./3.)
    return L

def computeRws(N, L):
    return L * (3. / (4. * jnp.pi * N)) ** (1./3.)

class Wavefunction(nn.Module):
    """
    Defines batched functions based on single-walker functions. This way you
    only need to define single-walker initialize and forward pass functions
    when defining wavefunctions.
    """
    def initBatch(self, rng, walkerRs):
        return self.init(rng, walkerRs[0,:,:])
    
    def applyBatch(self, parameters, walkerRs):
        vmapApply = jax.vmap(self.apply, in_axes=(None,0))
        return vmapApply(parameters, walkerRs)

def saveParameters(filename, parameters):
    """
    Saves a pytree object that contains a wavefunction's parameters.
    """
    with open(filename, "wb") as f:
        f.write(to_bytes(parameters))
    return True

def loadParameters(filename):
    """
    Loads a pytree FrozenDict from provided file. The file should have been
    saved using the `saveParameters` method above.
    """
    with open(filename, "rb") as f:
        parameters = from_bytes(FrozenDict, f.read())
    return parameters

def genKpoints(N, L):
    """
    Generate as many k-points as needed to represent N electrons in a cubic box
    with side length L. To make jax happy, we always return exactly N k-points,
    even if that doesn't lead to a full outer shell.
    
    NOTE: This code is a (only slightly) modified version of code taken from
    FermiNet! The original code is from the `make_kpoints` function from
    `ferminet/pbc/envelopes.py`.
    """
    
    recUnitVectorLength = 2 * jnp.pi / L
    
    dk = 1 + 1e-5
    
    max_k = int((N * dk)**(1/3) + 0.5)
    ordinals = sorted(range(-max_k, max_k+1), key=abs)
    ordinals = jnp.array(list(itertools.product(ordinals, repeat=3)))
    
    kpoints = ordinals * recUnitVectorLength
    kpointsOrder = jnp.argsort(jnp.linalg.norm(kpoints, axis=1))
    kpoints = kpoints[kpointsOrder]
    kNorms = jnp.linalg.norm(kpoints, axis=1)
    
    thresholdValue = kNorms[N-1] * dk
    chosenKpoints = kpoints[:N,:]
    return chosenKpoints

def testAntisymmetry(wavefunction, parameters, rs, tol=1e-6):
    """
    Tests that the `apply` method of a wavefunction object properly obeys
    Fermionic antisymmetry rules.
    """

    N = rs.shape[0]
    referenceValue = wavefunction.apply(parameters, rs)

    # Swapping spin up electrons
    for i in range(wavefunction.spins[0]):
        
        localBatch = []
        for j in range(wavefunction.spins[0]):
            if i == j:
                continue;
            localRs = np.array(rs)
            localRs[[i,j]] = localRs[[j,i]]
            localBatch.append(localRs)
        localBatch = jnp.array(localBatch)

        batchResults = wavefunction.applyBatch(parameters, localBatch)

        if any(jnp.abs(batchResults - referenceValue) > tol):
            raise Exception("This wavefunction is not antisymmetric!")
    
    # Swapping spin down electrons
    for i in range(wavefunction.spins[0],N):
        
        localBatch = []
        for j in range(wavefunction.spins[0],N):
            if i == j:
                continue;
            localRs = np.array(rs)
            localRs[[i,j]] = localRs[[j,i]]
            localBatch.append(localRs)
        localBatch = jnp.array(localBatch)

        batchResults = wavefunction.applyBatch(parameters, localBatch)

        if any(jnp.abs(batchResults - referenceValue) > tol):
            raise Exception("This wavefunction is not antisymmetric!")

    return True

class LogSimpleSlater(Wavefunction):
    """
    Creates a log-wavefunction that is just a simple Slater determinant of the
    input electron coordinates. The basis for the determinant is the lowest N
    k-points that satisfy periodic boundary conditions for the specified box
    length L (refer to `genKpoints`).
    
    NOTE: These determinants use the convention that different particle
    positions are in different rows. Columns correspond to plane wave orbitals.
    """
    N : int
    L : float

    def setup(self):
        self.kpoints = genKpoints(self.N, self.L)

    def __call__(self, rs):
        def makeSimpleSlaterRow(ri):
            def localKpointFunction(k):
                sign = jnp.sign(
                    1000. * jnp.sign(k[2]) +
                    100. * jnp.sign(k[1]) +
                    10. * jnp.sign(k[0]) +
                    1.
                )
                cosSwitch = (sign + 1.) / 2. * jnp.cos(jnp.dot(k, ri))
                sinSwitch = (1. - sign) / 2. * jnp.sin(jnp.dot(-k, ri))
                return cosSwitch + sinSwitch
            return jax.vmap(localKpointFunction)(self.kpoints)
        slaterMatrix = jax.vmap(makeSimpleSlaterRow)(rs)
        return jnp.linalg.slogdet(slaterMatrix)[1]

def coulombYukawa(r, A, F, L):
    """
    Coulomb-Yukawa two-body Jastrow function. Also has a decay term so that the
    Jastrow appropriately dies off before the boundary.

    """
    cy = (A/r) * (1 - jnp.exp(-r/F))
    r_cut = L/2
    x = jnp.clip(r / r_cut, a_min=0.0, a_max=1.0-1e-5)
    decay = jnp.exp(1 - 1 / (1 - x**2))
    return cy * decay

def getOffDiagonalFlat(x):
    """
    Given an NxN matrix `x`, returns a (N*(N-1),) jnp array with only the
    off-diagonal elements of `x`.
    """
    N = x.shape[0]
    i = jnp.arange(N)
    row = jnp.repeat(i, N-1)
    col = jnp.concatenate([
        jnp.concatenate([jnp.arange(j), jnp.arange(j+1, N)]) for j in range(N)
    ])
    return x[row, col]

class LogCYJastrow(Wavefunction):
    """
    Creates a log-wavefunction that is a Coulomb-Yukawa Jastrow term. The same
    and different spin electrons are handled by different $A$ parameters.
    Hence, this wavefunction has two variational parameters.

    The conventional $F$ parameters are set by the cusp conditions.

    NOTE: In this implementation, we always take the absolute values of the $A$
    parameters. This is so that even if negative values are encountered during
    optimization, the resulting wavefunction remains physical.
    """
    spins : (int,int)
    L : float

    def setup(self):

        N = self.spins[0] + self.spins[1]
        n = N / (self.L**3.)

        self.As = self.param(
            "As_same_diff",
            lambda rng : jnp.full(2, 1.0 / jnp.sqrt(4 * jnp.pi * n))
        )

    def __call__(self, rs):

        A_same = jnp.abs(self.As[0])
        A_diff = jnp.abs(self.As[1])
        
        F_same = jnp.sqrt(2 * A_same)
        F_diff = jnp.sqrt(A_diff)
        
        disps = rs[:,None,:] - rs[None,:,:]  # (N, N, 3)
        disps = (disps + self.L/2) % self.L - self.L/2
        mask = ~jnp.eye(disps.shape[0], dtype=bool)[:,:,None]
        disps = jnp.where(mask, disps, 0.0)
        r_ij = jnp.linalg.norm(disps, axis=-1)

        same_up = getOffDiagonalFlat(r_ij[:self.spins[0],:self.spins[0]])
        same_down = getOffDiagonalFlat(r_ij[self.spins[0]:,self.spins[0]:])
        sameDists = jnp.concatenate([same_up, same_down])
        sameCY = coulombYukawa(sameDists, A_same, F_same, self.L)
        
        diffDists = r_ij[:self.spins[0],self.spins[0]:].flatten()
        diffCY = coulombYukawa(diffDists, A_diff, F_diff, self.L)

        return -0.5 * jnp.sum(sameCY) - jnp.sum(diffCY)

class LogSlaterCYJastrow(Wavefunction):
    """
    Creates a log-wavefunction that is the product of two simple Slater
    determinant with the lowest k-points filled and a Coulomb-Yukawa Jastrow.

    There are 2 variational parameters, both in the Jastrow.
    """
    spins : (int,int)
    L : float

    def setup(self):
        self.slaterUp = LogSimpleSlater(self.spins[0], self.L)
        self.slaterDown = LogSimpleSlater(self.spins[1], self.L)
        self.CYJastrow = LogCYJastrow(self.spins, self.L)

    def __call__(self, rs):
        slaterUp = self.slaterUp(rs[:self.spins[0],:])
        slaterDown = self.slaterDown(rs[self.spins[0]:,:])
        CYJastrow = self.CYJastrow(rs)
        return slaterUp + slaterDown + CYJastrow

class NeuralDecayedFunction(nn.Module):
    """
    This is a neural function that takes in a list of one-dimensional inputs
    and spits out a list of the one-dimensional outputs.
    
    Importantly, the output is damped so that it fades to 0 for larger input
    values. The damping forces the function and all of its higher order
    derivatives to fade to 0 at L/2.

    Additionally, we make sure that the function is cuspless, in that it's
    first derivatve approaches 0 as the input approaches 0.
    """
    L : float
    hiddenLayers : int = 1
    hiddenFeatures : int = 32

    def setup(self):
        self.hiddenLayerStack = [
            nn.Dense(self.hiddenFeatures) for _ in range(self.hiddenLayers)
        ]
        self.outputLayer = nn.Dense(1)

    def __call__(self, x):
        
        x_cut = self.L / 2
        x = jnp.clip(x / x_cut, a_min=0.0, a_max=1.0-1e-5)
        decay = jnp.exp(1 - 1 / (1 - x**2))
        h = decay[:,None]
        #h = jnp.cos(jnp.pi * x)[:,None] # Makes function cuspless
        for layer in self.hiddenLayerStack:
            h = nn.swish(layer(h))
        networkOutput = self.outputLayer(h).flatten()
        return networkOutput * decay

class NeuralJastrow(Wavefunction):
    """
    This is a two-body Jastrow that sums over all pairs of non-identical
    particles. The individual two-body contributions come from two separated
    neural networks: one for matching spin electrons and one for opposite spin
    electrons.
    """
    spins: (int,int)
    L: float
    hiddenLayers : int
    hiddenFeatures : int

    def setup(self):
        N = self.spins[0] + self.spins[1]
        self.neuralFunctionSame = NeuralDecayedFunction(
            self.L, self.hiddenLayers, self.hiddenFeatures
        )
        self.neuralFunctionDiff = NeuralDecayedFunction(
            self.L, self.hiddenLayers, self.hiddenFeatures
        )

    def __call__(self, rs):
        
        disps = rs[:,None,:] - rs[None,:,:]  # (N, N, 3)
        disps = (disps + self.L/2) % self.L - self.L/2
        mask = ~jnp.eye(disps.shape[0], dtype=bool)[:,:,None]
        disps = jnp.where(mask, disps, 0.0)
        r_ij = jnp.linalg.norm(disps, axis=-1)

        same_up = getOffDiagonalFlat(r_ij[:self.spins[0],:self.spins[0]])
        same_down = getOffDiagonalFlat(r_ij[self.spins[0]:,self.spins[0]:])
        sameDists = jnp.concatenate([same_up, same_down])
        sameCY = self.neuralFunctionSame(sameDists)
        
        diffDists = r_ij[:self.spins[0],self.spins[0]:].flatten()
        diffCY = self.neuralFunctionDiff(diffDists)

        # Same spin electrons have been double counted above...
        return -0.5 * jnp.sum(sameCY) - jnp.sum(diffCY)
    
class NeuralSlaterJastrow(Wavefunction):
    """
    This log-wavefunction is a simple Slater determinant combined with a
    two-term Jastrow. The first term in the Jastrow is a standard two-body
    Coulomb-Yukawa Jastrow. The second term is a neural two-body corrective
    factor. The idea is basically to use a neural network to learn the best
    possible two-body Jastrow.

    The Coulomb-Yukawa Jastrow satsifies the cusp conditions, and the neural
    component is forced to be cuspless. So this wavefunction is guaranteed to
    satisfy the cusp conditions.
    """
    spins : (int,int)
    L : float
    hiddenLayers : int = 1
    hiddenFeatures : int = 32

    def setup(self):
        
        self.slaterUp = LogSimpleSlater(self.spins[0], self.L)
        self.slaterDown = LogSimpleSlater(self.spins[1], self.L)
        self.CYJastrow = LogCYJastrow(self.spins, self.L)
        self.neuralJastrow = NeuralJastrow(
            self.spins, self.L, self.hiddenLayers, self.hiddenFeatures
        )

    def __call__(self, rs):
        
        rsUp = rs[:self.spins[0],:]
        rsDown = rs[self.spins[0]:,:]
        
        slaterUp = self.slaterUp(rsUp)
        slaterDown = self.slaterDown(rsDown)
        CYJastrow = self.CYJastrow(rs)
        nqsJastrow = self.neuralJastrow(rs)
        return slaterUp + slaterDown + CYJastrow + nqsJastrow

class Backflow(nn.Module):
    """
    A pairwise neural backflow coordinate transform. The forward pass takes in
    electron positions and returns updated electron positions that are
    determined by a neural network acting on the scalar distance between a pair
    of electrons.

    The backflow transformation pushes coordinates in the direction of the
    vector $r_i - r_j$.
    """
    spins: (int,int)
    L: float
    hiddenLayers : int = 1
    hiddenFeatures : int = 32

    def setup(self):
        N = self.spins[0] + self.spins[1]
        self.neuralFunctionSame = NeuralDecayedFunction(
            self.L, self.hiddenLayers, self.hiddenFeatures
        )
        self.neuralFunctionDiff = NeuralDecayedFunction(
            self.L, self.hiddenLayers, self.hiddenFeatures
        )

    def __call__(self, rs):
            
        N = self.spins[0] + self.spins[1]
        disps = rs[:,None,:] - rs[None,:,:]  # (N, N, 3)
        disps = (disps + self.L/2) % self.L - self.L/2
        mask = ~jnp.eye(disps.shape[0], dtype=bool)[:,:,None]
        disps = jnp.where(mask, disps, 0.0)
        r_ij = jnp.sqrt(jnp.sum(disps**2, axis=-1) + 1e-15)

        bothUpr_ij = r_ij[:self.spins[0],:self.spins[0]]
        bothDownr_ij = r_ij[self.spins[0]:,self.spins[0]:]
        upDownr_ij = r_ij[self.spins[0]:,:self.spins[0]]

        bothUpForces = jnp.reshape(
            self.neuralFunctionSame(jnp.reshape(
                bothUpr_ij, (-1,)
            )),
            bothUpr_ij.shape
        )
        bothDownForces = jnp.reshape(
            self.neuralFunctionSame(jnp.reshape(
                bothDownr_ij, (-1,)
            )),
            bothDownr_ij.shape
        )
        upDownForces = jnp.reshape(
            self.neuralFunctionDiff(jnp.reshape(
                upDownr_ij, (-1,)
            )),
            upDownr_ij.shape
        )

        backflowForces = jnp.zeros((N,N))
        backflowForces = backflowForces.at[:self.spins[0],:self.spins[0]].set(bothUpForces)
        backflowForces = backflowForces.at[self.spins[0]:,self.spins[0]:].set(bothDownForces)
        backflowForces = backflowForces.at[self.spins[0]:,:self.spins[0]].set(upDownForces)
        backflowForces = backflowForces.at[:self.spins[0],self.spins[0]:].set(upDownForces.T)
        backflowForces = jnp.where(mask[:,:,0], backflowForces, 0.0)
        
        backflow = jnp.sum(backflowForces[:,:,None] * disps, axis=1)
        xs = rs + backflow

        return xs
    
class SlaterCYJastrowNeuralBackflow(Wavefunction):
    """
    Slater-Jastrow-Backflow wavefunction with the following:
    - Slater: two simple RHF slater determinants
    - Jastrow: Coulomb-Yukawa
    - Backflow: sum of neural pairwise backflow shifts
    """
    spins : (int,int)
    L : float
    hiddenLayers : int = 1
    hiddenFeatures : int = 32

    def setup(self):
        self.slaterUp = LogSimpleSlater(self.spins[0], self.L)
        self.slaterDown = LogSimpleSlater(self.spins[1], self.L)
        self.CYJastrow = LogCYJastrow(self.spins, self.L)
        self.backflow = Backflow(
            self.spins, self.L, self.hiddenLayers, self.hiddenFeatures
        )

    def __call__(self, rs):

        xs = self.backflow(rs)
        
        xsUp = xs[:self.spins[0],:]
        xsDown = xs[self.spins[0]:,:]
        
        slaterUp = self.slaterUp(xsUp)
        slaterDown = self.slaterDown(xsDown)
        CYJastrow = self.CYJastrow(rs)

        return slaterUp + slaterDown + CYJastrow
    
class PairwiseNeuralSlaterJastrowBackflow(Wavefunction):
    """
    Slater-Jastrow-Backflow wavefunction with the following:
    - Slater: two simple RHF slater determinants
    - Jastrow: Coulomb-Yukawa + neural correction acting on individual $r_ij$
    - Backflow: sum of neural pairwise backflow shifts
    """
    spins : (int,int)
    L : float
    hiddenLayers : int = 1
    hiddenFeatures : int = 32

    def setup(self):
        self.slaterUp = LogSimpleSlater(self.spins[0], self.L)
        self.slaterDown = LogSimpleSlater(self.spins[1], self.L)
        self.CYJastrow = LogCYJastrow(self.spins, self.L)
        self.neuralJastrow = NeuralJastrow(
            self.spins, self.L, self.hiddenLayers, self.hiddenFeatures
        )
        self.backflow = Backflow(
            self.spins, self.L, self.hiddenLayers, self.hiddenFeatures
        )

    def __call__(self, rs):

        xs = self.backflow(rs)
        
        xsUp = xs[:self.spins[0],:]
        xsDown = xs[self.spins[0]:,:]
        
        slaterUp = self.slaterUp(xsUp)
        slaterDown = self.slaterDown(xsDown)
        CYJastrow = self.CYJastrow(rs)
        nqsJastrow = self.neuralJastrow(rs)

        return slaterUp + slaterDown + CYJastrow + nqsJastrow

def uhfInitialization(r_ws, spins, numkPoints, seed=558):
    """
    Runs UHF for uniform electron gas with Wigner-Seitz radius `r_ws`.
    `numkPoints` specifies the number of k-points to use in the planewave
    basis for UHF.

    Returns the k-points used in the planewave basis and the UHF ground state
    coefficients for both spin-up and spin-down electrons.

    The coefficient matrices are of shape (K,N) where K is the number of
    planewaves and N is the number of particles of each spin.
    """
    
    np.random.seed(seed)
    
    system = qc.ueg_qc(r_ws, spins, numkPoints=numkPoints)
    kpoints = system.get_k_points()
    numkPoints = kpoints.shape[0]
    h1 = system.get_h1_real(kpoints)
    eri = system.get_eri_tensor_real(kpoints)
    
    mol = gto.M(verbose=0)
    mol.nelectron = system.n_particles
    mol.incore_anyway = True
    mol.energy_nuc = lambda *args: 0.0
    
    umf = scf.UHF(mol)
    umf.max_cycle = 200
    umf.get_hcore = lambda *args: [h1, h1]
    umf.get_ovlp = lambda *args: np.eye(numkPoints)
    umf._eri = ao2mo.restore(8, np.double(eri), numkPoints)
    
    dm0 = umf.get_init_guess()
    dm0[0] += np.random.randn(numkPoints, numkPoints) * 0.1
    dm0[1] += np.random.randn(numkPoints, numkPoints) * 0.1
    umf.level_shift = 0.4

    umf.kernel(dm0)
    
    mo1 = umf.stability(external=False)[0]
    umf = umf.newton().run(mo1, umf.mo_occ)
    mo1 = umf.stability(external=False)[0]
    umf = umf.newton().run(mo1, umf.mo_occ)
    umf.stability(external=False)

    energy = umf.e_tot
    kpoints = jnp.array(kpoints)
    spinUpCoeff = jnp.array(umf.mo_coeff[0][:,:spins[0]])
    spinDownCoeff = jnp.array(umf.mo_coeff[1][:,:spins[1]])

    return ( energy , kpoints , spinUpCoeff , spinDownCoeff )

class LogUHFSlaters(Wavefunction):
    """
    Creates a log-wavefunction that is the product of two unrestricted Slater
    determinants where the orbitals are sums of `numKpoints` planewaves.

    The `coeff` parameters contain the coefficients for this sum of planewaves,
    and it remains FIXED since they are not added as model parameters.
    """
    spins : (int,int)
    L : float
    kpoints : jnp.ndarray
    spinUpCoeff : jnp.ndarray
    spinDownCoeff : jnp.ndarray

    def __call__(self, rs):
        
        def makeSimpleSlaterRow(ri):
            def localKpointFunction(k):
                sign = jnp.sign(
                    1000. * jnp.sign(k[0]) +
                    100. * jnp.sign(k[1]) +
                    10. * jnp.sign(k[2]) +
                    1.
                )
                cosSwitch = (sign + 1.) / 2. * jnp.cos(jnp.dot(k, ri))
                sinSwitch = (1. - sign) / 2. * jnp.sin(jnp.dot(-k, ri))
                return cosSwitch + sinSwitch
            return jax.vmap(localKpointFunction)(self.kpoints)
            
        upMatrix = jax.vmap(makeSimpleSlaterRow)(rs[:self.spins[0],:])
        downMatrix = jax.vmap(makeSimpleSlaterRow)(rs[self.spins[0]:,:])

        upSlater = jnp.linalg.slogdet(upMatrix @ self.spinUpCoeff)[1]
        downSlater = jnp.linalg.slogdet(downMatrix @ self.spinDownCoeff)[1]
        
        return upSlater + downSlater

class LogUMPSlaters(Wavefunction):
    """
    Creates a log-wavefunction that is the product of two unrestricted Slater
    determinants where the orbitals are sums of `numKpoints` planewaves.

    The `coeff` parameters contain the coefficients for this sum of planewaves,
    and they will be optimized by any downstream optimizer, since they are
    added as parameters to the wavefunction.
    """
    spins : (int,int)
    L : float
    kpoints : jnp.ndarray
    spinUpInit : jnp.ndarray
    spinDownInit : jnp.ndarray

    def setup(self):

        N = self.spins[0] + self.spins[1]
        n = N / (self.L**3.)

        self.spinUpCoeff = self.param(
            "spinUpCoeff",
            lambda rng: self.spinUpInit
        )
        self.spinDownCoeff = self.param(
            "spinDownCoeff",
            lambda rng: self.spinDownInit
        )

    def __call__(self, rs):
        
        def makeSimpleSlaterRow(ri):
            def localKpointFunction(k):
                sign = jnp.sign(
                    1000. * jnp.sign(k[0]) +
                    100. * jnp.sign(k[1]) +
                    10. * jnp.sign(k[2]) +
                    1.
                )
                cosSwitch = (sign + 1.) / 2. * jnp.cos(jnp.dot(k, ri))
                sinSwitch = (1. - sign) / 2. * jnp.sin(jnp.dot(-k, ri))
                return cosSwitch + sinSwitch
            return jax.vmap(localKpointFunction)(self.kpoints)
            
        upMatrix = jax.vmap(makeSimpleSlaterRow)(rs[:self.spins[0],:])
        downMatrix = jax.vmap(makeSimpleSlaterRow)(rs[self.spins[0]:,:])

        upSlater = jnp.linalg.slogdet(upMatrix @ self.spinUpCoeff)[1]
        downSlater = jnp.linalg.slogdet(downMatrix @ self.spinDownCoeff)[1]
        
        return upSlater + downSlater
    
class LogUHFCY(Wavefunction):
    """
    Slater-Jastrow wavefunction with following specs:
    - Slater: UHF solution (fixed)
    - Jastrow: Coulomb-Yukawa
    """
    spins : (int,int)
    L : float
    kpoints : jnp.ndarray
    spinUpInit : jnp.ndarray
    spinDownInit : jnp.ndarray

    def setup(self):
        self.slater = LogUHFSlaters(
            self.spins, self.L,
            self.kpoints, self.spinUpInit, self.spinDownInit
        )
        self.CYJastrow = LogCYJastrow(self.spins, self.L)

    def __call__(self, rs):
        slater = self.slater(rs)
        CYJastrow = self.CYJastrow(rs)
        return slater + CYJastrow
    
class LogUMPCY(Wavefunction):
    """
    Slater-Jastrow wavefunction with following specs:
    - Slater: unrestricted sum of multiple planewaves
    - Jastrow: Coulomb-Yukawa
    """
    spins : (int,int)
    L : float
    kpoints : jnp.ndarray
    spinUpInit : jnp.ndarray
    spinDownInit : jnp.ndarray

    def setup(self):
        self.slater = LogUMPSlaters(
            self.spins, self.L,
            self.kpoints, self.spinUpInit, self.spinDownInit
        )
        self.CYJastrow = LogCYJastrow(self.spins, self.L)

    def __call__(self, rs):
        slater = self.slater(rs)
        CYJastrow = self.CYJastrow(rs)
        return slater + CYJastrow
    
class LogUMPSJB(Wavefunction):
    """
    Slater-Jastrow wavefunction with following specs:
    - Slater: unrestricted sum of multiple planewaves
    - Jastrow: Coulomb-Yukawa + neural pairwise Jastrow
    - Backflow: neural pairwise backflow shifts
    """
    spins : (int,int)
    L : float
    kpoints : jnp.ndarray
    spinUpInit : jnp.ndarray
    spinDownInit : jnp.ndarray
    hiddenLayers : int
    hiddenFeatures : int

    def setup(self):
        self.slater = LogUMPSlaters(
            self.spins, self.L,
            self.kpoints, self.spinUpInit, self.spinDownInit
        )
        self.CYJastrow = LogCYJastrow(self.spins, self.L)
        self.neuralJastrow = NeuralJastrow(
            self.spins, self.L, self.hiddenLayers, self.hiddenFeatures
        )
        self.backflow = Backflow(
            self.spins, self.L, self.hiddenLayers, self.hiddenFeatures
        )

    def __call__(self, rs):

        xs = self.backflow(rs)
        
        slater = self.slater(xs)
        CYJastrow = self.CYJastrow(rs)
        nqsJastrow = self.neuralJastrow(rs)
        
        return slater + CYJastrow + nqsJastrow
        
class DecayFunction():
    """
    Takes in a list of distances and computes the (cuspless) bump decay
    function.

    This is useful in computing Jastrows, since we want the effect of the
    Jastrow to die down before r_ij = L/2 so that the Jastrow remains smooth
    everywhere (except at the cusps ofc).
    """

    def __init__(self, L):
        self.L = L
    
    def __call__(self, dists):
        r_cut = self.L / 2
        xs = jnp.clip(dists / r_cut, a_min=0.0, a_max=0.99999)
        decay = jnp.exp(1 - 1 / (1 - xs**2))
        return decay

class LogTwoBodyJastrowRHF(Wavefunction):
    """
    Slater-Jastrow wavefunction with following specs:
    - Slater: RHF ground state
    - Jastrow: Coulomb-Yukawa + neural Jastrow with only one round of message
               passing
    """
    spins : (int,int)
    L : float
    hiddenFeatures : int

    def setup(self):
        
        self.slaterUp = LogSimpleSlater(self.spins[0], self.L)
        self.slaterDown = LogSimpleSlater(self.spins[1], self.L)
        self.CYJastrow = LogCYJastrow(self.spins, self.L)
        
        self.decay = DecayFunction(self.L)
        
        self.linearSelf1 = nn.Dense(self.hiddenFeatures)
        self.linearSelf2 = nn.Dense(1)

    def __call__(self, rs):
        
        slaterUp = self.slaterUp(rs[:self.spins[0],:])
        slaterDown = self.slaterDown(rs[self.spins[0]:,:])
        CYJastrow = self.CYJastrow(rs)
        
        disps = rs[:,None,:] - rs[None,:,:]  # (N, N, 3)
        disps = (disps + self.L/2) % self.L - self.L/2
        mask = ~jnp.eye(disps.shape[0], dtype=bool)[:,:,None]
        disps = jnp.where(mask, disps, 0.0)
        r_ij = jnp.linalg.norm(disps, axis=-1)
        decays = self.decay(r_ij)[:,:,None]
        decays = jnp.where(mask, decays, 0.0)
        
        cusplessDisps = jnp.cos(jnp.pi * disps / self.L)

        N = self.spins[0] + self.spins[1]
        electronIdxs = jnp.arange(N)
        electronSpins = jnp.where(electronIdxs < self.spins[0], 1, -1)
        matchMatrix = jnp.outer(electronSpins, electronSpins)[:,:,None]

        v_ij = jnp.concatenate([cusplessDisps, matchMatrix], axis=-1)
        
        selfTerm = decays * self.linearSelf2(nn.swish(self.linearSelf1(v_ij)))

        h_ij = selfTerm
        MBJastrow = jnp.sum(h_ij)
        
        return slaterUp + slaterDown + CYJastrow + MBJastrow

class LogThreeBodyJastrowRHF(Wavefunction):
    """
    Slater-Jastrow wavefunction with following specs:
    - Slater: RHF ground state
    - Jastrow: Coulomb-Yukawa + neural Jastrow with only one round of message
               passing
    """
    spins : (int,int)
    L : float
    hiddenFeatures : int

    def setup(self):
        
        self.slaterUp = LogSimpleSlater(self.spins[0], self.L)
        self.slaterDown = LogSimpleSlater(self.spins[1], self.L)
        self.CYJastrow = LogCYJastrow(self.spins, self.L)
        
        self.decay = DecayFunction(self.L)
        
        self.linearSelf1 = nn.Dense(self.hiddenFeatures)
        self.linearSelf2 = nn.Dense(1)
        self.linearThree1 = nn.Dense(self.hiddenFeatures)
        self.linearThree2 = nn.Dense(1)

    def __call__(self, rs):
        
        slaterUp = self.slaterUp(rs[:self.spins[0],:])
        slaterDown = self.slaterDown(rs[self.spins[0]:,:])
        CYJastrow = self.CYJastrow(rs)
        
        disps = rs[:,None,:] - rs[None,:,:]  # (N, N, 3)
        disps = (disps + self.L/2) % self.L - self.L/2
        mask = ~jnp.eye(disps.shape[0], dtype=bool)[:,:,None]
        disps = jnp.where(mask, disps, 0.0)
        r_ij = jnp.linalg.norm(disps, axis=-1)
        decays = self.decay(r_ij)[:,:,None]
        decays = jnp.where(mask, decays, 0.0)
        
        cusplessDisps = jnp.cos(jnp.pi * disps / self.L)

        N = self.spins[0] + self.spins[1]
        electronIdxs = jnp.arange(N)
        electronSpins = jnp.where(electronIdxs < self.spins[0], 1, -1)
        matchMatrix = jnp.outer(electronSpins, electronSpins)[:,:,None]

        v_ij = jnp.concatenate([cusplessDisps, matchMatrix], axis=-1)
        
        selfTerm = decays * self.linearSelf2(nn.swish(self.linearSelf1(v_ij)))

        """
        # Numpy reference implementation
        test_G = np.zeros((N,3))
        for l in range(N):
            for i in range(N):
                if i == l:
                    continue;
                localDecay = decays[i,l,0]
                localDisp = disps[i,l,:]
                localVij = v_ij[i,l,:]
                localCorrelator = self.linearThree2(nn.swish(self.linearThree1(localVij)))[0]
                test_G[l] += localDecay * localCorrelator * localDisp / N

        test_U3 = 0.0
        for l in range(N):
            test_U3 += test_G[l] @ test_G[l]
        """

        def compute_G_l(l):
            # map over i
            def body(i):
                localDecay = decays[i, l, 0]
                localDisp = disps[i, l, :]          # (3,)
                localVij = v_ij[i, l, :]            # (F,)
                hidden = nn.swish(self.linearThree1(localVij))
                localCorrelator = self.linearThree2(hidden)[0]   # scalar
                return localDecay * localCorrelator * localDisp  # (3,)
    
            G_l = jax.vmap(body)(jnp.arange(N))    # (N, 3)
            return jnp.average(G_l, axis=0)            # (3,)
    
        G = jax.vmap(compute_G_l)(jnp.arange(N))   # (N, 3)
        U3 = jnp.sum(jnp.sum(G * G, axis=-1))  # sum of squared norms
        
        MBJastrow = jnp.sum(selfTerm) + U3
        
        return slaterUp + slaterDown + CYJastrow + MBJastrow

class LogThreeDotJastrowRHF(Wavefunction):
    """
    Slater-Jastrow wavefunction with following specs:
    - Slater: RHF ground state
    - Jastrow: Coulomb-Yukawa + neural dot product version of Ceperley's correlator
    """
    spins : (int,int)
    L : float
    hiddenFeatures : int

    def setup(self):
        
        self.slaterUp = LogSimpleSlater(self.spins[0], self.L)
        self.slaterDown = LogSimpleSlater(self.spins[1], self.L)
        self.CYJastrow = LogCYJastrow(self.spins, self.L)
        
        self.decay = DecayFunction(self.L)

        self.queryMatrix = nn.Dense(self.hiddenFeatures)
        self.keyMatrix = nn.Dense(self.hiddenFeatures)
        self.value1 = nn.Dense(self.hiddenFeatures)
        self.value2 = nn.Dense(1)
        
        self.linearSelf1 = nn.Dense(self.hiddenFeatures)
        self.linearSelf2 = nn.Dense(1)

    def __call__(self, rs):
        
        slaterUp = self.slaterUp(rs[:self.spins[0],:])
        slaterDown = self.slaterDown(rs[self.spins[0]:,:])
        CYJastrow = self.CYJastrow(rs)
        
        disps = rs[:,None,:] - rs[None,:,:]  # (N, N, 3)
        disps = (disps + self.L/2) % self.L - self.L/2
        mask = ~jnp.eye(disps.shape[0], dtype=bool)[:,:,None]
        disps = jnp.where(mask, disps, 0.0)
        r_ij = jnp.linalg.norm(disps, axis=-1)
        decays = self.decay(r_ij)[:,:,None]
        decays = jnp.where(mask, decays, 0.0)
        
        cusplessDisps = jnp.cos(jnp.pi * disps / self.L)

        N = self.spins[0] + self.spins[1]
        electronIdxs = jnp.arange(N)
        electronSpins = jnp.where(electronIdxs < self.spins[0], 1, -1)
        matchMatrix = jnp.outer(electronSpins, electronSpins)[:,:,None]

        v_ij = jnp.concatenate([cusplessDisps, matchMatrix], axis=-1)
        
        selfTerm = jnp.sum(decays * self.linearSelf2(nn.swish(self.linearSelf1(v_ij))))

        queries = self.queryMatrix(v_ij)
        keys = self.keyMatrix(v_ij)
        values = self.value2(nn.swish(self.value1(v_ij)))

        print(decays.shape, queries.shape, keys.shape, values.shape)

        U3 = jnp.einsum(
            'ij,ik,ijd,ikd,ij,ik->',
            decays[..., 0],   # (N, N)
            decays[..., 0],   # (N, N)
            queries,          # (N, N, D)
            keys,             # (N, N, D)
            values[..., 0],   # (N, N)
            values[..., 0]    # (N, N)
        ) / N

        """
        # Numpy reference immplementation
        testU3 = 0
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    localDecay = decays[i,j,0] * decays[i,k,0]
                    localQuery = queries[i,j,:]
                    localKey = keys[i,k,:]
                    localValue = values[i,j,0] * values[i,k,0]
                    testU3 += localDecay * jnp.dot(localQuery, localKey) * localValue
        """
        
        MBJastrow = selfTerm + U3
        
        return slaterUp + slaterDown + CYJastrow + MBJastrow

class LogFewBodyJastrowRHF(Wavefunction):
    """
    Slater-Jastrow wavefunction with following specs:
    - Slater: RHF ground state
    - Jastrow: Coulomb-Yukawa + neural Jastrow with only one round of message
               passing
    """
    spins : (int,int)
    L : float
    hiddenFeatures : int

    def setup(self):
        
        self.slaterUp = LogSimpleSlater(self.spins[0], self.L)
        self.slaterDown = LogSimpleSlater(self.spins[1], self.L)
        self.CYJastrow = LogCYJastrow(self.spins, self.L)
        
        self.decay = DecayFunction(self.L)
        
        self.linearSelf1 = nn.Dense(self.hiddenFeatures)
        self.linearSelf2 = nn.Dense(1)
        self.linearCross1 = nn.Dense(self.hiddenFeatures)
        self.linearCross2 = nn.Dense(1)

    def __call__(self, rs):
        
        slaterUp = self.slaterUp(rs[:self.spins[0],:])
        slaterDown = self.slaterDown(rs[self.spins[0]:,:])
        CYJastrow = self.CYJastrow(rs)
        
        disps = rs[:,None,:] - rs[None,:,:]  # (N, N, 3)
        disps = (disps + self.L/2) % self.L - self.L/2
        mask = ~jnp.eye(disps.shape[0], dtype=bool)[:,:,None]
        disps = jnp.where(mask, disps, 0.0)
        r_ij = jnp.linalg.norm(disps, axis=-1)
        decays = self.decay(r_ij)[:,:,None]
        decays = jnp.where(mask, decays, 0.0)
        
        cusplessDisps = jnp.cos(jnp.pi * disps / self.L)

        N = self.spins[0] + self.spins[1]
        electronIdxs = jnp.arange(N)
        electronSpins = jnp.where(electronIdxs < self.spins[0], 1, -1)
        matchMatrix = jnp.outer(electronSpins, electronSpins)[:,:,None]

        v_ij = jnp.concatenate([cusplessDisps, matchMatrix], axis=-1)
        
        selfTerm = decays * self.linearSelf2(nn.swish(self.linearSelf1(v_ij)))

        """
        # Numpy reference implementation
        test = np.zeros((N,N,1))
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    localDecay = decays[i,j,0] * decays[i,k,0] * decays[j,k,0]
                    localInput = np.concatenate([v_ij[i,j,:], v_ij[i,k,:]], axis=-1)
                    localNeural = self.linearCross2(nn.swish(self.linearCross1(localInput)))[0]
                    test[i,j,0] += localDecay * localNeural
        """

        def f(i, j):
            d_ij = decays[i, j, 0]                         # scalar
            d_ik = decays[i, :, 0]                         # (N,)
            d_jk = decays[j, :, 0]                         # (N,)
            localDecay = d_ij * d_ik * d_jk                # (N,)
    
            localInput = jnp.concatenate(
                [jnp.repeat(v_ij[i, j, :][None, :], N, axis=0),  # (N, 4)
                 v_ij[i, :, :]],                                 # (N, 4)
                axis=-1                                          # (N, 8)
            )
    
            hidden = nn.swish(self.linearCross1(localInput))         # (N, H)
            localNeural = self.linearCross2(hidden)[:, 0]            # (N,)
            return jnp.average(localDecay * localNeural)

        f_vmap = jax.vmap(jax.vmap(f, in_axes=(None, 0)), in_axes=(0, None))
        crossTerm = f_vmap(jnp.arange(N), jnp.arange(N))[..., None]

        h_ij = selfTerm + crossTerm
        MBJastrow = jnp.sum(h_ij)
        
        return slaterUp + slaterDown + CYJastrow + MBJastrow

class LogManyBodyJastrowRHF(Wavefunction):
    """
    Slater-Jastrow wavefunction with following specs:
    - Slater: RHF ground state
    - Jastrow: Coulomb-Yukawa + neural many body Jastrow
    """
    spins : (int,int)
    L : float
    hiddenLayers : int
    hiddenFeatures : int

    def setup(self):
        
        self.slaterUp = LogSimpleSlater(self.spins[0], self.L)
        self.slaterDown = LogSimpleSlater(self.spins[1], self.L)
        self.CYJastrow = LogCYJastrow(self.spins, self.L)
        
        self.decay = DecayFunction(self.L)
        
        self.linear11 = nn.Dense(self.hiddenFeatures)
        self.linear12 = nn.Dense(self.hiddenFeatures)
        self.linear21 = nn.Dense(self.hiddenFeatures)
        self.linear22 = nn.Dense(self.hiddenFeatures)
        self.linear31 = nn.Dense(self.hiddenFeatures)
        self.linear32 = nn.Dense(self.hiddenFeatures)

        self.jastrowLinear1 = nn.Dense(self.hiddenFeatures)
        self.jastrowLinear2 = nn.Dense(1)

    def __call__(self, rs):
        
        slaterUp = self.slaterUp(rs[:self.spins[0],:])
        slaterDown = self.slaterDown(rs[self.spins[0]:,:])
        CYJastrow = self.CYJastrow(rs)
        
        disps = rs[:,None,:] - rs[None,:,:]  # (N, N, 3)
        disps = (disps + self.L/2) % self.L - self.L/2
        mask = ~jnp.eye(disps.shape[0], dtype=bool)[:,:,None]
        disps = jnp.where(mask, disps, 0.0)
        r_ij = jnp.linalg.norm(disps, axis=-1)
        decays = self.decay(r_ij)[:,:,None]
        decays = jnp.where(mask, decays, 0.0)
        
        cusplessDisps = jnp.cos(jnp.pi * disps / self.L)

        N = self.spins[0] + self.spins[1]
        electronIdxs = jnp.arange(N)
        electronSpins = jnp.where(electronIdxs < self.spins[0], 1, -1)
        matchMatrix = jnp.outer(electronSpins, electronSpins)[:,:,None]

        v_ij = jnp.concatenate([cusplessDisps, matchMatrix], axis=-1)

        selfTerm = self.linear12(nn.swish(self.linear11(v_ij)))

        def col_term_single_row(v_row, decay_row):
            """
            v_row: (N,4) -> v_ij[i,:]
            decay_row: (N,1) -> decays[i,:]
            returns: (N,8)
            """
            def f(j_idx):
                v_j = v_row[j_idx]
                v_k = v_row
                v_jk = jnp.concatenate([jnp.broadcast_to(v_j, (N,4)), v_k], axis=-1)
                return jnp.sum(decay_row * self.linear22(nn.swish(self.linear21(v_jk))), axis=0)
            return jax.vmap(f)(jnp.arange(N))  # (N,8)
        
        colTerm = jax.vmap(col_term_single_row, in_axes=(0,0))(v_ij, decays)  # (N,N,8)
        """

        col_vs = jnp.concatenate(
            [
                jnp.broadcast_to(v_ij[:,:,None,:], (N,N,N,4)),
                jnp.broadcast_to(v_ij[:,None,:,:], (N,N,N,4))
            ], axis=-1
        )
        colTerm = jnp.sum(
            decays[:,None,:,:] * self.linear22(nn.swish(self.linear21(col_vs))),
            axis=2
        )
        """

        """
        (i,j) = (3,8)

        myvij = jnp.broadcast_to(v_ij[i,j][None,:], (N,4))
        myviks = v_ij[i]
        myinputs = jnp.concatenate([myvij, myviks], axis=-1)
        mydecays = decays[i]
        myoutput = self.linear22(nn.swish(self.linear21(my     inputs)))
        myresult = jnp.sum(mydecays * myoutput, axis=0)

        print(myresult - colTermOld[i,j])

        print(selfTerm.shape, colTermOld.shape)
        """

        h_ij = selfTerm + colTerm
        U_ij = self.jastrowLinear2(nn.swish(self.jastrowLinear1(h_ij)))
        MBJastrow = jnp.nansum(decays * U_ij)
        
        return slaterUp + slaterDown + CYJastrow + MBJastrow
