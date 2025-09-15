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
    - Jastrow: Coulomb-Yukawa + neural function acting on pairwise information
        NOTE: This basically comes up with the "best" two-body Jastrow.
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
        MBJastrow = jnp.sum(selfTerm)
        
        return slaterUp + slaterDown + CYJastrow + MBJastrow

class LogThreeCeperleyJastrowRHF(Wavefunction):
    """
    Slater-Jastrow wavefunction with following specs:
    - Slater: RHF ground state
    - Jastrow: Coulomb-Yukawa + neural two-body + Ceperley's three-body correlator
        NOTE: This is slightly better than Ceperley's correlator since the eta
              function is neural and uses spin information.
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
            def body(i):
                localDecay = decays[i,l,0]
                localDisp = disps[i,l,:]
                localVij = v_ij[i,l,:]
                localCorrelator = self.linearThree2(
                    nn.swish(self.linearThree1(localVij))
                )[0]
                return localDecay * localCorrelator * localDisp
            G_l = jax.vmap(body)(jnp.arange(N))
            return jnp.average(G_l, axis=0)
    
        G = jax.vmap(compute_G_l)(jnp.arange(N))
        U3 = jnp.sum(G * G)
        
        MBJastrow = jnp.sum(selfTerm) + U3
        
        return slaterUp + slaterDown + CYJastrow + MBJastrow

class LogThreePavanJastrowRHF(Wavefunction):
    """
    Slater-Jastrow wavefunction with following specs:
    - Slater: RHF ground state
    - Jastrow: Coulomb-Yukawa + neural two-body + my three-body correlator
        NOTE: This tries to generalize Ceperley's three-body correlator
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

        self.queryMatrix = nn.Dense(self.hiddenFeatures)
        self.etaQuery1 = nn.Dense(self.hiddenFeatures)
        self.etaQuery2 = nn.Dense(1)
        
        self.keyMatrix = nn.Dense(self.hiddenFeatures)
        self.etaKey1 = nn.Dense(self.hiddenFeatures)
        self.etaKey2 = nn.Dense(1)

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

        """
        # Numpy reference implementation
        test_K = np.zeros((N,self.hiddenFeatures))
        test_Q = np.zeros((N,self.hiddenFeatures))
        
        for l in range(N):
        
            for i in range(N):
                
                localDecay = decays[i,l,0]
                localDisp = disps[i,l,:]
                localVij = v_ij[i,l,:]

                localKeyCorrelator = self.etaKey2(
                    nn.swish(self.etaKey1(localVij))
                )[0]
                localKeyDisp = self.keyMatrix(localDisp)
                localKey = localDecay * localKeyCorrelator * localKeyDisp

                localQueryCorrelator = self.etaQuery2(
                    nn.swish(self.etaQuery1(localVij))
                )[0]
                localQueryDisp = self.queryMatrix(localDisp)
                localQuery = localDecay * localQueryCorrelator * localQueryDisp

                test_K[l,:] += localKey
                test_Q[l,:] += localQuery

        test_U3 = 0.0
        for l in range(N):
            test_U3 += test_K[l] @ test_Q[l]
        """
        
        def compute_K(l):
            def key(i):
                localDecay = decays[i,l,0]
                localDisp = self.keyMatrix(disps[i,l,:])
                localVij = self.etaKey2(
                    nn.swish(self.etaKey1(v_ij[i,l,:]))
                )[0]
                return localDecay * localVij * localDisp
            K_l = jax.vmap(key)(jnp.arange(N))
            return jnp.sum(K_l, axis=0)

        def compute_Q(l):
            def query(i):
                localDecay = decays[i,l,0]
                localDisp = self.queryMatrix(disps[i,l,:])
                localVij = self.etaQuery2(
                    nn.swish(self.etaQuery1(v_ij[i,l,:]))
                )[0]
                return localDecay * localVij * localDisp
            Q_l = jax.vmap(query)(jnp.arange(N))
            return jnp.sum(Q_l, axis=0)

        K = jax.vmap(compute_K)(jnp.arange(N))
        Q = jax.vmap(compute_Q)(jnp.arange(N))
        U3 = jnp.sum(K * Q)
        
        MBJastrow = jnp.sum(selfTerm) + U3
        
        return slaterUp + slaterDown + CYJastrow + MBJastrow
        
