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

class LogTwoBodySJ(Wavefunction):
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
        
        self.linearSelf1 = nn.Dense(self.hiddenFeatures)
        self.linearSelf2 = nn.Dense(1)

    def __call__(self, rs):
        
        slaterUp = self.slaterUp(rs[:self.spins[0],:])
        slaterDown = self.slaterDown(rs[self.spins[0]:,:])
        CYJastrow = self.CYJastrow(rs)
        
        disps = rs[:,None,:] - rs[None,:,:]  # (N, N, 3)
        mask = ~jnp.eye(disps.shape[0], dtype=bool)[:,:,None]
        disps = jnp.where(mask, disps, 0.0)

        cosDisps = jnp.cos(2 * jnp.pi * disps / self.L)
        sinDisps = jnp.sin(2 * jnp.pi * disps / self.L)
        sinDispsMag = jnp.linalg.norm(
            jnp.sin(jnp.pi * disps / self.L),
            axis=-1, keepdims=True
        )
        sinDispsMag = jnp.where(mask, sinDispsMag, 0.0)
        
        N = self.spins[0] + self.spins[1]
        electronIdxs = jnp.arange(N)
        electronSpins = jnp.where(electronIdxs < self.spins[0], 1, -1)
        matchMatrix = jnp.outer(electronSpins, electronSpins)[:,:,None]
        
        v_ij = jnp.concatenate(
            [cosDisps, sinDisps, sinDispsMag, matchMatrix],
            axis=-1
        )
        
        selfTerm = self.linearSelf2(nn.swish(self.linearSelf1(v_ij)))
        neuralJastrow = jnp.average(selfTerm)
        
        return slaterUp + slaterDown + CYJastrow + neuralJastrow

class LogTwoBodySJB(Wavefunction):
    """
    Slater-Jastrow wavefunction with following specs:
    - Slater: RHF ground state
    - Jastrow: Coulomb-Yukawa + neural function acting on pairwise information
        NOTE: This basically comes up with the "best" two-body Jastrow.
    - Backflow: acts purely on pairwise information to produce arbitrary
                backflow coordinates
    """
    spins : (int,int)
    L : float
    hiddenFeatures : int

    def setup(self):
        
        self.slaterUp = LogSimpleSlater(self.spins[0], self.L)
        self.slaterDown = LogSimpleSlater(self.spins[1], self.L)
        self.CYJastrow = LogCYJastrow(self.spins, self.L)
        
        self.neuralJastrow1 = nn.Dense(self.hiddenFeatures)
        self.neuralJastrow2 = nn.Dense(1)
        
        self.neuralBackflow1 = nn.Dense(self.hiddenFeatures)
        self.neuralBackflow2 = nn.Dense(3)

    def __call__(self, rs):
        
        CYJastrow = self.CYJastrow(rs)
        
        disps = rs[:,None,:] - rs[None,:,:]  # (N, N, 3)
        mask = ~jnp.eye(disps.shape[0], dtype=bool)[:,:,None]
        disps = jnp.where(mask, disps, 0.0)

        cosDisps = jnp.cos(2 * jnp.pi * disps / self.L)
        sinDisps = jnp.sin(2 * jnp.pi * disps / self.L)
        sinDispsMag = jnp.linalg.norm(
            jnp.sin(jnp.pi * disps / self.L),
            axis=-1, keepdims=True
        )
        sinDispsMag = jnp.where(mask, sinDispsMag, 0.0)
        
        N = self.spins[0] + self.spins[1]
        electronIdxs = jnp.arange(N)
        electronSpins = jnp.where(electronIdxs < self.spins[0], 1, -1)
        matchMatrix = jnp.outer(electronSpins, electronSpins)[:,:,None]
        
        v_ij = jnp.concatenate(
            [cosDisps, sinDisps, sinDispsMag, matchMatrix],
            axis=-1
        )
        
        selfTerm = self.neuralJastrow2(nn.swish(self.neuralJastrow1(v_ij)))
        neuralJastrow = jnp.average(selfTerm)

        backflow = jnp.average(
            self.neuralBackflow2(nn.swish(self.neuralBackflow1(v_ij))),
            axis=1
        )
        xs = rs + backflow
        
        slaterUp = self.slaterUp(xs[:self.spins[0],:])
        slaterDown = self.slaterDown(xs[self.spins[0]:,:])
        
        return slaterUp + slaterDown + CYJastrow + neuralJastrow

class LogMessagePassingSJB(Wavefunction):
    """
    Slater-Jastrow wavefunction with following specs:
    - Slater: RHF ground state
    - Jastrow: Coulomb-Yukawa + neural message passing Jastrow
    - Backflow: neural message passing backflow

    """
    spins : (int,int)
    L : float
    T : int
    hiddenFeatures : int
    d1 : int
    d2 : int
    dv : int

    def setup(self):
        
        self.slaterUp = LogSimpleSlater(self.spins[0], self.L)
        self.slaterDown = LogSimpleSlater(self.spins[1], self.L)
        self.CYJastrow = LogCYJastrow(self.spins, self.L)
        
        self.neuralJastrow1 = nn.Dense(self.hiddenFeatures)
        self.neuralJastrow2 = nn.Dense(1)
        
        self.neuralBackflow1 = nn.Dense(self.hiddenFeatures)
        self.neuralBackflow2 = nn.Dense(3)

        self.hi0 = self.param(
            "hi0",
            lambda rng : jnp.zeros(self.d1)
        )
        self.hij0 = self.param(
            "hij0",
            lambda rng : jnp.zeros(self.d2)
        )

        self.Wqt = [
            nn.Dense(self.d2 + self.dv, use_bias=False) for _ in range(self.T)
        ]
        self.Wkt = [
            nn.Dense(self.d2 + self.dv, use_bias=False) for _ in range(self.T)
        ]

        self.Fmt = [
            (
                nn.Dense(self.hiddenFeatures),
                nn.Dense(self.d2 + self.dv)
            ) for _ in range(self.T)
        ]

        self.Alineart = [
            nn.Dense(self.d2 + self.dv) for _ in range(self.T)
        ]

        self.F1t = [
            (
                nn.Dense(self.hiddenFeatures),
                nn.Dense(self.d1)
            ) for _ in range(self.T)
        ]
        self.F2t = [
            (
                nn.Dense(self.hiddenFeatures),
                nn.Dense(self.d2)
            ) for _ in range(self.T)
        ]

    def __call__(self, rs):
        
        CYJastrow = self.CYJastrow(rs)
        
        disps = rs[:,None,:] - rs[None,:,:]  # (N, N, 3)
        mask = ~jnp.eye(disps.shape[0], dtype=bool)[:,:,None]
        disps = jnp.where(mask, disps, 0.0)

        cosDisps = jnp.cos(2 * jnp.pi * disps / self.L)
        sinDisps = jnp.sin(2 * jnp.pi * disps / self.L)
        sinDispsMag = jnp.linalg.norm(
            jnp.sin(jnp.pi * disps / self.L),
            axis=-1, keepdims=True
        )
        sinDispsMag = jnp.where(mask, sinDispsMag, 0.0)
        
        N = self.spins[0] + self.spins[1]
        electronIdxs = jnp.arange(N)
        electronSpins = jnp.where(electronIdxs < self.spins[0], 1, -1)
        matchMatrix = jnp.outer(electronSpins, electronSpins)[:,:,None]
        
        v_ij = jnp.concatenate(
            [cosDisps, sinDisps, sinDispsMag, matchMatrix],
            axis=-1
        )

        #hit = jnp.broadcast_to(self.hi0, (N,self.d1))
        hijt = jnp.broadcast_to(self.hij0, (N,N,self.d2))

        for t in range(self.T):

            #git = hit
            gijt = jnp.concatenate([hijt,v_ij], axis=-1)

            qijt = self.Wqt[t](gijt)
            kijt = self.Wkt[t](gijt)
            
            """
            # Numpy reference implementation
            np_Aijt = np.zeros((N,N,self.d2+self.dv))
            for i in range(N):
                for j in range(N):
                    for l in range(N):
                        np_Aijt[i,j,:] += qijt[i,l,:] * kijt[l,j,:] # / jnp.sqrt(N)
            """

            Aijt = self.Alineart[t](
                nn.swish(
                    jnp.einsum("ild,ljd->ijd", qijt, kijt) / jnp.sqrt(N)
                )
            )

            mijt = Aijt * self.Fmt[t][1](nn.swish(self.Fmt[t][0](gijt)))
            #acc_mijt = jnp.average(mijt, axis=1)
            
            #hit += self.F1t[t][1](nn.swish(self.F1t[t][0](
            #    jnp.concatenate([acc_mijt,git], axis=-1)
            #)))
            hijt += self.F2t[t][1](nn.swish(self.F2t[t][0](
                jnp.concatenate([mijt, gijt], axis=-1)
            )))

        #git = hit
        gijt = jnp.concatenate([hijt,v_ij], axis=-1)

        """
        MPJastrow = jnp.average(self.FJt[1](nn.swish(self.FJt[0](git))))
        MPBackflow = self.FBFt(git)
        """
        
        selfTerm = self.neuralJastrow2(nn.swish(self.neuralJastrow1(gijt)))
        neuralJastrow = jnp.average(selfTerm)

        backflow = jnp.average(
            self.neuralBackflow2(nn.swish(self.neuralBackflow1(gijt))),
            axis=1
        )
        xs = rs + backflow
        
        slaterUp = self.slaterUp(xs[:self.spins[0],:])
        slaterDown = self.slaterDown(xs[self.spins[0]:,:])
        
        return slaterUp + slaterDown + CYJastrow + neuralJastrow

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

class LogFixedCYJastrow(Wavefunction):
    """
    Creates a log-wavefunction that is a Coulomb-Yukawa Jastrow term. The same
    and different spin electrons are handled by different $A$ parameters.

    The conventional $F$ parameters are set by the cusp conditions.

    IMPORTANT: These $A$ parameters are *fixed* in this class! 

    NOTE: In this implementation, we always take the absolute values of the $A$
    parameters. This is so that even if negative values are encountered during
    optimization, the resulting wavefunction remains physical.
    """
    spins : (int,int)
    L : float

    def setup(self):

        N = self.spins[0] + self.spins[1]
        n = N / (self.L**3.)
        self.A = 1.0 / jnp.sqrt(4 * jnp.pi * n)

    def __call__(self, rs):

        A_same = self.A
        A_diff = self.A
        
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

class LogSlaterFixedCYJastrow(Wavefunction):
    """
    Creates a log-wavefunction that is the product of two simple Slater
    determinant with the lowest k-points filled and a Coulomb-Yukawa Jastrow.

    Since the CYJastrow's parameters are fixed, there are no variational
    parameters in this wavefunction class.
    """
    spins : (int,int)
    L : float

    def setup(self):
        self.slaterUp = LogSimpleSlater(self.spins[0], self.L)
        self.slaterDown = LogSimpleSlater(self.spins[1], self.L)
        self.CYJastrow = LogFixedCYJastrow(self.spins, self.L)

    def __call__(self, rs):
        slaterUp = self.slaterUp(rs[:self.spins[0],:])
        slaterDown = self.slaterDown(rs[self.spins[0]:,:])
        CYJastrow = self.CYJastrow(rs)
        return slaterUp + slaterDown + CYJastrow

class LogWignerCrystalSlater(Wavefunction):
    """
    Slater determinant reference for a 3D Wigner crystal with localized
    Gaussian orbitals on a BCC-like lattice, using a finite 3x3x3 image sum.

    Spin-up electrons  -> simple cubic sites
    Spin-down electrons -> BCC-shifted sites

    Orbital (for each center R_j):
        φ_j(r_i) = sum_{n in {-1,0,1}^3} exp( -α * | r_i - (R_j + n L) |^2 )

    We clip α to keep Gaussians localized enough that contributions at the
    edge of the 3×3×3 block are negligible; this avoids weird kinetic-energy
    artifacts from very broad orbitals at small r_s.

    - The ansatz is *not* meant to be good at small r_s (delocalized phase),
      but should be a decent WC reference at large r_s.
    """

    spins: tuple          # (N_up, N_dn)
    L: float              # box length

    def setup(self):
        N_up, N_dn = self.spins
    
        # Gaussian width parameter
        self.log_alpha = self.param(
            "log_alpha", lambda rng: jnp.array(-2.0)
        )

        # Lattice centers
        self.centers_up, self.centers_dn = self._create_centers(N_up, N_dn, self.L)
    
        # Precompute 3×3×3 image shifts as a NON-trainable variable
        offsets_1d = jnp.array([-1.0, 0.0, 1.0])
        off_grid = jnp.stack(
            jnp.meshgrid(offsets_1d, offsets_1d, offsets_1d, indexing="ij"),
            axis=-1
        )
        shifts = off_grid.reshape(-1, 3) * self.L  # shape (27,3)
    
        # Store as a non-trainable Flax variable
        self.image_shifts = self.variable(
            "constants", "image_shifts", lambda: shifts
        )

    # ---------------------------------------------------------------------
    # Lattice centers: SC for spin-up, BCC-shifted SC for spin-down
    # ---------------------------------------------------------------------
    @staticmethod
    def _create_centers(N_up, N_dn, L):
        """
        Construct cubic sites (for spin-up) and BCC-shifted cubic sites
        (for spin-down). Combined, these form a BCC-like pattern.
        """
        # Smallest n such that n^3 ≥ max(N_up, N_dn)
        n = 1
        maxN = max(N_up, N_dn)
        while n**3 < maxN:
            n += 1

        a = L / n  # SC lattice spacing
        # Equally spaced grid points in [0, L)
        coords = jnp.linspace(0.0, L - a, n)
        grid = jnp.stack(jnp.meshgrid(coords, coords, coords, indexing="ij"), axis=-1)
        grid = grid.reshape(-1, 3)  # (n^3, 3)

        centers_up = grid[:N_up]

        # BCC shift by (a/2, a/2, a/2)
        shift = jnp.array([a/2, a/2, a/2])
        centers_dn = (grid + shift)[:N_dn]

        return centers_up, centers_dn

    # ---------------------------------------------------------------------
    # Orbital building blocks
    # ---------------------------------------------------------------------
    def _orbital_single_center(self, ri, Rj, alpha):
        """
        Gaussian orbital centered at Rj with 3x3x3 image sum:

            φ_j(r_i) = Σ_{n∈{-1,0,1}^3} exp(-α |r_i - (Rj + nL)|^2)

        ri: (3,)
        Rj: (3,)
        image_shifts: (27, 3)
        """
        # (27, 3): positions of the center in neighboring cells
        centers_images = Rj + self.image_shifts.value  # broadcasting over (27, 3)

        # (27, 3): displacements from all images
        dr = ri - centers_images

        # (27,): squared distances
        r2 = jnp.sum(dr * dr, axis=-1)

        # Sum over images
        dr0 = self.image_shifts.value
        r20 = jnp.sum(dr0 * dr0, axis=-1)
        peak = jnp.sum(jnp.exp(-alpha * r20))
        return jnp.sum(jnp.exp(-alpha * r2)) / peak

    def _orbital_row(self, ri, centers, alpha):
        """
        All orbitals φ_j(r_i) for a single electron position ri.

        ri: (3,)
        centers: (N_spin, 3)
        returns: (N_spin,)
        """
        single = lambda Rj: self._orbital_single_center(ri, Rj, alpha)
        return jax.vmap(single)(centers)

    def _slater_logdet(self, rs_spin, centers, alpha):
        """
        Build and take logdet of the Slater matrix Φ_{ij} = φ_j(r_i).

        rs_spin: (N_spin, 3)
        centers: (N_spin, 3)
        """
        # Φ shape: (N_spin, N_spin)
        Phi = jax.vmap(lambda ri: self._orbital_row(ri, centers, alpha))(rs_spin)

        sign, logdet = jnp.linalg.slogdet(Phi)
        # For WC we can ignore the overall sign and just use |det|
        return logdet

    # ---------------------------------------------------------------------
    # Public call: log |det_up * det_dn|
    # ---------------------------------------------------------------------
    def __call__(self, rs):
        """
        rs shape = (N_up + N_dn, 3)
        Returns log |det_up * det_dn|.
        """
        N_up, N_dn = self.spins
        rs = rs % self.L # move everything into [0,L]
        rs_up = rs[:N_up]
        rs_dn = rs[N_up:N_up + N_dn]

        # Exponentiate and clip alpha to keep Gaussians localized
        alpha = jnp.exp(self.log_alpha)
        alpha_min = 55 / self.L**2
        alpha_max = 300 / self.L**2
        alpha = jnp.clip(alpha, alpha_min, alpha_max)

        log_up = self._slater_logdet(rs_up, self.centers_up, alpha)
        log_dn = self._slater_logdet(rs_dn, self.centers_dn, alpha)

        return log_up + log_dn

class LogWignerCrystalSlaterFixedCYJastrow(Wavefunction):
    """
    Creates a log-wavefunction that is the product of two simple Slater
    determinant with the Wigner crystal as the reference state.

    Since the CYJastrow's parameters are fixed, there are no variational
    parameters in this wavefunction class.
    """
    spins : (int,int)
    L : float

    def setup(self):
        self.wignerSlater = LogWignerCrystalSlater(self.spins, self.L)
        self.CYJastrow = LogFixedCYJastrow(self.spins, self.L)

    def __call__(self, rs):
        wignerSlater = self.wignerSlater(rs)
        CYJastrow = self.CYJastrow(rs)
        return wignerSlater + CYJastrow

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
