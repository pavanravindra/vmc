import jax
import jax.numpy as jnp
from jax.nn import initializers

from flax import linen as nn
from flax.serialization import to_bytes, from_bytes
from flax.core import FrozenDict

from pyscf import gto, scf, ao2mo
import qc

import numpy as np

import itertools

def computeLattice(N, r_ws, dim, basis_matrix=None):
    """
    Computes the scaled lattice matrix for a given number of particles and
    Wigner-Seitz radius.

    Args:
        N: Number of particles.
        r_ws: Wigner-Seitz radius (r_s).
        dim: 2 or 3.
        basis_matrix: (dim, dim) Optional template lattice. 
                      If None, defaults to the identity (cubic/square).
                      The output will be this matrix scaled to match the 
                      target density.
    """
    if dim == 2:
        target_vol = jnp.pi * (r_ws ** 2) * N
    elif dim == 3:
        target_vol = (4.0 / 3.0) * jnp.pi * (r_ws ** 3) * N
    else:
        raise ValueError("Only dim=2 or dim=3 are supported.")

    if basis_matrix is None:
        basis_matrix = jnp.eye(dim)
        template_vol = 1.0
    else:
        basis_matrix = jnp.array(basis_matrix)
        template_vol = jnp.abs(jnp.linalg.det(basis_matrix))

    scale = (target_vol / template_vol) ** (1.0 / dim)

    return basis_matrix * scale


def computeRws(N, lattice, dim):
    """
    Computes the effective Wigner-Seitz radius for a given lattice and 
    particle count.

    Args:
        N: Number of particles.
        lattice: (dim, dim) array defining the simulation cell.
        dim: 2 or 3.
    """
    vol = jnp.abs(jnp.linalg.det(lattice))
    
    if dim == 2:
        return jnp.sqrt(vol / (jnp.pi * N))
    elif dim == 3:
        return (vol / (N * (4.0 / 3.0) * jnp.pi)) ** (1.0 / 3.0)
    else:
        raise ValueError("Only dim=2 or dim=3 are supported.")

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

def genKpoints(N, lattice, dim):
    """
    Generates N k-points sorted by energy, ensuring +/- k pairs are adjacent.
    
    Sequence: [0, k1, -k1, k2, -k2, ...]
    
    Args:
        N: Number of k-points required.
        lattice: (dim, dim) numpy array (real space lattice vectors).
        dim: 2 or 3.
    """
    # Force lattice to concrete numpy for static integer generation
    lattice = np.array(lattice)
    
    if not (dim == 2 or dim == 3):
        raise ValueError("Only dim=2 or dim=3 are supported.")
    
    # 1. Reciprocal Lattice
    rec_lattice = 2 * np.pi * np.linalg.inv(lattice).T
    rec_norms = np.linalg.norm(rec_lattice, axis=1)

    # 2. Estimate Integer Bounds (with safety factor for anisotropy)
    vol_k_cell = np.abs(np.linalg.det(rec_lattice))
    
    if dim == 2:
        k_radius = (N * vol_k_cell / np.pi)**0.5
    else:
        k_radius = (N * vol_k_cell * 3 / (4 * np.pi))**(1.0/3.0)
        
    # Safety factor of 1.5 + buffer to handle anisotropic boxes
    max_integers = np.ceil(1.5 * k_radius / rec_norms).astype(int) + 2
    
    # 3. Generate Integer Grid
    ranges = [range(-m, m+1) for m in max_integers]
    int_candidates = np.array(list(itertools.product(*ranges)))
    
    # 4. Sort Candidates by PHYSICAL Norm
    phys_k = int_candidates @ rec_lattice
    norms = np.linalg.norm(phys_k, axis=1)
    
    # Sort indices based on energy (norm)
    sorted_indices = np.argsort(norms)
    candidates_sorted = int_candidates[sorted_indices]
    
    # 5. Selection Loop with +/- Pairing
    selected_ints = []
    seen_hashes = set()
    
    for k_vec in candidates_sorted:
        if len(selected_ints) >= N:
            break
            
        # Use tuple for efficient set lookup
        k_tuple = tuple(k_vec)
        
        if k_tuple in seen_hashes:
            continue
            
        # Found a new lowest-energy vector. Add it.
        selected_ints.append(k_vec)
        seen_hashes.add(k_tuple)
        
        # If we still need points, immediately try to add -k
        if len(selected_ints) < N:
            neg_k_vec = -k_vec
            neg_k_tuple = tuple(neg_k_vec)
            
            # Add -k if it hasn't been added (and isn't 0)
            if neg_k_tuple not in seen_hashes:
                selected_ints.append(neg_k_vec)
                seen_hashes.add(neg_k_tuple)
                
    # 6. Final Conversion to Physical Vectors
    selected_ints = np.array(selected_ints)
    chosenKpoints = selected_ints @ rec_lattice
    
    return jnp.array(chosenKpoints)

class LogSimpleSlater(Wavefunction):
    """
    Creates a log-wavefunction that is just a simple Slater determinant of the
    input electron coordinates. The basis for the determinant is specified by
    the provided kpoints.
    
    NOTE: These determinants use the convention that different particle
    positions are in different rows. Columns correspond to plane wave orbitals.
    """
    N : int
    dim : int
    kpoints : jnp.ndarray

    def setup(self):
        
        if not (self.dim == 2 or self.dim == 3):
            raise Exception("Only dim=2 or dim=3 are supported.")
            
        weights = 10.0 ** jnp.arange(1, self.dim + 1)
        weighted_sums = jnp.dot(jnp.sign(self.kpoints), weights)
        k_signs = jnp.sign(weighted_sums + 1.0)
        self.cos_switch = (k_signs + 1.0) / 2.0
        self.sin_switch = (1.0 - k_signs) / 2.0

    def __call__(self, rs):
        def makeSimpleSlaterRow(ri):
            def localKpointFunction(k, c_switch, s_switch):
                dot_val = jnp.dot(k, ri)
                term = (c_switch * jnp.cos(dot_val) + 
                        s_switch * jnp.sin(-dot_val))
                return term
            return jax.vmap(localKpointFunction)(
                self.kpoints, 
                self.cos_switch, 
                self.sin_switch
            )
        slaterMatrix = jax.vmap(makeSimpleSlaterRow)(rs)
        return jnp.linalg.slogdet(slaterMatrix)[1]

def coulombYukawa(r_real, r_frac, A, F):
    """
    Coulomb-Yukawa Jastrow with anisotropic Decay.
    
    Args:
        r_real: The physical distance (for the Coulomb/Yukawa interaction).
        r_frac: The norm of the fractional distance vector. 
                This is used to make sure the Jastrow dies at the unit cell
                boundaries.
    """
    r_safe = jnp.where(r_real < 1e-12, 1.0, r_real) 
    cy = (A / r_safe) * (1 - jnp.exp(-r_safe / F))
    
    r_cut_frac = 0.5
    x = jnp.clip(r_frac / r_cut_frac, a_min=0.0, a_max=1.0 - 1e-5)
    decay = jnp.exp(1 - 1 / (1 - x**2))
    
    return cy * decay

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
    lattice: jnp.ndarray

    def setup(self):

        N = self.spins[0] + self.spins[1]
        volume = jnp.abs(jnp.linalg.det(self.lattice))
        n = N / volume

        self.rec_lattice = jnp.linalg.inv(self.lattice)

        self.As = self.param(
            "As_same_diff",
            lambda rng: jnp.full(2, 1.0 / jnp.sqrt(4 * jnp.pi * n))
        )

    def __call__(self, rs):

        N = rs.shape[0]

        A_same = jnp.abs(self.As[0])
        A_diff = jnp.abs(self.As[1])
        
        F_same = jnp.sqrt(2 * A_same)
        F_diff = jnp.sqrt(A_diff)

        disp_real_raw = rs[:,None,:] - rs[None,:,:]
        disp_frac = jnp.dot(disp_real_raw, self.rec_lattice)
        disp_frac = (disp_frac + 0.5) % 1.0 - 0.5
        disp_real_mic = jnp.dot(disp_frac, self.lattice)

        eye_mask = jnp.eye(N)[:,:,None]
        r_ij_real = jnp.linalg.norm(disp_real_mic + eye_mask, axis=-1)
        r_ij_frac = jnp.linalg.norm(disp_frac + eye_mask, axis=-1)
        
        n_up, n_down = self.spins
        spin_mask = jnp.concatenate([jnp.zeros(n_up, dtype=int), jnp.ones(n_down, dtype=int)])
        
        mask_same = spin_mask[:,None] == spin_mask[None,:]
        mask_diff = spin_mask[:,None] != spin_mask[None,:]
        eye = jnp.eye(N, dtype=bool)
        mask_same = mask_same & (~eye)
        
        val_same = coulombYukawa(r_ij_real, r_ij_frac, A_same, F_same)
        val_same = jnp.where(mask_same, val_same, 0.0)

        val_diff = coulombYukawa(r_ij_real, r_ij_frac, A_diff, F_diff)
        val_diff = jnp.where(mask_diff, val_diff, 0.0)

        return -0.5 * (jnp.sum(val_same) + jnp.sum(val_diff))

class LogFixedCYJastrow(Wavefunction):
    """
    Creates a log-wavefunction that is a Coulomb-Yukawa Jastrow term. The same
    and different spin electrons are handled by different $A$ parameters.
    In this instance, the $A$ parameters are *fixed* at initialization.

    The conventional $F$ parameters are set by the cusp conditions.

    NOTE: In this implementation, we always take the absolute values of the $A$
    parameters. This is so that even if negative values are encountered during
    optimization, the resulting wavefunction remains physical.
    """
    spins : (int,int)
    lattice: jnp.ndarray
    As : jnp.ndarray

    def setup(self):

        N = self.spins[0] + self.spins[1]
        volume = jnp.abs(jnp.linalg.det(self.lattice))
        n = N / volume

        self.rec_lattice = jnp.linalg.inv(self.lattice)

    def __call__(self, rs):

        N = rs.shape[0]

        A_same = jnp.abs(self.As[0])
        A_diff = jnp.abs(self.As[1])
        
        F_same = jnp.sqrt(2 * A_same)
        F_diff = jnp.sqrt(A_diff)

        disp_real_raw = rs[:,None,:] - rs[None,:,:]
        disp_frac = jnp.dot(disp_real_raw, self.rec_lattice)
        disp_frac = (disp_frac + 0.5) % 1.0 - 0.5
        disp_real_mic = jnp.dot(disp_frac, self.lattice)

        eye_mask = jnp.eye(N)[:,:,None]
        r_ij_real = jnp.linalg.norm(disp_real_mic + eye_mask, axis=-1)
        r_ij_frac = jnp.linalg.norm(disp_frac + eye_mask, axis=-1)
        
        n_up, n_down = self.spins
        spin_mask = jnp.concatenate([jnp.zeros(n_up, dtype=int), jnp.ones(n_down, dtype=int)])
        
        mask_same = spin_mask[:,None] == spin_mask[None,:]
        mask_diff = spin_mask[:,None] != spin_mask[None,:]
        eye = jnp.eye(N, dtype=bool)
        mask_same = mask_same & (~eye)
        
        val_same = coulombYukawa(r_ij_real, r_ij_frac, A_same, F_same)
        val_same = jnp.where(mask_same, val_same, 0.0)

        val_diff = coulombYukawa(r_ij_real, r_ij_frac, A_diff, F_diff)
        val_diff = jnp.where(mask_diff, val_diff, 0.0)

        return -0.5 * (jnp.sum(val_same) + jnp.sum(val_diff))

class LogSimpleSlaters(Wavefunction):
    """
    Creates a log-wavefunction that is the product of two simple Slater
    determinant with the lowest k-points filled and a Coulomb-Yukawa Jastrow.

    There are 2 variational parameters, both in the Jastrow.
    """
    spins : (int,int)
    dim : int
    kpoints : jnp.ndarray

    def setup(self):
        self.slaterUp = LogSimpleSlater(self.spins[0], self.dim, self.kpoints)
        self.slaterDown = LogSimpleSlater(self.spins[1], self.dim, self.kpoints)

    def __call__(self, rs):
        slaterUp = self.slaterUp(rs[:self.spins[0],:])
        slaterDown = self.slaterDown(rs[self.spins[0]:,:])
        return slaterUp + slaterDown

class LogSlaterCYJastrow(Wavefunction):
    """
    Creates a log-wavefunction that is the product of two simple Slater
    determinant with the lowest k-points filled and a Coulomb-Yukawa Jastrow.

    There are 2 variational parameters, both in the Jastrow.
    """
    spins : (int,int)
    dim : int
    lattice: jnp.ndarray
    kpoints : jnp.ndarray

    def setup(self):
        self.slaterUp = LogSimpleSlater(self.spins[0], self.dim, self.kpoints)
        self.slaterDown = LogSimpleSlater(self.spins[1], self.dim, self.kpoints)
        self.CYJastrow = LogCYJastrow(self.spins, self.lattice)

    def __call__(self, rs):
        slaterUp = self.slaterUp(rs[:self.spins[0],:])
        slaterDown = self.slaterDown(rs[self.spins[0]:,:])
        CYJastrow = self.CYJastrow(rs)
        return slaterUp + slaterDown + CYJastrow

class LogTwoBodySJ(Wavefunction):
    """
    Two-Body Slater-Jastrow wavefunction for arbitrary (skewed) periodic cells.
    
    Args:
        lattice: (D, D) matrix defining the lattice vectors (rows).
                 e.g. [[Lx, 0], [Tx, Ty]] for a 2D tilted cell.
        kpoints: (N, D) matrix with the occupied k-points for the Slater
                 determinant.
    """
    spins : (int,int)
    dim : int
    lattice : jnp.ndarray
    kpoints : jnp.ndarray
    hiddenFeatures : int

    def setup(self):

        N = self.spins[0] + self.spins[1]
        
        self.slaterUp = LogSimpleSlater(self.spins[0], self.dim, self.kpoints)
        self.slaterDown = LogSimpleSlater(self.spins[1], self.dim, self.kpoints)
        self.CYJastrow = LogCYJastrow(self.spins, self.lattice)
        
        self.linearSelf1 = nn.Dense(self.hiddenFeatures)
        self.linearSelf2 = nn.Dense(
            1, initializers.zeros, bias_init=initializers.zeros
        )

    def __call__(self, rs):
        
        slaterUp = self.slaterUp(rs[:self.spins[0],:])
        slaterDown = self.slaterDown(rs[self.spins[0]:,:])
        CYJastrow = self.CYJastrow(rs)
        
        disps = rs[:,None,:] - rs[None,:,:]  # (N, N, 3)
        mask = ~jnp.eye(disps.shape[0], dtype=bool)[:,:,None]
        disps = jnp.where(mask, disps, 0.0)

        recLattice = jnp.linalg.inv(self.lattice)
        fracDisps = disps @ recLattice

        cosDisps = jnp.cos(2 * jnp.pi * fracDisps)
        sinDisps = jnp.sin(2 * jnp.pi * fracDisps)
        sinDispsMag = jnp.linalg.norm(
            jnp.sin(jnp.pi * fracDisps), axis=-1, keepdims=True
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
    Two-Body Slater-Jastrow-Backflow wavefunction for arbitrary (skewed)
    periodic cells.
    
    Args:
        lattice: (D, D) matrix defining the lattice vectors (rows).
                 e.g. [[Lx, 0], [Tx, Ty]] for a 2D tilted cell.
        kpoints: (N, D) matrix with the occupied k-points for the Slater
                 determinant.
    """
    spins : (int,int)
    dim : int
    lattice : jnp.ndarray
    kpoints : jnp.ndarray
    hiddenFeatures : int

    def setup(self):

        N = self.spins[0] + self.spins[1]
        
        self.slaterUp = LogSimpleSlater(self.spins[0], self.dim, self.kpoints)
        self.slaterDown = LogSimpleSlater(self.spins[1], self.dim, self.kpoints)
        self.CYJastrow = LogCYJastrow(self.spins, self.lattice)
        
        self.neuralJastrow1 = nn.Dense(self.hiddenFeatures)
        self.neuralJastrow2 = nn.Dense(1)
        
        self.neuralBackflow1 = nn.Dense(self.hiddenFeatures)
        self.neuralBackflow2 = nn.Dense(self.dim)

    def __call__(self, rs):
        
        CYJastrow = self.CYJastrow(rs)
        
        disps = rs[:,None,:] - rs[None,:,:]  # (N, N, 3)
        mask = ~jnp.eye(disps.shape[0], dtype=bool)[:,:,None]
        disps = jnp.where(mask, disps, 0.0)

        recLattice = jnp.linalg.inv(self.lattice)
        fracDisps = disps @ recLattice

        cosDisps = jnp.cos(2 * jnp.pi * fracDisps)
        sinDisps = jnp.sin(2 * jnp.pi * fracDisps)
        sinDispsMag = jnp.linalg.norm(
            jnp.sin(jnp.pi * fracDisps), axis=-1, keepdims=True
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

def uhfInitialization(spins, dim, lattice, kpoints, seed=558):
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

    cpu_device = jax.devices("cpu")[0]
    with jax.default_device(cpu_device):
        kpoints_cpu = jax.device_put(kpoints, cpu_device)
        lattice_cpu = jax.device_put(lattice, cpu_device)
        system = qc.ueg_qc(0, spins, dim=dim, e_cut_red=0, basis=lattice_cpu)
        numkPoints = kpoints_cpu.shape[0]
        h1_jax = system.get_h1(kpoints_cpu)
        eri_jax = system.get_eri_tensor_real(kpoints_cpu)
        h1 = np.asarray(h1_jax, dtype=np.double)
        eri = np.asarray(eri_jax, dtype=np.double)
    
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

class LogMPSlater(Wavefunction):
    """
    Creates a log-wavefunction that is a Slater determinant built from a
    multiple planewave basis.
    """
    N : int
    dim : int
    kpoints : jnp.ndarray
    init_coeffs : jnp.ndarray

    def setup(self):
        
        if not (self.dim == 2 or self.dim == 3):
            raise Exception("Only dim=2 or dim=3 are supported.")

        self.numKpoints = self.kpoints.shape[0]
        weights = 10.0 ** jnp.arange(1, self.dim + 1)
        weighted_sums = jnp.dot(jnp.sign(self.kpoints), weights)
        k_signs = jnp.sign(weighted_sums + 1.0)
        self.cos_switch = (k_signs + 1.0) / 2.0
        self.sin_switch = (1.0 - k_signs) / 2.0

        self.mp_coeffs = self.param(
            "MP_coefficients", lambda rng: self.init_coeffs
        )

    def __call__(self, rs):
        def makeBasisRow(ri):
            def localKpointFunction(k, c_switch, s_switch):
                dot_val = jnp.dot(k, ri)
                return c_switch * jnp.cos(dot_val) + s_switch * jnp.sin(-dot_val)
            terms = jax.vmap(localKpointFunction)(
                self.kpoints, self.cos_switch, self.sin_switch
            )
            return terms
        basisMatrix = jax.vmap(makeBasisRow)(rs)
        orbitalMatrix = jnp.dot(basisMatrix, self.mp_coeffs)
        return jnp.linalg.slogdet(orbitalMatrix)[1]

class LogMPTwoBodySJB(Wavefunction):
    """
    Two-Body Slater-Jastrow-Backflow wavefunction for arbitrary (skewed)
    periodic cells. The Slater determinant consists of a sum of multiple
    planewaves.
    
    Args:
        lattice: (D, D) matrix defining the lattice vectors (rows).
                 e.g. [[Lx, 0], [Tx, Ty]] for a 2D tilted cell.
        kpoints: (N_k, D) matrix with the k-points that make up the basis
                 for the Slater determinant.
    """
    spins : (int,int)
    dim : int
    lattice : jnp.ndarray
    kpoints : jnp.ndarray
    upCoeffs : jnp.ndarray
    downCoeffs : jnp.ndarray
    hiddenFeatures : int

    def setup(self):

        N = self.spins[0] + self.spins[1]
        
        self.slaterUp = LogMPSlater(self.spins[0], self.dim, self.kpoints, self.upCoeffs)
        self.slaterDown = LogMPSlater(self.spins[1], self.dim, self.kpoints, self.downCoeffs)
        self.CYJastrow = LogCYJastrow(self.spins, self.lattice)
        
        self.neuralJastrow1 = nn.Dense(self.hiddenFeatures)
        self.neuralJastrow2 = nn.Dense(1)
        
        self.neuralBackflow1 = nn.Dense(self.hiddenFeatures)
        self.neuralBackflow2 = nn.Dense(self.dim)

    def __call__(self, rs):
        
        CYJastrow = self.CYJastrow(rs)
        
        disps = rs[:,None,:] - rs[None,:,:]  # (N, N, 3)
        mask = ~jnp.eye(disps.shape[0], dtype=bool)[:,:,None]
        disps = jnp.where(mask, disps, 0.0)

        recLattice = jnp.linalg.inv(self.lattice)
        fracDisps = disps @ recLattice

        cosDisps = jnp.cos(2 * jnp.pi * fracDisps)
        sinDisps = jnp.sin(2 * jnp.pi * fracDisps)
        sinDispsMag = jnp.linalg.norm(
            jnp.sin(jnp.pi * fracDisps), axis=-1, keepdims=True
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

class LogFixedMPSlater(Wavefunction):
    """
    Creates a log-wavefunction that is a Slater determinant built from a
    multiple planewave basis.
    """
    N : int
    dim : int
    kpoints : jnp.ndarray
    coeffs : jnp.ndarray

    def setup(self):
        
        if not (self.dim == 2 or self.dim == 3):
            raise Exception("Only dim=2 or dim=3 are supported.")

        self.numKpoints = self.kpoints.shape[0]
        weights = 10.0 ** jnp.arange(1, self.dim + 1)
        weighted_sums = jnp.dot(jnp.sign(self.kpoints), weights)
        k_signs = jnp.sign(weighted_sums + 1.0)
        self.cos_switch = (k_signs + 1.0) / 2.0
        self.sin_switch = (1.0 - k_signs) / 2.0

    def __call__(self, rs):
        def makeBasisRow(ri):
            def localKpointFunction(k, c_switch, s_switch):
                dot_val = jnp.dot(k, ri)
                return c_switch * jnp.cos(dot_val) + s_switch * jnp.sin(-dot_val)
            terms = jax.vmap(localKpointFunction)(
                self.kpoints, self.cos_switch, self.sin_switch
            )
            return terms
        basisMatrix = jax.vmap(makeBasisRow)(rs)
        orbitalMatrix = jnp.dot(basisMatrix, self.coeffs)
        return jnp.linalg.slogdet(orbitalMatrix)[1]

class LogFixedMPTwoBodySJB(Wavefunction):
    """
    Two-Body Slater-Jastrow-Backflow wavefunction for arbitrary (skewed)
    periodic cells. The Slater determinant consists of a sum of multiple
    planewaves.
    
    Args:
        lattice: (D, D) matrix defining the lattice vectors (rows).
                 e.g. [[Lx, 0], [Tx, Ty]] for a 2D tilted cell.
        kpoints: (N_k, D) matrix with the k-points that make up the basis
                 for the Slater determinant.
    """
    spins : (int,int)
    dim : int
    lattice : jnp.ndarray
    kpoints : jnp.ndarray
    upCoeffs : jnp.ndarray
    downCoeffs : jnp.ndarray
    hiddenFeatures : int

    def setup(self):

        N = self.spins[0] + self.spins[1]
        
        self.slaterUp = LogFixedMPSlater(self.spins[0], self.dim, self.kpoints, self.upCoeffs)
        self.slaterDown = LogFixedMPSlater(self.spins[1], self.dim, self.kpoints, self.downCoeffs)
        self.CYJastrow = LogCYJastrow(self.spins, self.lattice)
        
        self.neuralJastrow1 = nn.Dense(self.hiddenFeatures)
        self.neuralJastrow2 = nn.Dense(1)
        
        self.neuralBackflow1 = nn.Dense(self.hiddenFeatures)
        self.neuralBackflow2 = nn.Dense(self.dim)

    def __call__(self, rs):
        
        CYJastrow = self.CYJastrow(rs)
        
        disps = rs[:,None,:] - rs[None,:,:]  # (N, N, 3)
        mask = ~jnp.eye(disps.shape[0], dtype=bool)[:,:,None]
        disps = jnp.where(mask, disps, 0.0)

        recLattice = jnp.linalg.inv(self.lattice)
        fracDisps = disps @ recLattice

        cosDisps = jnp.cos(2 * jnp.pi * fracDisps)
        sinDisps = jnp.sin(2 * jnp.pi * fracDisps)
        sinDispsMag = jnp.linalg.norm(
            jnp.sin(jnp.pi * fracDisps), axis=-1, keepdims=True
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

class LogMessagePassingSJBLayerNorm(Wavefunction):
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

        self.LNg = [
            nn.LayerNorm() for _ in range(self.T)
        ]
        self.LNm = [
            nn.LayerNorm() for _ in range(self.T)
        ]
        self.LNhij = [
            nn.LayerNorm() for _ in range(self.T)
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
            gijt = self.LNg[t](gijt)

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
            mijt = self.LNm[t](mijt)
            #acc_mijt = jnp.average(mijt, axis=1)

            #hit += self.F1t[t][1](nn.swish(self.F1t[t][0](
            #    jnp.concatenate([acc_mijt,git], axis=-1)
            #)))
            hijt += self.F2t[t][1](nn.swish(self.F2t[t][0](
                jnp.concatenate([mijt, gijt], axis=-1)
            )))
            hijt = self.LNhij[t](hijt)

        #git = hit
        gijt = jnp.concatenate([hijt,v_ij], axis=-1)

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

class LogWignerCrystalSlater(Wavefunction):
    """
    Slater determinant reference for a 3D Wigner crystal.

    Spin-up electrons  -> simple cubic sites
    Spin-down electrons -> BCC-shifted sites

    Orbital:
        φ_j(r_i) = exp( -alpha * | d_pbc(r_i - R_j) |^2 )
    where d_pbc uses periodic sine distance, NOT min-image,
    giving a smooth & exactly periodic Gaussian-like bump.

    This removes boundary cusps and yields a fully variational,
    JIT-friendly reference state.
    """

    spins: tuple
    L: float

    def setup(self):
        N_up, N_dn = self.spins

        # Variational Gaussian width alpha = exp(log_alpha)
        self.log_alpha = self.param(
            "log_alpha", lambda rng: jnp.array(
                jnp.log(jnp.sqrt(100.0) / self.L**2)
            )
        )

        # Lattice centers (SC grid for ↑, BCC shift for ↓)
        self.centers_up, self.centers_dn = self._create_centers(N_up, N_dn, self.L)

    @staticmethod
    def _create_centers(N_up, N_dn, L):
        """
        Construct cubic sites (for spin-up), and BCC-shifted cubic sites
        (for spin-down), just like the standard WC reference ansatz.
        """
        # Smallest n such that n^3 ≥ max(N_up, N_dn)
        n = 1
        while n**3 < max(N_up, N_dn):
            n += 1

        a = L / n
        coords = jnp.linspace(0, L - a, n)
        grid = jnp.stack(jnp.meshgrid(coords, coords, coords, indexing="ij"), axis=-1)
        grid = grid.reshape(-1, 3)

        centers_up = grid[:N_up]
        shift = jnp.array([a/2, a/2, a/2])
        centers_dn = (grid + shift)[:N_dn]

        return centers_up, centers_dn

    # ----------------------------------------------------------
    # Periodic SINE distance: smooth and analytic everywhere
    # ----------------------------------------------------------
    def _periodic_displacement(self, dr):
        """
        Component-wise periodic displacement using sine distance:
           d_per(x) = (L/pi) * sin(pi*x/L)
        This is exactly periodic and smooth; removes cusps.
        """
        return (self.L / jnp.pi) * jnp.sin(jnp.pi * dr / self.L)

    def _orbital_single_center(self, ri, Rj, alpha):
        """
        Gaussian-like orbital centered at Rj, using sine-periodic displacement.
        """
        dr = ri - Rj                     # (3,)
        dr_p = self._periodic_displacement(dr)  # smooth periodic displacement
        r2 = jnp.dot(dr_p, dr_p)
        return jnp.exp(-alpha * r2)

    def _orbital_row(self, ri, centers, alpha):
        """
        Returns all orbitals φ_j(ri) for a fixed ri.
        """
        return jax.vmap(lambda Rj: self._orbital_single_center(ri, Rj, alpha))(centers)

    def _slater_logdet(self, rs_spin, centers, alpha):
        """
        Build and take logdet of the Slater matrix Φ_{ij} = φ_j(r_i).
        """
        Phi = jax.vmap(lambda ri: self._orbital_row(ri, centers, alpha))(rs_spin)
        sign, logdet = jnp.linalg.slogdet(Phi)
        return logdet  # ignore sign: ground-state WF is real-positive gauge

    def __call__(self, rs):
        """
        rs shape = (N_up + N_dn, 3)
        Returns log |det_up * det_dn|.
        """
        N_up, N_dn = self.spins
        rs_up = rs[:N_up]
        rs_dn = rs[N_up:N_up + N_dn]

        alpha = jnp.exp(self.log_alpha)
        alpha_min = 0.5 / self.L**2
        alpha_max = 200.0 / self.L**2
        alpha = jnp.clip(alpha, alpha_min, alpha_max)

        log_up = self._slater_logdet(rs_up, self.centers_up, alpha)
        log_dn = self._slater_logdet(rs_dn, self.centers_dn, alpha)

        return log_up + log_dn

class LogTwoBodySJBGaussianWignerCrystal(Wavefunction):
    """
    Slater-Jastrow wavefunction with following specs:
    - Slater: Gaussians centered at BCC-like Wigner crystal sites
    - Jastrow: Coulomb-Yukawa + neural function acting on pairwise information
        NOTE: This basically comes up with the "best" two-body Jastrow.
    - Backflow: acts purely on pairwise information to produce arbitrary
                backflow coordinates
    """
    spins : (int,int)
    L : float
    hiddenFeatures : int

    def setup(self):
        
        self.wignerSlater = LogWignerCrystalSlater(self.spins, self.L)
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
        
        wignerSlater = self.wignerSlater(xs)
        
        return wignerSlater + CYJastrow + neuralJastrow

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
