import jax
import jax.numpy as jnp

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

def _canonical_pos_int(k_int: np.ndarray) -> bool:
    """
    Decide whether an integer k-vector is the canonical 'positive' representative
    of the ± pair, using lexicographic sign rule:
        first nonzero component must be positive.
    """
    for x in k_int:
        if x > 0:
            return True
        if x < 0:
            return False
    return True  # k=0 treated as positive

def genKpoints(N: int, lattice, dim: int, safety: float = 2.5, extra: int = 4):
    """
    Deterministic k-point generator with stable ordering and adjacent ± pairs.

    Ordering:
        [0, k1, -k1, k2, -k2, ...]
    where k_i are chosen by increasing |k|, with deterministic tie-breaking.

    Notes:
    - Uses integer lattice vectors m, maps to physical k = m @ rec_lattice.
    - Tie-break for equal |k| is lexicographic on integer components (dim-wise).
    - Picks only the canonical representative of each ± pair in the main loop,
      then appends its negative.

    Args:
        N: number of k-points to return
        lattice: (dim, dim) real-space lattice (rows are lattice vectors)
        dim: 2 or 3
        safety: multiplies radius estimate to ensure candidate set contains enough points
        extra: additive padding on integer bounds

    Returns:
        (N, dim) jnp.ndarray of k-points
    """
    if dim not in (2, 3):
        raise ValueError("Only dim=2 or dim=3 are supported.")
    lattice = np.array(lattice, dtype=np.float64)
    rec_lattice = 2 * np.pi * np.linalg.inv(lattice).T  # (dim, dim)

    # --- Choose a conservative integer search box ---
    # Estimate k-radius for N points in k-space volume, then map to integer bounds.
    vol_k_cell = abs(np.linalg.det(rec_lattice))
    if dim == 2:
        k_radius = np.sqrt(N * vol_k_cell / np.pi)
    else:
        k_radius = (N * vol_k_cell * 3.0 / (4.0 * np.pi)) ** (1.0 / 3.0)

    rec_norms = np.linalg.norm(rec_lattice, axis=1)  # norms of reciprocal basis vectors
    max_integers = np.ceil((safety * k_radius) / rec_norms).astype(int) + extra  # (dim,)

    ranges = [range(-m, m + 1) for m in max_integers]
    int_candidates = np.array(list(itertools.product(*ranges)), dtype=int)  # (M, dim)

    # Physical k and squared norms
    phys_k = int_candidates @ rec_lattice  # (M, dim)
    k2 = np.einsum("ij,ij->i", phys_k, phys_k)  # (M,)

    # Deterministic sort: primary key k2, then integer components lexicographically
    # np.lexsort uses last key as primary, so we pass reversed order.
    keys = [int_candidates[:, d] for d in range(dim-1, -1, -1)] + [k2]
    order = np.lexsort(keys)
    candidates_sorted = int_candidates[order]

    selected = []
    seen = set()

    # Ensure k=0 is first if present
    zero = tuple([0] * dim)
    if zero in map(tuple, candidates_sorted):
        selected.append(np.array(zero, dtype=int))
        seen.add(zero)

    for k_int in candidates_sorted:
        if len(selected) >= N:
            break

        kt = tuple(k_int)
        if kt in seen:
            continue
        if kt == zero:
            continue

        # Only consider canonical representative for each ± pair
        if not _canonical_pos_int(k_int):
            continue

        # Add k
        selected.append(k_int.copy())
        seen.add(kt)

        # Add -k if room
        if len(selected) < N:
            nk = tuple((-k_int).tolist())
            selected.append((-k_int).copy())
            seen.add(nk)

    selected = np.array(selected[:N], dtype=np.float64)
    chosen = selected @ rec_lattice
    return jnp.array(chosen)

class LogSimpleSlater(Wavefunction):
    """
    Creates a log-wavefunction that is just a simple Slater determinant of the
    input electron coordinates. The basis for the determinant is specified by
    the provided kpoints.

    IMPORTANT: This code assumes that the k-points come in order like
                [0, k1, -k1, k2, -k2, ...]
    
    NOTE: These determinants use the convention that different particle
    positions are in different rows. Columns correspond to plane wave orbitals.
    """
    N : int
    dim : int
    kpoints : jnp.ndarray

    def setup(self):
        
        if not (self.dim == 2 or self.dim == 3):
            raise Exception("Only dim=2 or dim=3 are supported.")
            
        Nk = self.kpoints.shape[0]
        cos = jnp.zeros(Nk).at[0].set(1.0)
        cos = cos.at[jnp.arange(1, Nk, 2)].set(1.0)
        self.cos_switch = cos
        self.sin_switch = 1.0 - cos

    def __call__(self, rs):
        def makeSimpleSlaterRow(ri):
            def localKpointFunction(k, c_switch, s_switch):
                dot_val = jnp.dot(k, ri)
                term = (c_switch * jnp.cos(dot_val) + 
                        s_switch * jnp.sin(dot_val))
                return term
            return jax.vmap(localKpointFunction)(
                self.kpoints, 
                self.cos_switch, 
                self.sin_switch
            )
        slaterMatrix = jax.vmap(makeSimpleSlaterRow)(rs)
        return jnp.linalg.slogdet(slaterMatrix)[1]

class LogMPSlater(Wavefunction):
    """
    Creates a log-wavefunction that is a Slater determinant built from a
    multiple planewave basis.

    IMPORTANT: This code assumes that the k-points come in order like
                [0, k1, -k1, k2, -k2, ...]
    """
    N : int
    dim : int
    kpoints : jnp.ndarray       # ( Nk , dim )
    init_coeffs : jnp.ndarray   # ( Nk , N )

    def setup(self):
        
        if not (self.dim == 2 or self.dim == 3):
            raise Exception("Only dim=2 or dim=3 are supported.")
            
        Nk = self.kpoints.shape[0]
        cos = jnp.zeros(Nk).at[0].set(1.0)
        cos = cos.at[jnp.arange(1, Nk, 2)].set(1.0)
        self.cos_switch = cos
        self.sin_switch = 1.0 - cos

        self.mp_coeffs = self.param(
            "MP_coefficients", lambda rng: self.init_coeffs
        )

    def __call__(self, rs):
        def makeBasisRow(ri):
            def localKpointFunction(k, c_switch, s_switch):
                dot_val = jnp.dot(k, ri)
                term = (c_switch * jnp.cos(dot_val) + 
                        s_switch * jnp.sin(dot_val))
                return term
            terms = jax.vmap(localKpointFunction)(
                self.kpoints, self.cos_switch, self.sin_switch
            )
            return terms
        basisMatrix = jax.vmap(makeBasisRow)(rs)               # ( N , Nk )
        orbitalMatrix = jnp.dot(basisMatrix, self.mp_coeffs)   # ( N , N )
        return jnp.linalg.slogdet(orbitalMatrix)[1]

class LogFixedMPSlater(Wavefunction):
    """
    Creates a log-wavefunction that is a Slater determinant built from a
    multiple planewave basis. The MP coefficients in this implementation are
    fixed at initialization.

    IMPORTANT: This code assumes that the k-points come in order like
                [0, k1, -k1, k2, -k2, ...]
    """
    N : int
    dim : int
    kpoints : jnp.ndarray       # ( Nk , dim )
    coeffs : jnp.ndarray        # ( Nk , N )

    def setup(self):
        
        if not (self.dim == 2 or self.dim == 3):
            raise Exception("Only dim=2 or dim=3 are supported.")
            
        Nk = self.kpoints.shape[0]
        cos = jnp.zeros(Nk).at[0].set(1.0)
        cos = cos.at[jnp.arange(1, Nk, 2)].set(1.0)
        self.cos_switch = cos
        self.sin_switch = 1.0 - cos

    def __call__(self, rs):
        def makeBasisRow(ri):
            def localKpointFunction(k, c_switch, s_switch):
                dot_val = jnp.dot(k, ri)
                term = (c_switch * jnp.cos(dot_val) + 
                        s_switch * jnp.sin(dot_val))
                return term
            terms = jax.vmap(localKpointFunction)(
                self.kpoints, self.cos_switch, self.sin_switch
            )
            return terms
        basisMatrix = jax.vmap(makeBasisRow)(rs)               # ( N , Nk )
        orbitalMatrix = jnp.dot(basisMatrix, self.coeffs)      # ( N , N )
        return jnp.linalg.slogdet(orbitalMatrix)[1]

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

def cyJastrowForwardFunction(rs, spins, lattice, rec_lattice, As):

        N = rs.shape[0]

        A_same = jnp.abs(As[0])
        A_diff = jnp.abs(As[1])
        
        F_same = jnp.sqrt(2 * A_same)
        F_diff = jnp.sqrt(A_diff)

        disp_real_raw = rs[:,None,:] - rs[None,:,:]
        disp_frac = jnp.dot(disp_real_raw, rec_lattice)
        disp_frac = (disp_frac + 0.5) % 1.0 - 0.5
        disp_real_mic = jnp.dot(disp_frac, lattice)

        eye_mask = jnp.eye(N)[:,:,None]
        r_ij_real = jnp.linalg.norm(disp_real_mic + eye_mask, axis=-1)
        r_ij_frac = jnp.linalg.norm(disp_frac + eye_mask, axis=-1)
        
        n_up, n_down = spins
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
        return cyJastrowForwardFunction(
            rs, self.spins, self.lattice, self.rec_lattice, self.As
        )

class LogFixedCYJastrow(Wavefunction):
    """
    Same as LogCYJastrow but the $A$ parameters are *fixed* at initialization.
    """
    spins : (int,int)
    lattice: jnp.ndarray
    As : jnp.ndarray

    def setup(self):
        volume = jnp.abs(jnp.linalg.det(self.lattice))
        self.rec_lattice = jnp.linalg.inv(self.lattice)

    def __call__(self, rs):
        return cyJastrowForwardFunction(
            rs, self.spins, self.lattice, self.rec_lattice, self.As
        )

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

def generateFeatures(rs, spins, lattice):
    """
    Takes in electron positions and generates two-body features that contain
    two-body information about electron displacements and spin matching.
    Importantly, the features are continuous at the provided unit cell
    boundaries, which make them useful as features for neural layers.

    If `rs` is shape (N,dim), then the output is shape (N,N,2*dim+2).
    """
        
    disps = rs[:,None,:] - rs[None,:,:]  # (N, N, dim)
    mask = ~jnp.eye(disps.shape[0], dtype=bool)[:,:,None]
    disps = jnp.where(mask, disps, 0.0)

    recLattice = jnp.linalg.inv(lattice)
    fracDisps = disps @ recLattice

    cosDisps = jnp.cos(2 * jnp.pi * fracDisps)
    sinDisps = jnp.sin(2 * jnp.pi * fracDisps)
    sinDispsMag = jnp.linalg.norm(
        jnp.sin(jnp.pi * fracDisps), axis=-1, keepdims=True
    )
    sinDispsMag = jnp.where(mask, sinDispsMag, 0.0)
    
    N = spins[0] + spins[1]
    electronIdxs = jnp.arange(N)
    electronSpins = jnp.where(electronIdxs < spins[0], 1, -1)
    matchMatrix = jnp.outer(electronSpins, electronSpins)[:,:,None]
    
    v_ij = jnp.concatenate(
        [cosDisps, sinDisps, sinDispsMag, matchMatrix],
        axis=-1
    )

    return v_ij

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

        self.slaterUp = LogSimpleSlater(self.spins[0], self.dim, self.kpoints)
        self.slaterDown = LogSimpleSlater(self.spins[1], self.dim, self.kpoints)
        self.CYJastrow = LogCYJastrow(self.spins, self.lattice)
        
        self.linear1 = nn.Dense(self.hiddenFeatures)
        self.linear2 = nn.Dense(
            1, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros
        )

    def __call__(self, rs):
        
        slaterUp = self.slaterUp(rs[:self.spins[0],:])
        slaterDown = self.slaterDown(rs[self.spins[0]:,:])
        CYJastrow = self.CYJastrow(rs)

        v_ij = generateFeatures(rs, self.spins, self.lattice)   # (N,N,2*dim+2)
        n_ij = self.linear2(nn.swish(self.linear1(v_ij)))       # (N,N,1)
        neuralJastrow = 0.5 * jnp.sum(n_ij) / sum(self.spins)
        
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
        
        self.slaterUp = LogSimpleSlater(self.spins[0], self.dim, self.kpoints)
        self.slaterDown = LogSimpleSlater(self.spins[1], self.dim, self.kpoints)
        self.CYJastrow = LogCYJastrow(self.spins, self.lattice)
        
        self.linear1 = nn.Dense(self.hiddenFeatures)
        self.linear2 = nn.Dense(
            self.dim+1, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros
        )

    def __call__(self, rs):

        v_ij = generateFeatures(rs, self.spins, self.lattice)   # (N,N,2*dim+2)
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
