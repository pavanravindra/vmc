import jax
import jax.numpy as jnp

from flax import linen as nn
from flax.serialization import to_bytes, from_bytes
from flax.core import FrozenDict

from pyscf import gto, scf, ao2mo

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

    def __call__(self, rs):
        def makeSimpleSlaterRow(ri):
            def localKpointFunction(k):
                return jnp.exp(1j * jnp.dot(k, ri))
            return jax.vmap(localKpointFunction)(self.kpoints)
        slaterMatrix = jax.vmap(makeSimpleSlaterRow)(rs)
        phase, logmag = jnp.linalg.slogdet(slaterMatrix)
        angle = jnp.angle(phase)
        return angle, logmag

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

        self.mp_coeffs = self.param(
            "MP_coefficients", lambda _ : self.init_coeffs
        )

    def __call__(self, rs):
        def makeBasisRow(ri):
            def localKpointFunction(k):
                return jnp.exp(1j * jnp.dot(k, ri))
            return jax.vmap(localKpointFunction)(self.kpoints)
        basisMatrix = jax.vmap(makeBasisRow)(rs)               # ( N , Nk )
        orbitalMatrix = jnp.dot(basisMatrix, self.mp_coeffs)   # ( N , N )
        phase, logmag = jnp.linalg.slogdet(orbitalMatrix)
        angle = jnp.angle(phase)
        return angle, logmag

def generateGaussianMPCoeffs(kpoints, centers, alpha):
    """
    TODO : This needs to be updated to handle complex wavefunctions.

    Generate fixed MP coefficients corresponding to a truncated Fourier
    representation of periodized Gaussian-like orbitals centered at `centers`,
    in the real cosine/sine basis convention

        [0, k1, -k1, k2, -k2, ...]

    where:
      - index 0 is the constant / cos(0) term
      - odd indices use cos(k · r)
      - even indices use sin(k · r)

    Parameters
    ----------
    kpoints : jnp.ndarray, shape (Nk, dim)
        Reciprocal vectors in the ordering [0, k1, -k1, k2, -k2, ...].
    centers : jnp.ndarray, shape (N, dim)
        Orbital centers R_j.
    alpha : float or scalar jnp.ndarray
        Positive real-space Gaussian parameter in exp(-alpha |r - R|^2).

    Returns
    -------
    coeffs : jnp.ndarray, shape (Nk, N)
        Coefficients for use in, e.g., LogFixedMPSlater.
    """
    Nk = kpoints.shape[0]

    cos_switch = jnp.zeros(Nk, dtype=kpoints.dtype).at[0].set(1.0)
    cos_switch = cos_switch.at[jnp.arange(1, Nk, 2)].set(1.0)
    sin_switch = 1.0 - cos_switch

    dots = jnp.dot(kpoints, centers.T)                 # (Nk, N)
    ksq = jnp.sum(kpoints**2, axis=1)                  # (Nk,)
    weights = jnp.exp(-ksq / (4.0 * alpha))[:, None]  # (Nk, 1)

    coeffs = weights * (
        cos_switch[:, None] * jnp.cos(dots) +
        sin_switch[:, None] * jnp.sin(dots)
    )

    return coeffs

class LogGaussianSlater(Wavefunction):
    """
    TODO : This needs to be updated to handle complex wavefunctions.

    Log-wavefunction for a Slater determinant whose orbitals are initialized as
    truncated Fourier representations of periodized Gaussian orbitals centered
    at the provided positions.

    IMPORTANT: This code assumes that the k-points come in order like
               [0, k1, -k1, k2, -k2, ...]
    """
    N: int
    dim: int
    r_ws: float
    kpoints: jnp.ndarray   # (Nk, dim)
    centers: jnp.ndarray   # (N, dim)

    def setup(self):

        if self.dim not in (2, 3):
            raise Exception("Only dim=2 or dim=3 are supported.")

        Nk = self.kpoints.shape[0]
        cos = jnp.zeros(Nk, dtype=self.kpoints.dtype).at[0].set(1.0)
        cos = cos.at[jnp.arange(1, Nk, 2)].set(1.0)
        self.cos_switch = cos
        self.sin_switch = 1.0 - cos

        self.log_alpha = self.param(
            "log_alpha",
            lambda _: jnp.array(
                jnp.log(1.0 / (self.r_ws ** 2)),
                dtype=self.kpoints.dtype
            )
        )

    def __call__(self, rs):

        def makeBasisRow(ri):
            def localKpointFunction(k, c_switch, s_switch):
                dot_val = jnp.dot(k, ri)
                return c_switch * jnp.cos(dot_val) + s_switch * jnp.sin(dot_val)

            return jax.vmap(localKpointFunction)(
                self.kpoints, self.cos_switch, self.sin_switch
            )

        basisMatrix = jax.vmap(makeBasisRow)(rs)      # (N, Nk)
        alpha = jnp.exp(self.log_alpha)
        coeffs = generateGaussianMPCoeffs(
            self.kpoints, self.centers, alpha
        )                                             # (Nk, N)
        orbitalMatrix = jnp.dot(basisMatrix, coeffs)  # (N, N)

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
        logmag = cyJastrowForwardFunction(
            rs, self.spins, self.lattice, self.rec_lattice, self.As
        )
        return logmag

class LogSimpleSlaters(Wavefunction):
    """    
    Creates a log-wavefunction that is the product of two simple Slater
    determinant with the lowest k-points filled.

    There are no variational parameters.
    """
    spins : (int,int)
    dim : int
    kpoints : jnp.ndarray

    def setup(self):
        self.slaterUp = LogSimpleSlater(self.spins[0], self.dim, self.kpoints)
        self.slaterDown = LogSimpleSlater(self.spins[1], self.dim, self.kpoints)

    def __call__(self, rs):
        angleUp, logmagUp = self.slaterUp(rs[:self.spins[0],:])
        angleDown, logmagDown = self.slaterDown(rs[self.spins[0]:,:])
        angle = angleUp + angleDown
        logmag = logmagUp + logmagDown
        return angle, logmag

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
        angleUp, logmagUp = self.slaterUp(rs[:self.spins[0],:])
        angleDown, logmagDown = self.slaterDown(rs[self.spins[0]:,:])
        logmagCY = self.CYJastrow(rs)
        angle = angleUp + angleDown
        logmag = logmagUp + logmagDown + logmagCY
        return angle, logmag

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
    TODO : This needs to be updated to handle complex wavefunctions.
    
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
    TODO : This needs to be updated to handle complex wavefunctions.
    
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
