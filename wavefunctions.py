import jax
import jax.numpy as jnp

from flax import linen as nn
from flax.serialization import to_bytes, from_bytes
from flax.core import FrozenDict

from pyscf import gto, scf, ao2mo
import qc

import numpy as np

import itertools


def compute_lattice(
    N: int, r_ws: float, dim: int, basis_matrix=None
) -> jnp.ndarray:
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


def compute_r_ws(N: int, lattice: jnp.ndarray, dim: int) -> jnp.ndarray:
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


def save_parameters(filename: str, parameters) -> bool:
    """Saves a pytree object that contains a wavefunction's parameters."""
    with open(filename, "wb") as f:
        f.write(to_bytes(parameters))
    return True


def load_parameters(filename: str):
    """Loads a pytree FrozenDict from a file written by save_parameters."""
    with open(filename, "rb") as f:
        parameters = from_bytes(FrozenDict, f.read())
    return parameters


class Wavefunction(nn.Module):
    """Base class; exposes batched init and forward pass over a walker array."""

    def init_batch(self, rng, walkerRs: jnp.ndarray):
        return self.init(rng, walkerRs[0, :, :])

    def apply_batch(self, parameters, walkerRs: jnp.ndarray) -> jnp.ndarray:
        vmapApply = jax.vmap(self.apply, in_axes=(None, 0))
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


def gen_k_points(
    N: int, lattice: jnp.ndarray, dim: int, safety: float = 2.5, extra: int = 4
) -> jnp.ndarray:
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

    vol_k_cell = abs(np.linalg.det(rec_lattice))
    if dim == 2:
        k_radius = np.sqrt(N * vol_k_cell / np.pi)
    else:
        k_radius = (N * vol_k_cell * 3.0 / (4.0 * np.pi)) ** (1.0 / 3.0)

    rec_norms = np.linalg.norm(rec_lattice, axis=1)
    max_integers = np.ceil((safety * k_radius) / rec_norms).astype(int) + extra

    ranges = [range(-m, m + 1) for m in max_integers]
    int_candidates = np.array(list(itertools.product(*ranges)), dtype=int)

    phys_k = int_candidates @ rec_lattice
    k2 = np.einsum("ij,ij->i", phys_k, phys_k)

    keys = [int_candidates[:, d] for d in range(dim - 1, -1, -1)] + [k2]
    order = np.lexsort(keys)
    candidates_sorted = int_candidates[order]

    selected = []
    seen = set()

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

        if not _canonical_pos_int(k_int):
            continue

        selected.append(k_int.copy())
        seen.add(kt)

        if len(selected) < N:
            nk = tuple((-k_int).tolist())
            selected.append((-k_int).copy())
            seen.add(nk)

    selected = np.array(selected[:N], dtype=np.float64)
    chosen = selected @ rec_lattice
    return jnp.array(chosen)


class LogSimpleSlater(Wavefunction):
    """
    Slater determinant of the N lowest-energy planewaves (fixed, no trainable params).

    IMPORTANT: Assumes k-points ordered as [0, k1, -k1, k2, -k2, ...].
    """
    N: int
    dim: int
    kpoints: jnp.ndarray

    def setup(self):

        if not (self.dim == 2 or self.dim == 3):
            raise Exception("Only dim=2 or dim=3 are supported.")

        Nk = self.kpoints.shape[0]
        cos = jnp.zeros(Nk).at[0].set(1.0)
        cos = cos.at[jnp.arange(1, Nk, 2)].set(1.0)
        self.cos_switch = cos
        self.sin_switch = 1.0 - cos

    def __call__(self, rs: jnp.ndarray) -> jnp.ndarray:
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
    Slater determinant from a trainable multiple-planewave orbital expansion.
    Each orbital is a linear combination of planewaves: φ_a(x) = Σ_k c_{ak} φ_k(x).

    IMPORTANT: Assumes k-points ordered as [0, k1, -k1, k2, -k2, ...].
    """
    N: int
    dim: int
    kpoints: jnp.ndarray      # (Nk, dim)
    init_coeffs: jnp.ndarray  # (Nk, N)

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

    def __call__(self, rs: jnp.ndarray) -> jnp.ndarray:
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
        basisMatrix = jax.vmap(makeBasisRow)(rs)             # (N, Nk)
        orbitalMatrix = jnp.dot(basisMatrix, self.mp_coeffs) # (N, N)
        return jnp.linalg.slogdet(orbitalMatrix)[1]


def occ_columns_from_theta(theta: jnp.ndarray) -> jnp.ndarray:
    """
    Smooth Grassmannian map from Thouless parameters to orbital coefficient matrix.

    Args:
        theta: (N_virt, N_occ) virtual-to-occupied mixing angles.

    Returns:
        C: (N_total, N_occ) orbital coefficient matrix.
           At theta=0 returns the Fermi sea (first N_occ columns of identity).

    Uses the matrix exponential of the block anti-symmetric generator
        K = [[0, -θᵀ], [θ, 0]]
    so that the gradient is finite everywhere, including at theta=0 (the Fermi
    Liquid optimum) where the SVD-based alternative is ill-conditioned.
    """
    n_virt, n_occ = theta.shape
    n_total = n_occ + n_virt
    K = jnp.zeros((n_total, n_total), dtype=theta.dtype)
    K = K.at[n_occ:, :n_occ].set(theta)      # virt-occ block
    K = K.at[:n_occ, n_occ:].set(-theta.T)   # occ-virt block (anti-symmetric)
    return jax.scipy.linalg.expm(K)[:, :n_occ]


def theta_from_occ_columns(C: jnp.ndarray) -> jnp.ndarray:
    """
    Recover a Thouless parameter matrix theta from an orbital coefficient matrix.

    Args:
        C: (N_total, N_occ) orbital coefficient matrix.

    Returns:
        theta: (N_virt, N_occ) Thouless parameters.

    Used only for initialization (runs outside JIT).
    """
    C = jnp.asarray(C)
    n_occ = C.shape[1]
    C_occ = C[:n_occ, :]
    C_vir = C[n_occ:, :]
    X = C_vir @ jnp.linalg.inv(C_occ)
    Ux, t, Vxh = jnp.linalg.svd(X, full_matrices=False)
    Vx = jnp.conj(Vxh.T)
    s = jnp.arctan(t)
    theta = (Vx @ jnp.diag(s) @ jnp.conj(Ux.T)).T
    return theta


class LogThoulessSlater(Wavefunction):
    """
    Slater determinant parameterized via Thouless rotation of the Fermi sea.

    The trainable parameter `thouless_rotation` has shape (N_virt, N_occ) and
    maps to orbital coefficients via occ_columns_from_theta (matrix-exponential
    based, well-conditioned at theta=0).

    IMPORTANT: Assumes k-points ordered as [0, k1, -k1, k2, -k2, ...].
    """
    N: int
    dim: int
    kpoints: jnp.ndarray      # (Nk, dim)
    init_coeffs: jnp.ndarray  # (Nk, N)

    def setup(self):

        if not (self.dim == 2 or self.dim == 3):
            raise Exception("Only dim=2 or dim=3 are supported.")

        Nk = self.kpoints.shape[0]
        cos = jnp.zeros(Nk).at[0].set(1.0)
        cos = cos.at[jnp.arange(1, Nk, 2)].set(1.0)
        self.cos_switch = cos
        self.sin_switch = 1.0 - cos

        thouless = theta_from_occ_columns(self.init_coeffs)
        self.thouless = self.param(
            "thouless_rotation", lambda _: thouless
        )

    def __call__(self, rs: jnp.ndarray) -> jnp.ndarray:
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
        basisMatrix = jax.vmap(makeBasisRow)(rs)           # (N, Nk)
        mp_coeffs = occ_columns_from_theta(self.thouless)  # (Nk, N)
        orbitalMatrix = jnp.dot(basisMatrix, mp_coeffs)    # (N, N)
        return jnp.linalg.slogdet(orbitalMatrix)[1]


class LogFixedMPSlater(Wavefunction):
    """
    Slater determinant from a fixed (non-trainable) multiple-planewave expansion.

    IMPORTANT: Assumes k-points ordered as [0, k1, -k1, k2, -k2, ...].
    """
    N: int
    dim: int
    kpoints: jnp.ndarray  # (Nk, dim)
    coeffs: jnp.ndarray   # (Nk, N)

    def setup(self):

        if not (self.dim == 2 or self.dim == 3):
            raise Exception("Only dim=2 or dim=3 are supported.")

        Nk = self.kpoints.shape[0]
        cos = jnp.zeros(Nk).at[0].set(1.0)
        cos = cos.at[jnp.arange(1, Nk, 2)].set(1.0)
        self.cos_switch = cos
        self.sin_switch = 1.0 - cos

    def __call__(self, rs: jnp.ndarray) -> jnp.ndarray:
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
        basisMatrix = jax.vmap(makeBasisRow)(rs)           # (N, Nk)
        orbitalMatrix = jnp.dot(basisMatrix, self.coeffs)  # (N, N)
        return jnp.linalg.slogdet(orbitalMatrix)[1]


def generate_gaussian_mp_coeffs(
    kpoints: jnp.ndarray, centers: jnp.ndarray, alpha: float
) -> jnp.ndarray:
    """
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
    weights = jnp.exp(-ksq / (4.0 * alpha))[:, None]   # (Nk, 1)

    coeffs = weights * (
        cos_switch[:, None] * jnp.cos(dots) +
        sin_switch[:, None] * jnp.sin(dots)
    )

    return coeffs


class LogGaussianSlater(Wavefunction):
    """
    Slater determinant with trainable Gaussian orbital widths initialized at r_ws.

    IMPORTANT: Assumes k-points ordered as [0, k1, -k1, k2, -k2, ...].
    """
    N: int
    dim: int
    r_ws: float
    kpoints: jnp.ndarray  # (Nk, dim)
    centers: jnp.ndarray  # (N, dim)

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

    def __call__(self, rs: jnp.ndarray) -> jnp.ndarray:

        def makeBasisRow(ri):
            def localKpointFunction(k, c_switch, s_switch):
                dot_val = jnp.dot(k, ri)
                return c_switch * jnp.cos(dot_val) + s_switch * jnp.sin(dot_val)

            return jax.vmap(localKpointFunction)(
                self.kpoints, self.cos_switch, self.sin_switch
            )

        basisMatrix = jax.vmap(makeBasisRow)(rs)       # (N, Nk)
        alpha = jnp.exp(self.log_alpha)
        coeffs = generate_gaussian_mp_coeffs(
            self.kpoints, self.centers, alpha
        )                                              # (Nk, N)
        orbitalMatrix = jnp.dot(basisMatrix, coeffs)   # (N, N)

        return jnp.linalg.slogdet(orbitalMatrix)[1]


def cy_jastrow_default_as(
    spins: tuple, lattice: jnp.ndarray
) -> jnp.ndarray:
    """
    RPA estimate of the CY Jastrow A parameters for the uniform electron gas.
    Returns shape (2,) array [A_same_spin, A_diff_spin], both set to 1/sqrt(4πn).
    """
    N = spins[0] + spins[1]
    volume = jnp.abs(jnp.linalg.det(lattice))
    n = N / volume
    return jnp.full(2, 1.0 / jnp.sqrt(4 * jnp.pi * n))


def _coulomb_yukawa(
    r_real: jnp.ndarray, A: jnp.ndarray, F: jnp.ndarray
) -> jnp.ndarray:
    """
    Coulomb-Yukawa pair potential u(r) = (A/r)(1 - exp(-r/F)).

    The cusp conditions fix F from A: F_same = sqrt(2A), F_diff = sqrt(A).
    No boundary decay is applied; smooth periodicity is ensured upstream by
    computing r_real via the sine-MIC approximation.
    """
    r_safe = jnp.where(r_real < 1e-12, 1.0, r_real)
    return (A / r_safe) * (1 - jnp.exp(-r_safe / F))


def _cy_jastrow_forward(
    rs: jnp.ndarray,
    spins: tuple,
    lattice: jnp.ndarray,
    rec_lattice: jnp.ndarray,
    As: jnp.ndarray,
) -> jnp.ndarray:
    """
    Forward pass for the Coulomb-Yukawa Jastrow factor.

    Cusp conditions: F_same = sqrt(2 A_same), F_diff = sqrt(A_diff).

    Distances are computed via the sine-MIC approximation:
        disp_mic = sin(π · disp_frac) @ lattice / π
    which is smooth and periodic across cell boundaries (no folx-incompatible
    rem/% operation), and equals the true MIC displacement for small separations.
    """
    N = rs.shape[0]

    A_same = jnp.abs(As[0])
    A_diff = jnp.abs(As[1])

    F_same = jnp.sqrt(2 * A_same)
    F_diff = jnp.sqrt(A_diff)

    disp_raw = rs[:, None, :] - rs[None, :, :]          # (N, N, dim)
    disp_frac = disp_raw @ rec_lattice                   # (N, N, dim)
    sin_disp_frac = jnp.sin(jnp.pi * disp_frac)         # (N, N, dim)

    eye_mask = jnp.eye(N, dtype=rs.dtype)[:, :, None]   # diagonal protection
    disp_mic = (sin_disp_frac @ lattice) / jnp.pi        # (N, N, dim)
    r_ij = jnp.linalg.norm(disp_mic + eye_mask, axis=-1) # (N, N)

    n_up, n_down = spins
    spin_mask = jnp.concatenate([jnp.zeros(n_up, dtype=int), jnp.ones(n_down, dtype=int)])

    mask_same = spin_mask[:, None] == spin_mask[None, :]
    mask_diff = spin_mask[:, None] != spin_mask[None, :]
    eye = jnp.eye(N, dtype=bool)
    mask_same = mask_same & (~eye)

    val_same = _coulomb_yukawa(r_ij, A_same, F_same)
    val_same = jnp.where(mask_same, val_same, 0.0)

    val_diff = _coulomb_yukawa(r_ij, A_diff, F_diff)
    val_diff = jnp.where(mask_diff, val_diff, 0.0)

    return -0.5 * (jnp.sum(val_same) + jnp.sum(val_diff))


class LogCYJastrow(Wavefunction):
    """
    Coulomb-Yukawa Jastrow factor with trainable A parameters.

    Same-spin and opposite-spin pairs use separate A values (parameter key
    "As_same_diff"). F parameters are set by cusp conditions. Absolute values
    of A are taken so optimization never produces an unphysical sign.
    """
    spins: tuple
    lattice: jnp.ndarray

    def setup(self):
        self.rec_lattice = jnp.linalg.inv(self.lattice)
        self.As = self.param(
            "As_same_diff",
            lambda _: cy_jastrow_default_as(self.spins, self.lattice)
        )

    def __call__(self, rs: jnp.ndarray) -> jnp.ndarray:
        return _cy_jastrow_forward(
            rs, self.spins, self.lattice, self.rec_lattice, self.As
        )


class LogFixedCYJastrow(Wavefunction):
    """Same as LogCYJastrow but A parameters are fixed at initialization."""
    spins: tuple
    lattice: jnp.ndarray
    As: jnp.ndarray

    def setup(self):
        self.rec_lattice = jnp.linalg.inv(self.lattice)

    def __call__(self, rs: jnp.ndarray) -> jnp.ndarray:
        return _cy_jastrow_forward(
            rs, self.spins, self.lattice, self.rec_lattice, self.As
        )


class LogSimpleSlaters(Wavefunction):
    """
    Product of two fixed Slater determinants (one per spin) with no trainable params.
    """
    spins: tuple
    dim: int
    kpoints: jnp.ndarray

    def setup(self):
        self.slaterUp = LogSimpleSlater(self.spins[0], self.dim, self.kpoints)
        self.slaterDown = LogSimpleSlater(self.spins[1], self.dim, self.kpoints)

    def __call__(self, rs: jnp.ndarray) -> jnp.ndarray:
        slaterUp = self.slaterUp(rs[:self.spins[0], :])
        slaterDown = self.slaterDown(rs[self.spins[0]:, :])
        return slaterUp + slaterDown


class LogSlaterCYJastrow(Wavefunction):
    """
    Product of two fixed Slater determinants and a trainable CY Jastrow (2 params).
    """
    spins: tuple
    dim: int
    lattice: jnp.ndarray
    kpoints: jnp.ndarray

    def setup(self):
        self.slaterUp = LogSimpleSlater(self.spins[0], self.dim, self.kpoints)
        self.slaterDown = LogSimpleSlater(self.spins[1], self.dim, self.kpoints)
        self.CYJastrow = LogCYJastrow(self.spins, self.lattice)

    def __call__(self, rs: jnp.ndarray) -> jnp.ndarray:
        slaterUp = self.slaterUp(rs[:self.spins[0], :])
        slaterDown = self.slaterDown(rs[self.spins[0]:, :])
        CYJastrow = self.CYJastrow(rs)
        return slaterUp + slaterDown + CYJastrow


def generate_features(
    rs: jnp.ndarray, spins: tuple, lattice: jnp.ndarray
) -> jnp.ndarray:
    """
    Generates smooth, periodic two-body features from electron positions.

    Returns shape (N, N, 2*dim+2) containing cosine/sine displacements,
    sine-magnitude distances, and spin-match indicators.
    """
    disps = rs[:, None, :] - rs[None, :, :]  # (N, N, dim)
    mask = ~jnp.eye(disps.shape[0], dtype=bool)[:, :, None]
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
    matchMatrix = jnp.outer(electronSpins, electronSpins)[:, :, None]

    v_ij = jnp.concatenate(
        [cosDisps, sinDisps, sinDispsMag, matchMatrix],
        axis=-1
    )

    return v_ij


class LogTwoBodySJ(Wavefunction):
    """
    Slater-Jastrow wavefunction: two fixed Slater determinants, a trainable CY
    Jastrow, and a two-body neural Jastrow term.
    """
    spins: tuple
    dim: int
    lattice: jnp.ndarray
    kpoints: jnp.ndarray
    hiddenFeatures: int

    def setup(self):

        self.slaterUp = LogSimpleSlater(self.spins[0], self.dim, self.kpoints)
        self.slaterDown = LogSimpleSlater(self.spins[1], self.dim, self.kpoints)
        self.CYJastrow = LogCYJastrow(self.spins, self.lattice)

        self.linear1 = nn.Dense(self.hiddenFeatures)
        self.linear2 = nn.Dense(
            1, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros
        )

    def __call__(self, rs: jnp.ndarray) -> jnp.ndarray:

        slaterUp = self.slaterUp(rs[:self.spins[0], :])
        slaterDown = self.slaterDown(rs[self.spins[0]:, :])
        CYJastrow = self.CYJastrow(rs)

        v_ij = generate_features(rs, self.spins, self.lattice)   # (N, N, 2*dim+2)
        n_ij = self.linear2(nn.swish(self.linear1(v_ij)))        # (N, N, 1)
        neuralJastrow = 0.5 * jnp.sum(n_ij) / sum(self.spins)

        return slaterUp + slaterDown + CYJastrow + neuralJastrow


class LogTwoBodySJB(Wavefunction):
    """
    Slater-Jastrow-Backflow wavefunction: two fixed Slater determinants (applied
    to backflow-shifted positions), a trainable CY Jastrow, and a two-body
    neural Jastrow term. Output features of a single two-body network provide
    both backflow displacements and the neural Jastrow.
    """
    spins: tuple
    dim: int
    lattice: jnp.ndarray
    kpoints: jnp.ndarray
    hiddenFeatures: int

    def setup(self):

        self.slaterUp = LogSimpleSlater(self.spins[0], self.dim, self.kpoints)
        self.slaterDown = LogSimpleSlater(self.spins[1], self.dim, self.kpoints)
        self.CYJastrow = LogCYJastrow(self.spins, self.lattice)

        self.linear1 = nn.Dense(self.hiddenFeatures)
        self.linear2 = nn.Dense(
            self.dim + 1, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros
        )

    def __call__(self, rs: jnp.ndarray) -> jnp.ndarray:

        v_ij = generate_features(rs, self.spins, self.lattice)   # (N, N, 2*dim+2)
        n_ij = self.linear2(nn.swish(self.linear1(v_ij)))        # (N, N, dim+1)

        jastrowFeatures = n_ij[:, :, 0]                          # (N, N)
        backflowFeatures = n_ij[:, :, 1:]                        # (N, N, dim)

        backflow = jnp.average(backflowFeatures, axis=1)         # (N, dim)
        xs = rs + backflow

        slaterUp = self.slaterUp(xs[:self.spins[0], :])
        slaterDown = self.slaterDown(xs[self.spins[0]:, :])
        CYJastrow = self.CYJastrow(rs)
        neuralJastrow = 0.5 * jnp.sum(jastrowFeatures) / sum(self.spins)

        return slaterUp + slaterDown + CYJastrow + neuralJastrow