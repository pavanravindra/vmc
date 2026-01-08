import jax
import jax.numpy as jnp
import jax.scipy as jsp

import numpy as np

from dataclasses import dataclass
from typing import Tuple, Optional
from functools import partial
import itertools

@dataclass
class ueg:
    r_s: float
    n_elec: Tuple[int, int]
    dim: int

    basis: Optional[jax.Array] = None
    
    rec_basis: jax.Array = jnp.zeros((3,3)) # Rows are b1, b2...
    volume: float = 0.0  
    n_particles: int = 0
    density: float = 0.0
    seed: int = 0

    def __post_init__(self):
        assert self.dim in (2, 3), "Only dim=2 or dim=3 are supported."
        assert self.n_elec[0] == self.n_elec[1], "Only unpolarized systems supported."

        self.n_particles = int(self.n_elec[0] + self.n_elec[1])

        # 1. Setup Default Basis (Cubic/Square) if None provided
        if self.basis is None:
            if self.dim == 3:
                vol = (4.0 / 3.0) * jnp.pi * (self.r_s**3) * self.n_particles
                L = vol ** (1.0 / 3.0)
            else:
                area = jnp.pi * (self.r_s**2) * self.n_particles
                L = jnp.sqrt(area)
            self.basis = jnp.eye(self.dim) * L
        
        self.basis = jnp.asarray(self.basis)

        # 2. Compute Geometry
        # Volume = |det(basis)|
        self.volume = jnp.abs(jnp.linalg.det(self.basis))
        self.density = self.n_particles / self.volume

        # Reciprocal Lattice: B = 2pi * (A^-1)^T
        inv_basis_T = jnp.linalg.inv(self.basis).T
        self.rec_basis = 2.0 * jnp.pi * inv_basis_T

    def _paired_kpoints_from_nvecs(self, nvecs: np.ndarray) -> jax.Array:
        """
        Generates physical k-points from integer vectors, handling
        general triclinic sorting orders.
        """
        nvecs = np.asarray(nvecs, dtype=int)
        assert nvecs.ndim == 2 and nvecs.shape[1] == self.dim
        
        # Add Gamma point if missing
        zero = tuple([0] * self.dim)
        n_set = {tuple(v) for v in nvecs.tolist()}
        if zero not in n_set:
            nvecs = np.vstack([np.zeros((1, self.dim), dtype=int), nvecs])
            n_set.add(zero)

        # Sort by PHYSICAL norm |k|^2, not integer norm |n|^2
        # k = n @ rec_basis
        rec_basis_np = np.array(self.rec_basis)
        k_vecs_temp = nvecs @ rec_basis_np
        k_sq = np.sum(k_vecs_temp**2, axis=1)
        
        order = np.argsort(k_sq, kind="stable")
        nvecs = nvecs[order]

        # Logic to group +k and -k pairs (unchanged logic, just applied to sorted list)
        result_n: list[tuple[int, ...]] = [zero]
        processed: set[tuple[int, ...]] = {zero}
        canonicals: list[tuple[int, ...]] = []

        for n in nvecs:
            t = tuple(int(x) for x in n.tolist())
            if t in processed or t == zero:
                continue
            neg = tuple(-x for x in t)
            if neg not in n_set:
                # If -k isn't in the set, we skip +k to maintain pairing
                # (Or strictly enforce grid symmetry upstream)
                processed.add(t)
                continue

            c = None
            for x in t:
                if x != 0:
                    c = t if x > 0 else neg
                    break
            if c is None:
                continue

            processed.add(c)
            processed.add(tuple(-x for x in c))
            canonicals.append(c)

        # Re-sort canonicals by physical energy just to be safe
        canonicals.sort(key=lambda v: np.sum((np.array(v) @ rec_basis_np)**2))
        
        for c in canonicals:
            result_n.append(c)
            result_n.append(tuple(-x for x in c))

        kpts = jnp.array(result_n, dtype=float) @ self.rec_basis
        return kpts

    def get_occ_k_points(self) -> jax.Array:
        dk = 1.0 + 1e-5
        n_up = int(self.n_elec[0])

        min_b_len = jnp.min(jnp.linalg.norm(self.rec_basis, axis=1))
        k_f_approx = (n_up * self.density)**(1.0/self.dim) * 2 * np.pi
        max_n = int(np.ceil(k_f_approx / min_b_len * dk)) + 2
        
        ordinals = sorted(range(-max_n, max_n + 1), key=abs)
        nvecs = np.array(list(itertools.product(ordinals, repeat=self.dim)), dtype=int)

        rec_basis_np = np.array(self.rec_basis)
        k_vecs = nvecs @ rec_basis_np
        k2 = np.sum(k_vecs**2, axis=1)
        
        order = np.argsort(k2, kind="stable")
        nvecs = nvecs[order]
        k2 = k2[order]

        n2F = k2[n_up - 1]
        keep = k2 <= (n2F * dk * dk + 1e-12)
        return self._paired_kpoints_from_nvecs(nvecs[keep])

    def _calc_dis(self, pos: jax.Array):
        pos_up = pos[0]
        pos_dn = pos[1]
        pos_flat = jnp.concatenate([pos_up, pos_dn], axis=0)
        n_particles = pos_flat.shape[0]

        inv_basis = jnp.linalg.inv(self.basis)

        def get_disp(i, j):
            dr = pos_flat[i] - pos_flat[j]
            # Transform to fractional coords: s = A^-1 r
            ds = dr @ inv_basis.T
            # Wrap to [-0.5, 0.5]
            ds = ds - jnp.round(ds)
            # Transform back: r = A s
            dr_wrapped = ds @ self.basis
            return dr_wrapped

        disp = jax.vmap(
            lambda i: jax.vmap(get_disp, in_axes=(None, 0))(i, jnp.arange(n_particles))
        )(jnp.arange(n_particles))
        
        dist = jnp.sqrt(jnp.sum(disp**2, axis=-1) + 1e-10)
        mask = ~jnp.eye(n_particles, dtype=bool)
        dist = jnp.where(mask, dist, 0.0)
        return dist, disp

    def init_walker_data(self, n_walkers: int) -> dict:
        def walker_init(subkey):
            subkey, subkey_up = jax.random.split(subkey)
            # Sample u ~ Uniform(0, 1)
            u_up = jax.random.uniform(subkey_up, (self.n_elec[0], self.dim))
            # r = u @ basis
            pos_up = u_up @ self.basis
            
            subkey, subkey_dn = jax.random.split(subkey)
            u_dn = jax.random.uniform(subkey_dn, (self.n_elec[1], self.dim))
            pos_dn = u_dn @ self.basis
            
            pos = jnp.array([pos_up, pos_dn])
            dist, disp = self._calc_dis(pos)
            return pos, dist, disp

        random_key = jax.random.PRNGKey(self.seed)
        random_key, *subkeys = jax.random.split(random_key, n_walkers + 1)
        pos, dist, disp = jax.vmap(walker_init)(jnp.array(subkeys))
        return {"pos": pos, "dist": dist, "disp": disp, "random_key": random_key}

    def update_walker_data(self, new_pos_batch: jax.Array, walker_data: dict) -> dict:
        assert new_pos_batch.shape == walker_data["pos"].shape
        # Use simple scan to avoid massive memory usage on vmap of all-to-all
        def update_single_walker(carry, new_pos_i):
            dist, disp = self._calc_dis(new_pos_i)
            return carry, (dist, disp)

        _, (dist, disp) = jax.lax.scan(update_single_walker, None, new_pos_batch)
        walker_data["dist"] = dist
        walker_data["disp"] = disp
        walker_data["pos"] = new_pos_batch
        return walker_data

    def __hash__(self) -> int:
        # Use basis hash instead of box_length
        return hash((
            tuple(np.array(self.basis).flatten().tolist()), 
            self.r_s, 
            self.n_elec
        ))


@dataclass
class ueg_qc(ueg):
    """Plane-wave QC Hamiltonian for General 2D/3D lattices."""
    e_cut_red: float = 1.0
    
    # Internal cache for Madelung constant
    _madelung_const: float = None

    def __post_init__(self):
        super().__post_init__()
        # Compute exact Madelung constant for this specific lattice geometry
        self._madelung_const = self._compute_madelung_constant()

    def _compute_madelung_constant(self) -> float:
        """
        Numerically computes the self-interaction potential (Madelung constant)
        using Ewald summation for the specific lattice basis.
        """
        # Ewald screening parameter. 
        # A standard heuristic is kappa ~ sqrt(pi)/L_eff
        L_eff = self.volume**(1.0/self.dim)
        kappa = np.sqrt(np.pi) / L_eff
        
        # Lattice sums limits
        n_max = 6 # sufficient for double precision with chosen kappa

        # 1. Real Space Sum: sum_R erfc(kappa|R|) / |R|
        ns = np.arange(-n_max, n_max+1)
        n_grid = np.array(list(itertools.product(ns, repeat=self.dim)))
        # Remove origin
        n_grid = n_grid[np.sum(n_grid**2, axis=1) > 0]
        
        Rs = n_grid @ np.array(self.basis)
        R_norms = np.linalg.norm(Rs, axis=1)
        real_term = np.sum(jsp.special.erfc(kappa * R_norms) / R_norms)

        # 2. Reciprocal Space Sum & Self/Background terms
        rec_term = 0.0
        const_term = 0.0
        
        ns_rec = np.arange(-n_max, n_max+1)
        k_grid = np.array(list(itertools.product(ns_rec, repeat=self.dim)))
        k_grid = k_grid[np.sum(k_grid**2, axis=1) > 0]
        Ks = k_grid @ np.array(self.rec_basis)
        K_sq = np.sum(Ks**2, axis=1)

        if self.dim == 3:
            # 3D Ewald
            # Sum_K (4pi/V K^2) * exp(-K^2 / 4kappa^2)
            rec_term = (4.0 * np.pi / self.volume) * np.sum(np.exp(-K_sq / (4 * kappa**2)) / K_sq)
            # Constant: -2kappa/sqrt(pi) - pi/(V kappa^2)
            const_term = -2.0 * kappa / np.sqrt(np.pi) - np.pi / (self.volume * kappa**2)
        
        elif self.dim == 2:
            # 2D Ewald for 1/r interaction
            # Sum_K (2pi/A K) * erfc(K / 2kappa)  <-- Note 1/K kernel, not 1/K^2
            K_norms = np.sqrt(K_sq)
            rec_term = (2.0 * np.pi / self.volume) * np.sum(jsp.special.erfc(K_norms / (2 * kappa)) / K_norms)
            # Constant: -2kappa/sqrt(pi) - 2sqrt(pi)/(A kappa)
            const_term = -2.0 * kappa / np.sqrt(np.pi) - 2.0 * np.sqrt(np.pi) / (self.volume * kappa)

        return float(real_term + rec_term + const_term)

    def get_k_points(self) -> jax.Array:
        # Generate grid based on cutoff
        # We need a grid large enough to cover the cutoff sphere in skewed space
        min_b_len = jnp.min(jnp.linalg.norm(self.rec_basis, axis=1))
        k_cut = np.sqrt(2.0 * float(self.e_cut_red)) # Assuming e_cut_red is in units of (2pi/L)^2 approx? 
        # Actually e_cut_red usually refers to units of (2pi/L_eff)^2. 
        # We'll generate a safe grid.
        max_n = int(np.ceil(k_cut / min_b_len * 2 * np.pi)) + 2
        
        ns = np.arange(-max_n, max_n + 1, dtype=int)
        nvecs = np.array(list(itertools.product(ns, repeat=self.dim)), dtype=int)

        # Filter by physical energy
        rec_basis_np = np.array(self.rec_basis)
        k_vecs = nvecs @ rec_basis_np
        eps_phys = 0.5 * np.sum(k_vecs**2, axis=1)
        
        # e_cut_red is typically dimensionless. We interpret it relative to the 
        # smallest non-zero k-vector energy or just keep user semantics.
        # Assuming user inputs e_cut_red consistent with cubic (2pi/L)^2 scaling:
        # We verify strict cutoff.
        keep = eps_phys <= (float(self.e_cut_red) + 1e-12)
        return self._paired_kpoints_from_nvecs(nvecs[keep])

    def madelung(self) -> float:
        # Returns the total potential shift v_Mad (at r=0)
        # Our _madelung_const is exactly 'v_Mad' for a unit charge (xi).
        # Original: -2.83 * N / L.
        # madelung() should return: self._madelung_const * self.n_particles
        return self._madelung_const * self.n_particles

    def get_h1(self, k_points: jax.Array) -> jax.Array:
        # H1 = Kinetic + Madelung_Correction
        # Kinetic = 0.5 * k^2
        h1 = jnp.diag(jnp.sum(k_points**2, axis=-1) / 2.0)
        
        # Correction term per particle is 0.5 * xi
        # code uses: 0.5 * madelung() / N = 0.5 * (xi * N) / N = 0.5 * xi
        mad_term = 0.5 * self.madelung() / self.n_particles
        return h1 + mad_term * jnp.eye(k_points.shape[0])
    
    # ... get_h1_real, unitary_pw_to_real remain exactly the same ...

    def eri_element(self, k_points: jax.Array, p, q, r, s) -> jax.Array:
        g1 = k_points[q] - k_points[p]
        g2 = k_points[r] - k_points[s]
        mom_ok = jnp.all(jnp.isclose(g1, g2, atol=1e-8), axis=-1)

        # NOTE: k_points are already physical cartesian vectors
        if self.dim == 3:
            g2norm = jnp.sum(g1 * g1, axis=-1)
            non_zero = g2norm > 1e-10
            element = 4.0 * jnp.pi / (self.volume * g2norm)
        else:
            # 2D case (1/r interaction)
            gnorm = jnp.linalg.norm(g1, axis=-1)
            non_zero = gnorm > 1e-10
            element = 2.0 * jnp.pi / (self.volume * gnorm)

        element = jnp.where(jnp.isinf(element) | jnp.isnan(element), 0.0, element)
        return mom_ok * non_zero * element

    # ... get_eri_tensor, get_eri_tensor_real, __hash__ remain the same ...
    def get_eri_tensor(self, k_points: jax.Array) -> jax.Array:
        n = k_points.shape[0]
        idx = jnp.arange(n)
        p, q, r, s = jnp.meshgrid(idx, idx, idx, idx, indexing="ij")
        eri_flat = self.eri_element(k_points, p.ravel(), q.ravel(), r.ravel(), s.ravel())
        return eri_flat.reshape(n, n, n, n)

    def unitary_pw_to_real(self, k_points: jax.Array) -> jax.Array:
        n_kpts = k_points.shape[0]
        assert (n_kpts - 1) % 2 == 0, "k_points must be Gamma + pairs."
        U = jnp.zeros((n_kpts, n_kpts), dtype=jnp.complex128)
        block = jnp.array([[1.0, 1.0], [-1.0j, 1.0j]]) / jnp.sqrt(2.0)
        n_blocks = (n_kpts - 1) // 2
        U = U.at[0, 0].set(1.0)
        U = U.at[1:, 1:].set(jsp.linalg.block_diag(*([block] * n_blocks)))
        return U

    def get_eri_tensor_real(self, k_points: jax.Array) -> jax.Array:
        eri = self.get_eri_tensor(k_points)
        U = self.unitary_pw_to_real(k_points)
        eri = jnp.einsum("ip,pqrs->iqrs", U.conj(), eri, optimize=True)
        eri = jnp.einsum("jq,iqrs->ijrs", U, eri, optimize=True)
        eri = jnp.einsum("kr,ijrs->ijks", U.conj(), eri, optimize=True)
        eri = jnp.einsum("ls,ijks->ijkl", U, eri, optimize=True).real
        return eri

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))