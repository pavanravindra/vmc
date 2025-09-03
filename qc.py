import jax
import jax.numpy as jnp

from jax import (
    jit,
    scipy as jsp
)

from dataclasses import dataclass
from typing import Tuple
from functools import partial
import itertools

@dataclass
class ueg:
    r_s: float
    n_elec: Tuple[int, int]
    box_length: float = 0.0
    rec_lattice: Tuple = ()
    dim: int = 3
    volume: float = 0.0
    n_particles: int = 0
    density: float = 0.0
    seed: int = 0

    def __post_init__(self):
        assert self.dim == 3, "Only 3D systems are supported."
        assert (
            self.n_elec[0] == self.n_elec[1]
        ), "Only unpolarized systems are supported."
        self.box_length = (
            4 / 3 * jnp.pi * self.r_s**3 * (self.n_elec[0] + self.n_elec[1])
        ) ** (1 / 3)
        self.rec_lattice = (2 * jnp.pi / self.box_length,) * 3
        self.volume = self.box_length**3
        self.n_particles = self.n_elec[0] + self.n_elec[1]
        self.density = self.n_particles / self.volume

    def get_occ_k_points(self) -> jax.Array:
        """Get the occupied k-points for the system."""
        dk = 1 + 1e-5
        max_k = int(jnp.ceil(self.n_elec[0] * dk) ** (1 / 3.0))
        ordinals = sorted(range(-max_k, max_k + 1), key=abs)
        ordinals = jnp.asarray(list(itertools.product(ordinals, repeat=3)))
        kpoints = ordinals @ (jnp.array(self.rec_lattice) * jnp.eye(3)).T
        kpoints = jnp.asarray(sorted(kpoints, key=jnp.linalg.norm))
        k_norms = jnp.linalg.norm(kpoints, axis=1)
        kpoints = kpoints[k_norms <= k_norms[self.n_elec[0] - 1] * dk]

        kpoints_list = kpoints.tolist()
        result = []
        result.append(kpoints_list[0])
        # remove gamma from consideration
        kpoints_list = [k for i, k in enumerate(kpoints_list) if i != 0]

        pairs = {}
        processed = set()
        for k in kpoints_list:
            k_tuple = tuple(k)
            if k_tuple in processed:
                continue

            neg_k = tuple(-x for x in k)
            processed.add(k_tuple)
            if neg_k in map(tuple, kpoints_list):
                processed.add(neg_k)

            canonical = None
            for i, val in enumerate(k):
                if abs(val) > 1e-10:
                    if val > 0:
                        canonical = k_tuple
                        partner = neg_k
                    else:
                        canonical = neg_k
                        partner = k_tuple
                    break

            if canonical is not None:
                pairs[canonical] = partner

        sorted_canonicals = sorted(pairs.keys(), key=lambda k: sum(x * x for x in k))
        for canonical in sorted_canonicals:
            result.append(canonical)
            result.append(pairs[canonical])
        return jnp.array(result)

    @partial(jit, static_argnums=(0,))
    def _calc_dis(self, pos: jax.Array) -> Tuple:
        box_length = jnp.array([self.box_length, self.box_length, self.box_length])
        pos_up = pos[0]
        pos_dn = pos[1]
        pos_flat = jnp.concatenate([pos_up, pos_dn], axis=0)
        n_particles = pos_flat.shape[0]

        def get_disp(i, j):
            dr = pos_flat[i] - pos_flat[j]
            dr = dr - box_length * jnp.round(dr / box_length)
            return dr

        disp = vmap(
            lambda i: vmap(get_disp, in_axes=(None, 0))(i, jnp.arange(n_particles))
        )(jnp.arange(n_particles))
        dist = jnp.sqrt(jnp.sum(disp**2, axis=-1) + 1e-10)
        mask = ~jnp.eye(n_particles, dtype=bool)
        dist = jnp.where(mask, dist, 0.0)
        return dist, disp

    def init_walker_data(self, n_walkers: int) -> dict:
        def walker_init(subkey):
            subkey, subkey_up = random.split(subkey)
            pos_up = random.uniform(subkey_up, (self.n_elec[0], 3)) * self.box_length
            subkey, subkey_dn = random.split(subkey)
            pos_dn = random.uniform(subkey_dn, (self.n_elec[1], 3)) * self.box_length
            pos = jnp.array([pos_up, pos_dn])
            dist, disp = self._calc_dis(pos)
            return pos, dist, disp

        random_key = random.PRNGKey(self.seed)
        random_key, *subkeys = random.split(random_key, n_walkers + 1)
        pos, dist, disp = vmap(walker_init)(jnp.array(subkeys))
        return {
            "pos": pos,
            "dist": dist,
            "disp": disp,
            "random_key": random_key,
        }

    @partial(jit, static_argnums=(0,))
    def update_walker_data(self, new_pos_batch: jax.Array, walker_data: dict) -> dict:
        assert new_pos_batch.shape == walker_data["pos"].shape

        def update_single_walker(carry, new_pos_i):
            dist, disp = self._calc_dis(new_pos_i)
            return carry, (dist, disp)

        _, (dist, disp) = lax.scan(update_single_walker, None, new_pos_batch)
        walker_data["dist"] = dist
        walker_data["disp"] = disp
        walker_data["pos"] = new_pos_batch
        return walker_data

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))

@dataclass
class ueg_qc(ueg):
    """Quantum chemistry class for the UEG."""

    numkPoints: int = 0

    def get_k_points(self) -> jax.Array:
        """Get the k-point basis for the system based on e_cut."""
        dk = 1 + 1e-5
        max_k = int((self.numkPoints * dk)**(1/3) + 0.5)
        ordinals = sorted(range(-max_k, max_k + 1), key=abs)
        ordinals = jnp.asarray(list(itertools.product(ordinals, repeat=3)))
        kpoints = ordinals @ (jnp.array(self.rec_lattice) * jnp.eye(3)).T
        kpoints = jnp.asarray(sorted(kpoints, key=jnp.linalg.norm))
        k_norms = jnp.linalg.norm(kpoints, axis=1) ** 2 / 2
        kpoints = kpoints[:self.numkPoints]

        kpoints_list = kpoints.tolist()
        result = []
        result.append(kpoints_list[0])
        # remove gamma from consideration
        kpoints_list = [k for i, k in enumerate(kpoints_list) if i != 0]

        pairs = {}
        processed = set()
        for k in kpoints_list:
            k_tuple = tuple(k)
            if k_tuple in processed:
                continue

            neg_k = tuple(-x for x in k)
            processed.add(k_tuple)
            if neg_k in map(tuple, kpoints_list):
                processed.add(neg_k)

            canonical = None
            for i, val in enumerate(k):
                if abs(val) > 1e-10:
                    if val > 0:
                        canonical = k_tuple
                        partner = neg_k
                    else:
                        canonical = neg_k
                        partner = k_tuple
                    break

            if canonical is not None:
                pairs[canonical] = partner

        sorted_canonicals = sorted(pairs.keys(), key=lambda k: sum(x * x for x in k))
        for canonical in sorted_canonicals:
            result.append(canonical)
            result.append(pairs[canonical])
        return jnp.array(result)

    def madelung(self):
        return (
            -2.837297
            * (3.0 / 4.0 / jnp.pi) ** (1.0 / 3.0)
            * self.n_particles ** (2.0 / 3.0)
            / self.r_s
        )

    @partial(jax.jit, static_argnums=(0,))
    def get_h1(self, k_points: jax.Array) -> jax.Array:
        """Get the one-body Hamiltonian in plane wave basis.
        Includes the Madelung constant."""
        h1 = jnp.diag(jnp.sum(k_points**2, axis=-1) / 2)
        madelung = 0.5 * self.madelung() / self.n_particles
        return h1 + madelung * jnp.eye(k_points.shape[0])

    @partial(jax.jit, static_argnums=(0,))
    def get_h1_real(self, k_points: jax.Array) -> jax.Array:
        """Get the one-body Hamiltonian in real basis."""
        h1_pw = self.get_h1(k_points)
        unitary = self.unitary_pw_to_real(k_points)
        h1 = unitary.conj() @ h1_pw @ unitary.T
        return h1.real

    @partial(jax.jit, static_argnums=(0,))
    def eri_element(
        self,
        k_points: jax.Array,
        p: jax.Array,
        q: jax.Array,
        r: jax.Array,
        s: jax.Array,
    ) -> jax.Array:
        """Periodic Coulomb interaction integral ( pq | rs )."""
        g1 = k_points[q] - k_points[p]
        g2 = k_points[r] - k_points[s]
        momentum_conserved = jnp.all(jnp.isclose(g1, g2), axis=-1)
        g1_squared = jnp.sum(g1 * g1, axis=-1)
        non_zero = g1_squared > 1e-10
        element = 4 * jnp.pi / g1_squared / self.volume
        element = jnp.where(jnp.isinf(element) | jnp.isnan(element), 0.0, element)
        return momentum_conserved * non_zero * element

    @partial(jax.jit, static_argnums=(0,))
    def get_eri_tensor(self, k_points: jax.Array) -> jax.Array:
        """Get the ERI tensor in plane wave basis."""
        numkPoints = k_points.shape[0]
        idx = jnp.arange(numkPoints)
        p_idx, q_idx, r_idx, s_idx = jnp.meshgrid(idx, idx, idx, idx, indexing="ij")
        p_flat = p_idx.flatten()
        q_flat = q_idx.flatten()
        r_flat = r_idx.flatten()
        s_flat = s_idx.flatten()
        eri_flat = self.eri_element(k_points, p_flat, q_flat, r_flat, s_flat)
        eri = eri_flat.reshape(numkPoints, numkPoints, numkPoints, numkPoints)
        return eri

    # @partial(jax.jit, static_argnums=(0,))
    def unitary_pw_to_real(self, k_points: jax.Array) -> jax.Array:
        """Unitary transformation from plane wave basis to real cos, sin basis.
        Assumes k_points arranged so that +k, -k pairs are adjacent.
        """
        numkPoints = k_points.shape[0]
        unitary = jnp.zeros((numkPoints, numkPoints), dtype=jnp.complex128)
        unitary_block = jnp.array([[1.0, 1.0], [-1.0j, 1.0j]]) / jnp.sqrt(2.0)
        n_blocks = (numkPoints - 1) // 2
        unitary = unitary.at[0, 0].set(1.0)
        unitary = unitary.at[1:, 1:].set(
            jsp.linalg.block_diag(*([unitary_block] * n_blocks))
        )
        return unitary

    @partial(jax.jit, static_argnums=(0,))
    def get_eri_tensor_real(self, k_points: jax.Array) -> jax.Array:
        """Calculate the ERI tensor in real basis using the unitary transformation."""
        eri = self.get_eri_tensor(k_points)
        unitary = self.unitary_pw_to_real(k_points)
        eri = jnp.einsum("ip,pqrs->iqrs", unitary.conj(), eri, optimize=True)
        eri = jnp.einsum("jq,iqrs->ijrs", unitary, eri, optimize=True)
        eri = jnp.einsum("kr,ijrs->ijks", unitary.conj(), eri, optimize=True)
        eri = jnp.einsum("ls,ijks->ijkl", unitary, eri, optimize=True).real
        return eri

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))