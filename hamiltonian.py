from dataclasses import dataclass
from typing import Callable, Any, Tuple, Union
import itertools

import jax
import jax.numpy as jnp

def vmap_chunked(
    fn: Callable[..., Any],
    n_chunks: int,
    *,
    in_axes: Union[int, Tuple[Union[int, None], ...]] = 0,
):
    """
    Memory friendly vmap: map over axis-0 in micro-batches using lax.map.

    Usage like vmap:
        out = vmap_chunked(fn, n_chunks, in_axes=...)(*args, **kwargs)
    """

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        g = lambda *a: fn(*a, **kwargs)

        if n_chunks == 1:
            return jax.vmap(g, in_axes=in_axes)(*args)

        if not isinstance(in_axes, tuple):
            in_axes_ = (in_axes,) * len(args)
        else:
            in_axes_ = in_axes

        mapped_pos = [i for i, ax in enumerate(in_axes_) if ax == 0]
        if not mapped_pos:
            return g(*args)

        nw = args[mapped_pos[0]].shape[0]
        batch_size = (nw + n_chunks - 1) // n_chunks

        mapped_args = tuple(args[i] for i in mapped_pos)

        def f(xi):
            full = list(args)
            for j, i in enumerate(mapped_pos):
                full[i] = xi[j]
            return g(*full)

        return jax.lax.map(f, mapped_args, batch_size=batch_size)

    return wrapped

class LocalEnergy:
    def configuration(self, parameters, rs):
        raise NotImplementedError

@dataclass
class Batched(LocalEnergy):
    term: LocalEnergy
    n_chunks: int = 1
    in_axes: tuple = (None, 0)

    def batch(self, parameters, walkerRs):
        mapper = vmap_chunked(self.term.configuration, self.n_chunks, in_axes=self.in_axes)
        return mapper(parameters, walkerRs)

    def configuration(self, parameters, rs):
        return self.term.configuration(parameters, rs)

def laplacian(logWavefunction, parameters, rs):
    """
    Computes the Laplacian of logWavefunction w.r.t. rs in a memory-heavy way.

    The Laplacian is computed all at once so this is very fast, but if you try
    to vmap this over all 1024 walkers, you will certainly run out of memory.

    The chunked `vmap` approach in `Batched` is designed to handle this issue.
    """

    def f_rs(x):
        return logWavefunction(parameters, x)

    grad_fn = jax.grad(f_rs)
    D = rs.size

    def second_deriv(i):
        v = jax.nn.one_hot(i, D, dtype=rs.dtype).reshape(rs.shape)
        hv = jax.jvp(grad_fn, (rs,), (v,))[1].reshape(-1)
        return hv[i]

    diag = jax.vmap(second_deriv)(jnp.arange(D, dtype=jnp.int32))  # float32
    # return jnp.sum(diag) # sum seems numerically unstable in float32?
    # Basically the sum contains a few very large numbers on a different order
    #     than the rest of the numbers

    # Compute sum in float64 since it's more stable
    diag64 = diag.astype(jnp.float64)
    lap64 = jax.lax.fori_loop(0, D, lambda i, acc: acc + diag64[i], jnp.float64(0.0))
    return lap64.astype(rs.dtype)

class LocalKineticEnergy(LocalEnergy):
    """
    Computes the local kinetic energy of an arbitrary log wavefunction.
    """

    def __init__(self, logWavefunction):
        self.logWavefunction = logWavefunction.apply

    def configuration(self, parameters, rs):

        grad2f = laplacian(self.logWavefunction, parameters, rs)

        gradFunction = jax.grad(self.logWavefunction, argnums=1)
        grad = gradFunction(parameters, rs)  # shape (N, D)
        gradf2 = jnp.sum(grad ** 2)

        kineticEnergy = -0.5 * (grad2f + gradf2)
        
        return kineticEnergy

class EwaldPotential(LocalEnergy):
    """
    This is heavily inspired by code from `make_ewald_potential` in FermiNet!
    
    Evaluates the infinite Coulomb sum for a bunch of electrons on a periodic lattice.
    """
    
    def __init__(self, lattice, truncationLimit=2):

        self.lattice = lattice
        self.truncationLimit = truncationLimit
        
        self.rec = 2 * jnp.pi * jnp.linalg.inv(self.lattice)
        self.volume = jnp.abs(jnp.linalg.det(lattice))
        self.gamma = (2.8 / self.volume**(1/3)) ** 2
        
        ordinals = sorted(range(-truncationLimit, truncationLimit + 1), key=abs)
        ordinals = jnp.array(list(itertools.product(ordinals, repeat=3)))
        
        self.lat_vectors = jnp.einsum('kj,ij->ik', self.lattice, ordinals)
        lat_vec_norm = jnp.linalg.norm(self.lat_vectors[1:], axis=-1)
        
        self.rec_vectors = jnp.einsum('jk,ij->ik', self.rec, ordinals[1:])
        self.rec_vec_square = jnp.einsum('ij,ij->i', self.rec_vectors, self.rec_vectors)
                
        self.madelung_const = (
            jnp.sum(jax.scipy.special.erfc(jnp.sqrt(self.gamma) * lat_vec_norm) / lat_vec_norm)
            - 2 * jnp.sqrt(self.gamma) / jnp.sqrt(jnp.pi)
            + (4 * jnp.pi / self.volume) * jnp.sum(jnp.exp(-self.rec_vec_square / (4 * self.gamma)) / self.rec_vec_square)
            - jnp.pi / (self.volume * self.gamma)
        )
    
        self.batch_ewald_sum = jax.vmap(self.ewald_sum, in_axes=0)
    
    def real_space_ewald(self, separation: jnp.ndarray):
        disp = jnp.linalg.norm(separation - self.lat_vectors, axis=1)
        return jnp.sum(jax.scipy.special.erfc(jnp.sqrt(self.gamma) * disp) / disp)

    def recp_space_ewald(self, separation: jnp.ndarray):
        phase = jnp.dot(self.rec_vectors, separation)
        factor = jnp.exp(-self.rec_vec_square / (4 * self.gamma)) / self.rec_vec_square
        return (4 * jnp.pi / self.volume) * jnp.sum(jnp.cos(phase) * factor)

    def ewald_sum(self, separation: jnp.ndarray):
        return  self.real_space_ewald(separation) +\
                self.recp_space_ewald(separation) -\
                jnp.pi / (self.volume * self.gamma)

    def configuration(self, parameters, rs):
        N = rs.shape[0]
        ee = rs[:, None, :] - rs[None, :, :]
        phase_ee = jnp.einsum('il,jkl->jki', self.rec / (2 * jnp.pi), ee)
        phase_prim_ee = phase_ee % 1
        prim_ee = jnp.einsum('il,jkl->jki', self.lattice, phase_prim_ee)
        prim_ee = prim_ee.reshape(-1, 3)
        vals = self.batch_ewald_sum(prim_ee).reshape(N,N)
        vals = vals.at[jnp.diag_indices(N)].set(0.0)
        return 0.5 * jnp.sum(vals) + 0.5 * N * self.madelung_const

class EwaldPotential2D(LocalEnergy):
    """
    This is heavily inspired by code from `make_ewald_potential` in FermiNet!
    
    Evaluates the infinite Coulomb sum for a bunch of electrons on a periodic lattice
    in 2 Dimensions.
    """
    
    def __init__(self, lattice, truncationLimit=2):

        self.lattice = lattice
        self.truncationLimit = truncationLimit
        
        self.rec = 2 * jnp.pi * jnp.linalg.inv(self.lattice)
        self.area = jnp.abs(jnp.linalg.det(lattice))
        self.gamma = (2.8 / self.area**(1/2)) ** 2
        
        ordinals = sorted(range(-truncationLimit, truncationLimit + 1), key=abs)
        ordinals = jnp.array(list(itertools.product(ordinals, repeat=2)))
        
        self.lat_vectors = jnp.einsum('kj,ij->ik', self.lattice, ordinals)
        lat_vec_norm = jnp.linalg.norm(self.lat_vectors[1:], axis=-1)
        
        self.rec_vectors = jnp.einsum('jk,ij->ik', self.rec, ordinals[1:])
        self.rec_vec_square = jnp.einsum('ij,ij->i', self.rec_vectors, self.rec_vectors)
        self.rec_vec_norm = jnp.sqrt(self.rec_vec_square)
                
        self.madelung_const = (
            jnp.sum(jax.scipy.special.erfc(jnp.sqrt(self.gamma) * lat_vec_norm) / lat_vec_norm)
            - 2 * jnp.sqrt(self.gamma) / jnp.sqrt(jnp.pi)
            + (2 * jnp.pi / self.area) * jnp.sum(jax.scipy.special.erfc(self.rec_vec_norm / (2 * jnp.sqrt(self.gamma))) / self.rec_vec_norm)
            - 2 * jnp.sqrt(jnp.pi) / (self.area * jnp.sqrt(self.gamma))
        )
    
        self.batch_ewald_sum = jax.vmap(self.ewald_sum, in_axes=0)
    
    def real_space_ewald(self, separation: jnp.ndarray):
        disp = jnp.linalg.norm(separation - self.lat_vectors, axis=1)
        return jnp.sum(jax.scipy.special.erfc(jnp.sqrt(self.gamma) * disp) / disp)

    def recp_space_ewald(self, separation: jnp.ndarray):
        phase = jnp.dot(self.rec_vectors, separation)
        k_norm = jnp.sqrt(jnp.sum(self.rec_vectors**2, axis=1))
        factor = jax.scipy.special.erfc(k_norm / (2 * jnp.sqrt(self.gamma))) / k_norm
        return (2 * jnp.pi / self.area) * jnp.sum(jnp.cos(phase) * factor)

    def ewald_sum(self, separation: jnp.ndarray):
        return  self.real_space_ewald(separation) +\
                self.recp_space_ewald(separation) -\
                2 * jnp.sqrt(jnp.pi) / (self.area * jnp.sqrt(self.gamma))

    def configuration(self, parameters, rs):
        N = rs.shape[0]
        ee = rs[:, None, :] - rs[None, :, :]
        phase_ee = jnp.einsum('il,jkl->jki', self.rec / (2 * jnp.pi), ee)
        phase_prim_ee = phase_ee % 1
        prim_ee = jnp.einsum('il,jkl->jki', self.lattice, phase_prim_ee)
        prim_ee = prim_ee.reshape(-1, 2)
        vals = self.batch_ewald_sum(prim_ee).reshape(N,N)
        vals = vals.at[jnp.diag_indices(N)].set(0.0)
        return 0.5 * jnp.sum(vals) + 0.5 * N * self.madelung_const

class LocalEnergyUEG(LocalEnergy):
    def __init__(
        self,
        logWavefunction,
        lattice,
        truncationLimit=2,
        kinetic_chunks: int = 1,
        potential_chunks: int = 1,
    ):
        kinetic = LocalKineticEnergy(logWavefunction)
        potential = EwaldPotential(lattice, truncationLimit)

        self.kinetic = Batched(kinetic, n_chunks=kinetic_chunks)
        self.potential = Batched(potential, n_chunks=potential_chunks)

    def batch(self, parameters, walkerRs):
        ke = self.kinetic.batch(parameters, walkerRs)
        pe = self.potential.batch(parameters, walkerRs)
        return ke + pe

    def configuration(self, parameters, rs):
        return (
            self.kinetic.configuration(parameters, rs)
            + self.potential.configuration(parameters, rs)
        )

class LocalEnergyUEG2D(LocalEnergy):
    def __init__(
        self,
        logWavefunction,
        lattice,
        truncationLimit=2,
        kinetic_chunks: int = 1,
        potential_chunks: int = 1,
    ):
        kinetic = LocalKineticEnergy(logWavefunction)
        potential = EwaldPotential2D(lattice, truncationLimit)

        self.kinetic = Batched(kinetic, n_chunks=kinetic_chunks)
        self.potential = Batched(potential, n_chunks=potential_chunks)

    def batch(self, parameters, walkerRs):
        ke = self.kinetic.batch(parameters, walkerRs)
        pe = self.potential.batch(parameters, walkerRs)
        return ke + pe

    def configuration(self, parameters, rs):
        return (
            self.kinetic.configuration(parameters, rs)
            + self.potential.configuration(parameters, rs)
        )
