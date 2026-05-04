import jax
import jax.numpy as jnp

import folx

import itertools


class LocalEnergy:

    def batch(self, parameters, walkerRs: jnp.ndarray) -> jnp.ndarray:
        batchFunction = folx.batched_vmap(
            self.configuration, in_axes=(None, 0), max_batch_size=64
        )
        return batchFunction(parameters, walkerRs)


class LocalKineticEnergy(LocalEnergy):
    """
    Local kinetic energy via folx forward-mode Laplacian computation.

    Evaluates:
        T_local = -½ ( ||∇ log ψ||² + ∇² log ψ )
    """

    def __init__(self, logWavefunction, sparsity_threshold: int):
        self.logWavefunction = logWavefunction.apply
        self.sparsity_threshold = sparsity_threshold

    def configuration(self, parameters, rs: jnp.ndarray) -> jnp.ndarray:

        def f_rs(x):
            return self.logWavefunction(parameters, x)

        folx_function = folx.forward_laplacian(
            f_rs, sparsity_threshold=self.sparsity_threshold
        )
        folx_output = folx_function(rs)

        lap = folx_output.laplacian
        jacobian = folx_output[1].dense_array
        gradf2 = jnp.sum(jacobian ** 2)

        kineticEnergy = -0.5 * (gradf2 + lap)

        return kineticEnergy


class EwaldPotential(LocalEnergy):
    """
    Ewald summation for the electron-electron Coulomb energy in 3D periodic cells.

    Based on `make_ewald_potential` from FermiNet.
    """

    def __init__(self, lattice: jnp.ndarray, truncationLimit: int = 2):

        self.lattice = lattice
        self.truncationLimit = truncationLimit

        self.rec = 2 * jnp.pi * jnp.linalg.inv(self.lattice)
        self.volume = jnp.abs(jnp.linalg.det(lattice))
        self.gamma = (2.8 / self.volume ** (1 / 3)) ** 2

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

    def real_space_ewald(self, separation: jnp.ndarray) -> jnp.ndarray:
        disp = jnp.linalg.norm(separation - self.lat_vectors, axis=1)
        return jnp.sum(jax.scipy.special.erfc(jnp.sqrt(self.gamma) * disp) / disp)

    def recp_space_ewald(self, separation: jnp.ndarray) -> jnp.ndarray:
        phase = jnp.dot(self.rec_vectors, separation)
        factor = jnp.exp(-self.rec_vec_square / (4 * self.gamma)) / self.rec_vec_square
        return (4 * jnp.pi / self.volume) * jnp.sum(jnp.cos(phase) * factor)

    def ewald_sum(self, separation: jnp.ndarray) -> jnp.ndarray:
        return (self.real_space_ewald(separation)
                + self.recp_space_ewald(separation)
                - jnp.pi / (self.volume * self.gamma))

    def configuration(self, parameters, rs: jnp.ndarray) -> jnp.ndarray:
        N = rs.shape[0]
        ee = rs[:, None, :] - rs[None, :, :]
        phase_ee = jnp.einsum('il,jkl->jki', self.rec / (2 * jnp.pi), ee)
        phase_prim_ee = phase_ee % 1
        prim_ee = jnp.einsum('il,jkl->jki', self.lattice, phase_prim_ee)
        prim_ee = prim_ee.reshape(-1, 3)
        vals = self.batch_ewald_sum(prim_ee).reshape(N, N)
        vals = vals.at[jnp.diag_indices(N)].set(0.0)
        return 0.5 * jnp.sum(vals) + 0.5 * N * self.madelung_const


class EwaldPotential2D(LocalEnergy):
    """
    Ewald summation for the electron-electron Coulomb energy in 2D periodic cells.

    Based on `make_ewald_potential` from FermiNet.
    """

    def __init__(self, lattice: jnp.ndarray, truncationLimit: int = 2):

        self.lattice = lattice
        self.truncationLimit = truncationLimit

        self.rec = 2 * jnp.pi * jnp.linalg.inv(self.lattice)
        self.area = jnp.abs(jnp.linalg.det(lattice))
        self.gamma = (2.8 / self.area ** (1 / 2)) ** 2

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

    def real_space_ewald(self, separation: jnp.ndarray) -> jnp.ndarray:
        disp = jnp.linalg.norm(separation - self.lat_vectors, axis=1)
        return jnp.sum(jax.scipy.special.erfc(jnp.sqrt(self.gamma) * disp) / disp)

    def recp_space_ewald(self, separation: jnp.ndarray) -> jnp.ndarray:
        phase = jnp.dot(self.rec_vectors, separation)
        k_norm = jnp.sqrt(jnp.sum(self.rec_vectors ** 2, axis=1))
        factor = jax.scipy.special.erfc(k_norm / (2 * jnp.sqrt(self.gamma))) / k_norm
        return (2 * jnp.pi / self.area) * jnp.sum(jnp.cos(phase) * factor)

    def ewald_sum(self, separation: jnp.ndarray) -> jnp.ndarray:
        return (self.real_space_ewald(separation)
                + self.recp_space_ewald(separation)
                - 2 * jnp.sqrt(jnp.pi) / (self.area * jnp.sqrt(self.gamma)))

    def configuration(self, parameters, rs: jnp.ndarray) -> jnp.ndarray:
        N = rs.shape[0]
        ee = rs[:, None, :] - rs[None, :, :]
        phase_ee = jnp.einsum('il,jkl->jki', self.rec / (2 * jnp.pi), ee)
        phase_prim_ee = phase_ee % 1
        prim_ee = jnp.einsum('il,jkl->jki', self.lattice, phase_prim_ee)
        prim_ee = prim_ee.reshape(-1, 2)
        vals = self.batch_ewald_sum(prim_ee).reshape(N, N)
        vals = vals.at[jnp.diag_indices(N)].set(0.0)
        return 0.5 * jnp.sum(vals) + 0.5 * N * self.madelung_const


class LocalEnergyUEG3D(LocalEnergy):

    def __init__(
        self, logWavefunction, lattice: jnp.ndarray,
        sparsity_threshold: int, truncationLimit: int = 2
    ):
        self.kineticEnergy = LocalKineticEnergy(logWavefunction, sparsity_threshold)
        self.potentialEnergy = EwaldPotential(lattice, truncationLimit)

    def configuration(self, parameters, rs: jnp.ndarray) -> jnp.ndarray:
        kineticEnergy = self.kineticEnergy.configuration(parameters, rs)
        potentialEnergy = self.potentialEnergy.configuration(parameters, rs)
        return kineticEnergy + potentialEnergy


class LocalEnergyUEG2D(LocalEnergy):

    def __init__(
        self, logWavefunction, lattice: jnp.ndarray,
        sparsity_threshold: int, truncationLimit: int = 2
    ):
        self.kineticEnergy = LocalKineticEnergy(logWavefunction, sparsity_threshold)
        self.potentialEnergy = EwaldPotential2D(lattice, truncationLimit)

    def configuration(self, parameters, rs: jnp.ndarray) -> jnp.ndarray:
        kineticEnergy = self.kineticEnergy.configuration(parameters, rs)
        potentialEnergy = self.potentialEnergy.configuration(parameters, rs)
        return kineticEnergy + potentialEnergy