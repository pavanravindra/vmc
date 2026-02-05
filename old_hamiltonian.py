import jax
import jax.numpy as jnp

import itertools

class LocalEnergy:
    
    def batch(self, parameters, walkerRs):
        batchFunction = jax.vmap(self.configuration, in_axes=(None,0))
        return batchFunction(parameters, walkerRs)

def laplacian(logWavefunction, parameters, rs):
    """
    Computes the Laplacian of logWavefunction w.r.t. rs in a memory-efficient
    way. Uses forward-over-reverse AD (jax.jvp) and jax.lax.scan to accumulate
    the sum over coordinates without blowing up memory.
    """
    def f_rs(rs):
        return logWavefunction(parameters, rs)

    grad_fn = jax.grad(f_rs)
    rsFlat = rs.reshape(-1)
    numCoords = rsFlat.size

    def body(carry, i):
        e_i = jnp.eye(numCoords)[i].reshape(rs.shape)
        secondDerivative = jax.jvp(grad_fn, (rs,), (e_i,))[1].reshape(-1)[i]
        lap = carry + secondDerivative
        return ( lap , None )

    lap, _ = jax.lax.scan(body, 0.0, jnp.arange(numCoords))
    
    return lap

class LocalKineticEnergy(LocalEnergy):
    """
    Computes the local kinetic energy of an arbitrary log wavefunction.
    """

    def __init__(self, logWavefunction):
        self.logWavefunction = logWavefunction.apply

    def configuration(self, parameters, rs):

        grad2f = laplacian(self.logWavefunction, parameters, rs)

        gradFunction = jax.grad(self.logWavefunction, argnums=1)
        grad = gradFunction(parameters, rs)  # shape (N, 3)
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

    def __init__(self, logWavefunction, lattice, truncationLimit=2):
        self.kineticEnergy = LocalKineticEnergy(logWavefunction)
        self.potentialEnergy = EwaldPotential(lattice, truncationLimit)

    def configuration(self, parameters, rs):
        kineticEnergy = self.kineticEnergy.configuration(parameters, rs)
        potentialEnergy = self.potentialEnergy.configuration(parameters, rs)
        return kineticEnergy + potentialEnergy

class LocalEnergyUEG2D(LocalEnergy):

    def __init__(self, logWavefunction, lattice, truncationLimit=2):
        self.kineticEnergy = LocalKineticEnergy(logWavefunction)
        self.potentialEnergy = EwaldPotential2D(lattice, truncationLimit)

    def configuration(self, parameters, rs):
        kineticEnergy = self.kineticEnergy.configuration(parameters, rs)
        potentialEnergy = self.potentialEnergy.configuration(parameters, rs)
        return kineticEnergy + potentialEnergy
        