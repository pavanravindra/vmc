import sys, os

sys.path.append("/burg-archive/ccce/users/phr2114/vmc")
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
#os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core.frozen_dict import freeze, unfreeze
import numpy as np

import wavefunctions, folx_hamiltonian, trajectory, optimization

class DynamicWavefunction(wavefunctions.Wavefunction):
    """
    TODO
    """
    spins : (int,int)
    dim : int
    lattice : jnp.ndarray
    kpoints : jnp.ndarray
    hiddenFeatures : int

    def setup(self):

        self.slaterUp = wavefunctions.LogSimpleSlater(self.spins[0], self.dim, self.kpoints)
        self.slaterDown = wavefunctions.LogSimpleSlater(self.spins[1], self.dim, self.kpoints)
        self.CYJastrow = wavefunctions.LogCYJastrow(self.spins, self.lattice)

        self.weights1_nb = nn.Dense(self.hiddenFeatures)
        self.weights2_nb = nn.Dense(1)

        self.linear1_nb = nn.Dense(self.hiddenFeatures)
        self.linear2_nb = nn.Dense(self.hiddenFeatures)

        self.jastrow1_nb = nn.Dense(self.hiddenFeatures)
        self.jastrow2_nb = nn.Dense(
            1, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros
        )

        self.backflow1_nb = nn.Dense(self.hiddenFeatures)
        self.backflow2_nb = nn.Dense(
            self.dim, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros
        )

    def __call__(self, rs):

        N = sum(self.spins)

        eye_mask = jnp.eye(N, dtype=bool)[:,:,None]
        v_ij = wavefunctions.generateFeatures(rs, self.spins, self.lattice)   # (N,N,2*dim+2)

        g_ij = nn.sigmoid(
            self.weights2_nb(nn.swish(self.weights1_nb(v_ij)))
        )                                                       # (N,N,1)
        h_ij = self.linear2_nb(nn.swish(self.linear1_nb(v_ij))) # (N,N,d)

        h_i = jnp.sum(
            jnp.where(~eye_mask, g_ij * h_ij, 0.0), axis=1
        ) / N                                                   # (N,d)

        jastrow_nb = jnp.sum(
            self.jastrow2_nb(nn.swish(self.jastrow1_nb(h_i)))
        ) / N

        backflow_nb = self.backflow2_nb(nn.swish(self.backflow1_nb(h_i))) # (N,dim)

        xs = rs + backflow_nb

        CYJastrow = self.CYJastrow(rs)
        slaterUp = self.slaterUp(xs[:self.spins[0],:])
        slaterDown = self.slaterDown(xs[self.spins[0]:,:])

        return slaterUp + slaterDown + CYJastrow + jastrow_nb

class DynamicStaticWavefunction(wavefunctions.Wavefunction):
    """
    TODO
    """
    spins : (int,int)
    dim : int
    lattice : jnp.ndarray
    kpoints : jnp.ndarray
    upCoeffs : jnp.ndarray
    downCoeffs : jnp.ndarray
    hiddenFeatures : int

    def setup(self):

        self.slaterUp = wavefunctions.LogMPSlater(
            self.spins[0], self.dim, self.kpoints, self.upCoeffs
        )
        self.slaterDown = wavefunctions.LogMPSlater(
            self.spins[1], self.dim, self.kpoints, self.downCoeffs
        )
        self.CYJastrow = wavefunctions.LogCYJastrow(self.spins, self.lattice)

        self.weights1_nb = nn.Dense(self.hiddenFeatures)
        self.weights2_nb = nn.Dense(1)

        self.linear1_nb = nn.Dense(self.hiddenFeatures)
        self.linear2_nb = nn.Dense(self.hiddenFeatures)

        self.jastrow1_nb = nn.Dense(self.hiddenFeatures)
        self.jastrow2_nb = nn.Dense(
            1, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros
        )

        self.backflow1_nb = nn.Dense(self.hiddenFeatures)
        self.backflow2_nb = nn.Dense(
            self.dim, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros
        )

    def __call__(self, rs):

        N = sum(self.spins)

        eye_mask = jnp.eye(N, dtype=bool)[:,:,None]
        v_ij = wavefunctions.generateFeatures(rs, self.spins, self.lattice)   # (N,N,2*dim+2)

        g_ij = nn.sigmoid(
            self.weights2_nb(nn.swish(self.weights1_nb(v_ij)))
        )                                                       # (N,N,1)
        h_ij = self.linear2_nb(nn.swish(self.linear1_nb(v_ij))) # (N,N,d)

        h_i = jnp.sum(
            jnp.where(~eye_mask, g_ij * h_ij, 0.0), axis=1
        ) / N                                                   # (N,d)

        jastrow_nb = jnp.sum(
            self.jastrow2_nb(nn.swish(self.jastrow1_nb(h_i)))
        ) / N

        backflow_nb = self.backflow2_nb(nn.swish(self.backflow1_nb(h_i))) # (N,dim)

        xs = rs + backflow_nb

        CYJastrow = self.CYJastrow(rs)
        slaterUp = self.slaterUp(xs[:self.spins[0],:])
        slaterDown = self.slaterDown(xs[self.spins[0]:,:])

        return slaterUp + slaterDown + CYJastrow + jastrow_nb

class DynamicRoundsWavefunction(wavefunctions.Wavefunction):
    """
    T-round neural-backflow wavefunction.

    For t = 1, this reduces exactly to the original one-layer architecture.

    For t >= 2, each additional round builds new pair features from:
        v_ij,
        h_i^(t-1) - h_j^(t-1),
        h_i^(t-1) * h_j^(t-1)

    Naming convention is chosen so that:
        T=1 uses the original parameter names
        T=2 matches the explicit two-layer version exactly

    In particular:
        round 1: weights1_nb, weights2_nb, ...
        round 2: weights1_nb_2, weights2_nb_2, ...
        round 3: weights1_nb_3, weights2_nb_3, ...
        etc.
    """
    spins: tuple[int, int]
    dim: int
    lattice: jnp.ndarray
    kpoints: jnp.ndarray
    hiddenFeatures: int
    T: int = 1

    def setup(self):

        if self.T < 1:
            raise ValueError(f"T must be >= 1, got T={self.T}")

        self.slaterUp = wavefunctions.LogSimpleSlater(
            self.spins[0], self.dim, self.kpoints
        )
        self.slaterDown = wavefunctions.LogSimpleSlater(
            self.spins[1], self.dim, self.kpoints
        )
        self.CYJastrow = wavefunctions.LogCYJastrow(self.spins, self.lattice)

        for t in range(1, self.T + 1):

            suffix = "" if t == 1 else f"_{t}"

            setattr(self, f"weights1_nb{suffix}", nn.Dense(self.hiddenFeatures))
            setattr(self, f"weights2_nb{suffix}", nn.Dense(1))

            setattr(self, f"linear1_nb{suffix}", nn.Dense(self.hiddenFeatures))
            setattr(self, f"linear2_nb{suffix}", nn.Dense(self.hiddenFeatures))

            setattr(self, f"jastrow1_nb{suffix}", nn.Dense(self.hiddenFeatures))
            setattr(
                self,
                f"jastrow2_nb{suffix}",
                nn.Dense(
                    1,
                    kernel_init=nn.initializers.zeros,
                    bias_init=nn.initializers.zeros,
                ),
            )

            setattr(self, f"backflow1_nb{suffix}", nn.Dense(self.hiddenFeatures))
            setattr(
                self,
                f"backflow2_nb{suffix}",
                nn.Dense(
                    self.dim,
                    kernel_init=nn.initializers.zeros,
                    bias_init=nn.initializers.zeros,
                ),
            )

    def __call__(self, rs):

        N = sum(self.spins)

        eye_mask = jnp.eye(N, dtype=bool)[:, :, None]
        v_ij = wavefunctions.generateFeatures(rs, self.spins, self.lattice)  # (N,N,2*dim+2)

        total_jastrow_nb = 0.0
        total_backflow_nb = jnp.zeros_like(rs)

        h_i_prev = None

        for t in range(1, self.T + 1):

            suffix = "" if t == 1 else f"_{t}"

            weights1 = getattr(self, f"weights1_nb{suffix}")
            weights2 = getattr(self, f"weights2_nb{suffix}")
            linear1 = getattr(self, f"linear1_nb{suffix}")
            linear2 = getattr(self, f"linear2_nb{suffix}")
            jastrow1 = getattr(self, f"jastrow1_nb{suffix}")
            jastrow2 = getattr(self, f"jastrow2_nb{suffix}")
            backflow1 = getattr(self, f"backflow1_nb{suffix}")
            backflow2 = getattr(self, f"backflow2_nb{suffix}")

            if t == 1:
                x_ij = v_ij
            else:
                h_i_row = h_i_prev[:, None, :]   # (N,1,hiddenFeatures)
                h_i_col = h_i_prev[None, :, :]   # (1,N,hiddenFeatures)

                dh_ij = h_i_row - h_i_col        # (N,N,hiddenFeatures)
                ph_ij = h_i_row * h_i_col        # (N,N,hiddenFeatures)

                x_ij = jnp.concatenate([v_ij, dh_ij, ph_ij], axis=-1)

            g_ij = nn.sigmoid(
                weights2(nn.swish(weights1(x_ij)))
            )  # (N,N,1)

            h_ij = linear2(
                nn.swish(linear1(x_ij))
            )  # (N,N,hiddenFeatures)

            h_i = jnp.sum(
                jnp.where(~eye_mask, g_ij * h_ij, 0.0), axis=1
            ) / N  # (N,hiddenFeatures)

            jastrow_nb_t = jnp.sum(
                jastrow2(nn.swish(jastrow1(h_i)))
            ) / N

            backflow_nb_t = backflow2(
                nn.swish(backflow1(h_i))
            )  # (N,dim)

            total_jastrow_nb = total_jastrow_nb + jastrow_nb_t
            total_backflow_nb = total_backflow_nb + backflow_nb_t

            h_i_prev = h_i

        xs = rs + total_backflow_nb

        CYJastrow = self.CYJastrow(rs)
        slaterUp = self.slaterUp(xs[:self.spins[0], :])
        slaterDown = self.slaterDown(xs[self.spins[0]:, :])

        return slaterUp + slaterDown + CYJastrow + total_jastrow_nb
