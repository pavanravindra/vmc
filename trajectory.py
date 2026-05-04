import jax
import jax.numpy as jnp
import numpy as np

import os, re


class MetropolisUpdater:

    def __init__(self, logWavefunction):
        self.logWavefunction = logWavefunction.apply
        self._update_walkers = jax.vmap(
            self._update_configuration,
            in_axes=(None, 0, 0, None)
        )

    def _update_configuration(
        self, parameters, rs1: jnp.ndarray, rng, tau: float
    ) -> jnp.ndarray:

        shiftRng, acceptRng = jax.random.split(rng, 2)

        randomShift = jax.random.normal(shiftRng, shape=rs1.shape) * jnp.sqrt(tau)
        rs2 = rs1 + randomShift

        probRatio = jnp.exp(
            2 * (self.logWavefunction(parameters, rs2) - self.logWavefunction(parameters, rs1))
        )
        acceptProb = jnp.minimum(1., probRatio)
        accept = jax.random.bernoulli(acceptRng, acceptProb)

        return jnp.where(accept, rs2, rs1)

    def update_batch(
        self, parameters, rs: jnp.ndarray, rng, tau: float
    ) -> jnp.ndarray:
        walkers = rs.shape[0]
        rngs = jax.random.split(rng, walkers)
        return self._update_walkers(parameters, rs, rngs, tau)


class MALAUpdater:

    def __init__(self, logWavefunction, r_ws: float, clipGradients: bool = True):
        self.logWavefunction = logWavefunction.apply
        self.r_ws = r_ws
        self.gradLogWavefunction = jax.grad(logWavefunction.apply, argnums=1)
        self.clipGradients = clipGradients
        self._update_walkers = jax.vmap(
            self._update_configuration,
            in_axes=(None, 0, 0, None)
        )

    def _update_configuration(
        self, parameters, rs1: jnp.ndarray, rng, tau: float
    ) -> jnp.ndarray:

        def clip(jnpArray):
            maxValue = self.r_ws / 25
            norms = jnp.linalg.norm(jnpArray, axis=-1, keepdims=True)
            scale = jnp.minimum(1.0, maxValue / (norms + 1e-12))
            return jnpArray * scale

        def proposalProb(Ri, Rf):
            localGrad = self.gradLogWavefunction(parameters, Ri)
            gradShift = localGrad * tau
            if self.clipGradients:
                gradShift = clip(gradShift)
            deviation = Rf - Ri - gradShift
            return jnp.exp(-jnp.sum(deviation ** 2) / (2 * tau))

        shiftRng, acceptRng = jax.random.split(rng, 2)

        localGrad = self.gradLogWavefunction(parameters, rs1)
        gradShift = localGrad * tau
        if self.clipGradients:
            gradShift = clip(gradShift)

        randomShift = jax.random.normal(shiftRng, shape=rs1.shape) * jnp.sqrt(tau)
        rs2 = rs1 + gradShift + randomShift

        proposalRatio = proposalProb(rs2, rs1) / proposalProb(rs1, rs2)
        probRatio = jnp.exp(
            2. * (self.logWavefunction(parameters, rs2) - self.logWavefunction(parameters, rs1))
        )
        acceptProb = jnp.minimum(1., proposalRatio * probRatio)
        accept = jax.random.bernoulli(acceptRng, acceptProb)

        return jnp.where(accept, rs2, rs1)

    def update_batch(
        self, parameters, rs: jnp.ndarray, rng, tau: float
    ) -> jnp.ndarray:
        walkers = rs.shape[0]
        rngs = jax.random.split(rng, walkers)
        return self._update_walkers(parameters, rs, rngs, tau)


def generate_bcc(spins: tuple, lattice: jnp.ndarray, dim: int) -> jnp.ndarray:
    """
    Returns the lattice positions of the BCC-like Wigner crystal phase in 3D.

    Args:
        spins: Tuple (n_up, n_down).
        lattice: (dim, dim) matrix where rows are lattice vectors.
        dim: must be 3 for BCC-like lattice.
    """
    if dim != 3:
        raise Exception("BCC-like crystal phase is a 3D phase.")

    (NUp, NDown) = spins

    numLatticePoints = int(jnp.ceil(jnp.maximum(NUp, NDown) ** (1 / 3)))

    points = jnp.linspace(0, 1, numLatticePoints, endpoint=False)

    grids = jnp.meshgrid(*([points] * dim), indexing="ij")
    frac_up = jnp.stack(grids, axis=-1).reshape(-1, dim)

    frac_shift = (1.0 / numLatticePoints) / 2.0
    frac_down = (frac_up + frac_shift) % 1.0

    upPositions = frac_up @ lattice
    downPositions = frac_down @ lattice

    coordinates = jnp.concatenate([
        upPositions[:NUp], downPositions[:NDown]
    ], axis=0)

    return coordinates


def generate_striped_afm(
    spins: tuple, lattice: jnp.ndarray, dim: int, gridShape=None
) -> jnp.ndarray:
    """
    Returns the lattice positions of the striped anti-ferromagnetic (AFM)
    Wigner crystal phase in 2D.

    Args:
        spins: Tuple (n_up, n_down).
        lattice: (dim, dim) matrix where rows are lattice vectors.
        dim: must be 2 for striped AFM lattice.
        gridShape: specifies number of grid points along each dimension.
    """
    NUp = spins[0]
    NDown = spins[1]

    if dim != 2:
        raise Exception("Striped AFM phase is a 2D phase.")

    if gridShape is None:
        n_side = int(jnp.ceil(jnp.sqrt(jnp.maximum(NUp, NDown))))
        n_side_x = n_side
        n_side_y = n_side
    else:
        (n_side_x, n_side_y) = gridShape

    x_points = jnp.linspace(0, 1, n_side_x, endpoint=False)
    y_points = jnp.linspace(0, 1, n_side_y, endpoint=False)

    xx, yy = jnp.meshgrid(x_points, y_points, indexing='ij')
    frac_up = jnp.stack([xx, yy], axis=-1).reshape(-1, 2)

    dx = 1.0 / n_side_x
    dy = 1.0 / n_side_y

    shift_vec = jnp.array([dx / 2.0, dy / 2.0])

    frac_down = (frac_up + shift_vec) % 1.0

    upPositions = frac_up @ lattice
    downPositions = frac_down @ lattice

    coordinates = jnp.concatenate([
        upPositions[:NUp], downPositions[:NDown]
    ], axis=0)

    return coordinates


def acceptance_array(rs1: jnp.ndarray, rs2: jnp.ndarray) -> jnp.ndarray:
    """
    Returns a boolean array indicating which walkers moved to new positions.

    Args:
        rs1, rs2: Walker positions with shape (W, N, D).
    """
    matchBools = rs1 == rs2
    match = jnp.all(matchBools, axis=(1, 2))
    return ~match


def acceptance_rate(rs1: jnp.ndarray, rs2: jnp.ndarray) -> float:
    """
    Returns the Metropolis acceptance rate.

    Args:
        rs1, rs2: Walker positions with shape (W, N, D).
    """
    matchBools = rs1 == rs2
    match = jnp.all(matchBools, axis=(1, 2))
    return 1 - jnp.average(match)


def blocking_analysis(energies: np.ndarray, neql: int = 0, printQ=False, writeQ=False):
    weights = np.ones(energies.shape)
    nSamples = weights.shape[0] - neql
    weights = weights[neql:]
    energies = energies[neql:]
    weightedEnergies = np.multiply(weights, energies)
    meanEnergy = weightedEnergies.sum() / weights.sum()
    outputString = ""
    outputString += f"#\n# Mean: {meanEnergy:.8e}\n"
    outputString += "# Block size    # of blocks         Mean                Error\n"
    blockSizes = np.array([1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500, 1000, 10000])
    prevError = 0.0
    plateauError = None
    for i in blockSizes[blockSizes < nSamples / 2.0]:
        nBlocks = nSamples // i
        blockedWeights = np.zeros(nBlocks)
        blockedEnergies = np.zeros(nBlocks)
        for j in range(nBlocks):
            blockedWeights[j] = weights[j * i: (j + 1) * i].sum()
            blockedEnergies[j] = (
                weightedEnergies[j * i: (j + 1) * i].sum() / blockedWeights[j]
            )
        v1 = blockedWeights.sum()
        v2 = (blockedWeights ** 2).sum()
        mean = np.multiply(blockedWeights, blockedEnergies).sum() / v1
        error = (
            np.multiply(blockedWeights, (blockedEnergies - mean) ** 2).sum()
            / (v1 - v2 / v1)
            / (nBlocks - 1)
        ) ** 0.5
        outputString += f"  {i:5d}           {nBlocks:6d}       {mean:.8e}       {error:.6e}\n"
        if error < 1.05 * prevError and plateauError is None:
            plateauError = max(error, prevError)
        prevError = error

    if plateauError is not None:
        outputString += f"# Stocahstic error estimate: {plateauError:.6e}\n#\n"
    if printQ:
        print(outputString)
    if writeQ:
        file = open(writeQ, "w")
        file.write(outputString)
        file.close()

    return meanEnergy, plateauError


def get_statistics(path: str):

    (mean, error) = (None, None)

    if os.path.isfile(path):
        with open(path, 'r') as f:
            for line in f:
                if line.startswith("# Mean:"):
                    match = re.search(r"# Mean:\s+([-+eE0-9\.]+)", line)
                    if match:
                        mean = float(match.group(1))
                elif "Stocahstic error estimate" in line:
                    match = re.search(r"Stocahstic error estimate:\s+([-+eE0-9\.]+)", line)
                    if match:
                        error = float(match.group(1))

    return (mean, error)