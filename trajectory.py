import jax
import jax.numpy as jnp
import numpy as np

import os, re

class MetropolisUpdater:

    def __init__(self, logWavefunction):
        self.logWavefunction = logWavefunction.apply
        self.updateWalkers = jax.vmap(
            self.updateConfiguration,
            in_axes=(None,0,0,None)
        )

    def updateConfiguration(self, parameters, rs1, rng, tau):
                
        shiftRng, acceptRng = jax.random.split(rng, 2)
        
        randomShift = jax.random.normal(shiftRng, shape=rs1.shape) * jnp.sqrt(tau)
        rs2 = rs1 + randomShift
        
        probRatio = jnp.exp(
            2 * (self.logWavefunction(parameters,rs2) - self.logWavefunction(parameters,rs1))
        )
        acceptProb = jnp.minimum(1., probRatio)
        accept = jax.random.bernoulli(acceptRng, acceptProb)
        
        return jnp.where(accept, rs2, rs1)

    def updateBatch(self, parameters, rs, rng, tau):
        walkers = rs.shape[0]
        rngs = jax.random.split(rng, walkers)
        return self.updateWalkers(parameters, rs, rngs, tau)

class MALAUpdater:

    def __init__(self, logWavefunction, r_ws, clipGradients=True):
        self.logWavefunction = logWavefunction.apply
        self.r_ws = r_ws
        self.gradLogWavefunction = jax.grad(logWavefunction.apply, argnums=1)
        self.clipGradients = clipGradients
        self.updateWalkers = jax.vmap(
            self.updateConfiguration,
            in_axes=(None,0,0,None)
        )

    def updateConfiguration(self, parameters, rs1, rng, tau):

        def clip(jnpArray):
            # Clips elements of `jnpArray` to have absolute value <= r_ws / 10
            maxValue = self.r_ws / 10
            mask = jnp.abs(jnpArray) > maxValue
            return jnp.where(mask, maxValue * jnp.sign(jnpArray), jnpArray)
    
        def proposalProb(Ri, Rf):
            # Returns (unnormalized) proposal probability P(Rf | Ri)
            localGrad = self.gradLogWavefunction(parameters,Ri)
            gradShift = localGrad * tau
            if self.clipGradients:
                gradShift = clip(gradShift)
            deviation = Rf - Ri - gradShift
            return jnp.exp(-jnp.sum(deviation**2) / (2 * tau))
                
        shiftRng, acceptRng = jax.random.split(rng, 2)

        localGrad = self.gradLogWavefunction(parameters, rs1)
        gradShift = localGrad * tau
        if self.clipGradients:
            gradShift = clip(gradShift)
        
        randomShift = jax.random.normal(shiftRng, shape=rs1.shape) * jnp.sqrt(tau)
        rs2 = rs1 + gradShift + randomShift

        proposalRatio = proposalProb(rs2,rs1) / proposalProb(rs1,rs2)
        probRatio = jnp.exp(
            2. * (self.logWavefunction(parameters,rs2) - self.logWavefunction(parameters,rs1))
        )
        acceptProb = jnp.minimum(1., proposalRatio * probRatio)
        accept = jax.random.bernoulli(acceptRng, acceptProb)
        
        return jnp.where(accept, rs2, rs1)

    def updateBatch(self, parameters, rs, rng, tau):
        walkers = rs.shape[0]
        rngs = jax.random.split(rng, walkers)
        return self.updateWalkers(parameters, rs, rngs, tau)

def wignerCrystal(spins, lattice, walkers, rng, dim=3, gridShape=None):
    """
    Generates a Wigner Crystal initial guess for arbitrary unit cells.
    
    Args:
        spins: Tuple (n_up, n_down)
        lattice: (dim, dim) matrix where rows are lattice vectors.
                 e.g. [[Lx, 0], [Tx, Ty]]
        walkers: Number of walkers to generate
        rng: JAX random key
        dim: 2 or 3
        gridShape: specifies number of grid points along each dimension
    """
    
    NUp = spins[0]
    NDown = spins[1]

    if dim == 2:
        
        # STRIPED TRIANGULAR PHASE

        if gridShape is None:
            n_side = int(jnp.ceil(jnp.sqrt(jnp.maximum(NUp, NDown))))
            n_side_x = n_side
            n_side_y = n_side
        else:
            ( n_side_x , n_side_y ) = gridShape
            
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
        
        singleWalker = jnp.concatenate([
            upPositions[:NUp], downPositions[:NDown]
        ], axis=0)

    elif dim == 3:
        
        # BCC-LIKE LATTICE
        
        numLatticePoints = int(jnp.ceil(jnp.maximum(NUp, NDown) ** (1/3)))
        
        points = jnp.linspace(0, 1, numLatticePoints, endpoint=False)
        
        grids = jnp.meshgrid(*([points]*dim), indexing="ij")
        frac_up = jnp.stack(grids, axis=-1).reshape(-1, dim)
        
        frac_shift = (1.0 / numLatticePoints) / 2.0
        frac_down = (frac_up + frac_shift) % 1.0
        
        upPositions = frac_up @ lattice
        downPositions = frac_down @ lattice
        
        singleWalker = jnp.concatenate([
            upPositions[:NUp], downPositions[:NDown]
        ], axis=0)

    else:
        raise Exception("Only dim=2 or dim=3 are supported.")

    allWalkers = jnp.broadcast_to(
        singleWalker, (walkers,) + singleWalker.shape
    )
    
    return allWalkers

def acceptanceArray(rs1, rs2):
    """
    Returns an array that shows which walkers have been updated to new
    positions (True) and which have not (False).
    
    The inputs should be lists of electron positions with shape (W,N,D), where
    W is the number of walkers, N is the number of electrons, D is the number
    of dimensions.
    """
    matchBools = rs1 == rs2
    match = jnp.all(matchBools, axis=(1,2))
    return ~match

def acceptanceRate(rs1, rs2):
    """
    Returns the Metropolis acceptance rate given the before and after electron
    positions.
    
    The inputs should be lists of electron positions with shape (W,N,D), where
    W is the number of walkers, N is the number of electrons, D is the number
    of dimensions.
    """
    matchBools = rs1 == rs2
    match = jnp.all(matchBools, axis=(1,2))
    return 1 - jnp.average(match)

def blockingAnalysis(energies, neql=0, printQ=False, writeQ=False):
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
            blockedWeights[j] = weights[j * i : (j + 1) * i].sum()
            blockedEnergies[j] = (
                weightedEnergies[j * i : (j + 1) * i].sum() / blockedWeights[j]
            )
        v1 = blockedWeights.sum()
        v2 = (blockedWeights**2).sum()
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

def getStatistics(path):
    
    (mean,error) = (None,None)
    
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

    return (mean,error)
