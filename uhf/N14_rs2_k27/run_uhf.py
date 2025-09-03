import sys

sys.path.append("/burg/ccce/users/phr2114/vmc")

import numpy as np

import wavefunctions, hamiltonian, trajectory, optimization

###############################################################################
#   Settings for whole simulation                                             #
###############################################################################

numTrials = 20

N = 14
r_ws = 2
numKpts = 27

NUp = N // 2
NDown = N - NUp
spins = ( NUp , NDown )

###############################################################################
#   Creating wavefunction, walker updater, optimizer, and local energy        #
###############################################################################

bestEnergy = np.inf
bestKpoints = None
bestUpCoeff = None
bestDownCoeff = None

for _ in range(numTrials):

    ( energy , kpoints , upCoeff , downCoeff ) = wavefunctions.uhfInitialization(
        r_ws, spins, numKpts
    )

    print("ENERGY: " + str(energy))

    if energy < bestEnergy:
        bestEnergy = energy
        bestKpoints = kpoints
        bestUpCoeff = upCoeff
        bestDownCoeff = downCoeff

###############################################################################
#   Write relevant results out to files                                       #
###############################################################################

np.save("energy", bestEnergy)
np.save("kpoints", bestKpoints)
np.save("upCoeff", bestUpCoeff)
np.save("downCoeff", bestDownCoeff)
