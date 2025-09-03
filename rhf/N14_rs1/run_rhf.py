import sys

sys.path.append("/burg/ccce/users/phr2114/vmc")

import numpy as np
from pyscf import gto, scf, ao2mo, cc

import wavefunctions, hamiltonian, trajectory, optimization, qc

###############################################################################
#   Settings for whole simulation                                             #
###############################################################################

numTrials = 20

N = 14
r_ws = 1
numKpts = 27

NUp = N // 2
NDown = N - NUp
spins = ( NUp , NDown )

###############################################################################
#   Creating wavefunction, walker updater, optimizer, and local energy        #
###############################################################################

system = qc.ueg_qc(r_ws, spins, numkPoints=numKpts)
kpoints = system.get_k_points()
numkPoints = kpoints.shape[0]
h1 = system.get_h1_real(kpoints)
eri = system.get_eri_tensor_real(kpoints)

mol = gto.M(verbose=0)
mol.nelectron = system.n_particles
mol.incore_anyway = True
mol.energy_nuc = lambda *args: 0.0
mol.verbose = 3
mf = scf.RHF(mol)
mf.get_hcore = lambda *args: h1
mf.get_ovlp = lambda *args: np.eye(numKpts)
mf._eri = ao2mo.restore(8, np.double(eri), numKpts)
mf.init_guess = "1e"
energy = mf.kernel()
print("ENERGY: " + str(energy))


###############################################################################
#   Write relevant results out to files                                       #
###############################################################################

np.save("energy", energy)
