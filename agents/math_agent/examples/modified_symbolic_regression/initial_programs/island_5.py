import numpy as np


# EVOLVE-BLOCK-START
def wavefunction(z1, z2, alpha=1.0):
    return (z1 - z2) * np.exp(-alpha * (np.abs(z1) ** 2 + np.abs(z2) ** 2))
# EVOLVE-BLOCK-END
