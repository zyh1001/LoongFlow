import numpy as np


# EVOLVE-BLOCK-START
def wavefunction(z1, z2):
    return (z1 - z2) * np.abs(z1 - z2)
# EVOLVE-BLOCK-END
