import numpy as np


# EVOLVE-BLOCK-START
def wavefunction(z1, z2, a=0.25, b=0.25, c=0.0, d=0.0):
    return (z1 - z2)**3 * (1 + c * np.abs(z1 - z2)**2 + d * np.abs(z1 - z2)**4) * np.exp(-a * np.abs(z1 - z2)**2 - b * np.abs(z1 + z2)**2)
# EVOLVE-BLOCK-END