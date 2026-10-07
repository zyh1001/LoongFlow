import numpy as np


# EVOLVE-BLOCK-START
def wavefunction(COORDINATES, kapa, alpha=1.0):
    return PAIR_PRODUCT * np.exp(-alpha * RADIUS_SUM / (1 + kapa))
# EVOLVE-BLOCK-END
