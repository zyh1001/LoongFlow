import numpy as np


# EVOLVE-BLOCK-START
def wavefunction(COORDINATES, alpha=1.0):
    return PAIR_PRODUCT * np.exp(-alpha * RADIUS_SUM)
# EVOLVE-BLOCK-END
