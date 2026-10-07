import numpy as np


# EVOLVE-BLOCK-START
def wavefunction(z1, z2, c0_re=1.0, c0_im=0.0):
    return (c0_re + 1j * c0_im) * (z1 - z2)
# EVOLVE-BLOCK-END
