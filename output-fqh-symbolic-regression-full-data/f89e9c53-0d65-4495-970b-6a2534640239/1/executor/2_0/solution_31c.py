import numpy as np


# EVOLVE-BLOCK-START
def wavefunction(z1, z2, g=0.25, h=0.25, a0=1.0, a1=0.0, a2=0.0,
                 b0=0.0, b1=0.0, b2=0.0, b3=0.0, c0=0.0, c1=0.0):
    d = z1 - z2
    u = np.abs(d) ** 2
    S = np.abs(z1) ** 2 + np.abs(z2) ** 2
    Z = z1 + z2
    env = np.exp(-(g * S + h * u))
    pref = (a0 * d ** 3
            + a1 * d ** 3 * u
            + a2 * d ** 3 * u ** 2
            + (b0 + 1j * b1) * d * Z ** 2
            + (b2 + 1j * b3) * d * Z ** 2 * u
            + (c0 + 1j * c1) * d ** 3 * S)
    return pref * env
# EVOLVE-BLOCK-END