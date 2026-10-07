import numpy as np


# EVOLVE-BLOCK-START
def wavefunction(z1, z2, g=0.25, h=0.0,
                 a0=1.0, a1=0.0, a2=0.0, a3=0.0, a4=0.0, a5=0.0,
                 cr=0.0, ci=0.0):
    d = z1 - z2
    W = z1 + z2
    S = np.abs(z1) ** 2 + np.abs(z2) ** 2
    u = np.abs(d) ** 2
    env = np.exp(-(g * S + h * u))
    poly = a0 + a1 * u + a2 * u ** 2 + a3 * u ** 3 + a4 * u ** 4 + a5 * u ** 5
    return (d ** 3 * poly + (cr + 1j * ci) * d * W ** 2) * env
# EVOLVE-BLOCK-END