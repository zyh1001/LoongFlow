import numpy as np


# EVOLVE-BLOCK-START
def wavefunction(z1, z2, g=0.25, h=0.0,
                 c1_re=0.0, c1_im=0.0,
                 c2_re=0.0, c2_im=0.0,
                 c3_re=0.0, c3_im=0.0,
                 c4_re=0.0, c4_im=0.0,
                 c5_re=0.0, c5_im=0.0,
                 c6_re=0.0, c6_im=0.0):
    d = z1 - z2
    Z = z1 + z2
    u = np.abs(d) ** 2
    v = np.abs(Z) ** 2
    S = np.abs(z1) ** 2 + np.abs(z2) ** 2
    env = np.exp(-(g * S + h * u))
    pref = (d ** 3
            + (c1_re + 1j * c1_im) * d ** 3 * u
            + (c2_re + 1j * c2_im) * d ** 3 * u ** 2
            + (c3_re + 1j * c3_im) * d ** 3 * v
            + (c4_re + 1j * c4_im) * d * Z ** 2
            + (c5_re + 1j * c5_im) * d * Z ** 2 * u
            + (c6_re + 1j * c6_im) * d * Z ** 2 * u ** 2)
    return pref * env
# EVOLVE-BLOCK-END