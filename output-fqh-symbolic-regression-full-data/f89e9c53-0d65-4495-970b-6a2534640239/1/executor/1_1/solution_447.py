import numpy as np


# EVOLVE-BLOCK-START
def wavefunction(z1, z2, g=0.25, h=0.25,
                 a0=1.0, a1=0.0,
                 cnl_re=0.0, cnl_im=0.0,
                 ccm_re=0.0, ccm_im=0.0):
    d = z1 - z2
    S = np.abs(z1) ** 2 + np.abs(z2) ** 2
    u = np.abs(d) ** 2
    Z = z1 + z2
    env = np.exp(-(g * S + h * u))
    pref = (a0 * d ** 3
            + a1 * d ** 3 * u
            + (cnl_re + 1j * cnl_im) * d ** 3 * u ** 2
            + (ccm_re + 1j * ccm_im) * d * Z ** 2)
    return pref * env
# EVOLVE-BLOCK-END