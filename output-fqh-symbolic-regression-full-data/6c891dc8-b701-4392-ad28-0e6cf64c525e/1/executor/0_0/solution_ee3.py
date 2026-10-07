import numpy as np


# EVOLVE-BLOCK-START
def wavefunction(z1, z2, a=0.125, b=0.125, c=0.0, d=0.0):
    # Relative and center-of-mass coordinates in the z = x - i y convention.
    rel = z1 - z2
    cm = z1 + z2

    # Rotationally invariant squared lengths.
    r2 = np.abs(rel) ** 2       # |z1 - z2|^2
    R2 = np.abs(cm) ** 2        # |z1 + z2|^2

    # (z1 - z2)^3 : odd-power relative factor  ->  fermionic antisymmetry
    #                                          -> single angular-momentum sector M = 3
    # (1 + c*r2 + d*r2^2) : radial LL-mixing polynomial (M = 3 preserved)
    # exp(-a*r2 - b*R2)   : relative + center-of-mass Gaussian envelopes
    return rel ** 3 * (1 + c * r2 + d * r2 ** 2) * np.exp(-a * r2 - b * R2)
# EVOLVE-BLOCK-END