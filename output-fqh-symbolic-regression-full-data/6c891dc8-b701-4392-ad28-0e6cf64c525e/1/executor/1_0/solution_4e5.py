import numpy as np


# EVOLVE-BLOCK-START
def wavefunction(z1, z2, a=0.125, b=0.125,
                 c1=0.0, c2=0.0, c3=0.0, c4=0.0,
                 d1=0.0, d2=0.0, d3=0.0,
                 e1=0.0, e2=0.0, e3=0.0,
                 alpha=0.0, beta=0.0):
    r2 = np.abs(z1 - z2)**2
    R2 = np.abs(z1 + z2)**2
    r = np.sqrt(r2)
    R = np.sqrt(R2)
    return (z1 - z2)**3 * (1 + c1*r2 + c2*r2**2 + c3*r2**3 + c4*r2**4) * (1 + d1*R2 + d2*R2**2 + d3*R2**3) * (1 + e1*r2*R2 + e2*r2**2*R2 + e3*r2*R2**2) * np.exp(-a*r2 - b*R2 - alpha*r - beta*R)
# EVOLVE-BLOCK-END