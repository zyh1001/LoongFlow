import numpy as np


# EVOLVE-BLOCK-START
def wavefunction(z1, z2, g=0.25, h=0.0, r1=0.0, i1=0.0, r2=0.0, i2=0.0, r3=0.0, i3=0.0, r4=0.0, i4=0.0, r5=0.0, i5=0.0):
    return np.exp(-(g * (np.abs(z1) ** 2 + np.abs(z2) ** 2) + h * np.abs(z1 - z2) ** 2)) * ((z1 - z2) ** 3 * (1 + (r1 + 1j * i1) * np.abs(z1 - z2) ** 2 + (r2 + 1j * i2) * np.abs(z1 - z2) ** 4 + (r3 + 1j * i3) * np.abs(z1 - z2) ** 6) + (r4 + 1j * i4) * (z1 - z2) ** 3 * np.abs(z1 + z2) ** 2 + (r5 + 1j * i5) * (z1 - z2) * (z1 + z2) ** 2)
# EVOLVE-BLOCK-END