import numpy as np


# EVOLVE-BLOCK-START
def wavefunction(z1, z2, z3, z4, z5, z6):
    return ((((z1 - z2) * ((z1 - z3) * (z1 - z4))) * (((z1 - z5) * (z1 - z6)) * ((z2 - z3) * (z2 - z4)))) * ((((z2 - z5) * (z2 - z6)) * ((z3 - z4) * (z3 - z5))) * (((z3 - z6) * (z4 - z5)) * ((z4 - z6) * (z5 - z6)))))
# EVOLVE-BLOCK-END
