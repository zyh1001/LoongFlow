import numpy as np


# EVOLVE-BLOCK-START
def wavefunction(z1, z2, z3, z4, z5, z6, alpha=0.25, gamma=0.0, eta=0.0, delta=0.0, beta=0.0, lam=0.0):
    # Laughlin core with non-holomorphic and anharmonic corrections.
    # J = product of all pair differences (Vandermonde).
    d12 = z1 - z2
    d13 = z1 - z3
    d14 = z1 - z4
    d15 = z1 - z5
    d16 = z1 - z6
    d23 = z2 - z3
    d24 = z2 - z4
    d25 = z2 - z5
    d26 = z2 - z6
    d34 = z3 - z4
    d35 = z3 - z5
    d36 = z3 - z6
    d45 = z4 - z5
    d46 = z4 - z6
    d56 = z5 - z6

    J = ((((d12 * d13) * (d14 * d15)) * ((d16 * d23) * (d24 * d25))) *
         (((d26 * d34) * (d35 * d36)) * ((d45 * d46) * d56)))

    core = (J * J) * J                       # J^3 : odd power -> antisymmetric, M=45

    S = (((np.abs(z1) ** 2 + np.abs(z2) ** 2) +
          (np.abs(z3) ** 2 + np.abs(z4) ** 2)) +
         (np.abs(z5) ** 2 + np.abs(z6) ** 2))       # sum |z_i|^2
    Z = z1 + z2 + z3 + z4 + z5 + z6
    G = np.abs(Z) ** 2                              # |sum z_i|^2
    P = z1 * z2 * z3 * z4 * z5 * z6                 # product z_i
    quartic = (np.abs(z1) ** 4 + np.abs(z2) ** 4 + np.abs(z3) ** 4 +
               np.abs(z4) ** 4 + np.abs(z5) ** 4 + np.abs(z6) ** 4)

    # Symmetric, rotational-invariant envelope and non-holomorphic corrections.
    # eta controls |J|^{2 eta}, delta controls |P|^{2 delta},
    # beta controls anharmonic one-body quartic, lam gives a sample-dependent phase.
    return core * np.abs(J) ** (2 * eta) * np.abs(P) ** (2 * delta) * np.exp(
        -alpha * S - beta * quartic - gamma * G + 1j * lam * S
    )
# EVOLVE-BLOCK-END