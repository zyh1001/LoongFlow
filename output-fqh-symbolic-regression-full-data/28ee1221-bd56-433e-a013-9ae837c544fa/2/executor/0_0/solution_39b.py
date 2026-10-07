import numpy as np


# EVOLVE-BLOCK-START
def wavefunction(z1, z2, z3, z4, z5, z6,
                 alpha=1.3818, beta=0.01231, beta2=0.0, gamma=0.2577,
                 lam=0.4903, eta=10.071, delta=-0.8372):
    # ------------------------------------------------------------------
    # Fusion candidate (island_0): holomorphic Vandermonde core J^3
    # (island_0 identity)  x  non-holomorphic LL-mixing screen (island_1
    # motif, |J|^{2 eta})  +  NEW two-body quartic Jastrow (R4).
    #   Psi = J^3 * prod_{i<j} r_ij^{2 eta} * |P|^{2 delta}
    #         * exp(-alpha*S - beta*Q - beta2*R4 - gamma*G + 1j*lam*S)
    # ANTISYMMETRIC: J^3 is odd; all other factors are real invariants.
    # SINGLE SECTOR: M = 3*N(N-1)/2 = 45 (only J^3 carries angular weight).
    # ------------------------------------------------------------------
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
    core = (J * J) * J                       # J^3 -> odd -> antisymmetric, M=45

    S = (np.abs(z1) ** 2 + np.abs(z2) ** 2 + np.abs(z3) ** 2 +
         np.abs(z4) ** 2 + np.abs(z5) ** 2 + np.abs(z6) ** 2)       # M = 0
    Q = (np.abs(z1) ** 4 + np.abs(z2) ** 4 + np.abs(z3) ** 4 +
         np.abs(z4) ** 4 + np.abs(z5) ** 4 + np.abs(z6) ** 4)       # M = 0

    Z = z1 + z2 + z3 + z4 + z5 + z6
    G = np.abs(Z) ** 2                                              # M = 0
    P = z1 * z2 * z3 * z4 * z5 * z6

    R4 = (np.abs(d12) ** 4 + np.abs(d13) ** 4 + np.abs(d14) ** 4 +
          np.abs(d15) ** 4 + np.abs(d16) ** 4 + np.abs(d23) ** 4 +
          np.abs(d24) ** 4 + np.abs(d25) ** 4 + np.abs(d26) ** 4 +
          np.abs(d34) ** 4 + np.abs(d35) ** 4 + np.abs(d36) ** 4 +
          np.abs(d45) ** 4 + np.abs(d46) ** 4 + np.abs(d56) ** 4)    # two-body, M=0

    return (core * np.abs(J) ** (2 * eta) * np.abs(P) ** (2 * delta)
            * np.exp(-alpha * S - beta * Q - beta2 * R4 - gamma * G
                     + 1j * lam * S))
# EVOLVE-BLOCK-END