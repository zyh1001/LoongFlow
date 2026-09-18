# EVOLVE-BLOCK-START
def candidate_spec():
    """
    Candidate for the two-electron nu=1/3 FQH disk wavefunction in the
    convention z_i = x_i - i*y_i.

    Structural choice (minimal M=3 Laughlin core):

        Psi(z1,z2) = c0 * z12**3 * exp(-|p1| * (|z1|**2 + |z2|**2))

    with z12 = z1 - z2.

    Analytic guarantees for EVERY fitted parameter value X = (c0, p1):

    * Fermionic antisymmetry (C1): under exchange z12 -> -z12, so
      z12**3 -> -z12**3, while |z1|,|z2| magnitudes are exchange
      invariant => Psi(r2,r1) = -Psi(r1,r2). exchange_error = 0.

    * Single angular momentum sector M = 3 (C2): under a counterclockwise
      physical rotation z_i -> exp(-i*phi) z_i we have z12 -> exp(-i*phi)
      z12 and every magnitude is invariant, hence
      Psi -> exp(-3i*phi) Psi. No sector mixing => rotation_error ~ 0.

    * Finiteness (C3): |p1| is placed INSIDE the exponent so the
      Gaussian factor has modulus <= 1 for any fitted real p1; the
      polynomial prefactor is exponentially suppressed at infinity and
      has no poles. finite_fraction = 1.0.

    * Complexity: only 3 fitted reals (c0 real/imag + one width).
      This maximizes the complexity_penalty while the M=3 Laughlin core
      plus the Gaussian envelope carry essentially all of the fidelity.
    """
    return {
        "expression": "c0 * z12**3 * exp(-abs(p1) * (abs(z1)**2 + abs(z2)**2))",
        "parameters": {
            "c0_re": 1.0,   # global complex amplitude (real part)
            "c0_im": 0.0,   # global complex amplitude (imag part)
            "p1": 0.25,     # envelope width; analytic Laughlin/disk anchor 1/4
        },
    }
# EVOLVE-BLOCK-END