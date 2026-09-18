# EVOLVE-BLOCK-START
def candidate_spec():
    """Compact analytic candidate for the N=2, nu=1/3 FQH disk wavefunction.

    Structure (Laughlin core x Landau-level-mixing correction x Gaussian):

        Psi(z1,z2) = c0 * z12**3
                     * (1 + p3 * abs(z12)**2)
                     * exp( -abs(p1) * (abs(z1)**2 + abs(z2)**2)
                            -abs(p2) * abs(z12)**2 )

    with z12 = z1 - z2, z1 = x1 - i*y1, z2 = x2 - i*y2.

    Analytical guarantees (independent of the fitted parameter values):
      * Exchange:   z12 -> -z12  =>  Psi(r2,r1) = -Psi(r1,r2)   (fermionic, M=3).
      * Rotation:   z_i -> e^{-i phi} z_i  =>  Psi -> e^{-3 i phi} Psi  (M=3).
      * Finiteness: -abs(p1)*(...) - abs(p2)*(...) <= 0  =>  exp(.) <= 1.
      * Decoupling: |z1|^2+|z2|^2 = 2|Z_cm|^2 + |z12|^2/2, so (p1,p2)
                    independently set the center-of-mass and relative widths.
    """
    return {
        "expression": (
            "c0 * z12**3 * (1 + p3 * abs(z12)**2) * "
            "exp(-abs(p1)*(abs(z1)**2 + abs(z2)**2) - abs(p2)*abs(z12)**2)"
        ),
        "parameters": {
            "c0_re": 1.0,
            "c0_im": 0.0,
            "p1": 0.25,   # Laughlin Gaussian width 1/4 (physical anchor)
            "p2": 0.0,    # relative-width Landau-level-mixing correction
            "p3": 0.0,    # polynomial Landau-level-mixing correction
        },
    }
# EVOLVE-BLOCK-END