# EVOLVE-BLOCK-START
def candidate_spec():
    # Physical structured ansatz for the two-electron nu=1/3 FQH disk state.
    #
    #   z_rel = z1 - z2  (convention z_i = x_i - i y_i)
    #   Psi = c0 * (z1-z2)^3 * (1 + p3*|z1-z2|^2)
    #             * exp( -|p1|*(|z1|^2+|z2|^2) - |p2|*|z1-z2|^2 )
    #
    # Hard-constraint identities (hold for EVERY fitted numeric value):
    #   C1 fermionic antisymmetry: z_rel -> -z_rel flips only the odd
    #      (z1-z2)^3 factor; all |.| factors are exchange invariant, so
    #      Psi(r2,r1) = -Psi(r1,r2)  =>  exchange_error = 0.
    #   C2 single angular-momentum eigenstate M=3: under z_i -> e^{-i phi} z_i,
    #      (z1-z2)^3 -> e^{-3i phi}(z1-z2)^3 and every modulus is invariant,
    #      so Psi -> e^{-3i phi} Psi  =>  rotation_error ~ 0, M = 3.
    #   C3 finiteness: the exponent is -|p1|*(nonneg) - |p2|*(nonneg) <= 0,
    #      hence exp() <= 1 for any fitted p1,p2 and the polynomial prefactor
    #      is exponentially suppressed at infinity. No poles.
    #
    #   Identity |z1|^2+|z2|^2 = 2|Z_cm|^2 + (1/2)|z1-z2|^2 shows that
    #   p1 and p2 independently set the center-of-mass and relative widths
    #   (p2 acts as a Landau-level-mixing radial correction around the
    #   Laughlin limit p2 = 0, p3 = 0).
    #
    # Initial point: p1 = 0.25 (Laughlin width 1/4), p2 = p3 = 0 makes the
    # very first evaluated candidate the guaranteed-valid Laughlin state;
    # the fitter can only add corrections that lower the validation NMSE.
    return {
        "expression": (
            "c0 * (z1 - z2)**3 * (1 + p3 * abs(z1 - z2)**2) * "
            "exp(-abs(p1)*(abs(z1)**2 + abs(z2)**2)"
            " - abs(p2)*abs(z1 - z2)**2)"
        ),
        "parameters": {
            "c0_re": 1.0,
            "c0_im": 0.0,
            "p1": 0.25,
            "p2": 0.0,
            "p3": 0.0,
        },
    }
# EVOLVE-BLOCK-END