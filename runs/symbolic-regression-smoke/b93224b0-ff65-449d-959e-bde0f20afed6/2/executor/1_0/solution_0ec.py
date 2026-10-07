# EVOLVE-BLOCK-START
def candidate_spec():
    return {
        "expression": "c0 * z12**3 * exp(-a * (abs(z1)**2 + abs(z2)**2))",
        "parameters": {
            "c0_re": 1.0,
            "c0_im": 0.0,
            "a": 0.25,
        },
    }
# EVOLVE-BLOCK-END