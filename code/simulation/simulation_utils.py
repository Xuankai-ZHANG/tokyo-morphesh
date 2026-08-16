"""Small utilities shared by the inter-community energy-sharing simulations."""


def restore_lambda_totals(serialized_totals, lambdas):
    """Restore lambda-keyed totals after JSON converts numeric keys to strings."""
    return {
        lam: float(serialized_totals.get(str(lam), 0.0))
        for lam in lambdas
    }
