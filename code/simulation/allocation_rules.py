"""Reusable allocation rules for the inter-community energy-sharing simulation engines."""

import numpy as np


def conditional_donor_normalized(weight_matrix_raw, surplus, deficit):
    """Allocate each donor's surplus only across receivers with current deficit.

    The raw sparse matrix is receiver-by-donor. Each active donor column is
    normalized over receivers whose deficit is positive at the current
    timestep. Receiver imports are capped at their current deficit.
    """
    surplus = np.asarray(surplus, dtype=np.float64)
    deficit = np.asarray(deficit, dtype=np.float64)
    active_receivers = deficit > 0

    active_weight = np.asarray(
        weight_matrix_raw.T.dot(active_receivers.astype(np.float64))
    ).ravel()
    scaled_surplus = np.zeros_like(surplus)
    np.divide(
        surplus,
        active_weight,
        out=scaled_surplus,
        where=active_weight > 0,
    )

    received = np.asarray(weight_matrix_raw.dot(scaled_surplus)).ravel()
    received[~active_receivers] = 0.0
    imported = np.minimum(deficit, received)
    return received, imported
