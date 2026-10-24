"""Model support classification, separate from actual holding/trade tolerances."""
import numpy as np

MODEL_WEIGHT_ZERO_TOLERANCE = 1e-6


def is_model_zero(weights):
    """Classify model residuals without modifying the raw objective weights."""
    return np.abs(weights) <= MODEL_WEIGHT_ZERO_TOLERANCE
