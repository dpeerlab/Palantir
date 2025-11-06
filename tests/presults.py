import numpy as np
import pandas as pd
import pytest
import scipy
import palantir

# Check scipy version for pygam compatibility
# pygam has known issues with scipy >= 1.13 due to sparse matrix changes
SCIPY_VERSION = tuple(map(int, scipy.__version__.split('.')[:2]))
SCIPY_INCOMPATIBLE_WITH_PYGAM = SCIPY_VERSION >= (1, 13)


def test_PResults():
    # Create some dummy data
    pseudotime = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    entropy = None
    branch_probs = pd.DataFrame(
        {"branch1": [0.1, 0.2, 0.3, 0.4, 0.5], "branch2": [0.5, 0.4, 0.3, 0.2, 0.1]}
    )
    waypoints = None

    # Initialize PResults object
    presults = palantir.presults.PResults(pseudotime, entropy, branch_probs, waypoints)

    # Asserts to check attributes
    assert np.array_equal(presults.pseudotime, pseudotime)
    assert presults.entropy is None
    assert presults.waypoints is None
    assert np.array_equal(presults.branch_probs, branch_probs.values)


@pytest.mark.skipif(
    SCIPY_INCOMPATIBLE_WITH_PYGAM,
    reason="pygam is incompatible with scipy >= 1.13 (sparse matrix .A attribute removed)"
)
def test_gam_fit_predict():
    # Create some dummy data
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    weights = None
    pred_x = None
    n_splines = 4
    spline_order = 2

    # Call the function
    y_pred, stds = palantir.presults.gam_fit_predict(
        x, y, weights, pred_x, n_splines, spline_order
    )

    # Asserts to check the output
    assert isinstance(y_pred, np.ndarray)
    assert isinstance(stds, np.ndarray)
