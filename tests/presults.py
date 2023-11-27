import numpy as np
import pandas as pd
import palantir

def test_PResults():
    # Create some dummy data
    pseudotime = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    entropy = None
    branch_probs = pd.DataFrame({'branch1': [0.1, 0.2, 0.3, 0.4, 0.5], 'branch2': [0.5, 0.4, 0.3, 0.2, 0.1]})
    waypoints = None

    # Initialize PResults object
    presults = palantir.presults.PResults(pseudotime, entropy, branch_probs, waypoints)

    # Asserts to check attributes
    assert np.array_equal(presults.pseudotime, pseudotime)
    assert presults.entropy is None
    assert presults.waypoints is None
    assert np.array_equal(presults.branch_probs, branch_probs.values)

def test_gam_fit_predict():
    # Create some dummy data
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    weights = None
    pred_x = None
    n_splines = 4
    spline_order = 2

    # Call the function
    y_pred, stds = palantir.presults.gam_fit_predict(x, y, weights, pred_x, n_splines, spline_order)

    # Asserts to check the output
    assert isinstance(y_pred, np.ndarray)
    assert isinstance(stds, np.ndarray)