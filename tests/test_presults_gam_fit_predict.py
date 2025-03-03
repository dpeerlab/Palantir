import pytest
import numpy as np
import pandas as pd

# Skip all tests in this file if pygam is not installed
try:
    import pygam
except ImportError:
    pytestmark = pytest.mark.skip(reason="pygam not installed")

# Handle scipy compatibility issues
try:
    import scipy.sparse as sp
    test_matrix = sp.csr_matrix((1, 1))
    if not hasattr(test_matrix, 'A'):
        pytestmark = pytest.mark.skip(reason="scipy/pygam compatibility issue")
except:
    pass

from palantir.presults import gam_fit_predict


def test_gam_fit_predict_basic():
    """Test basic functionality of gam_fit_predict"""
    # Create test data
    x = np.linspace(0, 1, 50)
    y = np.sin(2 * np.pi * x) + 0.1 * np.random.randn(50)

    # Run gam_fit_predict
    y_pred, stds = gam_fit_predict(x, y)

    # Check output shapes
    assert len(y_pred) == len(x)
    assert len(stds) == len(x)

    # Check that predictions follow the general trend
    assert np.corrcoef(y, y_pred)[0, 1] > 0.8  # Strong correlation


def test_gam_fit_predict_with_weights():
    """Test gam_fit_predict with weights"""
    # Create test data
    x = np.linspace(0, 1, 50)
    y = np.sin(2 * np.pi * x) + 0.1 * np.random.randn(50)

    # Create weights biased toward the beginning
    weights = np.exp(-3 * x)

    # Run gam_fit_predict with weights
    y_pred_weighted, _ = gam_fit_predict(x, y, weights=weights)
    # Run without weights for comparison
    y_pred_unweighted, _ = gam_fit_predict(x, y)

    # Check that predictions differ when using weights
    assert not np.allclose(y_pred_weighted, y_pred_unweighted)

    # Early points should be fitted better with weights
    early_idx = x < 0.3
    early_mse_weighted = np.mean((y[early_idx] - y_pred_weighted[early_idx]) ** 2)
    early_mse_unweighted = np.mean((y[early_idx] - y_pred_unweighted[early_idx]) ** 2)
    assert early_mse_weighted <= early_mse_unweighted


def test_gam_fit_predict_with_pred_x():
    """Test gam_fit_predict with custom prediction points"""
    # Create test data
    x = np.linspace(0, 1, 50)
    y = np.sin(2 * np.pi * x) + 0.1 * np.random.randn(50)

    # Create custom prediction points
    pred_x = np.linspace(0, 1, 100)  # Higher resolution

    # Run gam_fit_predict with custom prediction points
    y_pred, stds = gam_fit_predict(x, y, pred_x=pred_x)

    # Check that output shapes match the custom prediction points
    assert len(y_pred) == len(pred_x)
    assert len(stds) == len(pred_x)


def test_gam_fit_predict_spline_params():
    """Test gam_fit_predict with different spline parameters"""
    # Create test data
    x = np.linspace(0, 1, 50)
    y = np.sin(2 * np.pi * x) + 0.1 * np.random.randn(50)

    # Run with default spline parameters
    y_pred_default, _ = gam_fit_predict(x, y)

    # Run with custom spline parameters
    y_pred_custom, _ = gam_fit_predict(x, y, n_splines=8, spline_order=3)

    # Check that predictions differ with different spline parameters
    assert not np.allclose(y_pred_default, y_pred_custom)
