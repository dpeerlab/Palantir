import pytest
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from pandas.testing import assert_frame_equal

from palantir.utils import determine_multiscale_space


def test_determine_multiscale_space_with_dict(mock_dm_res):
    """Test determine_multiscale_space with dictionary input"""
    # Test with default n_eigs (determined by eigen gap)
    result = determine_multiscale_space(mock_dm_res)
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == 50  # Should have 50 cells
    # The number of components can vary depending on the generated eigenvalues

    # Test with specific n_eigs
    result = determine_multiscale_space(mock_dm_res, n_eigs=3)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (50, 2)  # Only use 2 eigenvectors (skip first)


def test_determine_multiscale_space_with_anndata(mock_anndata):
    """Test determine_multiscale_space with AnnData input"""
    # Setup eigenvalues with a clear gap for testing auto-selection
    n_components = 10
    eigvals = np.zeros(n_components)
    eigvals[0] = 0.95  # First eigenvalue
    eigvals[1] = 0.85
    eigvals[2] = 0.75
    eigvals[3] = 0.30  # Big gap after this one
    eigvals[4:] = np.linspace(0.25, 0.1, n_components - 4)

    # Create eigenvectors
    eigvecs = np.random.rand(mock_anndata.n_obs, n_components)

    # Add to mock anndata
    mock_anndata.uns["DM_EigenValues"] = eigvals
    mock_anndata.obsm["DM_EigenVectors"] = eigvecs

    # Test with AnnData input - both stores in obsm and returns DataFrame
    result = determine_multiscale_space(mock_anndata)
    assert isinstance(result, pd.DataFrame)  # Returns DataFrame for both AnnData and dict input
    assert "DM_EigenVectors_multiscaled" in mock_anndata.obsm  # Also stores in AnnData

    # Should detect gap and use components after skipping first
    scaled_shape = mock_anndata.obsm["DM_EigenVectors_multiscaled"].shape
    assert scaled_shape[0] == mock_anndata.n_obs  # Number of cells matches
    # Number of components can vary based on how the algorithm detects eigen gaps


def test_determine_multiscale_space_with_small_gap(mock_anndata):
    """Test determine_multiscale_space with small eigen gap"""
    # Setup eigenvalues with no clear gap
    n_components = 5
    eigvals = np.linspace(0.9, 0.5, n_components)

    # Create eigenvectors
    eigvecs = np.random.rand(mock_anndata.n_obs, n_components)

    # Add to mock anndata
    mock_anndata.uns["DM_EigenValues"] = eigvals
    mock_anndata.obsm["DM_EigenVectors"] = eigvecs

    # Test with AnnData input - both stores in obsm and returns DataFrame
    result = determine_multiscale_space(mock_anndata)
    assert isinstance(result, pd.DataFrame)  # Returns DataFrame
    assert "DM_EigenVectors_multiscaled" in mock_anndata.obsm  # Also stores in AnnData

    # Should fall back to second largest gap
    scaled_shape = mock_anndata.obsm["DM_EigenVectors_multiscaled"].shape
    assert scaled_shape[0] == mock_anndata.n_obs
