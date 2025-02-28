import pytest
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from unittest.mock import patch, MagicMock

from palantir.utils import run_low_density_variability, run_density_evaluation


@pytest.fixture
def mock_anndata_with_density(mock_anndata):
    """Create anndata with density for testing low_density_variability"""
    # Add density values
    mock_anndata.obs["mellon_log_density"] = np.random.rand(mock_anndata.n_obs)

    # Add local variability
    mock_anndata.layers["local_variability"] = np.random.rand(
        mock_anndata.n_obs, mock_anndata.n_vars
    )

    # Add branch masks
    mock_anndata.obsm["branch_masks"] = pd.DataFrame(
        np.random.randint(0, 2, size=(mock_anndata.n_obs, 2)),
        columns=["branch1", "branch2"],
        index=mock_anndata.obs_names,
    )

    # Also add branch mask in obs
    mock_anndata.obs["obs_branch"] = np.random.randint(0, 2, size=mock_anndata.n_obs)

    return mock_anndata


def test_run_low_density_variability_with_obsm(mock_anndata_with_density):
    """Test run_low_density_variability function with obsm branch masks"""
    ad = mock_anndata_with_density

    # Test with default parameters (branch_masks in obsm)
    result = run_low_density_variability(ad)

    # Check results
    assert result.shape == (ad.n_vars, 2)  # 2 branches
    assert "low_density_gene_variability_branch1" in ad.var.columns
    assert "low_density_gene_variability_branch2" in ad.var.columns

    # Test with custom parameters
    result = run_low_density_variability(
        ad,
        cell_mask="branch_masks",
        density_key="mellon_log_density",
        localvar_key="local_variability",
        score_key="test_prefix",
    )

    assert "test_prefix_branch1" in ad.var.columns
    assert "test_prefix_branch2" in ad.var.columns


def test_run_low_density_variability_with_obs(mock_anndata_with_density):
    """Test run_low_density_variability function with obs column"""
    ad = mock_anndata_with_density

    # Test with obs column
    result = run_low_density_variability(ad, cell_mask="obs_branch")

    # Check results
    assert result.shape == (ad.n_vars, 1)
    assert "low_density_gene_variability__obs_branch" in ad.var.columns


def test_run_low_density_variability_with_array(mock_anndata_with_density):
    """Test run_low_density_variability function with array input"""
    ad = mock_anndata_with_density

    # Test with np.array mask
    mask = np.zeros(ad.n_obs, dtype=bool)
    mask[:10] = True
    result = run_low_density_variability(ad, cell_mask=mask)
    assert "low_density_gene_variability_" in ad.var.columns

    # Test with list of cell names
    cell_list = ad.obs_names[:10].tolist()
    result = run_low_density_variability(ad, cell_mask=cell_list)
    assert "low_density_gene_variability_" in ad.var.columns


def test_run_low_density_variability_errors(mock_anndata_with_density):
    """Test error handling in run_low_density_variability"""
    ad = mock_anndata_with_density

    # Test missing density key
    with pytest.raises(ValueError, match="not_a_key' not found in ad.obs"):
        run_low_density_variability(ad, density_key="not_a_key")

    # Test missing layer key
    with pytest.raises(ValueError, match="not_a_key' not found in ad.layers"):
        run_low_density_variability(ad, localvar_key="not_a_key")

    # Test missing cell_mask key
    with pytest.raises(ValueError, match="not_a_key' not found in ad.obsm or ad.obs"):
        run_low_density_variability(ad, cell_mask="not_a_key")

    # Test invalid cell_mask type
    with pytest.raises(ValueError, match="cell_mask must be either a string key"):
        run_low_density_variability(ad, cell_mask=42)  # Integer is invalid


@patch("mellon.Predictor.from_dict")
def test_run_density_evaluation(mock_predictor_from_dict):
    """Test run_density_evaluation function"""
    # Create input and output anndata objects
    in_ad = AnnData(X=np.random.rand(20, 10))
    out_ad = AnnData(X=np.random.rand(15, 10))

    # Setup predictor mock
    mock_predictor = MagicMock()
    mock_predictor.return_value = np.random.rand(15)
    mock_predictor_from_dict.return_value = mock_predictor

    # Add required fields
    in_ad.uns["mellon_log_density_predictor"] = {"mock": "predictor"}
    out_ad.obsm["DM_EigenVectors"] = np.random.rand(15, 5)

    # Run the function
    result = run_density_evaluation(in_ad, out_ad)

    # Check results
    assert len(result) == 15
    assert "cross_log_density" in out_ad.obs.columns
    assert "cross_log_density_clipped" in out_ad.obs.columns

    # Verify predictor was called
    mock_predictor_from_dict.assert_called_once_with(in_ad.uns["mellon_log_density_predictor"])
    mock_predictor.assert_called_once_with(out_ad.obsm["DM_EigenVectors"])

    # Test with custom parameters
    result = run_density_evaluation(
        in_ad,
        out_ad,
        predictor_key="mellon_log_density_predictor",
        repr_key="DM_EigenVectors",
        density_key="custom_density",
    )

    assert "custom_density" in out_ad.obs.columns
    assert "custom_density_clipped" in out_ad.obs.columns


def test_run_density_evaluation_errors():
    """Test error handling in run_density_evaluation"""
    # Create input and output anndata objects
    in_ad = AnnData(X=np.random.rand(20, 10))
    out_ad = AnnData(X=np.random.rand(15, 10))

    # Test missing repr_key
    with pytest.raises(ValueError, match="'DM_EigenVectors' not found in out_ad.obsm"):
        run_density_evaluation(in_ad, out_ad)

    # Add eigenvectors but no predictor
    out_ad.obsm["DM_EigenVectors"] = np.random.rand(15, 5)

    # Test missing predictor_key
    with pytest.raises(ValueError, match="'mellon_log_density_predictor' not found in in_ad.uns"):
        run_density_evaluation(in_ad, out_ad)
