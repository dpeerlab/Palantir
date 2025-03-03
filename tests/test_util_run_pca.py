import pytest
import pandas as pd
import scanpy as sc
from anndata import AnnData
import numpy as np

from palantir.utils import run_pca


@pytest.fixture
def mock_data():
    n_cells = 50
    n_genes = 500
    return pd.DataFrame(
        np.random.rand(n_cells, n_genes),
        columns=[f"gene_{i}" for i in range(n_genes)],
        index=[f"cell_{i}" for i in range(n_cells)],
    )


@pytest.fixture
def mock_anndata(mock_data):
    ad = AnnData(X=mock_data)
    ad.obsm["DM_EigenVectors_multiscaled"] = mock_data
    ad.var["highly_variable"] = np.random.choice([True, False], size=mock_data.shape[1])
    return ad


# Test with DataFrame
def test_run_pca_dataframe(mock_data):
    pca_results, var_ratio = run_pca(mock_data, use_hvg=False)
    assert isinstance(pca_results, pd.DataFrame)
    assert isinstance(var_ratio, np.ndarray)
    assert pca_results.shape[1] <= 300  # Check n_components


# Test with AnnData
def test_run_pca_anndata(mock_anndata):
    pca_results, var_ratio = run_pca(mock_anndata)
    assert "X_pca" in mock_anndata.obsm.keys()
    assert mock_anndata.obsm["X_pca"].shape[1] <= 300


# Test n_components parameter
def test_run_pca_components(mock_data):
    pca_results, _ = run_pca(mock_data, n_components=5, use_hvg=False)
    assert pca_results.shape[1] == 5


# Test use_hvg parameter
def test_run_pca_hvg(mock_anndata):
    pca_results, _ = run_pca(mock_anndata, use_hvg=True)
    assert pca_results.shape[1] <= 300


# Test pca_key parameter
def test_run_pca_pca_key(mock_anndata):
    run_pca(mock_anndata, pca_key="custom_key")
    assert "custom_key" in mock_anndata.obsm.keys()
    assert mock_anndata.obsm["custom_key"].shape[1] <= 300
