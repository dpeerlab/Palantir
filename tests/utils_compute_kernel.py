from scipy.sparse import find, csr_matrix
import pytest
import pandas as pd
import scanpy as sc
import numpy as np

from palantir.utils import compute_kernel


@pytest.fixture
def mock_data():
    n_cells = 50
    n_genes = 10
    return pd.DataFrame(
        np.random.rand(n_cells, n_genes),
        columns=[f"gene_{i}" for i in range(n_genes)],
        index=[f"cell_{i}" for i in range(n_cells)],
    )


@pytest.fixture
def mock_anndata(mock_data):
    ad = sc.AnnData(X=mock_data)
    ad.obsm["X_pca"] = mock_data
    return ad


# Test with DataFrame
def test_compute_kernel_dataframe(mock_data):
    kernel = compute_kernel(mock_data)
    assert isinstance(kernel, csr_matrix)


# Test with AnnData
def test_compute_kernel_anndata(mock_anndata):
    kernel = compute_kernel(mock_anndata)
    assert "DM_Kernel" in mock_anndata.obsp.keys()


# Test knn parameter
def test_compute_kernel_knn(mock_data):
    kernel = compute_kernel(mock_data, knn=10)
    assert isinstance(kernel, csr_matrix)


# Test alpha parameter
def test_compute_kernel_alpha(mock_data):
    kernel = compute_kernel(mock_data, alpha=0.5)
    assert isinstance(kernel, csr_matrix)


# Test pca_key parameter
def test_compute_kernel_pca_key(mock_anndata):
    mock_anndata.obsm["custom_pca"] = np.random.rand(mock_anndata.shape[0], 10)
    kernel = compute_kernel(mock_anndata, pca_key="custom_pca")
    assert "DM_Kernel" in mock_anndata.obsp.keys()


# Test kernel_key parameter
def test_compute_kernel_kernel_key(mock_anndata):
    kernel = compute_kernel(mock_anndata, kernel_key="custom_kernel")
    assert "custom_kernel" in mock_anndata.obsp.keys()
