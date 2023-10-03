import scanpy as sc
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from palantir.utils import run_local_variability

# Mock data for dense matrix
def mock_anndata_dense(n_cells, n_genes, layer_keys, obsp_keys):
    ad = sc.AnnData(np.random.rand(n_cells, n_genes))
    for key in layer_keys:
        ad.layers[key] = np.random.rand(n_cells, n_genes)
    for key in obsp_keys:
        ad.obsp[key] = np.random.rand(n_cells, n_cells)
    return ad


# Mock data for sparse matrix
def mock_anndata_sparse(n_cells, n_genes, layer_keys, obsp_keys):
    ad = sc.AnnData(csr_matrix(np.random.rand(n_cells, n_genes)))
    for key in layer_keys:
        ad.layers[key] = csr_matrix(np.random.rand(n_cells, n_genes))
    for key in obsp_keys:
        ad.obsp[key] = csr_matrix(np.random.rand(n_cells, n_cells))
    return ad


# Test with default keys, dense
@pytest.mark.filterwarnings("ignore:invalid value encountered in divide")
def test_run_local_variability_default_dense():
    ad = mock_anndata_dense(50, 20, ["MAGIC_imputed_data"], ["distances"])
    _test_run_local_variability(ad)


# Test with default keys, sparse
@pytest.mark.filterwarnings("ignore:invalid value encountered in divide")
def test_run_local_variability_default_sparse():
    ad = mock_anndata_sparse(50, 20, ["MAGIC_imputed_data"], ["distances"])
    _test_run_local_variability(ad)


# Test with custom keys, dense
@pytest.mark.filterwarnings("ignore:invalid value encountered in divide")
def test_run_local_variability_custom_keys_dense():
    ad = mock_anndata_dense(50, 20, ["custom_expression"], ["custom_distances"])
    _test_run_local_variability(
        ad, "custom_expression", "custom_distances", "custom_local_var"
    )


# Test with custom keys, sparse
@pytest.mark.filterwarnings("ignore:invalid value encountered in divide")
def test_run_local_variability_custom_keys_sparse():
    ad = mock_anndata_sparse(50, 20, ["custom_expression"], ["custom_distances"])
    _test_run_local_variability(
        ad, "custom_expression", "custom_distances", "custom_local_var"
    )


# Helper function for assertions
def _test_run_local_variability(
    ad,
    expression_key="MAGIC_imputed_data",
    distances_key="distances",
    localvar_key="local_variability",
):
    result = run_local_variability(ad, expression_key, distances_key, localvar_key)

    assert localvar_key in ad.layers
    assert isinstance(result, np.ndarray) or isinstance(result, csr_matrix)
    assert result.shape == (50, 20)


# Test missing keys
def test_run_local_variability_missing_keys():
    ad = mock_anndata_dense(50, 20, ["MAGIC_imputed_data"], ["distances"])

    with pytest.raises(KeyError):
        run_local_variability(ad, "missing_expression", "distances")

    with pytest.raises(KeyError):
        run_local_variability(ad, "MAGIC_imputed_data", "missing_distances")
