import pytest
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.sparse import csr_matrix
from anndata import AnnData

from palantir.utils import run_magic_imputation


@pytest.fixture
def mock_dm_res():
    return {"T": csr_matrix(np.random.rand(50, 50))}


# Test with numpy ndarray
def test_run_magic_imputation_ndarray(mock_dm_res):
    data = np.random.rand(50, 20)
    # With default sparse=True
    result = run_magic_imputation(data, dm_res=mock_dm_res)
    assert isinstance(result, csr_matrix)
    # With sparse=False
    result = run_magic_imputation(data, dm_res=mock_dm_res, sparse=False)
    assert isinstance(result, np.ndarray)


# Test with pandas DataFrame
def test_run_magic_imputation_dataframe(mock_dm_res):
    data = pd.DataFrame(np.random.rand(50, 20))
    # With default sparse=True, converts to DataFrame
    result = run_magic_imputation(data, dm_res=mock_dm_res)
    assert isinstance(result, pd.DataFrame)
    # With sparse=False
    result = run_magic_imputation(data, dm_res=mock_dm_res, sparse=False)
    assert isinstance(result, pd.DataFrame)


# Test with csr_matrix
def test_run_magic_imputation_csr(mock_dm_res):
    data = csr_matrix(np.random.rand(50, 20))
    # With default sparse=True
    result = run_magic_imputation(data, dm_res=mock_dm_res)
    assert isinstance(result, csr_matrix)
    # With sparse=False
    result = run_magic_imputation(data, dm_res=mock_dm_res, sparse=False)
    assert isinstance(result, np.ndarray)


# Test with AnnData
def test_run_magic_imputation_anndata():
    data = AnnData(np.random.rand(50, 20))
    data.obsp["DM_Similarity"] = np.random.rand(50, 50)
    # With default sparse=True
    result = run_magic_imputation(data)
    assert "MAGIC_imputed_data" in data.layers
    assert isinstance(result, csr_matrix)
    
    # With sparse=False
    data2 = AnnData(np.random.rand(50, 20))
    data2.obsp["DM_Similarity"] = np.random.rand(50, 50)
    result2 = run_magic_imputation(data2, sparse=False)
    assert "MAGIC_imputed_data" in data2.layers
    assert isinstance(result2, np.ndarray)


# Test with AnnData and custom keys
def test_run_magic_imputation_anndata_custom_keys():
    data = AnnData(np.random.rand(50, 20))
    data.layers["custom_expr"] = np.random.rand(50, 20)
    data.obsp["custom_sim"] = np.random.rand(50, 50)
    result = run_magic_imputation(
        data,
        expression_key="custom_expr",
        sim_key="custom_sim",
        imputation_key="custom_imp",
    )
    assert "custom_imp" in data.layers
    assert isinstance(result, csr_matrix)
    
    # With sparse=False
    data2 = AnnData(np.random.rand(50, 20))
    data2.layers["custom_expr"] = np.random.rand(50, 20)
    data2.obsp["custom_sim"] = np.random.rand(50, 50)
    result2 = run_magic_imputation(
        data2,
        expression_key="custom_expr",
        sim_key="custom_sim",
        imputation_key="custom_imp2",
        sparse=False,
    )
    assert "custom_imp2" in data2.layers
    assert isinstance(result2, np.ndarray)


# Test with missing dm_res and not AnnData
def test_run_magic_imputation_missing_dm_res():
    data = np.random.rand(50, 20)
    with pytest.raises(ValueError):
        run_magic_imputation(data)


# Test with missing expression_key in AnnData
def test_run_magic_imputation_missing_expression_key():
    data = AnnData(np.random.rand(50, 20))
    data.obsp["DM_Similarity"] = np.random.rand(50, 50)
    with pytest.raises(ValueError):
        run_magic_imputation(data, expression_key="missing_key")
