import pytest
import pandas as pd
import numpy as np
import scanpy as sc
from anndata import AnnData
from pandas.testing import assert_frame_equal, assert_series_equal
from anndata import AnnData

from palantir.validation import (
    _validate_obsm_key,
    _validate_varm_key,
    _validate_gene_trend_input,
)


@pytest.fixture
def mock_anndata_with_obsm():
    """Create anndata with obsm for testing validation functions"""
    n_cells = 20
    n_genes = 10
    ad = AnnData(X=np.random.rand(n_cells, n_genes))

    # Add DataFrame in obsm
    ad.obsm["df_key"] = pd.DataFrame(
        np.random.rand(n_cells, 3), columns=["c1", "c2", "c3"], index=ad.obs_names
    )

    # Add numpy array in obsm with column names in uns
    ad.obsm["np_key"] = np.random.rand(n_cells, 3)
    ad.uns["np_key_columns"] = ["c1", "c2", "c3"]

    return ad


@pytest.fixture
def mock_anndata_with_varm():
    """Create anndata with varm for testing validation functions"""
    n_cells = 20
    n_genes = 10
    ad = AnnData(X=np.random.rand(n_cells, n_genes))

    # Add DataFrame in varm
    ad.varm["df_key"] = pd.DataFrame(
        np.random.rand(n_genes, 5), columns=[0.1, 0.2, 0.3, 0.4, 0.5], index=ad.var_names
    )

    # Add numpy array in varm with pseudotime in uns
    ad.varm["np_key"] = np.random.rand(n_genes, 5)
    ad.uns["np_key_pseudotime"] = [0.1, 0.2, 0.3, 0.4, 0.5]

    return ad


@pytest.fixture
def mock_anndata_with_gene_trends():
    """Create anndata with gene trends for testing validation functions"""
    n_cells = 20
    n_genes = 10
    ad = AnnData(X=np.random.rand(n_cells, n_genes))

    # Add branch masks in various locations
    # 1. as DataFrame in obsm
    ad.obsm["branch_masks"] = pd.DataFrame(
        np.random.randint(0, 2, size=(n_cells, 3)),
        columns=["branch1", "branch2", "branch3"],
        index=ad.obs_names,
    )

    # 2. as list in uns
    ad.uns["branch_list"] = ["branch1", "branch2", "branch3"]

    # 3. as numpy array with columns in uns
    ad.obsm["branch_array"] = np.random.randint(0, 2, size=(n_cells, 3))
    ad.uns["branch_array_columns"] = ["branch1", "branch2", "branch3"]

    # Add gene trends for each branch
    for branch in ["branch1", "branch2", "branch3"]:
        trend_key = f"gene_trends_{branch}"
        ad.varm[trend_key] = pd.DataFrame(
            np.random.rand(n_genes, 5), columns=[0.1, 0.2, 0.3, 0.4, 0.5], index=ad.var_names
        )

    return ad


def test_validate_obsm_key_with_df(mock_anndata_with_obsm):
    """Test _validate_obsm_key with DataFrame input"""
    ad = mock_anndata_with_obsm

    # Test DataFrame as_df=True (default)
    data, data_names = _validate_obsm_key(ad, "df_key")
    assert isinstance(data, pd.DataFrame)
    assert data.shape == (ad.n_obs, 3)
    assert list(data_names) == ["c1", "c2", "c3"]

    # Test DataFrame as_df=False
    data, data_names = _validate_obsm_key(ad, "df_key", as_df=False)
    assert isinstance(data, np.ndarray)
    assert data.shape == (ad.n_obs, 3)
    assert list(data_names) == ["c1", "c2", "c3"]


def test_validate_obsm_key_with_array(mock_anndata_with_obsm):
    """Test _validate_obsm_key with numpy array input"""
    ad = mock_anndata_with_obsm

    # Test numpy array as_df=True
    data, data_names = _validate_obsm_key(ad, "np_key")
    assert isinstance(data, pd.DataFrame)
    assert data.shape == (ad.n_obs, 3)
    assert list(data_names) == ["c1", "c2", "c3"]

    # Test numpy array as_df=False
    data, data_names = _validate_obsm_key(ad, "np_key", as_df=False)
    assert isinstance(data, np.ndarray)
    assert data.shape == (ad.n_obs, 3)
    assert list(data_names) == ["c1", "c2", "c3"]


def test_validate_obsm_key_errors(mock_anndata_with_obsm):
    """Test _validate_obsm_key error handling"""
    ad = mock_anndata_with_obsm

    # Test key not in obsm
    with pytest.raises(KeyError, match="not_a_key not found in ad.obsm"):
        _validate_obsm_key(ad, "not_a_key")

    # Test numpy array without columns in uns
    ad.obsm["bad_key"] = np.random.rand(ad.n_obs, 3)
    with pytest.raises(KeyError, match="bad_key_columns not found"):
        _validate_obsm_key(ad, "bad_key")


def test_validate_varm_key_with_df(mock_anndata_with_varm):
    """Test _validate_varm_key with DataFrame input"""
    ad = mock_anndata_with_varm

    # Test DataFrame as_df=True (default)
    data, data_names = _validate_varm_key(ad, "df_key")
    assert isinstance(data, pd.DataFrame)
    assert data.shape == (ad.n_vars, 5)
    assert list(data_names) == [0.1, 0.2, 0.3, 0.4, 0.5]

    # Test DataFrame as_df=False
    data, data_names = _validate_varm_key(ad, "df_key", as_df=False)
    assert isinstance(data, np.ndarray)
    assert data.shape == (ad.n_vars, 5)
    assert list(data_names) == [0.1, 0.2, 0.3, 0.4, 0.5]


def test_validate_varm_key_with_array(mock_anndata_with_varm):
    """Test _validate_varm_key with numpy array input"""
    ad = mock_anndata_with_varm

    # Test numpy array as_df=True
    data, data_names = _validate_varm_key(ad, "np_key")
    assert isinstance(data, pd.DataFrame)
    assert data.shape == (ad.n_vars, 5)
    assert np.allclose(data_names, [0.1, 0.2, 0.3, 0.4, 0.5])

    # Test numpy array as_df=False
    data, data_names = _validate_varm_key(ad, "np_key", as_df=False)
    assert isinstance(data, np.ndarray)
    assert data.shape == (ad.n_vars, 5)
    assert np.allclose(data_names, [0.1, 0.2, 0.3, 0.4, 0.5])


def test_validate_varm_key_errors(mock_anndata_with_varm):
    """Test _validate_varm_key error handling"""
    ad = mock_anndata_with_varm

    # Test key not in varm
    with pytest.raises(KeyError, match="not_a_key not found in ad.varm"):
        _validate_varm_key(ad, "not_a_key")

    # Test numpy array without pseudotime in uns
    ad.varm["bad_key"] = np.random.rand(ad.n_vars, 3)
    with pytest.raises(KeyError, match="bad_key_pseudotime not found"):
        _validate_varm_key(ad, "bad_key")


def test_validate_gene_trend_input_anndata(mock_anndata_with_gene_trends):
    """Test _validate_gene_trend_input with AnnData input"""
    ad = mock_anndata_with_gene_trends

    # Test with default parameters (branch_masks in obsm)
    gene_trends = _validate_gene_trend_input(ad)
    assert isinstance(gene_trends, dict)
    assert len(gene_trends) == 3
    assert "branch1" in gene_trends
    assert "branch2" in gene_trends
    assert "branch3" in gene_trends

    # Test with branch_names as a string key in uns
    gene_trends = _validate_gene_trend_input(ad, branch_names="branch_list")
    assert isinstance(gene_trends, dict)
    assert len(gene_trends) == 3

    # Test with branch_names as a key in obsm with DataFrame
    gene_trends = _validate_gene_trend_input(ad, branch_names="branch_masks")
    assert isinstance(gene_trends, dict)
    assert len(gene_trends) == 3

    # Test with branch_names as a key with columns in uns
    gene_trends = _validate_gene_trend_input(ad, branch_names="branch_array")
    assert isinstance(gene_trends, dict)
    assert len(gene_trends) == 3


def test_validate_gene_trend_input_dict():
    """Test _validate_gene_trend_input with dict input"""
    # Create test dictionary
    trends1 = pd.DataFrame(np.random.rand(10, 5), columns=[0.1, 0.2, 0.3, 0.4, 0.5])
    trends2 = pd.DataFrame(np.random.rand(10, 5), columns=[0.1, 0.2, 0.3, 0.4, 0.5])

    input_dict = {"branch1": {"trends": trends1}, "branch2": {"trends": trends2}}

    gene_trends = _validate_gene_trend_input(input_dict)
    assert gene_trends is input_dict  # Should return the same dict


def test_validate_gene_trend_input_errors(mock_anndata_with_gene_trends):
    """Test _validate_gene_trend_input error handling"""
    ad = mock_anndata_with_gene_trends

    # Test invalid branch_names key
    with pytest.raises(KeyError, match="not_a_key.*not found"):
        _validate_gene_trend_input(ad, branch_names="not_a_key")

    # Test invalid data type
    with pytest.raises(ValueError, match="must be an instance of either AnnData"):
        _validate_gene_trend_input([1, 2, 3])  # List is not valid input
