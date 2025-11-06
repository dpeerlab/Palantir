import pytest
import pandas as pd
import numpy as np
import warnings
import scanpy as sc
from anndata import AnnData
from pandas.testing import assert_frame_equal, assert_series_equal

from palantir.validation import (
    _validate_obsm_key,
    _validate_varm_key,
    _validate_gene_trend_input,
    normalize_cell_identifiers,
)
from palantir import config


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


# Tests for normalize_cell_identifiers
@pytest.fixture
def mock_obs_names_str():
    """Create string obs_names for testing"""
    return pd.Index(["cell_A", "cell_B", "cell_C", "cell_D", "cell_E"])


@pytest.fixture
def mock_obs_names_int():
    """Create integer obs_names for testing"""
    return pd.Index([100, 200, 300, 400, 500])


def test_normalize_cell_identifiers_list_exact_match(mock_obs_names_str):
    """Test normalization with list of cells that match exactly"""
    cells = ["cell_A", "cell_C", "cell_E"]
    result, n_req, n_found = normalize_cell_identifiers(cells, mock_obs_names_str, context="test")

    assert n_req == 3
    assert n_found == 3
    assert set(result.keys()) == {"cell_A", "cell_C", "cell_E"}
    assert all(v == "" for v in result.values())


def test_normalize_cell_identifiers_dict(mock_obs_names_str):
    """Test normalization with dict input"""
    cells = {"cell_A": "type1", "cell_B": "type2"}
    result, n_req, n_found = normalize_cell_identifiers(cells, mock_obs_names_str, context="test")

    assert n_req == 2
    assert n_found == 2
    assert result == {"cell_A": "type1", "cell_B": "type2"}


def test_normalize_cell_identifiers_series(mock_obs_names_str):
    """Test normalization with pd.Series input"""
    cells = pd.Series(["label1", "label2"], index=["cell_A", "cell_C"])
    result, n_req, n_found = normalize_cell_identifiers(cells, mock_obs_names_str, context="test")

    assert n_req == 2
    assert n_found == 2
    assert result == {"cell_A": "label1", "cell_C": "label2"}


def test_normalize_cell_identifiers_int_to_str_conversion(mock_obs_names_str):
    """Test automatic conversion of integer cell IDs to strings"""
    # User provides integers, obs_names are strings
    cells = pd.Series(["pDC", "ERP"], index=[123, 456])

    # Should warn about no cells found (which mentions the type mismatch)
    with pytest.warns(UserWarning, match="None of the 2 requested cells were found"):
        result, n_req, n_found = normalize_cell_identifiers(cells, mock_obs_names_str, context="test")

    assert n_req == 2
    assert n_found == 0  # Won't match because integers don't exist in string obs_names
    # Will convert to strings "123" and "456" but won't match obs_names
    assert "123" in result
    assert "456" in result


def test_normalize_cell_identifiers_str_to_int_data(mock_obs_names_int):
    """Test with string cells and integer obs_names"""
    # User provides strings, obs_names are integers
    cells = ["100", "300", "500"]

    with pytest.warns(UserWarning, match="Cell identifiers are strings but data.obs_names contains integers"):
        result, n_req, n_found = normalize_cell_identifiers(cells, mock_obs_names_int, context="test")

    assert n_req == 3
    assert n_found == 3
    assert set(result.keys()) == {"100", "300", "500"}


def test_normalize_cell_identifiers_int_cells_int_obs():
    """Test with integer cells and integer obs_names - should match perfectly"""
    obs_names = pd.Index([100, 200, 300, 400, 500])
    cells = [100, 300, 500]

    # Should not warn when types match
    result, n_req, n_found = normalize_cell_identifiers(cells, obs_names, context="test")

    assert n_req == 3
    assert n_found == 3
    assert set(result.keys()) == {100, 300, 500}


def test_normalize_cell_identifiers_partial_match():
    """Test when only some cells are found"""
    obs_names = pd.Index(["cell_A", "cell_B", "cell_C"])
    cells = ["cell_A", "cell_X", "cell_Y"]  # Only cell_A exists

    with pytest.warns(UserWarning, match="Only 1/3 requested cells were found"):
        result, n_req, n_found = normalize_cell_identifiers(cells, obs_names, context="test")

    assert n_req == 3
    assert n_found == 1
    assert "cell_A" in result


def test_normalize_cell_identifiers_no_match():
    """Test when no cells are found"""
    obs_names = pd.Index(["cell_A", "cell_B", "cell_C"])
    cells = ["cell_X", "cell_Y", "cell_Z"]

    with pytest.warns(UserWarning, match="None of the 3 requested cells were found"):
        result, n_req, n_found = normalize_cell_identifiers(cells, obs_names, context="test")

    assert n_req == 3
    assert n_found == 0


def test_normalize_cell_identifiers_config_disable_warnings():
    """Test that warnings can be disabled via config"""
    obs_names = pd.Index(["cell_A", "cell_B"])
    cells = ["cell_X", "cell_Y"]  # Won't be found

    # Save original config
    original_warn = config.WARN_ON_CELL_ID_CONVERSION

    try:
        # Disable warnings
        config.WARN_ON_CELL_ID_CONVERSION = False

        # Should not raise warning
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            result, n_req, n_found = normalize_cell_identifiers(cells, obs_names, context="test")

        assert n_req == 2
        assert n_found == 0

    finally:
        # Restore original config
        config.WARN_ON_CELL_ID_CONVERSION = original_warn


def test_normalize_cell_identifiers_config_disable_auto_convert():
    """Test that auto-conversion can be disabled via config"""
    obs_names = pd.Index(["100", "200", "300"])
    cells = [100, 200]  # Integers

    # Save original configs
    original_auto = config.AUTO_CONVERT_CELL_IDS_TO_STR
    original_warn = config.WARN_ON_CELL_ID_CONVERSION

    try:
        # Disable auto-conversion
        config.AUTO_CONVERT_CELL_IDS_TO_STR = False
        config.WARN_ON_CELL_ID_CONVERSION = True

        with pytest.warns(UserWarning, match="Automatic conversion is disabled"):
            result, n_req, n_found = normalize_cell_identifiers(cells, obs_names, context="test")

        assert n_req == 2
        assert n_found == 0  # Won't match because conversion is disabled

    finally:
        # Restore original configs
        config.AUTO_CONVERT_CELL_IDS_TO_STR = original_auto
        config.WARN_ON_CELL_ID_CONVERSION = original_warn


def test_normalize_cell_identifiers_pd_index():
    """Test with pd.Index input"""
    obs_names = pd.Index(["cell_A", "cell_B", "cell_C"])
    cells = pd.Index(["cell_A", "cell_C"])

    result, n_req, n_found = normalize_cell_identifiers(cells, obs_names, context="test")

    assert n_req == 2
    assert n_found == 2
    assert set(result.keys()) == {"cell_A", "cell_C"}


def test_normalize_cell_identifiers_numpy_array():
    """Test with numpy array input"""
    obs_names = pd.Index(["cell_A", "cell_B", "cell_C"])
    cells = np.array(["cell_A", "cell_C"])

    result, n_req, n_found = normalize_cell_identifiers(cells, obs_names, context="test")

    assert n_req == 2
    assert n_found == 2
    assert set(result.keys()) == {"cell_A", "cell_C"}


def test_normalize_cell_identifiers_empty_input():
    """Test with empty input"""
    obs_names = pd.Index(["cell_A", "cell_B"])
    cells = []

    with pytest.raises(ValueError, match="No cells provided"):
        normalize_cell_identifiers(cells, obs_names, context="test")


def test_normalize_cell_identifiers_invalid_type():
    """Test with invalid input type"""
    obs_names = pd.Index(["cell_A", "cell_B"])
    cells = 12345  # Invalid type

    with pytest.raises(ValueError, match="Invalid cells format"):
        normalize_cell_identifiers(cells, obs_names, context="test")
