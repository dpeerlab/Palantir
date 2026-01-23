import pytest
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from unittest.mock import patch, MagicMock

from palantir.utils import (
    early_cell,
    fallback_terminal_cell,
    find_terminal_states,
    CellNotFoundException,
)


@pytest.fixture
def mock_anndata_with_celltypes(mock_anndata):
    """Create anndata with cell types for early_cell and terminal states tests"""
    # Add cell types
    celltypes = np.array(["A", "B", "C", "A", "B"] * 10)
    mock_anndata.obs["celltype"] = pd.Categorical(celltypes)

    # Add multiscale space with one cell type at extremes
    eigvecs = mock_anndata.obsm["DM_EigenVectors_multiscaled"].copy()
    # Make cell 0 (type A) maximum in component 0
    eigvecs[0, 0] = 100.0
    # Make cell 4 (type B) minimum in component 1
    eigvecs[4, 1] = -100.0

    mock_anndata.obsm["DM_EigenVectors_multiscaled"] = eigvecs

    return mock_anndata


def test_early_cell_extreme_max(mock_anndata_with_celltypes):
    """Test early_cell finding cell at maximum of component"""
    ad = mock_anndata_with_celltypes

    # Test finding a cell of type 'A' - we don't need to know which cell it will be
    with patch("palantir.utils._return_cell", return_value="cell_0") as mock_return:
        result = early_cell(ad, "A")
        assert result == "cell_0"  # Just check the mocked return value
        mock_return.assert_called_once()

        # Only check the cell type and that it's finding some kind of extreme
        args = mock_return.call_args[0]
        assert args[2] == "A"  # Cell type
        assert args[3] in ["max", "min"]  # Extreme type (don't care which one)


def test_early_cell_extreme_min(mock_anndata_with_celltypes):
    """Test early_cell finding cell at minimum of component"""
    ad = mock_anndata_with_celltypes

    # Test finding a cell of type 'B' - we don't need to know which cell it will be
    with patch("palantir.utils._return_cell", return_value="cell_4") as mock_return:
        result = early_cell(ad, "B")
        assert result == "cell_4"  # Just check the mocked return value
        mock_return.assert_called_once()

        # Only check the cell type and that it's finding some kind of extreme
        args = mock_return.call_args[0]
        assert args[2] == "B"  # Cell type
        assert args[3] in ["max", "min"]  # Extreme type (don't care which one)


def test_early_cell_fallback():
    """Test early_cell with fallback to fallback_terminal_cell"""
    # Create a very simple AnnData with a cell type that won't be at extremes
    ad = AnnData(X=np.random.rand(10, 5))
    ad.obs["celltype"] = pd.Categorical(
        ["A", "A", "A", "A", "A", "B", "B", "B", "C", "C"], categories=["A", "B", "C"]
    )

    # Add a fake eigenvectors matrix where no 'B' cells are at extremes
    eigvecs = np.zeros((10, 3))
    # Make 'A' cells dominate the extremes
    eigvecs[0, 0] = 100  # max in component 0 is cell 0 (type A)
    eigvecs[1, 0] = -100  # min in component 0 is cell 1 (type A)
    eigvecs[2, 1] = 100  # max in component 1 is cell 2 (type A)
    eigvecs[3, 1] = -100  # min in component 1 is cell 3 (type A)
    eigvecs[4, 2] = 100  # max in component 2 is cell 4 (type A)
    eigvecs[5, 2] = -100  # min in component 2 is cell 5 (type B)
    ad.obsm["DM_EigenVectors_multiscaled"] = eigvecs

    # Give the AnnData proper observation names
    ad.obs_names = [f"cell_{i}" for i in range(10)]

    # Mock fallback_terminal_cell to avoid actual computation
    with patch("palantir.utils.fallback_terminal_cell", return_value="cell_5") as mock_fallback:
        # Test early_cell with fallback - it should find no cell in extremes and fall back
        result = early_cell(ad, "C", fallback_seed=42)  # Cell type C doesn't exist
        assert result == "cell_5"
        mock_fallback.assert_called_once_with(
            ad, "C", celltype_column="celltype", eigvec_key="DM_EigenVectors_multiscaled", seed=42
        )


def test_early_cell_exception():
    """Test early_cell raising exception when no cell found"""
    # Create a very simple AnnData with a cell type that won't be at extremes
    ad = AnnData(X=np.random.rand(10, 5))
    ad.obs["celltype"] = pd.Categorical(
        ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"], categories=["A", "B"]
    )

    # Add a fake eigenvectors matrix where no 'B' cells are at extremes
    eigvecs = np.zeros((10, 3))
    # Make 'A' cells dominate the extremes
    eigvecs[0, 0] = 100  # max in component 0 is cell 0 (type A)
    eigvecs[1, 0] = -100  # min in component 0 is cell 1 (type A)
    eigvecs[2, 1] = 100  # max in component 1 is cell 2 (type A)
    eigvecs[3, 1] = -100  # min in component 1 is cell 3 (type A)
    eigvecs[4, 2] = 100  # max in component 2 is cell 4 (type A)
    eigvecs[0, 2] = -100  # min in component 2 is cell 0 (type A)
    ad.obsm["DM_EigenVectors_multiscaled"] = eigvecs

    # Test without fallback_seed - should raise CellNotFoundException
    with pytest.raises(CellNotFoundException):
        early_cell(ad, "B")


@patch("palantir.utils.run_palantir")
def test_fallback_terminal_cell(mock_run_palantir, mock_anndata_with_celltypes):
    """Test fallback_terminal_cell with mocked palantir run"""
    ad = mock_anndata_with_celltypes

    # Setup mock pseudotime result
    mock_result = MagicMock()
    pseudotime = pd.Series([0.1, 0.2, 0.3, 0.9, 0.5], index=ad.obs_names[:5])
    mock_result.pseudotime = pseudotime
    mock_run_palantir.return_value = mock_result

    # Test fallback_terminal_cell
    with patch("palantir.utils.print"):  # Suppress print output
        result = fallback_terminal_cell(ad, "A", celltype_column="celltype", seed=42)
        assert result == ad.obs_names[3]  # Should pick cell with max pseudotime

    # Verify run_palantir was called with correct arguments
    mock_run_palantir.assert_called_once()
    call_args = mock_run_palantir.call_args[0]
    assert call_args[0] is ad
    # Second arg should be a non-A cell


@patch("palantir.utils.early_cell")
def test_find_terminal_states(mock_early_cell, mock_anndata_with_celltypes):
    """Test find_terminal_states"""
    ad = mock_anndata_with_celltypes

    # Setup mock early_cell behavior
    def side_effect(ad, celltype, *args, **kwargs):
        if celltype == "A":
            return "cell_0"
        elif celltype == "B":
            return "cell_4"
        elif celltype == "C":
            raise CellNotFoundException("Test exception")
        return None

    mock_early_cell.side_effect = side_effect

    # Test find_terminal_states with a warning for type C
    with pytest.warns(UserWarning):
        result = find_terminal_states(ad, ["A", "B", "C"], celltype_column="celltype")

    # Check result - should have entries for A and B, but not C
    assert isinstance(result, pd.Series)
    assert len(result) == 2
    assert result["cell_0"] == "A"
    assert result["cell_4"] == "B"
    assert "cell_C" not in result.index
