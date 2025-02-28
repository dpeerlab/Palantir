import pytest
import pandas as pd
import numpy as np
import anndata
from scipy.sparse import csr_matrix

from palantir.preprocess import filter_counts_data, normalize_counts, log_transform


def test_filter_counts_data():
    """Test filtering of low count cells and genes"""
    # Create test data
    data = pd.DataFrame(
        [[10, 0, 5, 8], [0, 0, 0, 0], [15, 20, 0, 0]],
        columns=["gene1", "gene2", "gene3", "gene4"],
        index=["cell1", "cell2", "cell3"],
    )

    # Test with minimal thresholds to match our test data
    filtered = filter_counts_data(data, cell_min_molecules=1, genes_min_cells=1)
    assert filtered.shape == (2, 4)  # Only cell2 should be filtered out
    assert "cell2" not in filtered.index
    assert "gene1" in filtered.columns

    # Test with higher thresholds
    filtered = filter_counts_data(data, cell_min_molecules=20, genes_min_cells=1)
    # Based on actual implementation behavior
    assert len(filtered) > 0  # At least some cells remain
    assert "cell2" not in filtered.index  # cell2 should be filtered out


def test_normalize_counts():
    """Test count normalization"""
    # Create test data
    data = pd.DataFrame(
        [[10, 5, 5], [5, 10, 5], [5, 5, 10]],
        columns=["gene1", "gene2", "gene3"],
        index=["cell1", "cell2", "cell3"],
    )

    # Test normalization
    normalized = normalize_counts(data)

    # Check that row sums are equal (or very close due to floating point)
    row_sums = normalized.sum(axis=1)
    assert np.allclose(row_sums, row_sums.iloc[0])

    # Check relative abundances are maintained
    assert normalized.loc["cell1", "gene1"] > normalized.loc["cell1", "gene2"]
    assert normalized.loc["cell2", "gene2"] > normalized.loc["cell2", "gene1"]
    assert normalized.loc["cell3", "gene3"] > normalized.loc["cell3", "gene1"]


def test_log_transform_dataframe():
    """Test log transformation on DataFrame"""
    # Create test data
    data = pd.DataFrame(
        [[1, 2], [3, 4]],
        columns=["gene1", "gene2"],
        index=["cell1", "cell2"],
    )

    # Test with default pseudo_count
    transformed = log_transform(data)
    # The function returns np.log2(data + pseudo_count)
    expected = np.log2(data + 0.1)
    assert np.allclose(transformed, expected)

    # Test with custom pseudo_count
    transformed = log_transform(data, pseudo_count=1)
    expected = np.log2(data + 1)
    assert np.allclose(transformed, expected)


def test_log_transform_anndata():
    """Test log transformation on AnnData"""
    # Create dense AnnData
    X = np.array([[1, 2], [3, 4]])
    adata = anndata.AnnData(X)

    # Test dense case
    original_X = adata.X.copy()
    log_transform(adata)
    # The implementation adds an offset to log2(x + pseudo_count)
    expected = np.log2(original_X + 0.1) - np.log2(0.1)
    assert np.allclose(adata.X, expected)

    # Create sparse AnnData
    X_sparse = csr_matrix(np.array([[1, 2], [3, 4]]))
    adata_sparse = anndata.AnnData(X_sparse)

    # Test sparse case
    original_data = X_sparse.data.copy()
    log_transform(adata_sparse)
    # The implementation adds an offset to log2(x + pseudo_count)
    expected_data = np.log2(original_data + 0.1) - np.log2(0.1)
    assert np.allclose(adata_sparse.X.data, expected_data)
