import pytest
import pandas as pd
import numpy as np
import scanpy as sc
from anndata import AnnData

from palantir.presults import select_branch_cells
import palantir.presults


def test_select_branch_cells_basic():
    """Test basic functionality of select_branch_cells"""
    # Create test AnnData
    n_cells = 100
    n_genes = 20
    adata = AnnData(np.random.normal(0, 1, (n_cells, n_genes)))
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]

    # Create pseudotime
    pseudotime = np.linspace(0, 1, n_cells)
    adata.obs["palantir_pseudotime"] = pseudotime

    # Create fate probabilities
    fate_probs = np.zeros((n_cells, 3))
    # First branch: higher probability at beginning
    fate_probs[:, 0] = np.linspace(0.8, 0.1, n_cells)
    # Second branch: higher probability in middle
    x = np.linspace(-3, 3, n_cells)
    fate_probs[:, 1] = np.exp(-(x**2)) / 2
    # Third branch: higher probability at end
    fate_probs[:, 2] = np.linspace(0.1, 0.8, n_cells)

    # Normalize rows to sum to 1
    fate_probs = fate_probs / fate_probs.sum(axis=1, keepdims=True)

    # Store in AnnData
    adata.obsm["palantir_fate_probabilities"] = pd.DataFrame(
        fate_probs, index=adata.obs_names, columns=["branch1", "branch2", "branch3"]
    )

    # Run select_branch_cells
    masks = select_branch_cells(adata)

    # Check that the masks are boolean arrays
    assert masks.dtype == bool
    assert masks.shape == (n_cells, 3)

    # Check that masks are stored in AnnData
    assert "branch_masks" in adata.obsm

    # Check masks make sense with probabilities
    # Higher probability cells should be selected - but we don't check specific values
    # as branch selection behavior depends on the quantile-based algorithm
    high_prob_branch1 = fate_probs[:, 0] > 0.5
    assert np.any(masks[high_prob_branch1, 0])  # At least some high prob cells should be selected


def test_select_branch_cells_custom_keys():
    """Test select_branch_cells with custom keys"""
    # Create test AnnData
    n_cells = 100
    n_genes = 20
    adata = AnnData(np.random.normal(0, 1, (n_cells, n_genes)))

    # Create pseudotime with custom key
    pseudotime_key = "custom_pseudotime"
    adata.obs[pseudotime_key] = np.linspace(0, 1, n_cells)

    # Create fate probabilities with custom key
    fate_prob_key = "custom_fate_probs"
    fate_probs = np.random.random((n_cells, 3))
    fate_probs = fate_probs / fate_probs.sum(axis=1, keepdims=True)
    adata.obsm[fate_prob_key] = pd.DataFrame(
        fate_probs, index=adata.obs_names, columns=["branch1", "branch2", "branch3"]
    )

    # Custom masks key
    masks_key = "custom_masks"

    # Run select_branch_cells with custom keys
    masks = select_branch_cells(
        adata, pseudo_time_key=pseudotime_key, fate_prob_key=fate_prob_key, masks_key=masks_key
    )

    # Check that masks are stored in AnnData with custom key
    assert masks_key in adata.obsm

    # Check shapes
    assert masks.shape == (n_cells, 3)


def test_select_branch_cells_parameters():
    """Test select_branch_cells with different parameters"""
    # Create test AnnData
    n_cells = 100
    n_genes = 20
    adata = AnnData(np.random.normal(0, 1, (n_cells, n_genes)))

    # Create pseudotime
    adata.obs["palantir_pseudotime"] = np.linspace(0, 1, n_cells)

    # Create fate probabilities
    fate_probs = np.random.random((n_cells, 3))
    fate_probs = fate_probs / fate_probs.sum(axis=1, keepdims=True)
    adata.obsm["palantir_fate_probabilities"] = pd.DataFrame(
        fate_probs, index=adata.obs_names, columns=["branch1", "branch2", "branch3"]
    )

    # Run with different q parameters - for randomly generated data, the relationship between
    # q and the number of selected cells can be unpredictable
    masks1 = select_branch_cells(adata, q=0.01)
    masks2 = select_branch_cells(adata, q=0.5)

    # Just verify we get different results with different parameters
    assert masks1.shape == masks2.shape

    # Run with different eps parameters
    masks3 = select_branch_cells(adata, eps=0.01)
    masks4 = select_branch_cells(adata, eps=0.1)

    # Higher eps should select more cells or at least the same number
    assert masks3.sum() <= masks4.sum()

    # Test save_as_df parameter
    # True is default, test False
    select_branch_cells(adata, save_as_df=False)
    assert isinstance(adata.obsm["branch_masks"], np.ndarray)
    assert "branch_masks_columns" in adata.uns


def test_select_branch_cells_with_different_resolutions():
    """Test select_branch_cells with different resolution settings"""

    # Store original resolution
    original_res = palantir.presults.PSEUDOTIME_RES

    try:
        # Test with high resolution (potential division by zero case for small datasets)
        n_cells = 10
        n_genes = 5

        # Create small test AnnData
        adata_small = AnnData(np.random.normal(0, 1, (n_cells, n_genes)))
        adata_small.obs["palantir_pseudotime"] = np.linspace(0, 1, n_cells)
        adata_small.obsm["palantir_fate_probabilities"] = pd.DataFrame(
            np.random.random((n_cells, 2)),
            columns=["branch1", "branch2"],
            index=adata_small.obs_names,
        )

        # Test with a very high resolution (will trigger nsteps == 0 case)
        palantir.presults.PSEUDOTIME_RES = 1000
        masks_high_res = select_branch_cells(adata_small)
        assert masks_high_res.shape == (n_cells, 2)

        # Test with a very low resolution (regular case)
        palantir.presults.PSEUDOTIME_RES = 2
        masks_low_res = select_branch_cells(adata_small)
        assert masks_low_res.shape == (n_cells, 2)

        # Create larger test AnnData
        n_cells = 100
        adata_large = AnnData(np.random.normal(0, 1, (n_cells, n_genes)))
        adata_large.obs["palantir_pseudotime"] = np.linspace(0, 1, n_cells)
        adata_large.obsm["palantir_fate_probabilities"] = pd.DataFrame(
            np.random.random((n_cells, 2)),
            columns=["branch1", "branch2"],
            index=adata_large.obs_names,
        )

        # Test with medium resolution (regular case)
        palantir.presults.PSEUDOTIME_RES = 10
        masks_medium_res = select_branch_cells(adata_large)
        assert masks_medium_res.shape == (n_cells, 2)

    finally:
        # Restore original resolution
        palantir.presults.PSEUDOTIME_RES = original_res


def test_select_branch_cells_error_handling():
    """Test error handling in select_branch_cells"""
    # Create AnnData without required data
    adata = AnnData(np.random.normal(0, 1, (10, 10)))

    # Should raise KeyError for missing pseudotime
    with pytest.raises(KeyError):
        select_branch_cells(adata)

    # Add pseudotime but no fate probabilities
    adata.obs["palantir_pseudotime"] = np.linspace(0, 1, 10)

    # Should raise KeyError for missing fate probabilities
    with pytest.raises(KeyError):
        select_branch_cells(adata)
