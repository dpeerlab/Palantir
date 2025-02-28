import pytest
import pandas as pd
import numpy as np
import scanpy as sc
from anndata import AnnData
import os
import tempfile
import anndata
import warnings

import palantir


@pytest.fixture
def sample_data():
    """Load the sample data from the data directory"""
    # Get the data directory relative to the test file
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    file_path = os.path.join(data_dir, "marrow_sample_scseq_counts.h5ad")

    # Skip test if the data file doesn't exist
    if not os.path.exists(file_path):
        pytest.skip(f"Sample data file {file_path} not found")

    # Load the data
    ad = anndata.read_h5ad(file_path)
    return ad


@pytest.fixture
def processed_data(sample_data):
    """Process the sample data for Palantir"""
    ad = sample_data.copy()

    # Normalize and log transform
    sc.pp.normalize_per_cell(ad)
    palantir.preprocess.log_transform(ad)

    # Select highly variable genes
    sc.pp.highly_variable_genes(ad, n_top_genes=1500, flavor="cell_ranger")

    # Run PCA
    sc.pp.pca(ad)

    # Run diffusion maps
    palantir.utils.run_diffusion_maps(ad, n_components=5)

    # Determine multiscale space
    palantir.utils.determine_multiscale_space(ad)

    # Set up neighbors for visualization
    sc.pp.neighbors(ad)
    sc.tl.umap(ad)

    # Run MAGIC imputation
    palantir.utils.run_magic_imputation(ad)

    return ad


def test_palantir_reproducibility(processed_data):
    """Test that Palantir results are reproducible"""
    ad = processed_data.copy()

    # Set up terminal states (same as sample notebook)
    terminal_states = pd.Series(
        ["DC", "Mono", "Ery"],
        index=["Run5_131097901611291", "Run5_134936662236454", "Run4_200562869397916"],
    )

    # Set start cell (same as sample notebook)
    start_cell = "Run5_164698952452459"

    # Run Palantir
    pr_res = palantir.core.run_palantir(
        ad, start_cell, num_waypoints=500, terminal_states=terminal_states
    )

    # Expected values for the start cell
    # These are expected probabilities for the start cell from the sample notebook
    expected_probs = {"Ery": 0.33, "DC": 0.33, "Mono": 0.33}

    # Get actual values
    actual_probs = pr_res.branch_probs.loc[start_cell]

    # Check that probabilities are close to expected (start cell should be roughly equal probabilities)
    for branch, expected in expected_probs.items():
        assert (
            np.abs(actual_probs[branch] - expected) < 0.15
        ), f"Branch {branch} probability differs more than expected"

    # Expected values for terminal state cells
    for term_cell, term_name in terminal_states.items():
        # Terminal state cell should have high probability for its own fate
        assert (
            pr_res.branch_probs.loc[term_cell, term_name] > 0.7
        ), f"Terminal state {term_name} doesn't have high probability"

    # Pseudotime should be 0 for start cell (or very close)
    assert pr_res.pseudotime[start_cell] < 0.05, "Start cell pseudotime should be close to 0"

    # Entropy should be high for start cell (multipotent state)
    assert pr_res.entropy[start_cell] > 0.8, "Start cell entropy should be high"

    # Terminal states should have low entropy
    for term_cell in terminal_states.index:
        assert (
            pr_res.entropy[term_cell] < 0.5
        ), f"Terminal state {term_cell} should have low entropy"


def test_branch_selection(processed_data):
    """Test the branch selection functionality"""
    ad = processed_data.copy()

    # Set up terminal states
    terminal_states = pd.Series(
        ["DC", "Mono", "Ery"],
        index=["Run5_131097901611291", "Run5_134936662236454", "Run4_200562869397916"],
    )

    # Run Palantir
    start_cell = "Run5_164698952452459"
    palantir.core.run_palantir(ad, start_cell, num_waypoints=500, terminal_states=terminal_states)

    # Run branch selection
    masks = palantir.presults.select_branch_cells(ad, eps=0)

    # Check that the masks were computed correctly
    assert masks.shape[1] == 3, "Should have 3 branches selected"
    assert masks.shape[0] == ad.n_obs, "Should have a mask for each cell"

    # Check that the masks were stored in the AnnData object
    assert "branch_masks" in ad.obsm, "Branch masks should be stored in obsm"

    # Check that terminal cells are selected in their respective branches
    for term_cell, term_name in terminal_states.items():
        branch_idx = list(ad.obsm["palantir_fate_probabilities"].columns).index(term_name)
        assert masks[ad.obs_names == term_cell, branch_idx][
            0
        ], f"Terminal cell {term_name} should be selected in its branch"


def test_gene_trends(processed_data):
    """Test gene trend computation"""
    ad = processed_data.copy()

    # Set up terminal states
    terminal_states = pd.Series(
        ["DC", "Mono", "Ery"],
        index=["Run5_131097901611291", "Run5_134936662236454", "Run4_200562869397916"],
    )

    # Run Palantir
    start_cell = "Run5_164698952452459"
    palantir.core.run_palantir(ad, start_cell, num_waypoints=500, terminal_states=terminal_states)

    # Select branch cells
    palantir.presults.select_branch_cells(ad, eps=0)

    # Compute gene trends
    gene_trends = palantir.presults.compute_gene_trends(
        ad,
        expression_key="MAGIC_imputed_data",
    )

    # Expected gene expression patterns
    # CD34 should decrease along all lineages (stem cell marker)
    # GATA1 should increase in erythroid lineage
    # MPO should increase in monocyte lineage
    # IRF8 should increase in DC lineage

    # Check that gene trends were computed for all branches
    assert "Ery" in gene_trends, "Erythroid gene trends missing"
    assert "DC" in gene_trends, "DC gene trends missing"
    assert "Mono" in gene_trends, "Monocyte gene trends missing"

    # Check that gene trends were stored in the AnnData object
    assert "gene_trends_Ery" in ad.varm, "Erythroid gene trends not stored in varm"

    # Get the trend data for specific genes
    cd34_ery = ad.varm["gene_trends_Ery"].loc["CD34"].values
    gata1_ery = ad.varm["gene_trends_Ery"].loc["GATA1"].values

    # CD34 should decrease in erythroid lineage (end lower than start)
    assert cd34_ery[0] > cd34_ery[-1], "CD34 should decrease along erythroid lineage"

    # GATA1 should increase in erythroid lineage (end higher than start)
    assert gata1_ery[0] < gata1_ery[-1], "GATA1 should increase along erythroid lineage"


def test_clustering_gene_trends(processed_data):
    """Test clustering of gene trends"""
    ad = processed_data.copy()

    # Set up terminal states
    terminal_states = pd.Series(
        ["DC", "Mono", "Ery"],
        index=["Run5_131097901611291", "Run5_134936662236454", "Run4_200562869397916"],
    )

    # Run Palantir
    start_cell = "Run5_164698952452459"
    palantir.core.run_palantir(ad, start_cell, num_waypoints=500, terminal_states=terminal_states)

    # Select branch cells
    palantir.presults.select_branch_cells(ad, eps=0)

    # Compute gene trends
    palantir.presults.compute_gene_trends(
        ad,
        expression_key="MAGIC_imputed_data",
    )

    # Select a subset of genes for clustering
    genes = ["CD34", "MPO", "GATA1", "IRF8", "CSF1R", "ITGA2B", "CD79A", "CD79B"]

    # Cluster gene trends
    clusters = palantir.presults.cluster_gene_trends(ad, "Ery", genes)

    # Check that all genes were clustered
    assert len(clusters) == len(genes), "Not all genes were clustered"

    # Check that clusters were stored in the AnnData object
    assert "gene_trends_clusters" in ad.var, "Clusters should be stored in var"

    # Related genes should be clustered together
    # For example, CD79A and CD79B should be in the same cluster
    cd79a_cluster = clusters.loc["CD79A"]
    cd79b_cluster = clusters.loc["CD79B"]
    assert cd79a_cluster == cd79b_cluster, "CD79A and CD79B should be in the same cluster"
