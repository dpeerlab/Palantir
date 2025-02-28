import pytest
import pandas as pd
import numpy as np
import scanpy as sc
from anndata import AnnData

from palantir.presults import cluster_gene_trends


def test_cluster_gene_trends_basic():
    """Test basic functionality of cluster_gene_trends"""
    # Create a simple DataFrame of gene trends
    n_genes = 30
    n_timepoints = 50

    # Create some patterns that should cluster together
    timepoints = np.linspace(0, 1, n_timepoints)

    # Create random trends with some patterns
    np.random.seed(42)
    trends = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)], columns=timepoints)

    # First 10 genes follow similar pattern (increasing)
    for i in range(10):
        trends.iloc[i] = np.linspace(0, 1, n_timepoints) + np.random.normal(0, 0.1, n_timepoints)

    # Next 10 genes follow another pattern (decreasing)
    for i in range(10, 20):
        trends.iloc[i] = np.linspace(1, 0, n_timepoints) + np.random.normal(0, 0.1, n_timepoints)

    # Last 10 genes follow a third pattern (bell curve)
    for i in range(20, 30):
        trends.iloc[i] = np.sin(np.linspace(0, np.pi, n_timepoints)) + np.random.normal(
            0, 0.1, n_timepoints
        )

    # Test with DataFrame
    clusters = cluster_gene_trends(trends, "branch1")

    # Check output
    assert isinstance(clusters, pd.Series)
    assert len(clusters) == n_genes
    assert clusters.index.equals(trends.index)

    # There should be at least 2 clusters found
    assert len(clusters.unique()) >= 2

    # Check that similar genes are clustered together
    # First 10 genes should mostly be in the same cluster
    first_cluster = clusters.iloc[:10].mode().iloc[0]
    assert (
        clusters.iloc[:10] == first_cluster
    ).mean() > 0.5  # More than half should be in the same cluster


def test_cluster_gene_trends_anndata():
    """Test cluster_gene_trends with AnnData input"""
    # Create AnnData object
    n_cells = 100
    n_genes = 30
    adata = AnnData(np.random.normal(0, 1, (n_cells, n_genes)))
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]

    # Create gene trends for the branch
    n_timepoints = 50
    timepoints = np.linspace(0, 1, n_timepoints)
    branch_key = "test_branch"

    # Same trends as before
    trends = np.zeros((n_genes, n_timepoints))
    # First 10 genes
    for i in range(10):
        trends[i] = np.linspace(0, 1, n_timepoints) + np.random.normal(0, 0.1, n_timepoints)
    # Next 10 genes
    for i in range(10, 20):
        trends[i] = np.linspace(1, 0, n_timepoints) + np.random.normal(0, 0.1, n_timepoints)
    # Last 10 genes
    for i in range(20, 30):
        trends[i] = np.sin(np.linspace(0, np.pi, n_timepoints)) + np.random.normal(
            0, 0.1, n_timepoints
        )

    # Store the trends in AnnData
    adata.varm[f"gene_trends_{branch_key}"] = pd.DataFrame(
        trends, index=adata.var_names, columns=[str(t) for t in timepoints]
    )

    # Run clustering
    clusters = cluster_gene_trends(adata, branch_key, gene_trend_key="gene_trends")

    # Check output
    assert isinstance(clusters, pd.Series)
    assert len(clusters) == n_genes
    assert clusters.index.equals(adata.var_names)

    # The clusters should be stored in the var annotation
    assert "gene_trends_clusters" in adata.var
    assert np.all(adata.var["gene_trends_clusters"] == clusters)


def test_cluster_gene_trends_custom_genes():
    """Test cluster_gene_trends with subset of genes"""
    # Create a simple DataFrame of gene trends
    n_genes = 30
    n_timepoints = 50
    timepoints = np.linspace(0, 1, n_timepoints)

    # Create trends
    np.random.seed(42)
    trends = pd.DataFrame(
        np.random.normal(0, 1, (n_genes, n_timepoints)),
        index=[f"gene_{i}" for i in range(n_genes)],
        columns=timepoints,
    )

    # Select a subset of genes
    selected_genes = [f"gene_{i}" for i in range(0, n_genes, 2)]  # Every other gene

    # Test with subset of genes
    clusters = cluster_gene_trends(trends, "branch1", genes=selected_genes)

    # Check output
    assert isinstance(clusters, pd.Series)
    assert len(clusters) == len(selected_genes)
    assert set(clusters.index) == set(selected_genes)


def test_cluster_gene_trends_parameters():
    """Test cluster_gene_trends with custom parameters"""
    # Create a simple DataFrame of gene trends
    n_genes = 30
    n_timepoints = 50
    timepoints = np.linspace(0, 1, n_timepoints)

    # Create trends
    np.random.seed(42)
    trends = pd.DataFrame(
        np.random.normal(0, 1, (n_genes, n_timepoints)),
        index=[f"gene_{i}" for i in range(n_genes)],
        columns=timepoints,
    )

    # Test with custom parameters
    clusters1 = cluster_gene_trends(trends, "branch1", n_neighbors=10)
    clusters2 = cluster_gene_trends(trends, "branch1", n_neighbors=20)

    # The clusters should be different with different parameters
    assert (clusters1 != clusters2).any()


def test_cluster_gene_trends_error_handling():
    """Test error handling of cluster_gene_trends"""
    # Create AnnData without varm data
    adata = AnnData(np.random.normal(0, 1, (10, 10)))

    # Should raise KeyError for missing gene_trend_key
    with pytest.raises(KeyError):
        cluster_gene_trends(adata, "branch1", gene_trend_key=None)

    # Should raise KeyError for missing branch data
    with pytest.raises(KeyError):
        cluster_gene_trends(adata, "nonexistent_branch", gene_trend_key="some_key")
