import pytest
from unittest.mock import patch
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib
import matplotlib.pyplot as plt

from palantir.plot import (
    plot_molecules_per_cell_and_gene,
    cell_types,
    highlight_cells_on_umap,
    plot_tsne_by_cell_sizes,
    plot_gene_expression,
    plot_diffusion_components,
    plot_palantir_results,
)
from palantir.presults import PResults

# Fixtures for the UMAP DataFrame
@pytest.fixture
def mock_umap_df():
    return pd.DataFrame({'x': np.random.rand(100), 'y': np.random.rand(100)}, index=[f'cell_{i}' for i in range(100)])

# Fixtures for AnnData object
@pytest.fixture
def mock_anndata(mock_umap_df):
    adata = sc.AnnData(X=np.random.randn(100, 5))
    adata.obs_names = mock_umap_df.index
    adata.obs['palantir_pseudotime'] = np.random.rand(100)
    adata.obs['palantir_entropy'] = np.random.rand(100)
    adata.obsm['X_umap'] = mock_umap_df.values
    adata.obsm['DM_EigenVectors'] = np.random.randn(100, 3)
    adata.obsm['palantir_fate_probabilities'] = pd.DataFrame(
        np.random.randn(100, 3),
        columns=["a", "b", "c"],
        index=mock_umap_df.index,
    )
    return adata


@pytest.fixture
def mock_tsne():
    return pd.DataFrame({'x': np.random.rand(100), 'y': np.random.rand(100)})

@pytest.fixture
def mock_data():
    return pd.DataFrame(np.random.rand(100, 20))

@pytest.fixture
def mock_clusters():
    return pd.Series(['Cluster_1']*50 + ['Cluster_2']*50)

@pytest.fixture
def mock_cluster_colors():
    return pd.Series({'Cluster_1': '#FF0000', 'Cluster_2': '#00FF00'})

@pytest.fixture
def mock_gene_data():
    return pd.DataFrame(np.random.rand(100, 5), columns=[f'gene_{i}' for i in range(5)])

@pytest.fixture
def mock_dm_res():
    return pd.DataFrame(np.random.rand(100, 3))

@pytest.fixture
def mock_presults():
    return PResults(
        pseudotime=pd.Series(np.random.rand(100)),
        entropy=pd.Series(np.random.rand(100)),
        branch_probs=pd.DataFrame(np.random.rand(100, 3)),
        waypoints=None
    )

def test_plot_molecules_per_cell_and_gene():
    # Create synthetic data
    data = np.random.rand(100, 20)

    # Generate plot
    fig, ax = plot_molecules_per_cell_and_gene(data)

    # Validate plot properties
    assert isinstance(fig, plt.Figure), "Output should include a matplotlib Figure"

    # Validate subplots
    assert len(fig.get_axes()) == 3, "Should have 3 subplots"

    # Validate each subplot
    for i, ax in enumerate(fig.get_axes()):
        if i == 0:
            assert ax.get_xlabel() == "Molecules per cell (log10 scale)"
        elif i == 1:
            assert ax.get_xlabel() == "Nonzero cells per gene (log10 scale)"
        else:
            assert ax.get_xlabel() == "Molecules per gene (log10 scale)"

        assert ax.get_ylabel() == "Frequency"

def test_cell_types_default_colors(mock_tsne, mock_clusters):
    fig, axs = cell_types(mock_tsne, mock_clusters)
    assert len(axs) == 2, "Number of axes should match number of clusters"


# Test with custom cluster_colors
def test_cell_types_custom_colors(mock_tsne, mock_clusters, mock_cluster_colors):
    fig, axs = cell_types(mock_tsne, mock_clusters, cluster_colors=mock_cluster_colors)
    assert len(axs) == 2, "Number of axes should match number of clusters"

    # Check if colors match
    colors = pd.Series(sc.get_edgecolor() for ax in axs.values() for sc in ax.collections)
    colors = set(colors.apply(matplotlib.colors.to_rgba))
    expected_colors = set(mock_cluster_colors.apply(matplotlib.colors.to_rgba))
    expected_colors.add(matplotlib.colors.to_rgba("lightgrey"))
    assert set(colors) == expected_colors, "Cluster colors should match"


# Test n_cols parameter
def test_cell_types_n_cols(mock_tsne, mock_clusters):
    fig, axs = cell_types(mock_tsne, mock_clusters, n_cols=1)
    assert len(axs) == 2, "Number of axes should match number of clusters"
    ncols = len(set(ax.get_subplotspec().colspan for ax in fig.axes))
    assert ncols == 1, "Number of columns should be 1"

# Test highlight_cells_on_umap
def test_highlight_cells_on_umap(mock_umap_df, mock_anndata):
    # Define cells to highlight
    highlight_cells_dict = {'cell_1': 'A', 'cell_3': 'B'}

    # Test with DataFrame
    fig, ax = highlight_cells_on_umap(mock_umap_df, highlight_cells_dict)
    assert isinstance(fig, plt.Figure), "Output should include a matplotlib Figure"
    assert ax.collections, "Should have scatter plots"

    # Test with AnnData
    fig, ax = highlight_cells_on_umap(mock_anndata, highlight_cells_dict)
    assert isinstance(fig, plt.Figure), "Output should include a matplotlib Figure"
    assert ax.collections, "Should have scatter plots"

    # Test annotation_offset
    fig, ax = highlight_cells_on_umap(mock_umap_df, highlight_cells_dict, annotation_offset=0.05)
    assert isinstance(fig, plt.Figure), "Output should include a matplotlib Figure"

    # Test size of highlighted points
    fig, ax = highlight_cells_on_umap(mock_umap_df, highlight_cells_dict, s_highlighted=20)
    assert np.any([p.get_sizes()[0] == 20 for p in ax.collections]), "Highlighted scatter point size should be 20"

    # Test errors
    with pytest.raises(KeyError):
        highlight_cells_on_umap(mock_anndata, highlight_cells_dict, embedding_basis='X_invalid')

    with pytest.raises(TypeError):
        highlight_cells_on_umap(mock_anndata, 3)  # Invalid 'cells' argument

# Test plot_tsne_by_cell_sizes
def test_plot_tsne_by_cell_sizes(mock_data, mock_tsne):
    fig, ax = plot_tsne_by_cell_sizes(mock_data, mock_tsne)
    
    # Validate plot and axis
    assert isinstance(fig, plt.Figure), "Output should include a matplotlib Figure"
    assert ax.collections, "Should have scatter plots"
    assert ax.collections[0].get_array().data.shape[0] == 100, "Scatter plot color should be based on 100 data points"

    # Test colorbar
    cbar = fig.colorbar(ax.collections[0], ax=ax)
    assert cbar is not None, "Colorbar should exist"

    # Test vmin, vmax
    fig, ax = plot_tsne_by_cell_sizes(mock_data, mock_tsne, vmin=0.2, vmax=0.8)
    cbar_mappable = ax.collections[0]
    assert cbar_mappable.get_clim() == (0.2, 0.8), "Color limits should be set to vmin and vmax"

def test_plot_gene_expression(mock_gene_data, mock_tsne):
    genes = ['gene_0', 'gene_1']
    fig, axs = plot_gene_expression(mock_gene_data, mock_tsne, genes)
    assert isinstance(fig, plt.Figure)

def test_plot_gene_expression_missing_genes(mock_gene_data, mock_tsne):
    genes = ['gene_0', 'nonexistent_gene']
    fig, axs = plot_gene_expression(mock_gene_data, mock_tsne, genes)
    assert isinstance(fig, plt.Figure)  # Expect a warning but still a plot

def test_plot_gene_expression_no_genes(mock_gene_data, mock_tsne):
    with pytest.raises(ValueError):
        plot_gene_expression(mock_gene_data, mock_tsne, ['nonexistent_gene'])

def test_plot_diffusion_components_with_anndata(mock_anndata, mock_dm_res):
    fig, axs = plot_diffusion_components(mock_anndata)
    assert isinstance(fig, plt.Figure)
    for ax in axs.values():
        assert isinstance(ax, plt.Axes)

def test_plot_diffusion_components_with_dataframe(mock_tsne, mock_dm_res):
    dm_res_dict = {"EigenVectors": mock_dm_res}
    fig, axs = plot_diffusion_components(mock_tsne, dm_res=dm_res_dict)
    assert isinstance(fig, plt.Figure)
    for ax in axs.values():
        assert isinstance(ax, plt.Axes)

def test_plot_diffusion_components_key_error_embedding(mock_anndata):
    with pytest.raises(KeyError):
        plot_diffusion_components(mock_anndata, embedding_basis='NonexistentKey')

def test_plot_diffusion_components_key_error_dm_res(mock_anndata):
    with pytest.raises(KeyError):
        plot_diffusion_components(mock_anndata, dm_res='NonexistentKey')

def test_plot_diffusion_components_default_args(mock_anndata):
    fig, axs = plot_diffusion_components(mock_anndata)
    for ax in axs.values():
        assert ax.collections[0].get_array().data.shape[0] == 100  # Checking data points

def test_plot_diffusion_components_custom_args(mock_anndata):
    fig, axs = plot_diffusion_components(mock_anndata, s=10, edgecolors='r')
    for ax in axs.values():
        assert ax.collections[0].get_edgecolors().all() == np.array([1, 0, 0, 1]).all()
        assert ax.collections[0].get_sizes()[0] == 10

# Test with AnnData and all keys available
def test_plot_palantir_results_anndata(mock_anndata):
    fig = plot_palantir_results(mock_anndata)
    assert isinstance(fig, plt.Figure)

# Test with DataFrame and PResults
def test_plot_palantir_results_dataframe(mock_tsne, mock_presults):
    fig = plot_palantir_results(mock_tsne, pr_res=mock_presults)
    assert isinstance(fig, plt.Figure)

# Test KeyError for missing embedding_basis
def test_plot_palantir_results_key_error_embedding(mock_anndata):
    with pytest.raises(KeyError):
        plot_palantir_results(mock_anndata, embedding_basis='NonexistentKey')

# Test KeyError for missing Palantir results in AnnData
def test_plot_palantir_results_key_error_palantir(mock_anndata):
    mock_anndata.obs = pd.DataFrame(index=mock_anndata.obs_names)  # Clearing obs
    with pytest.raises(KeyError):
        plot_palantir_results(mock_anndata)

# Test plotting with custom arguments
def test_plot_palantir_results_custom_args(mock_anndata):
    fig = plot_palantir_results(mock_anndata, s=10, edgecolors='r')
    ax = fig.axes[0]  # Assuming first subplot holds the first scatter plot
    assert np.all(ax.collections[0].get_edgecolors() == [1, 0, 0, 1])
    assert ax.collections[0].get_sizes()[0] == 10
