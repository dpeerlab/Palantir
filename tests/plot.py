import pytest
from unittest.mock import patch
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle

from palantir.plot import (
    density_2d,
    plot_molecules_per_cell_and_gene,
    cell_types,
    highlight_cells_on_umap,
    plot_tsne_by_cell_sizes,
    plot_gene_expression,
    plot_diffusion_components,
    plot_palantir_results,
    plot_terminal_state_probs,
    plot_branch_selection,
    plot_gene_trends_legacy,
    plot_gene_trends,
    plot_stats,
    plot_branch,
    plot_trend,
    plot_gene_trend_heatmaps,
    plot_gene_trend_clusters,
    gene_score_histogram,
)
from palantir.presults import PResults


# Fixtures for the UMAP DataFrame
@pytest.fixture
def mock_umap_df():
    return pd.DataFrame(
        {"x": np.random.rand(100), "y": np.random.rand(100)},
        index=[f"cell_{i}" for i in range(100)],
    )


@pytest.fixture
def mock_tsne():
    return pd.DataFrame(
        {"x": np.random.rand(100), "y": np.random.rand(100)},
        index=[f"cell_{i}" for i in range(100)],
    )


@pytest.fixture
def mock_data():
    return pd.DataFrame(
        np.random.rand(100, 20),
        index=[f"cell_{i}" for i in range(100)],
    )


@pytest.fixture
def mock_clusters():
    return pd.Series(
        ["Cluster_1"] * 50 + ["Cluster_2"] * 50,
        index=[f"cell_{i}" for i in range(100)],
    )


@pytest.fixture
def mock_cluster_colors():
    return pd.Series({"Cluster_1": "#FF0000", "Cluster_2": "#00FF00"})


@pytest.fixture
def mock_gene_data():
    return pd.DataFrame(
        np.random.rand(100, 5),
        columns=[f"gene_{i}" for i in range(5)],
        index=[f"cell_{i}" for i in range(100)],
    )


@pytest.fixture
def mock_dm_res():
    return pd.DataFrame(
        np.random.rand(100, 3),
        index=[f"cell_{i}" for i in range(100)],
    )


@pytest.fixture
def mock_presults():
    cell_index = [f"cell_{i}" for i in range(100)]
    return PResults(
        pseudotime=pd.Series(np.random.rand(100), index=cell_index),
        entropy=pd.Series(np.random.rand(100), index=cell_index),
        branch_probs=pd.DataFrame(
            np.random.rand(100, 3),
            index=cell_index,
        ),
        waypoints=None,
    )


@pytest.fixture
def mock_cells():
    return [f"cell_{i}" for i in range(10)]


@pytest.fixture
def mock_gene_trends():
    return {
        "Branch_1": {
            "trends": pd.DataFrame(
                {"0.0": [0.2, 0.3], "1.0": [0.4, 0.5]}, index=["Gene1", "Gene2"]
            ),
            "std": pd.DataFrame(
                {"0.0": [0.02, 0.03], "1.0": [0.04, 0.05]}, index=["Gene1", "Gene2"]
            ),
        },
        "Branch_2": {
            "trends": pd.DataFrame(
                {"0.0": [0.1, 0.2], "1.0": [0.2, 0.3]}, index=["Gene1", "Gene2"]
            ),
            "std": pd.DataFrame(
                {"0.0": [0.01, 0.02], "1.0": [0.02, 0.03]}, index=["Gene1", "Gene2"]
            ),
        },
    }


# Fixtures for AnnData object
@pytest.fixture
def mock_anndata(mock_umap_df):
    adata = sc.AnnData(X=np.random.randn(100, 5))
    adata.obs_names = mock_umap_df.index
    adata.var_names = [f"gene_{i}" for i in range(5)]
    adata.obs["palantir_pseudotime"] = np.random.rand(100)
    adata.obs["palantir_entropy"] = np.random.rand(100)
    adata.obsm["X_umap"] = mock_umap_df.values
    adata.obsm["DM_EigenVectors"] = np.random.randn(100, 3)
    adata.obsm["palantir_fate_probabilities"] = pd.DataFrame(
        np.random.randn(100, 3),
        columns=["a", "b", "c"],
        index=mock_umap_df.index,
    )
    adata.obsm["branch_masks"] = pd.DataFrame(
        np.random.randint(2, size=(100, 3)),
        columns=["a", "b", "c"],
        index=mock_umap_df.index,
        dtype=bool,
    )
    for branch in ["a", "b", "c"]:
        adata.uns[f"gene_trends_{branch}_pseudotime"] = np.linspace(0, 1, 10)
        adata.varm[f"gene_trends_{branch}"] = pd.DataFrame(
            np.random.rand(5, 10),
            index=adata.var_names,
            columns=adata.uns[f"gene_trends_{branch}_pseudotime"],
        )
    adata.var["clusters"] = pd.Series(
        ["A", "A", "B", "B", "B"],
        index=adata.var_names,
    )
    adata.var["gene_score"] = np.random.rand(5)
    return adata


def test_density_2d():
    # Test with random data
    x = np.random.rand(100)
    y = np.random.rand(100)
    x_out, y_out, z_out = density_2d(x, y)

    # Validate output shape and types
    assert x_out.shape == x.shape
    assert y_out.shape == y.shape
    assert z_out.shape == x.shape


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
    plt.close()


def test_cell_types_default_colors(mock_tsne, mock_clusters):
    fig, axs = cell_types(mock_tsne, mock_clusters)
    assert len(axs) == 2, "Number of axes should match number of clusters"


# Test with custom cluster_colors
def test_cell_types_custom_colors(mock_tsne, mock_clusters, mock_cluster_colors):
    fig, axs = cell_types(mock_tsne, mock_clusters, cluster_colors=mock_cluster_colors)
    assert len(axs) == 2, "Number of axes should match number of clusters"

    # Check if colors match
    colors = pd.Series(
        sc.get_edgecolor() for ax in axs.values() for sc in ax.collections
    )
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


def test_highlight_cells_on_umap(mock_anndata, mock_umap_df):
    # Test KeyError
    with pytest.raises(KeyError):
        highlight_cells_on_umap(
            mock_anndata, ["cell_1"], embedding_basis="unknown_basis"
        )

    # Test TypeError for data
    with pytest.raises(TypeError):
        highlight_cells_on_umap("InvalidType", ["cell_1"])

    # Test TypeError for cells
    with pytest.raises(TypeError):
        highlight_cells_on_umap(mock_anndata, 123)

    # Test normal use case with AnnData
    fig, ax = highlight_cells_on_umap(mock_anndata, ["cell_1", "cell_2"])
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)

    # Test normal use case with DataFrame
    fig, ax = highlight_cells_on_umap(mock_umap_df, ["cell_1", "cell_2"])
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)

    # Test with different types for cells parameter
    fig, ax = highlight_cells_on_umap(
        mock_anndata, {"cell_1": "label1", "cell_2": "label2"}
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)

    mask = np.array([True if i < 2 else False for i in range(100)])
    fig, ax = highlight_cells_on_umap(mock_anndata, mask)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)

    cell_series = pd.Series({"cell_1": "label1", "cell_2": "label2"})
    fig, ax = highlight_cells_on_umap(mock_anndata, cell_series)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)


# Test plot_tsne_by_cell_sizes
def test_plot_tsne_by_cell_sizes(mock_data, mock_tsne):
    fig, ax = plot_tsne_by_cell_sizes(mock_data, mock_tsne)

    # Validate plot and axis
    assert isinstance(fig, plt.Figure), "Output should include a matplotlib Figure"
    assert ax.collections, "Should have scatter plots"
    assert (
        ax.collections[0].get_array().data.shape[0] == 100
    ), "Scatter plot color should be based on 100 data points"

    # Test colorbar
    cbar = fig.colorbar(ax.collections[0], ax=ax)
    assert cbar is not None, "Colorbar should exist"

    # Test vmin, vmax
    fig, ax = plot_tsne_by_cell_sizes(mock_data, mock_tsne, vmin=0.2, vmax=0.8)
    cbar_mappable = ax.collections[0]
    assert cbar_mappable.get_clim() == (
        0.2,
        0.8,
    ), "Color limits should be set to vmin and vmax"
    plt.close()


def test_plot_gene_expression(mock_gene_data, mock_tsne):
    genes = ["gene_0", "gene_1"]
    fig, axs = plot_gene_expression(mock_gene_data, mock_tsne, genes, plot_scale=True)
    assert isinstance(fig, plt.Figure)
    plt.close()


def test_plot_gene_expression_missing_genes(mock_gene_data, mock_tsne):
    genes = ["gene_0", "nonexistent_gene"]
    fig, axs = plot_gene_expression(mock_gene_data, mock_tsne, genes)
    assert isinstance(fig, plt.Figure)  # Expect a warning but still a plot
    plt.close()


def test_plot_gene_expression_no_genes(mock_gene_data, mock_tsne):
    with pytest.raises(ValueError):
        plot_gene_expression(mock_gene_data, mock_tsne, ["nonexistent_gene"])
    plt.close()


def test_plot_diffusion_components_with_anndata(mock_anndata, mock_dm_res):
    fig, axs = plot_diffusion_components(mock_anndata)
    assert isinstance(fig, plt.Figure)
    for ax in axs.values():
        assert isinstance(ax, plt.Axes)
    plt.close()


def test_plot_diffusion_components_with_dataframe(mock_tsne, mock_dm_res):
    dm_res_dict = {"EigenVectors": mock_dm_res}
    fig, axs = plot_diffusion_components(mock_tsne, dm_res=dm_res_dict)
    assert isinstance(fig, plt.Figure)
    for ax in axs.values():
        assert isinstance(ax, plt.Axes)
    plt.close()


def test_plot_diffusion_components_key_error_embedding(mock_anndata):
    with pytest.raises(KeyError):
        plot_diffusion_components(mock_anndata, embedding_basis="NonexistentKey")
    plt.close()


def test_plot_diffusion_components_key_error_dm_res(mock_anndata):
    with pytest.raises(KeyError):
        plot_diffusion_components(mock_anndata, dm_res="NonexistentKey")
    plt.close()


def test_plot_diffusion_components_default_args(mock_anndata):
    fig, axs = plot_diffusion_components(mock_anndata)
    for ax in axs.values():
        assert (
            ax.collections[0].get_array().data.shape[0] == 100
        )  # Checking data points
    plt.close()


def test_plot_diffusion_components_custom_args(mock_anndata):
    fig, axs = plot_diffusion_components(mock_anndata, s=10, edgecolors="r")
    for ax in axs.values():
        assert ax.collections[0].get_edgecolors().all() == np.array([1, 0, 0, 1]).all()
        assert ax.collections[0].get_sizes()[0] == 10
    plt.close()


# Test with AnnData and all keys available
def test_plot_palantir_results_anndata(mock_anndata):
    fig = plot_palantir_results(mock_anndata)
    assert isinstance(fig, plt.Figure)
    plt.close()


# Test with DataFrame and PResults
def test_plot_palantir_results_dataframe(mock_tsne, mock_presults):
    fig = plot_palantir_results(mock_tsne, pr_res=mock_presults)
    assert isinstance(fig, plt.Figure)
    plt.close()


# Test KeyError for missing embedding_basis
def test_plot_palantir_results_key_error_embedding(mock_anndata):
    with pytest.raises(KeyError):
        plot_palantir_results(mock_anndata, embedding_basis="NonexistentKey")
    plt.close()


# Test KeyError for missing Palantir results in AnnData
def test_plot_palantir_results_key_error_palantir(mock_anndata):
    mock_anndata.obs = pd.DataFrame(index=mock_anndata.obs_names)  # Clearing obs
    with pytest.raises(KeyError):
        plot_palantir_results(mock_anndata)
    plt.close()


# Test plotting with custom arguments
def test_plot_palantir_results_custom_args(mock_anndata):
    fig = plot_palantir_results(mock_anndata, s=10, edgecolors="r")
    ax = fig.axes[0]  # Assuming first subplot holds the first scatter plot
    assert np.all(ax.collections[0].get_edgecolors() == [1, 0, 0, 1])
    assert ax.collections[0].get_sizes()[0] == 10
    plt.close()


# Test with AnnData and all keys available
def test_plot_terminal_state_probs_anndata(mock_anndata, mock_cells):
    fig = plot_terminal_state_probs(mock_anndata, mock_cells)
    assert isinstance(fig, plt.Figure)
    plt.close()


# Test with DataFrame and PResults
def test_plot_terminal_state_probs_dataframe(mock_data, mock_presults, mock_cells):
    fig = plot_terminal_state_probs(mock_data, mock_cells, pr_res=mock_presults)
    assert isinstance(fig, plt.Figure)
    plt.close()


# Test ValueError for missing pr_res in DataFrame input
def test_plot_terminal_state_probs_value_error(mock_data, mock_cells):
    with pytest.raises(ValueError):
        plot_terminal_state_probs(mock_data, mock_cells)
    plt.close()


# Test plotting with custom arguments
def test_plot_terminal_state_probs_custom_args(mock_anndata, mock_cells):
    fig = plot_terminal_state_probs(mock_anndata, mock_cells, linewidth=2.0)
    ax = fig.axes[0]  # Assuming first subplot holds the first bar plot
    assert ax.patches[0].get_linewidth() == 2.0
    plt.close()


# Test if the function uses the correct keys and raises appropriate errors
def test_plot_branch_selection_keys(mock_anndata):
    # This will depend on how your mock_anndata is structured
    with pytest.raises(KeyError):
        plot_branch_selection(mock_anndata, pseudo_time_key="invalid_key")
    plt.close()

    with pytest.raises(KeyError):
        plot_branch_selection(mock_anndata, fate_prob_key="invalid_key")
    plt.close()

    with pytest.raises(KeyError):
        plot_branch_selection(mock_anndata, embedding_basis="invalid_basis")
    plt.close()


# Test the scatter custom arguments
def test_plot_branch_selection_custom_args(mock_anndata):
    fig = plot_branch_selection(mock_anndata, marker="x", alpha=0.5)
    ax1, ax2 = (
        fig.axes[0],
        fig.axes[1],
    )  # Assuming the first two axes correspond to the first fate

    # Extract the scatter plots, assuming that the plot with custom markers is the last one
    scatter1, scatter2 = ax1.collections[-1], ax2.collections[-1]

    alpha1 = scatter1.get_alpha()
    assert alpha1 == 0.5
    plt.close()


# Test 1: Basic functionality
def test_plot_gene_trends_legacy_basic(mock_gene_trends):
    fig = plot_gene_trends_legacy(mock_gene_trends)
    axes = fig.axes
    # Check if the number of subplots matches the number of genes
    assert len(axes) == 2
    plt.close()


# Test 2: Custom gene list
def test_plot_gene_trends_legacy_custom_genes(mock_gene_trends):
    fig = plot_gene_trends_legacy(mock_gene_trends, genes=["Gene1"])
    axes = fig.axes
    # Check if the number of subplots matches the number of custom genes
    assert len(axes) == 1
    # Check if the title of the subplot matches the custom gene
    assert axes[0].get_title() == "Gene1"
    plt.close()


# Test 3: Color consistency
def test_plot_gene_trends_legacy_color_consistency(mock_gene_trends):
    fig = plot_gene_trends_legacy(mock_gene_trends)
    axes = fig.axes
    colors_1 = [line.get_color() for line in axes[0].lines]
    colors_2 = [line.get_color() for line in axes[1].lines]
    # Check if the colors are consistent across different genes
    assert colors_1 == colors_2
    plt.close()


# Test 1: Basic Functionality with AnnData
def test_plot_gene_trends_basic_anndata(mock_anndata):
    fig = plot_gene_trends(mock_anndata)
    axes = fig.axes
    assert len(axes) == mock_anndata.n_vars
    plt.close()


# Test 2: Basic Functionality with Dictionary
def test_plot_gene_trends_basic_dict(mock_gene_trends):
    fig = plot_gene_trends(mock_gene_trends)
    axes = fig.axes
    assert len(axes) == 2  # Mock data contains 2 genes
    plt.close()


# Test 3: Custom Genes
def test_plot_gene_trends_custom_genes(mock_anndata):
    fig = plot_gene_trends(mock_anndata, genes=["gene_1"])
    axes = fig.axes
    assert len(axes) == 1
    assert axes[0].get_title() == "gene_1"
    plt.close()


# Test 4: Custom Branch Names
def test_plot_gene_trends_custom_branch_names(mock_anndata):
    fig = plot_gene_trends(mock_anndata, branch_names=["a", "b"])
    axes = fig.axes
    assert len(axes) == mock_anndata.n_vars
    plt.close()


# Test 5: Error Handling - Invalid Data Type
def test_plot_gene_trends_invalid_data_type():
    with pytest.raises(ValueError):
        plot_gene_trends("invalid_data_type")
    plt.close()


# Test 6: Error Handling - Missing Key
def test_plot_gene_trends_missing_key(mock_anndata):
    with pytest.raises(KeyError):
        plot_gene_trends(
            mock_anndata, gene_trend_key="missing_key", branch_names="missing_branch"
        )
    plt.close()


@pytest.mark.parametrize("wrong_type", [123, True, 1.23, "unknown_key"])
def test_plot_stats_key_errors(mock_anndata, wrong_type):
    with pytest.raises(KeyError):
        plot_stats(mock_anndata, x=wrong_type, y="palantir_pseudotime")
    plt.close()


def test_plot_stats_basic(mock_anndata):
    fig, ax = plot_stats(mock_anndata, x="palantir_pseudotime", y="palantir_entropy")
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)


def test_plot_stats_optional_parameters(mock_anndata):
    fig, ax = plot_stats(
        mock_anndata,
        x="palantir_pseudotime",
        y="palantir_entropy",
        color="palantir_entropy",
    )
    plt.close()


def test_plot_stats_masking(mock_anndata):
    # Create a condition here that you want to mask
    mask_condition = mock_anndata.obs["palantir_pseudotime"] > 0.5
    mock_anndata.obsm["branch_masks"] = pd.DataFrame({"mock_branch": mask_condition})
    fig, ax = plot_stats(
        mock_anndata,
        x="palantir_pseudotime",
        y="palantir_entropy",
        masks_key="branch_masks",
    )
    plt.close()


@pytest.mark.parametrize(
    "branch_name, position, pseudo_time_key, should_fail",
    [
        ("a", "gene_1", "palantir_pseudotime", False),
        (123, "gene_1", "palantir_pseudotime", True),
        ("b", "gene_1", 123, True),
    ],
)
def test_plot_branch_input_validation(
    mock_anndata, branch_name, position, pseudo_time_key, should_fail
):
    if should_fail:
        with pytest.raises((TypeError, ValueError)):
            plot_branch(
                mock_anndata, branch_name, position, pseudo_time_key=pseudo_time_key
            )
    else:
        plot_branch(
            mock_anndata, branch_name, position, pseudo_time_key=pseudo_time_key
        )
        plt.close()


def test_plot_branch_functionality(mock_anndata):
    fig, ax = plot_branch(mock_anndata, "a", "gene_1")
    assert ax.get_xlabel() == "Pseudotime"


def test_plot_trend_type_validation(mock_anndata):
    with pytest.raises(TypeError):
        plot_trend("string_instead_of_anndata", "a", "gene_1")
    plt.close()
    with pytest.raises(TypeError):
        plot_trend(mock_anndata, 123, "gene_1")
    plt.close()


def test_plot_trend_value_validation(mock_anndata):
    with pytest.raises((ValueError, KeyError)):
        plot_trend(mock_anndata, "nonexistent_branch", "gene_1")
    plt.close()


def test_plot_trend_plotting(mock_anndata):
    fig, ax = plot_trend(mock_anndata, "a", "gene_1")
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close()


def test_plot_gene_trend_heatmaps(mock_anndata):
    fig = plot_gene_trend_heatmaps(
        mock_anndata, genes=["gene_1", "gene_2"], scaling="z-score"
    )

    # Test returned type
    assert isinstance(fig, plt.Figure)

    # Test number of subplots (should be same as number of branches)
    assert len(fig.axes) == len(mock_anndata.obsm["branch_masks"].columns) * 2

    plt.close(fig)


def test_plot_gene_trend_clusters(mock_anndata):
    # Test with AnnData object
    fig = plot_gene_trend_clusters(mock_anndata, branch_name="a", clusters="clusters")
    assert isinstance(fig, plt.Figure)

    # Verify number of subplots
    unique_clusters = mock_anndata.var["clusters"].unique()
    expected_subplots = len(unique_clusters)
    assert len(fig.axes) == expected_subplots

    # Test DataFrame input
    trends_df = mock_anndata.varm["gene_trends_a"]
    clusters_series = mock_anndata.var["clusters"]
    fig_df = plot_gene_trend_clusters(trends_df, clusters=clusters_series)

    assert isinstance(fig_df, plt.Figure)
    assert len(fig_df.axes) == expected_subplots

    plt.close(fig)
    plt.close(fig_df)


def test_gene_score_histogram(mock_anndata):
    # Test with minimum required parameters
    fig = gene_score_histogram(mock_anndata, "gene_score")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    # Test with optional parameters
    fig = gene_score_histogram(
        mock_anndata,
        "gene_score",
        genes=["gene_0", "gene_1"],
        bins=50,
        quantile=0.9,
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    # Test with None quantile
    fig = gene_score_histogram(
        mock_anndata,
        "gene_score",
        quantile=None,
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_gene_score_histogram_errors(mock_anndata):
    # Test with invalid AnnData
    with pytest.raises(ValueError):
        gene_score_histogram(None, "gene_score")

    # Test with invalid score_key
    with pytest.raises(ValueError):
        gene_score_histogram(mock_anndata, "invalid_key")

    # Test with invalid gene
    with pytest.raises(ValueError):
        gene_score_histogram(mock_anndata, "gene_score", genes=["invalid_gene"])

    # Test with invalid quantile
    with pytest.raises(ValueError):
        gene_score_histogram(mock_anndata, "gene_score", quantile=1.5)
