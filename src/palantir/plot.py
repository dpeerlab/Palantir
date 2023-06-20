from typing import Union, Optional, List
import warnings
import os
import numpy as np
import pandas as pd
from itertools import chain
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
import scanpy as sc

import matplotlib
from matplotlib import font_manager

try:
    os.environ["DISPLAY"]
except KeyError:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

with warnings.catch_warnings():
    # catch experimental ipython widget warning
    warnings.simplefilter("ignore")
    import seaborn as sns

    sns.set(context="paper", style="ticks", font_scale=1.5, font="Bitstream Vera Sans")

# set plotting defaults
with warnings.catch_warnings():
    # catch warnings that system can't find fonts
    warnings.simplefilter("ignore")
    import seaborn as sns

    fm = font_manager.fontManager
    fm.findfont("Raleway")
    fm.findfont("Lato")

warnings.filterwarnings(action="ignore", message="remove_na is deprecated")


class FigureGrid:
    """
    Generates a grid of axes for plotting
    axes can be iterated over or selected by number. e.g.:
    >>> # iterate over axes and plot some nonsense
    >>> fig = FigureGrid(4, max_cols=2)
    >>> for i, ax in enumerate(fig):
    >>>     plt.plot(np.arange(10) * i)
    >>> # select axis using indexing
    >>> ax3 = fig[3]
    >>> ax3.set_title("I'm axis 3")
    """

    # Figure Grid is favorable for displaying multiple graphs side by side.

    def __init__(self, n: int, max_cols=3, scale=3):
        """
        :param n: number of axes to generate
        :param max_cols: maximum number of axes in a given row
        """

        self.n = n
        self.nrows = int(np.ceil(n / max_cols))
        self.ncols = int(min((max_cols, n)))
        figsize = self.ncols * scale, self.nrows * scale

        # create figure
        self.gs = plt.GridSpec(nrows=self.nrows, ncols=self.ncols)
        self.figure = plt.figure(figsize=figsize)

        # create axes
        self.axes = {}
        for i in range(n):
            row = int(i // self.ncols)
            col = int(i % self.ncols)
            self.axes[i] = plt.subplot(self.gs[row, col])

    def __getitem__(self, item):
        return self.axes[item]

    def __iter__(self):
        for i in range(self.n):
            yield self[i]


def get_fig(fig=None, ax=None, figsize=[4, 4]):
    """fills in any missing axis or figure with the currently active one
    :param ax: matplotlib Axis object
    :param fig: matplotlib Figure object
    """
    if not fig:
        fig = plt.figure(figsize=figsize)
    if not ax:
        ax = plt.gca()
    return fig, ax


def density_2d(x, y):
    """return x and y and their density z, sorted by their density (smallest to largest)
    :param x:
    :param y:
    :return:
    """
    xy = np.vstack([np.ravel(x), np.ravel(y)])
    z = gaussian_kde(xy)(xy)
    i = np.argsort(z)
    return np.ravel(x)[i], np.ravel(y)[i], np.arcsinh(z[i])


def plot_molecules_per_cell_and_gene(data, fig=None, ax=None):

    height = 4
    width = 12
    fig = plt.figure(figsize=[width, height])
    gs = plt.GridSpec(1, 3)
    colsum = np.log10(data.sum(axis=0))
    rowsum = np.log10(data.sum(axis=1))
    for i in range(3):
        ax = plt.subplot(gs[0, i])

        if i == 0:
            n, bins, patches = ax.hist(rowsum, bins="auto")
            plt.xlabel("Molecules per cell (log10 scale)")
        elif i == 1:
            temp = np.log10(data.astype(bool).sum(axis=0))
            n, bins, patches = ax.hist(temp, bins="auto")
            plt.xlabel("Nonzero cells per gene (log10 scale)")
        else:
            n, bins, patches = ax.hist(colsum, bins="auto")
            plt.xlabel("Molecules per gene (log10 scale)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        ax.tick_params(axis="x", labelsize=8)
    sns.despine()

    return fig, ax


def cell_types(tsne, clusters, cluster_colors=None, n_cols=5):
    """Plot cell clusters on the tSNE map
    :param tsne: tSNE map
    :param clusters: Results of the determine_cell_clusters function
    """

    # Cluster colors
    if cluster_colors is None:
        cluster_colors = pd.Series(
            sns.color_palette("hls", len(set(clusters))), index=set(clusters)
        )
    n_clusters = len(cluster_colors)

    # Cell types
    fig = FigureGrid(n_clusters, n_cols)
    for ax, cluster in zip(fig, cluster_colors.index):
        ax.scatter(tsne.loc[:, "x"], tsne.loc[:, "y"], s=3, color="lightgrey")
        cells = clusters.index[clusters == cluster]
        ax.scatter(
            tsne.loc[cells, "x"],
            tsne.loc[cells, "y"],
            s=5,
            color=cluster_colors[cluster],
        )
        ax.set_axis_off()
        ax.set_title(cluster, fontsize=10)


def plot_cell_clusters(plot_embedding, clusters):
    """Plot cell clusters on the tSNE map
    :param plot_embedding: tSNE map
    :param clusters: Results of the determine_cell_clusters function
    """
    tsne = plot_embedding.copy()
    tsne.columns = ["x", "y"]

    # Cluster colors
    n_clusters = len(set(clusters))
    cluster_colors = pd.Series(
        sns.color_palette("hls", n_clusters), index=set(clusters)
    )

    # Set up figure
    n_cols = 6
    n_rows = int(np.ceil(n_clusters / n_cols))
    plt.figure(figsize=[2 * n_cols, 2 * (n_rows + 2)])
    gs = plt.GridSpec(
        n_rows + 2, n_cols, height_ratios=np.append([0.75, 0.75], np.repeat(1, n_rows))
    )

    # Clusters
    ax = plt.subplot(gs[0:2, 2:4])
    ax.scatter(tsne["x"], tsne["y"], s=3, color=cluster_colors[clusters[tsne.index]])
    ax.set_axis_off()

    # Branch probabilities
    for i, cluster in enumerate(set(clusters)):
        row = int(np.floor(i / n_cols))
        ax = plt.subplot(gs[row + 2, i % n_cols])
        ax.scatter(tsne.loc[:, "x"], tsne.loc[:, "y"], s=3, color="lightgrey")
        cells = clusters.index[clusters == cluster]
        ax.scatter(
            tsne.loc[cells, "x"],
            tsne.loc[cells, "y"],
            s=3,
            color=cluster_colors[cluster],
        )
        ax.set_axis_off()
        ax.set_title(cluster, fontsize=10)


def plot_tsne(tsne, fig=None, ax=None):
    """Plot tSNE projections of the data
    :param fig: matplotlib Figure object
    :param ax: matplotlib Axis object
    :param title: Title for the plot
    """
    fig, ax = get_fig(fig=fig, ax=ax)
    ax.scatter(tsne["x"], tsne["y"], s=5)
    ax.set_axis_off()
    return fig, ax


def highlight_cells_on_tsne(plot_tsne, cells, fig=None, ax=None):
    """Function to highlight specific cells on the tSNE map"""
    fig, ax = get_fig(fig=fig, ax=ax)
    tsne = plot_tsne.copy()
    tsne.columns = ["x", "y"]
    ax.scatter(tsne["x"], tsne["y"], s=5, color="lightgrey")
    ax.scatter(tsne.loc[cells, "x"], tsne.loc[cells, "y"], s=30)
    ax.set_axis_off()
    return fig, ax


def plot_tsne_by_cell_sizes(data, tsne, fig=None, ax=None, vmin=None, vmax=None):
    """Plot tSNE projections of the data with cells colored by molecule counts
    :param fig: matplotlib Figure object
    :param ax: matplotlib Axis object
    :param vmin: Minimum molecule count for plotting
    :param vmax: Maximum molecule count for plotting
    :param title: Title for the plot
    """

    sizes = data.sum(axis=1)
    fig, ax = get_fig(fig, ax)
    plt.scatter(tsne["x"], tsne["y"], s=3, c=sizes, cmap=matplotlib.cm.Spectral_r)
    ax.set_axis_off()
    plt.colorbar()
    return fig, ax


def plot_gene_expression(
    data,
    tsne,
    genes,
    plot_scale=False,
    n_cols=5,
    percentile=0,
    cmap=matplotlib.cm.Spectral_r,
):
    """Plot gene expression on tSNE maps
    :param genes: Iterable of strings to plot on tSNE
    """

    not_in_dataframe = set(genes).difference(data.columns)
    if not_in_dataframe:
        if len(not_in_dataframe) < len(genes):
            print(
                "The following genes were either not observed in the experiment, "
                "or the wrong gene symbol was used: {!r}".format(not_in_dataframe)
            )
        else:
            print(
                "None of the listed genes were observed in the experiment, or the "
                "wrong symbols were used."
            )
            return

    # remove genes missing from experiment
    genes = pd.Series(genes)[pd.Series(genes).isin(data.columns)]

    # Plot
    cells = data.index.intersection(tsne.index)
    fig = FigureGrid(len(genes), n_cols)

    for g, ax in zip(genes, fig):
        # Data
        c = data.loc[cells, g]
        vmin = np.percentile(c[~np.isnan(c)], percentile)
        vmax = np.percentile(c[~np.isnan(c)], 100 - percentile)

        ax.scatter(tsne["x"], tsne["y"], s=3, color="lightgrey")
        ax.scatter(
            tsne.loc[cells, "x"],
            tsne.loc[cells, "y"],
            s=3,
            c=c,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_axis_off()
        ax.set_title(g)

        if plot_scale:
            normalize = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            cax, _ = matplotlib.colorbar.make_axes(ax)
            matplotlib.colorbar.ColorbarBase(cax, norm=normalize, cmap=cmap)


def plot_diffusion_components(tsne, dm_res):
    """Plots the diffusion components on tSNE maps
    :return: fig, ax
    """

    # Please run tSNE before plotting diffusion components. #
    # Please run diffusion maps using run_diffusion_map before plotting #

    # Plot
    fig = FigureGrid(dm_res["EigenVectors"].shape[1], 5)

    for i, ax in enumerate(fig):
        ax.scatter(
            tsne.iloc[:, 0],
            tsne.iloc[:, 1],
            c=dm_res["EigenVectors"].loc[tsne.index, i],
            cmap=matplotlib.cm.Spectral_r,
            edgecolors="none",
            s=3,
        )
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
        ax.set_aspect("equal")
        ax.set_title("Component %d" % i, fontsize=10)
        ax.set_axis_off()


def plot_palantir_results(pr_res, tsne, s=3):
    """Plot Palantir results on tSNE"""

    # Set up figure
    n_branches = pr_res.branch_probs.shape[1]
    n_cols = 6
    n_rows = int(np.ceil(n_branches / n_cols))
    plt.figure(figsize=[2 * n_cols, 2 * (n_rows + 2)])
    gs = plt.GridSpec(
        n_rows + 2, n_cols, height_ratios=np.append([0.75, 0.75], np.repeat(1, n_rows))
    )
    cmap = matplotlib.cm.plasma
    # Pseudotime
    ax = plt.subplot(gs[0:2, 1:3])
    c = pr_res.pseudotime[tsne.index]
    ax.scatter(tsne.iloc[:, 0], tsne.iloc[:, 1], s=s, cmap=matplotlib.cm.plasma, c=c)
    normalize = matplotlib.colors.Normalize(vmin=np.min(c), vmax=np.max(c))
    cax, _ = matplotlib.colorbar.make_axes(ax)
    matplotlib.colorbar.ColorbarBase(cax, norm=normalize, cmap=cmap)
    ax.set_axis_off()
    ax.set_title("Pseudotime")

    # Entropy
    ax = plt.subplot(gs[0:2, 3:5])
    c = pr_res.entropy[tsne.index]
    ax.scatter(tsne.iloc[:, 0], tsne.iloc[:, 1], s=s, cmap=matplotlib.cm.plasma, c=c)
    normalize = matplotlib.colors.Normalize(vmin=np.min(c), vmax=np.max(c))
    cax, _ = matplotlib.colorbar.make_axes(ax)
    matplotlib.colorbar.ColorbarBase(cax, norm=normalize, cmap=cmap)
    ax.set_axis_off()
    ax.set_title("Differentiation potential")

    for i, branch in enumerate(pr_res.branch_probs.columns):
        row = int(np.floor(i / n_cols))
        ax = plt.subplot(gs[row + 2, np.remainder(i, n_cols)])
        c = pr_res.branch_probs.loc[tsne.index, branch]
        ax.scatter(
            tsne.iloc[:, 0], tsne.iloc[:, 1], s=s, cmap=matplotlib.cm.plasma, c=c
        )
        normalize = matplotlib.colors.Normalize(vmin=np.min(c), vmax=np.max(c))
        cax, _ = matplotlib.colorbar.make_axes(ax)
        matplotlib.colorbar.ColorbarBase(cax, norm=normalize, cmap=cmap)
        ax.set_axis_off()
        ax.set_title(branch, fontsize=10)


def plot_terminal_state_probs(pr_res, cells):
    """Function to plot barplot for probabilities for each cell in the list
    :param: pr_res: Palantir results object
    :param: cells: List of cell for which the barplots need to be plotted
    """
    n_cols = 5
    n_rows = int(np.ceil(len(cells) / n_cols))
    if len(cells) < n_cols:
        n_cols = len(cells)
    fig = plt.figure(figsize=[3 * n_cols, 3 * n_rows])

    # Branch colors
    set1_colors = sns.color_palette("Set1", 8).as_hex()
    set2_colors = sns.color_palette("Set2", 8).as_hex()
    cluster_colors = np.array(list(chain(*[set1_colors, set2_colors])))
    branch_colors = pd.Series(
        cluster_colors[range(pr_res.branch_probs.shape[1])],
        index=pr_res.branch_probs.columns,
    )

    for i, cell in enumerate(cells):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)

        # Probs
        df = pd.DataFrame(pr_res.branch_probs.loc[cell, :])
        df.loc[:, "x"] = pr_res.branch_probs.columns
        df.columns = ["y", "x"]

        # Plot
        sns.barplot(x="x", y="y", data=df, ax=ax, palette=branch_colors)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_ylim([0, 1])
        ax.set_yticks([0, 1])
        ax.set_yticklabels([0, 1])
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
        ax.set_title(cell, fontsize=10)
    sns.despine()

def plot_branch_selection(
    ad: sc.AnnData,
    pseudo_time_key: str = "palantir_pseudotime",
    fate_prob_key: str = "palantir_fate_probabilities",
    selection_key: str = "branch_mask",
    embedding_basis: str = "X_umap",
    **kwargs,
):
    """
    Plot cells along specific branches of pseudotime ordering and the UMAP embedding.

    Parameters
    ----------
    ad : sc.AnnData
        Annotated data matrix. The pseudotime and fate probabilities should be stored under the keys provided.
    pseudo_time_key : str, optional
        Key to access the pseudotime from obs of the AnnData object. Default is 'palantir_pseudotime'.
    fate_prob_key : str, optional
        Key to access the fate probabilities from obsm of the AnnData object. Default is 'palantir_fate_probabilities'.
    selection_key : str, optional
        Key under which the branch cell selection mask is stored in the AnnData object. Default is 'branch_mask'.
    embedding_basis : str, optional
        Key to access the UMAP embedding from obsm of the AnnData object. Default is 'X_umap'.
    **kwargs
        Additional arguments passed to `matplotlib.pyplot.scatter`.

    Returns
    -------
    matplotlib.pyplot.Figure
        A matplotlib Figure object representing the plot of the branch selections.
        
    """
    assert pseudo_time_key in ad.obs, f"{pseudo_time_key} not found in ad.obs"
    assert fate_prob_key in ad.obsm, f"{fate_prob_key} not found in ad.obsm"
    assert (
        fate_prob_key + "_columns" in ad.uns
    ), f"{fate_prob_key}_columns not found in ad.uns"
    assert embedding_basis in ad.obsm, f"{embedding_basis} not found in ad.obsm"

    fate_probs = ad.obsm[fate_prob_key]
    fate_names = ad.uns[fate_prob_key + "_columns"]
    pt = ad.obs[pseudo_time_key]
    umap = ad.obsm[embedding_basis]

    fig, axes = plt.subplots(len(fate_names), 2, figsize=(15, 5*len(fate_names)), width_ratios=[2, 1])

    colors = {True: "#003366", False: "#f0f8ff"}

    for i, fate in enumerate(fate_names):
        ax1 = axes[i, 0]
        ax2 = axes[i, 1]
        mask = ad.obs[selection_key + "_" + fate].astype(bool)

        # plot cells along pseudotime
        ax1.scatter(pt[~mask], fate_probs[~mask, i], c='#f0f8ff', label='Other Cells', **kwargs)
        ax1.scatter(pt[mask], fate_probs[mask, i], c='#003366', label='Selected Cells', **kwargs)
        ax1.set_title(f"Branch: {fate}")
        ax1.set_xlabel("Pseudotime")
        ax1.set_ylabel("Fate Probability")
        ax1.legend()

        # plot UMAP
        ax2.scatter(umap[~mask, 0], umap[~mask, 1], c='#f0f8ff', label='Other Cells', **kwargs)
        ax2.scatter(umap[mask, 0], umap[mask, 1], c='#003366', label='Selected Cells', **kwargs)
        ax2.set_title(f"Branch: {fate}")
        ax2.axis("off")

    plt.tight_layout()
    sns.despine()

    return fig


def plot_gene_trends(
    data: Union[sc.AnnData, pd.DataFrame],
    genes: Optional[List[str]] = None,
    gene_trend_key: Optional[str] = "palantir_gene_trends",
) -> plt.Figure:
    """
    Plot the gene trends: each gene is plotted in a different panel.

    Parameters
    ----------
    data : Union[sc.AnnData, pd.DataFrame]
        Either a Scanpy AnnData object or a DataFrame of gene expression trends.
    genes : list of str, optional
        List of gene names to plot. If None, all genes in the DataFrame or AnnData object are plotted. Default is None.
    gene_trend_key : str, optional
        Key to access gene trends from varm of the AnnData object. If data is a DataFrame,
        gene_trend_key should be None. If gene_trend_key is None when data is an AnnData,
        a KeyError will be raised. Default is None.

    Returns
    -------
    matplotlib.pyplot.Figure
        A matplotlib Figure object representing the plot of the gene trends.

    Raises
    ------
    KeyError
        If gene_trend_key is None when data is an AnnData object.
    """
    # Retrieve the gene expression trends from the AnnData object or DataFrame
    if isinstance(data, sc.AnnData):
        if gene_trend_key is None:
            raise KeyError(
                "Must provide a gene_trend_key when data is an AnnData object."
            )
        pseudotimes = data.uns[gene_trend_key + "_pseudotime"]
        gene_trends = pd.DataFrame(
            data.varm[gene_trend_key], index=data.var_names, columns=pseudotimes
        )
    else:
        gene_trends = data

    if genes is None:
        genes = gene_trends.index.tolist()

    # Set up figure
    fig = plt.figure(figsize=[7, 3 * len(genes)])
    for i, gene in enumerate(genes):
        ax = fig.add_subplot(len(genes), 1, i + 1)
        trend = gene_trends.loc[gene, :]
        ax.plot(
            trend.index, trend.values, label=gene
        )
        ax.set_xticks([trend.index.min(), trend.index.max()])
        ax.set_title(gene)

    sns.despine()

    return fig

def plot_gene_trend_heatmaps(
    data: Union[sc.AnnData, pd.DataFrame],
    genes: Optional[List[str]] = None,
    gene_trend_key: Optional[str] = "palantir_gene_trends",
) -> plt.Figure:
    """
    Plot the gene trends on heatmap: a heatmap is generated for each gene.

    Parameters
    ----------
    data : Union[sc.AnnData, pd.DataFrame]
        Either a Scanpy AnnData object or a DataFrame of gene expression trends.
    genes : list of str, optional
        List of gene names to plot. If None, all genes in the DataFrame or AnnData object are plotted. Default is None.
    gene_trend_key : str, optional
        Key to access gene trends from varm of the AnnData object. If data is a DataFrame,
        gene_trend_key should be None. If gene_trend_key is None when data is an AnnData,
        a KeyError will be raised. Default is None.

    Returns
    -------
    matplotlib.pyplot.Figure
        A matplotlib Figure object representing the heatmap of the gene trends.

    Raises
    ------
    KeyError
        If gene_trend_key is None when data is an AnnData object.
    """
    # Retrieve the gene expression trends from the AnnData object or DataFrame
    if isinstance(data, sc.AnnData):
        if gene_trend_key is None:
            raise KeyError(
                "Must provide a gene_trend_key when data is an AnnData object."
            )
        pseudotimes = data.uns[gene_trend_key + "_pseudotime"]
        gene_trends = pd.DataFrame(
            data.varm[gene_trend_key], index=data.var_names, columns=pseudotimes
        )
    else:
        gene_trends = data

    if genes is None:
        genes = gene_trends.index.tolist()

    gene_trends = gene_trends.loc[genes, :]

    # Standardize the matrix
    mat = pd.DataFrame(
        StandardScaler().fit_transform(gene_trends.T).T,
        index=gene_trends.index,
        columns=gene_trends.columns,
    )

    #  Set up plot
    fig = plt.figure(figsize=[7, 0.7 * len(genes)])
    sns.heatmap(mat, xticklabels=False, cmap=matplotlib.cm.Spectral_r)

    return fig



def plot_gene_trend_clusters(
    data: Union[sc.AnnData, pd.DataFrame],
    clusters: Optional[Union[pd.Series, str]] = None,
    gene_trend_key: Optional[str] = "palantir_gene_trends",
) -> plt.Figure:
    """
    Plot the gene trend clusters.

    This function takes either a Scanpy AnnData object or a DataFrame of gene expression trends,
    and a Series of clusters or a key to clusters in AnnData object's var, and creates a plot of the gene trend clusters.

    Parameters
    ----------
    data : Union[sc.AnnData, pd.DataFrame]
        Either a Scanpy AnnData object or a DataFrame of gene expression trends.
    clusters : pd.Series or str, optional
        A Series of clusters indexed by gene names, or a string key to access clusters from var of the AnnData object.
        If data is a DataFrame, clusters should be a Series. If clusters is None, it is set to gene_trend_key+"_clusters".
        Default is None.
    gene_trend_key : str, optional
        Key to access gene trends from varm of the AnnData object. If data is a DataFrame,
        gene_trend_key should be None. If gene_trend_key is None when data is an AnnData,
        a KeyError will be raised. Default is None.

    Returns
    -------
    matplotlib.pyplot.Figure
        A matplotlib Figure object representing the plot of the gene trend clusters.

    Raises
    ------
    KeyError
        If gene_trend_key is None when data is an AnnData object.
    """
    # Retrieve the gene expression trends from the AnnData object or DataFrame
    if isinstance(data, sc.AnnData):
        if gene_trend_key is None:
            raise KeyError(
                "Must provide a gene_trend_key when data is an AnnData object."
            )
        pseudotimes = data.uns[gene_trend_key + "_pseudotime"]
        trends = pd.DataFrame(
            data.varm[gene_trend_key], index=data.var_names, columns=pseudotimes
        )
        if clusters is None:
            clusters = gene_trend_key + "_clusters"
        if isinstance(clusters, str):
            clusters = data.var[clusters]
    else:
        trends = data

    # Standardize the trends
    trends = pd.DataFrame(
        StandardScaler().fit_transform(trends.T).T,
        index=trends.index,
        columns=trends.columns,
    )

    # Check if clusters is a categorical series
    if isinstance(clusters, pd.Series) and pd.api.types.is_categorical_dtype(clusters):
        cluster_labels = clusters.cat.categories
    else:
        cluster_labels = set(clusters)

    # Plot the gene trend clusters
    n_rows = int(np.ceil(len(cluster_labels) / 3))
    fig = plt.figure(figsize=[5.5 * 3, 2.5 * n_rows])

    for i, c in enumerate(cluster_labels):
        ax = fig.add_subplot(n_rows, 3, i + 1)
        means = trends.loc[clusters.index[clusters == c], :].mean()
        std = trends.loc[clusters.index[clusters == c], :].std()

        # Plot all trends
        for g in clusters.index[clusters == c]:
            ax.plot(
                means.index,
                np.ravel(trends.loc[g, :]),
                linewidth=0.5,
                color="lightgrey",
            )

        # Mean
        ax.plot(means.index, np.ravel(means), color="#377eb8")
        ax.plot(
            means.index,
            np.ravel(means - std),
            linestyle="--",
            color="#377eb8",
            linewidth=0.75,
        )
        ax.plot(
            means.index,
            np.ravel(means + std),
            linestyle="--",
            color="#377eb8",
            linewidth=0.75,
        )
        ax.set_title("Cluster {}".format(c), fontsize=12)
        ax.tick_params("both", length=2, width=1, which="major")
        ax.tick_params(axis="both", which="major", labelsize=8, direction="in")
        ax.set_xticklabels([])
    sns.despine()

    return fig
