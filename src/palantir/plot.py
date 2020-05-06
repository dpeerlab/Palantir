import warnings
import os
import numpy as np
import pandas as pd
from itertools import chain
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde

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


def plot_cell_clusters(tsne, clusters):
    """Plot cell clusters on the tSNE map
    :param tsne: tSNE map
    :param clusters: Results of the determine_cell_clusters function
    """

    # Cluster colors
    n_clusters = len(set(clusters))
    cluster_colors = pd.Series(
        sns.color_palette("hls", n_clusters), index=set(clusters)
    )

    # Set up figure
    n_cols = 6
    n_rows = int(np.ceil(n_clusters / n_cols))
    fig = plt.figure(figsize=[2 * n_cols, 2 * (n_rows + 2)])
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


def highlight_cells_on_tsne(tsne, cells, fig=None, ax=None):
    """    Function to highlight specific cells on the tSNE map
    """
    fig, ax = get_fig(fig=fig, ax=ax)
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
    """ Plot gene expression on tSNE maps
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
    """ Plots the diffusion components on tSNE maps
    :return: fig, ax
    """

    # Please run tSNE before plotting diffusion components. #
    # Please run diffusion maps using run_diffusion_map before plotting #

    # Plot
    fig = FigureGrid(dm_res["EigenVectors"].shape[1], 5)

    for i, ax in enumerate(fig):
        ax.scatter(
            tsne["x"],
            tsne["y"],
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


def plot_palantir_results(pr_res, tsne):
    """ Plot Palantir results on tSNE
    """

    # Set up figure
    n_branches = pr_res.branch_probs.shape[1]
    n_cols = 6
    n_rows = int(np.ceil(n_branches / n_cols))
    fig = plt.figure(figsize=[2 * n_cols, 2 * (n_rows + 2)])
    gs = plt.GridSpec(
        n_rows + 2, n_cols, height_ratios=np.append([0.75, 0.75], np.repeat(1, n_rows))
    )
    cmap = matplotlib.cm.plasma
    # Pseudotime
    ax = plt.subplot(gs[0:2, 1:3])
    c = pr_res.pseudotime[tsne.index]
    ax.scatter(tsne.loc[:, "x"], tsne.loc[:, "y"], s=3, cmap=matplotlib.cm.plasma, c=c)
    normalize = matplotlib.colors.Normalize(vmin=np.min(c), vmax=np.max(c))
    cax, _ = matplotlib.colorbar.make_axes(ax)
    cbar = matplotlib.colorbar.ColorbarBase(cax, norm=normalize, cmap=cmap)
    ax.set_axis_off()
    ax.set_title("Pseudotime")

    # Entropy
    ax = plt.subplot(gs[0:2, 3:5])
    c = pr_res.entropy[tsne.index]
    ax.scatter(tsne.loc[:, "x"], tsne.loc[:, "y"], s=3, cmap=matplotlib.cm.plasma, c=c)
    normalize = matplotlib.colors.Normalize(vmin=np.min(c), vmax=np.max(c))
    cax, _ = matplotlib.colorbar.make_axes(ax)
    cbar = matplotlib.colorbar.ColorbarBase(cax, norm=normalize, cmap=cmap)
    ax.set_axis_off()
    ax.set_title("Differentiation potential")

    for i, branch in enumerate(pr_res.branch_probs.columns):
        row = int(np.floor(i / n_cols))
        ax = plt.subplot(gs[row + 2, np.remainder(i, n_cols)])
        c = pr_res.branch_probs.loc[tsne.index, branch]
        ax.scatter(
            tsne.loc[:, "x"], tsne.loc[:, "y"], s=3, cmap=matplotlib.cm.plasma, c=c
        )
        normalize = matplotlib.colors.Normalize(vmin=np.min(c), vmax=np.max(c))
        cax, _ = matplotlib.colorbar.make_axes(ax)
        cbar = matplotlib.colorbar.ColorbarBase(cax, norm=normalize, cmap=cmap)
        ax.set_axis_off()
        ax.set_title(branch, fontsize=10)


def plot_terminal_state_probs(pr_res, cells):
    """ Function to plot barplot for probabilities for each cell in the list
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
        sns.barplot("x", "y", data=df, ax=ax, palette=branch_colors)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_ylim([0, 1])
        ax.set_yticks([0, 1])
        ax.set_yticklabels([0, 1])
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
        ax.set_title(cell, fontsize=10)
    sns.despine()


def plot_gene_trends(gene_trends, genes=None):
    """ Plot the gene trends: each gene is plotted in a different panel
    :param: gene_trends: Results of the compute_marker_trends function
    """

    # Branches and genes
    branches = list(gene_trends.keys())
    colors = pd.Series(
        sns.color_palette("Set2", len(branches)).as_hex(), index=branches
    )
    if genes is None:
        genes = gene_trends[branches[0]]["trends"].index

    # Set up figure
    fig = plt.figure(figsize=[7, 3 * len(genes)])
    for i, gene in enumerate(genes):
        ax = fig.add_subplot(len(genes), 1, i + 1)
        for branch in branches:
            trends = gene_trends[branch]["trends"]
            stds = gene_trends[branch]["std"]
            ax.plot(
                trends.columns, trends.loc[gene, :], color=colors[branch], label=branch
            )
            ax.set_xticks([0, 1])
            ax.fill_between(
                trends.columns,
                trends.loc[gene, :] - stds.loc[gene, :],
                trends.loc[gene, :] + stds.loc[gene, :],
                alpha=0.1,
                color=colors[branch],
            )
            ax.set_title(gene)
        # Add legend
        if i == 0:
            ax.legend()

    sns.despine()


def plot_gene_trend_heatmaps(gene_trends):
    """ Plot the gene trends on heatmap: a heatmap is generated or each branch
    :param: gene_trends: Results of the compute_marker_trends function
    """

    # Plot height
    branches = list(gene_trends.keys())
    genes = gene_trends[branches[0]]["trends"].index
    height = 0.7 * len(genes) * len(branches)

    #  Set up plot
    fig = plt.figure(figsize=[7, height])
    for i, branch in enumerate(branches):
        ax = fig.add_subplot(len(branches), 1, i + 1)

        # Standardize the matrix
        mat = gene_trends[branch]["trends"]
        mat = pd.DataFrame(
            StandardScaler().fit_transform(mat.T).T,
            index=mat.index,
            columns=mat.columns,
        )
        sns.heatmap(mat, xticklabels=False, ax=ax, cmap=matplotlib.cm.Spectral_r)
        ax.set_title(branch, fontsize=12)


def plot_gene_trend_clusters(trends, clusters):
    """ Plot the gene trend clusters
    """

    # Standardize the trends
    trends = pd.DataFrame(
        StandardScaler().fit_transform(trends.T).T,
        index=trends.index,
        columns=trends.columns,
    )

    n_rows = int(np.ceil(len(set(clusters)) / 3))
    fig = plt.figure(figsize=[5.5 * 3, 2.5 * n_rows])
    for i, c in enumerate(set(clusters)):
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
        # ax.set_xticklabels( ax.get_xticklabels(), fontsize=8 )
        # ax.set_yticklabels( ax.get_yticklabels(), fontsize=8 )
    sns.despine()
