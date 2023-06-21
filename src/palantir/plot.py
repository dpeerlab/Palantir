from typing import Union, Optional, List, Tuple, Dict
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
import matplotlib.patheffects as PathEffects
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .presults import PResults

import matplotlib.pyplot as plt

# set plotting defaults
with warnings.catch_warnings():
    # catch warnings that system can't find fonts
    warnings.simplefilter("ignore")
    import seaborn as sns

    sns.set(context="paper", style="ticks", font_scale=1.5, font="Bitstream Vera Sans")
    fm = font_manager.fontManager
    fm.findfont("Raleway")
    fm.findfont("Lato")

SELECTED_COLOR = "#377eb8"
DESELECTED_COLOR = "#CFD5E2"


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


def highlight_cells_on_umap(
    data: Union[sc.AnnData, pd.DataFrame],
    cells: Union[List[str], Dict[str, str], pd.Series],
    annotation_offset: float = 0.03,
    s: float = 1,
    s_highlighted: float = 10,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    embedding_bases: str = "X_umap",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Highlights and annotates specific cells on a UMAP plot.

    Parameters
    ----------
    data : Union[sc.AnnData, pd.DataFrame]
        Either a Scanpy AnnData object or a DataFrame of UMAP coordinates.
    cells : Union[List[str], Dict[str, str], pd.Series]
        List of cells to highlight on the UMAP. If provided as a dictionary or pd.Series,
        keys are used as cell names and values are used as annotations.
    annotation_offset : float, optional
        Offset for the annotations in proportion to the data range. Default is 0.03.
    s : float, optional
        Size of the points in the scatter plot. Default is 0.1.
    s_highlighted : float, optional
        Size of the points in the highlighted cells. Default is 10.
    fig : Optional[plt.Figure], optional
        Matplotlib Figure object. If None, a new figure is created. Default is None.
    ax : Optional[plt.Axes], optional
        Matplotlib Axes object. If None, a new Axes object is created. Default is None.
    embedding_bases : str, optional
        The key to retrieve UMAP results from the AnnData object. Default is 'X_umap'.

    Returns
    -------
    fig : plt.Figure
        A matplotlib Figure object containing the UMAP plot.
    ax : plt.Axes
        The corresponding Axes object.

    Raises
    ------
    KeyError
        If 'embedding_bases' is not found in 'data.obsm'.
    TypeError
        If 'cells' is neither list, dict nor pd.Series.
    """
    if isinstance(data, sc.AnnData):
        if embedding_bases not in data.obsm:
            raise KeyError(f"'{embedding_bases}' not found in .obsm.")
        umap = pd.DataFrame(
            data.obsm[embedding_bases], index=data.obs_names, columns=["x", "y"]
        )
    elif isinstance(data, pd.DataFrame):
        umap = data.copy()
    else:
        raise TypeError("'data' should be either sc.AnnData or pd.DataFrame.")

    if isinstance(cells, list):
        cells = {cell: "" for cell in cells}
    elif isinstance(cells, (dict, pd.Series)):
        cells = dict(cells)
    else:
        raise TypeError("'cells' should be either list, dict or pd.Series.")

    xpad, ypad = (umap.max() - umap.min()) * annotation_offset

    fig, ax = get_fig(fig=fig, ax=ax)
    ax.scatter(umap["x"], umap["y"], s=s, color=DESELECTED_COLOR)

    for cell, annotation in cells.items():
        if cell in umap.index:
            x, y = umap.loc[cell, ["x", "y"]]
            ax.scatter(x, y, c=SELECTED_COLOR, s=s_highlighted)
            if annotation:
                ax.annotate(annotation, (x, y), (x + xpad, y + ypad), "data")
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


def plot_diffusion_components(
    data: Union[sc.AnnData, pd.DataFrame],
    dm_res: Optional[Union[pd.DataFrame, str]] = "DM_EigenVectors",
    embedding_bases: str = "X_umap",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Visualize diffusion components on tSNE or UMAP plots.

    Parameters
    ----------
    data : Union[sc.AnnData, pd.DataFrame]
        Either a Scanpy AnnData object or a DataFrame of tSNE or UMAP results.
    dm_res : pd.DataFrame or str, optional
        DataFrame containing diffusion map results or a string key to access diffusion map
        results from the AnnData object's obsm. Default is 'DM_Eigenvectors'.
    embedding_bases : str, optional
        The key to retrieve UMAP results from the AnnData object. Defaults to 'X_umap'.

    Returns
    -------
    matplotlib.pyplot.Figure, matplotlib.pyplot.Axes
        A matplotlib Figure and Axes objects representing the plot of the diffusion components.

    Raises
    ------
    KeyError
        If `embedding_bases` or `dm_res` is not found when `data` is an AnnData object.
    """
    # Retrieve the embedding data
    if isinstance(data, sc.AnnData):
        if embedding_bases not in data.obsm:
            raise KeyError(f"'{embedding_bases}' not found in .obsm.")
        embedding_data = pd.DataFrame(data.obsm[embedding_bases], index=data.obs_names)
        if isinstance(dm_res, str):
            if dm_res not in data.obsm:
                raise KeyError(f"'{dm_res}' not found in .obsm.")
            dm_res = {
                "EigenVectors": pd.DataFrame(data.obsm[dm_res], index=data.obs_names)
            }
    else:
        embedding_data = data

    fig = FigureGrid(dm_res["EigenVectors"].shape[1], 5)

    for i, ax in enumerate(fig):
        ax.scatter(
            embedding_data.iloc[:, 0],
            embedding_data.iloc[:, 1],
            c=dm_res["EigenVectors"].loc[embedding_data.index, i],
            cmap=matplotlib.cm.Spectral_r,
            edgecolors="none",
            s=3,
        )
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
        ax.set_aspect("equal")
        ax.set_title(f"Component {i}", fontsize=10)
        ax.set_axis_off()

    return fig, ax


def plot_palantir_results(
    data: Union[sc.AnnData, pd.DataFrame],
    pr_res: Optional[PResults] = None,
    embedding_bases: str = "X_umap",
    pseudo_time_key: str = "palantir_pseudotime",
    entropy_key: str = "palantir_entropy",
    fate_prob_key: str = "palantir_fate_probabilities",
    **kwargs,
):
    """
    Plot Palantir results on t-SNE or UMAP plots.

    Parameters
    ----------
    data : Union[sc.AnnData, pd.DataFrame]
        Either a Scanpy AnnData object or a DataFrame of tSNE or UMAP results.
    pr_res : Optional[PResults]
        Optional PResults object containing Palantir results. If None, results are expected to be found in the provided AnnData object.
    embedding_bases : str, optional
        The key to retrieve UMAP results from the AnnData object. Defaults to 'X_umap'.
    pseudo_time_key : str, optional
        Key to access the pseudotime from obs of the AnnData object. Default is 'palantir_pseudotime'.
    entropy_key : str, optional
        Key to access the entropy from obs of the AnnData object. Default is 'palantir_entropy'.
    fate_prob_key : str, optional
        Key to access the fate probabilities from obsm of the AnnData object. Default is 'palantir_fate_probabilities'.
    **kwargs
        Additional keyword arguments passed to `ax.scatter`.

    Returns
    -------
    matplotlib.pyplot.Figure
        A matplotlib Figure object representing the plot of the Palantir results.
    """
    if isinstance(data, sc.AnnData):
        if embedding_bases not in data.obsm:
            raise KeyError(f"'{embedding_bases}' not found in .obsm.")
        embedding_data = pd.DataFrame(data.obsm[embedding_bases], index=data.obs_names)
        if pr_res is None:
            if (
                pseudo_time_key not in data.obs
                or entropy_key not in data.obs
                or fate_prob_key not in data.obsm
            ):
                raise KeyError("Required Palantir results not found in .obs or .obsm.")
            pr_res = PResults(
                data.obs[pseudo_time_key],
                data.obs[entropy_key],
                pd.DataFrame(
                    data.obsm[fate_prob_key],
                    index=data.obs_names,
                    columns=data.uns[fate_prob_key + "_columns"],
                ),
                None,
            )
    else:
        embedding_data = data

    n_branches = pr_res.branch_probs.shape[1]
    n_cols = 6
    n_rows = int(np.ceil(n_branches / n_cols))
    plt.figure(figsize=[2 * n_cols, 2 * (n_rows + 2)])
    gs = plt.GridSpec(
        n_rows + 2, n_cols, height_ratios=np.append([0.75, 0.75], np.repeat(1, n_rows))
    )
    cmap = matplotlib.cm.plasma

    def scatter_with_colorbar(ax, x, y, c, cmap):
        sc = ax.scatter(x, y, c=c, cmap=cmap, **kwargs)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(sc, cax=cax, orientation="vertical")

    # Pseudotime
    ax = plt.subplot(gs[0:2, 1:3])
    scatter_with_colorbar(
        ax,
        embedding_data.iloc[:, 0],
        embedding_data.iloc[:, 1],
        pr_res.pseudotime[embedding_data.index],
        cmap,
    )
    ax.set_axis_off()
    ax.set_title("Pseudotime")

    # Entropy
    ax = plt.subplot(gs[0:2, 3:5])
    scatter_with_colorbar(
        ax,
        embedding_data.iloc[:, 0],
        embedding_data.iloc[:, 1],
        pr_res.entropy[embedding_data.index],
        cmap,
    )
    ax.set_axis_off()
    ax.set_title("Entropy")

    # Branch probabilities
    for i, branch in enumerate(pr_res.branch_probs.columns):
        ax = plt.subplot(gs[i // n_cols + 2, i % n_cols])
        scatter_with_colorbar(
            ax,
            embedding_data.iloc[:, 0],
            embedding_data.iloc[:, 1],
            pr_res.branch_probs[branch][embedding_data.index],
            cmap,
        )
        ax.set_axis_off()
        ax.set_title(branch, fontsize=10)

    plt.tight_layout()
    return plt.gcf()


def plot_terminal_state_probs(
    data: Union[sc.AnnData, pd.DataFrame],
    cells: List[str],
    pr_res: Optional[PResults] = None,
    fate_prob_key: str = "palantir_fate_probabilities",
):
    """Function to plot barplot for probabilities for each cell in the list

    Parameters
    ----------
    data : Union[sc.AnnData, pd.DataFrame]
        Either a Scanpy AnnData object or a DataFrame of fate probabilities.
    cells : List[str]
        List of cell for which the barplots need to be plotted.
    pr_res : Optional[PResults]
        Optional PResults object containing Palantir results. If None, results are expected to be found in the provided AnnData object.
    fate_prob_key : str, optional
        Key to access the fate probabilities from obsm of the AnnData object. Default is 'palantir_fate_probabilities'.

    Returns
    -------
    matplotlib.pyplot.Figure
        A matplotlib Figure object representing the plot of the cell fate probabilities.
    """
    if isinstance(data, sc.AnnData):
        if pr_res is None:
            if fate_prob_key not in data.obsm:
                raise KeyError(f"'{fate_prob_key}' not found in .obsm.")
            branch_probs = pd.DataFrame(
                data.obsm[fate_prob_key],
                index=data.obs_names,
                columns=data.uns[fate_prob_key + "_columns"],
            )
    else:
        if pr_res is None:
            raise ValueError("pr_res must be provided when data is a DataFrame.")
        branch_probs = pr_res.branch_probs

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
        cluster_colors[range(branch_probs.shape[1])],
        index=branch_probs.columns,
    )

    for i, cell in enumerate(cells):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)

        # Probs
        df = pd.DataFrame(branch_probs.loc[cell, :])
        df.loc[:, "x"] = branch_probs.columns
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

    return fig


def plot_branch_selection(
    ad: sc.AnnData,
    pseudo_time_key: str = "palantir_pseudotime",
    fate_prob_key: str = "palantir_fate_probabilities",
    masks_key: str = "branch_masks",
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
        Key to access the fate probabilities from obsm of the AnnData object.
        Default is 'palantir_fate_probabilities'.
    masks_key : str, optional
        Key to access the branch cell selection masks from obsm of the AnnData object.
        Default is 'branch_masks'.
    embedding_basis : str, optional
        Key to access the UMAP embedding from obsm of the AnnData object. Default is 'X_umap'.
    **kwargs
        Additional arguments passed to `matplotlib.pyplot.scatter`.

    Returns
    -------
    matplotlib.pyplot.Figure
        A matplotlib Figure object representing the plot of the branch selections.

    """
    if pseudo_time_key not in ad.obs:
        raise KeyError(f"{pseudo_time_key} not found in ad.obs")
    if fate_prob_key not in ad.obsm:
        raise KeyError(f"{fate_prob_key} not found in ad.obsm")
    if fate_prob_key + "_columns" not in ad.uns:
        raise KeyError(f"{fate_prob_key}_columns not found in ad.uns")
    if masks_key not in ad.obsm:
        raise KeyError(f"{masks_key} not found in ad.obsm")
    if masks_key + "_columns" not in ad.uns:
        raise KeyError(f"{masks_key}_columns not found in ad.uns")
    if embedding_basis not in ad.obsm:
        raise KeyError(f"{embedding_basis} not found in ad.obsm")

    fate_probs = ad.obsm[fate_prob_key]
    fate_names = ad.uns[fate_prob_key + "_columns"]
    fate_mask = ad.obsm[masks_key]
    fate_mask_names = ad.uns[masks_key + "_columns"]
    assert set(fate_names) == set(
        fate_mask_names
    ), f"Fates in .uns['{fate_prob_key}_columns'] and .uns['{masks_key}_columns'] must be the same."
    fate_mask = pd.DataFrame(fate_mask, columns=fate_mask_names, index=ad.obs_names)
    pt = ad.obs[pseudo_time_key]
    umap = ad.obsm[embedding_basis]

    fig, axes = plt.subplots(
        len(fate_names), 2, figsize=(15, 5 * len(fate_names)), width_ratios=[2, 1]
    )

    for i, fate in enumerate(fate_names):
        ax1 = axes[i, 0]
        ax2 = axes[i, 1]
        mask = fate_mask[fate].astype(bool)

        # plot cells along pseudotime
        ax1.scatter(
            pt[~mask],
            fate_probs[~mask, i],
            c=DESELECTED_COLOR,
            label="Other Cells",
            **kwargs,
        )
        ax1.scatter(
            pt[mask],
            fate_probs[mask, i],
            c=SELECTED_COLOR,
            label="Selected Cells",
            **kwargs,
        )
        ax1.set_title(f"Branch: {fate}")
        ax1.set_xlabel("Pseudotime")
        ax1.set_ylabel(f"{fate}-Fate Probability")
        ax1.legend()

        # plot UMAP
        ax2.scatter(
            umap[~mask, 0],
            umap[~mask, 1],
            c=DESELECTED_COLOR,
            label="Other Cells",
            **kwargs,
        )
        ax2.scatter(
            umap[mask, 0],
            umap[mask, 1],
            c=SELECTED_COLOR,
            label="Selected Cells",
            **kwargs,
        )
        ax2.set_title(f"Branch: {fate}")
        ax2.axis("off")

    plt.tight_layout()
    sns.despine()

    return fig


def plot_gene_trends_legacy(gene_trends, genes=None):
    """Plot the gene trends: each gene is plotted in a different panel
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


def _validate_gene_trend_input(
    data: Union[sc.AnnData, Dict],
    gene_trend_key: str = "gene_trends",
    branch_names: Union[str, List] = "branch_masks_columns",
) -> Dict:
    """
    Validates the input for gene trend plots, and converts it into a dictionary of gene trends.

    Parameters
    ----------
    data : Union[sc.AnnData, Dict]
        AnnData object or dictionary of gene trends.
    gene_trend_key : str, optional
        Key to access gene trends in the AnnData object's varm. Default is 'gene_trends'.
    branch_names : Union[str, List], optional
        Key to access branch names from AnnData object or list of branch names. If a string is provided,
        it is assumed to be a key in AnnData.uns. Default is 'branch_masks_columns'.

    Returns
    -------
    gene_trends : Dict
        Dictionary of gene trends.
    """
    if isinstance(data, sc.AnnData):
        if isinstance(branch_names, str):
            if branch_names not in data.uns.keys():
                raise KeyError(
                    f"'{branch_names}' not found in .uns. "
                    "'branch_names' must either be in .uns or a list of branch names."
                )
            branch_names = data.uns[branch_names]

        gene_trends = dict()
        for branch in branch_names:
            gene_names = data.var_names
            varm_name = gene_trend_key + "_" + branch
            if varm_name not in data.varm:
                raise KeyError(
                    f"'gene_trend_key + \"_\" + branch_name' = '{varm_name}' not found in .varm. "
                )
            pt_grid_name = gene_trend_key + "_" + branch + "_pseudotime"
            if pt_grid_name not in data.uns.keys():
                raise KeyError(
                    '\'gene_trend_key + "_" + branch_name + "_pseudotime"\' '
                    f"= '{pt_grid_name}' not found in .uns. "
                )
            pt_grid = data.uns[pt_grid_name]
            trends = data.varm[varm_name]
            gene_trends[branch] = {
                "trends": pd.DataFrame(trends, columns=pt_grid, index=gene_names)
            }
    elif isinstance(data, Dict):
        gene_trends = data
    else:
        raise ValueError("Input should be an AnnData object or a dictionary.")

    return gene_trends


def plot_gene_trends(
    data: Union[Dict, sc.AnnData],
    genes: Optional[List[str]] = None,
    gene_trend_key: str = "gene_trends",
    branch_names: Union[str, List] = "branch_masks_columns",
) -> plt.Figure:
    """Plot the gene trends: each gene is plotted in a different panel.

    Parameters
    ----------
    data : Union[Dict, sc.AnnData]
        AnnData object or dictionary of gene trends.
    genes : Union[List, Set, Tuple], optional
        List of genes to plot. If None, plot all genes. Default is None.
    gene_trend_key : str, optional
        Key to access gene trends in the AnnData object's varm. Default is 'gene_trends'.
    branch_names : Union[str, List], optional
        Key to access branch names from AnnData object or list of branch names. If a string is provided,
        it is assumed to be a key in AnnData.uns. Default is 'branch_masks_columns'.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib figure object of the plot.

    Raises
    ------
    KeyError
        If 'branch_names' is not found in .uns when it's a string or 'gene_trend_key + "_" + branch_name'
        is not found in .varm.
    ValueError
        If 'data' is neither an AnnData object nor a dictionary.
    """

    gene_trends = _validate_gene_trend_input(data, gene_trend_key, branch_names)

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
            ax.plot(
                trends.columns, trends.loc[gene, :], color=colors[branch], label=branch
            )
            ax.set_xticks([0, 1])
            ax.set_title(gene)

        # Add legend
        if i == 0:
            ax.legend()

    sns.despine()

    return fig


def plot_gene_trend_heatmaps(
    data: Union[sc.AnnData, Dict],
    genes: Optional[List[str]] = None,
    gene_trend_key: str = "gene_trends",
    branch_names: Union[str, List] = "branch_masks_columns",
) -> plt.Figure:
    """
    Plot the gene trends on heatmaps: a heatmap is generated for each branch.

    Parameters
    ----------
    data : Union[sc.AnnData, Dict]
        AnnData object or dictionary of gene trends.
    genes : Optional[List[str]], optional
        List of genes to include in the plot. If None, all genes are included.
        Default is None.
    gene_trend_key : str, optional
        Key to access gene trends in the AnnData object's varm. Default is 'gene_trends'.
    branch_names : Union[str, List], optional
        Key to access branch names from AnnData object or list of branch names. If a string is provided,
        it is assumed to be a key in AnnData.uns. Default is 'branch_masks_columns'.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib figure object of the plot.
    """

    gene_trends = _validate_gene_trend_input(data, gene_trend_key, branch_names)

    # Plot height
    branches = list(gene_trends.keys())
    if genes is None:
        genes = gene_trends[branches[0]]["trends"].index
    height = 0.7 * len(genes) * len(branches)

    if genes is None:
        genes = gene_trends[branches[0]]["trends"].index

    fig = plt.figure(figsize=[7, height])
    for i, branch in enumerate(branches):
        ax = fig.add_subplot(len(branches), 1, i + 1)

        # Standardize the matrix
        mat = gene_trends[branch]["trends"].loc[genes, :]
        mat = pd.DataFrame(
            StandardScaler().fit_transform(mat.T).T,
            index=mat.index,
            columns=mat.columns,
        )
        sns.heatmap(mat, xticklabels=False, ax=ax, cmap=plt.cm.Spectral_r)
        ax.set_title(branch, fontsize=12)

    return fig


def plot_gene_trend_clusters(
    data: Union[sc.AnnData, pd.DataFrame],
    branch_name: str = "",
    clusters: Optional[Union[pd.Series, str]] = None,
    gene_trend_key: Optional[str] = "gene_trends",
) -> plt.Figure:
    """
    Visualize gene trend clusters.

    This function accepts either a Scanpy AnnData object or a DataFrame of gene
    expression trends, along with a Series of clusters or a key to clusters in
    AnnData object's var. It creates a plot representing gene trend clusters.

    Parameters
    ----------
    data : Union[sc.AnnData, pd.DataFrame]
        AnnData object or DataFrame of gene expression trends.
    branch_name : str, optional
        Name of the branch for which to plot gene trends.
        It is added to the gene_trend_key when accessing gene trends from varm.
        Defaults to "".
    clusters : Union[pd.Series, str], optional
        Series of clusters indexed by gene names or string key to access clusters
        from the AnnData object's var. If data is a DataFrame, clusters should be
        a Series. Default key is 'gene_trend_key + "_" + branch_name + "_clusters"'.
    gene_trend_key : str, optional
        Key to access gene trends in the AnnData object's varm. Default is 'gene_trends'.

    Returns
    -------
    matplotlib.pyplot.Figure
        Matplotlib Figure object representing the plot of the gene trend clusters.

    Raises
    ------
    KeyError
        If `gene_trend_key` is None when `data` is an AnnData object.
    """
    # Process inputs and standardize trends
    if isinstance(data, sc.AnnData):
        if gene_trend_key is None:
            raise KeyError(
                "Must provide a gene_trend_key when data is an AnnData object."
            )

        varm_name = gene_trend_key + "_" + branch_name
        if varm_name not in data.varm:
            raise KeyError(
                f"'gene_trend_key + \"_\" + branch_name' = '{varm_name}' not found in .varm."
            )

        pt_grid_name = gene_trend_key + "_" + branch_name + "_pseudotime"
        if pt_grid_name not in data.uns.keys():
            raise KeyError(
                '\'gene_trend_key + "_" + branch_name + "_pseudotime"\' '
                f"= '{pt_grid_name}' not found in .uns."
            )

        pseudotimes = data.uns[pt_grid_name]
        trends = pd.DataFrame(
            data.varm[varm_name], index=data.var_names, columns=pseudotimes
        )

        if clusters is None:
            clusters = gene_trend_key + "_clusters"
        if isinstance(clusters, str):
            clusters = data.var[clusters]
    else:
        trends = data

    trends = pd.DataFrame(
        StandardScaler().fit_transform(trends.T).T,
        index=trends.index,
        columns=trends.columns,
    )

    # Obtain unique clusters and prepare figure
    cluster_labels = (
        clusters.cat.categories
        if pd.api.types.is_categorical_dtype(clusters)
        else set(clusters)
    )
    n_rows = int(np.ceil(len(cluster_labels) / 3))
    fig = plt.figure(figsize=[5.5 * 3, 2.5 * n_rows])

    # Plot each cluster
    for i, c in enumerate(cluster_labels):
        ax = fig.add_subplot(n_rows, 3, i + 1)
        cluster_trends = trends.loc[clusters.index[clusters == c], :]
        means = cluster_trends.mean()
        std = cluster_trends.std()

        ax.plot(
            cluster_trends.columns,
            cluster_trends.T,
            linewidth=0.5,
            color=DESELECTED_COLOR,
        )
        ax.plot(means.index, means, color=SELECTED_COLOR)
        ax.plot(
            means.index,
            means - std,
            linestyle="--",
            color=SELECTED_COLOR,
            linewidth=0.75,
        )
        ax.plot(
            means.index,
            means + std,
            linestyle="--",
            color=SELECTED_COLOR,
            linewidth=0.75,
        )

        ax.set_title(f"Cluster {c}", fontsize=12)
        ax.tick_params("both", length=2, width=1, which="major")
        ax.tick_params(axis="both", which="major", labelsize=8, direction="in")
        ax.set_xticklabels([])

    sns.despine()
    return fig


def gene_score_histogram(
    ad: sc.AnnData,
    score_key: str,
    genes: Optional[List[str]] = None,
    bins: int = 100,
    quantile: Optional[float] = 0.95,
    extra_offset_fraction: float = 0.1,
    anno_min_diff_fraction: float = 0.05,
) -> plt.Figure:
    """
    Draw a histogram of gene scores with percentile line and annotations for specific genes.

    Parameters
    ----------
    ad : sc.AnnData
        Annotated data matrix.
    score_key : str
        The key in `ad.var` data frame for the gene score.
    genes : Optional[List[str]], default=None
        List of genes to be annotated. If None, no genes are annotated.
    bins : int, default=100
        The number of bins for the histogram.
    quantile : Optional[float], default=0.95
        Quantile line to draw on the histogram. If None, no line is drawn.
    extra_offset_fraction : float, default=0.1
        Fraction of max height to use as extra offset for annotation.
    anno_min_diff_fraction : float, default=0.05
        Fraction of the range of the scores to be used as minimum difference for annotation.

    Returns
    -------
    fig : matplotlib Figure
        Figure object with the histogram.

    Raises
    ------
    ValueError
        If input parameters are not as expected.
    """
    if not isinstance(ad, sc.AnnData):
        raise ValueError("Input data should be of type sc.AnnData.")
    if score_key not in ad.var.columns:
        raise ValueError(f"Score key {score_key} not found in ad.var columns.")
    scores = ad.var[score_key]

    if genes is not None:
        if not all(gene in scores for gene in genes):
            raise ValueError("All genes must be present in the scores.")

    fig, ax = plt.subplots(figsize=(10, 6))
    n_markers = len(genes) if genes is not None else 0

    heights, bins, _ = ax.hist(scores, bins=bins, zorder=-n_markers - 2)

    if quantile is not None:
        if quantile < 0 or quantile > 1:
            raise ValueError("Quantile should be a float between 0 and 1.")
        ax.vlines(
            np.quantile(scores, quantile),
            0,
            np.max(heights),
            alpha=0.5,
            color="red",
            label=f"{quantile:.0%} percentile",
        )

    ax.legend()
    ax.set_xlabel(f"{score_key} score")
    ax.set_ylabel("# of genes")

    ax.spines[["right", "top"]].set_visible(False)
    plt.locator_params(axis="x", nbins=3)

    if genes is None:
        return fig

    previous_value = -np.inf
    extra_offset = extra_offset_fraction * np.max(heights)
    min_diff = anno_min_diff_fraction * (np.max(bins) - np.min(bins))
    marks = scores[genes].sort_values()
    ranks = scores.rank(ascending=False)
    for k, (highlight_gene, value) in enumerate(marks.items()):
        hl_rank = int(ranks[highlight_gene])
        i = np.searchsorted(bins, value)
        text_offset = -np.inf if value - previous_value > min_diff else previous_value
        previous_value = value
        height = heights[i - 1]
        text_offset = max(text_offset + extra_offset, height + 1.8 * extra_offset)
        txt = ax.annotate(
            f"{highlight_gene} #{hl_rank}",
            (value, height),
            (value, text_offset),
            arrowprops=dict(facecolor="black", width=1, alpha=0.5),
            rotation=90,
            horizontalalignment="center",
            zorder=-k,
        )
        txt.set_path_effects(
            [PathEffects.withStroke(linewidth=2, foreground="w", alpha=0.8)]
        )

    return fig
