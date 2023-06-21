from typing import Union, Optional, List, Dict
import numpy as np
import pandas as pd
import pickle
import time

from collections import OrderedDict
from joblib import delayed, Parallel
from sklearn.preprocessing import StandardScaler
from pygam import LinearGAM, s
import scanpy as sc
import mellon


# Used for trend computation and branch selection
PSEUDOTIME_RES = 500


class PResults(object):
    """
    Container of palantir results
    """

    def __init__(self, pseudotime, entropy, branch_probs, waypoints):

        # Initialize
        self._pseudotime = (pseudotime - pseudotime.min()) / (
            pseudotime.max() - pseudotime.min()
        )
        self._entropy = entropy
        self._branch_probs = branch_probs
        self._branch_probs[self._branch_probs < 0.01] = 0
        self._waypoints = waypoints

    # Getters and setters
    @property
    def pseudotime(self):
        return self._pseudotime

    @property
    def branch_probs(self):
        return self._branch_probs

    @branch_probs.setter
    def branch_probs(self, branch_probs):
        self._branch_probs = branch_probs

    @property
    def entropy(self):
        return self._entropy

    @entropy.setter
    def entropy(self, entropy):
        self._entropy = entropy

    @property
    def waypoints(self):
        return self._waypoints

    @classmethod
    def load(cls, pkl_file):
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)

        # Set up object
        presults = cls(
            data["_pseudotime"],
            data["_entropy"],
            data["_branch_probs"],
            data["_waypoints"],
        )
        return presults

    def save(self, pkl_file: str):
        pickle.dump(vars(self), pkl_file)


def compute_gene_trends_legacy(
    data: Union[sc.AnnData, PResults],
    gene_exprs: Optional[pd.DataFrame] = None,
    lineages: Optional[List[str]] = None,
    n_splines: int = 4,
    spline_order: int = 2,
    n_jobs: int = -1,
    expression_key: str = "MAGIC_imputed_data",
    pseudo_time_key: str = "palantir_pseudotime",
    fate_prob_key: str = "palantir_fate_probabilities",
    gene_trend_key: str = "palantir_gene_trends",
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Compute gene expression trends along pseudotemporal trajectory.

    This function calculates gene expression trends and their standard deviations
    along the pseudotemporal trajectory computed by Palantir.

    Parameters
    ----------
    data : Union[sc.AnnData, palantir.presults.PResults]
        Either a Scanpy AnnData object or a Palantir results object.
    gene_exprs : pd.DataFrame, optional
        DataFrame of gene expressions, shape (cells, genes).
    lineages : List[str], optional
        Subset of lineages for which to compute the trends.
        If None uses all columns of the fate probability matrix.
        Default is None.
    n_splines : int, optional
        Number of splines to use. Must be non-negative. Default is 4.
    spline_order : int, optional
        Order of the splines to use. Must be non-negative. Default is 2.
    n_jobs : int, optional
        Number of cores to use. Default is -1.
    expression_key : str, optional
        Key to access gene expression matrix from a layer of the AnnData object. Default is 'MAGIC_imputed_data'.
        If `gene_exprs` is None, this key is used to fetch the gene expressions from `data.X`.
    pseudo_time_key : str, optional
        Key to access pseudotime from obs of the AnnData object. Default is 'palantir_pseudotime'.
    fate_prob_key : str, optional
        Key to access fate probabilities from obsm of the AnnData object. Default is 'palantir_fate_probabilities'.
    gene_trend_key : str, optional
        Starting key to store the gene trends in the varm attribute of the AnnData object. The default is 'palantir_gene_trends'.
        The gene trend matrices for each fate will be stored under 'varm[gene_trend_key + "_" + lineage_name]'.
        The pseudotime points at which the gene trends are computed, corresponding to the columns of the gene trend matrices,
        are stored in the uns attribute under 'uns[gene_trend_key + "_" + lineage_name + "_pseudotime"]'.


    Returns
    -------
    Dict[str, Dict[str, pd.DataFrame]]
        Dictionary of gene expression trends and standard deviations for each branch.
    """
    # Extract palantir results from AnnData if necessary
    if isinstance(data, sc.AnnData):
        gene_exprs = data.to_df(expression_key)
        pseudo_time = pd.Series(data.obs[pseudo_time_key], index=data.obs_names)
        fate_probs = pd.DataFrame(
            data.obsm[fate_prob_key],
            index=data.obs_names,
            columns=data.uns[fate_prob_key + "_columns"],
        )

        pr_res = PResults(pseudo_time, None, fate_probs, None)
    else:
        pr_res = data

    # Compute for all lineages if branch is not speicified
    if lineages is None:
        lineages = pr_res.branch_probs.columns

    # Results Container
    results = OrderedDict()
    for branch in lineages:
        results[branch] = OrderedDict()
        # Bins along the pseudotime
        br_cells = pr_res.branch_probs.index[pr_res.branch_probs.loc[:, branch] > 0.7]
        bins = np.linspace(0, pr_res.pseudotime[br_cells].max(), PSEUDOTIME_RES)

        # Branch results container
        results[branch]["trends"] = pd.DataFrame(
            0.0, index=gene_exprs.columns, columns=bins
        )
        results[branch]["std"] = pd.DataFrame(
            0.0, index=gene_exprs.columns, columns=bins
        )

    # Compute for each branch
    for branch in lineages:
        print(branch)
        start = time.time()

        # Branch cells and weights
        weights = pr_res.branch_probs.loc[gene_exprs.index, branch].values
        bins = np.array(results[branch]["trends"].columns)
        res = Parallel(n_jobs=n_jobs)(
            delayed(gam_fit_predict)(
                pr_res.pseudotime[gene_exprs.index].values,
                gene_exprs.loc[:, gene].values,
                weights,
                bins,
                n_splines,
                spline_order,
            )
            for gene in gene_exprs.columns
        )

        # Fill in the matrices
        for i, gene in enumerate(gene_exprs.columns):
            results[branch]["trends"].loc[gene, :] = res[i][0]
            results[branch]["std"].loc[gene, :] = res[i][1]
        if isinstance(data, sc.AnnData):
            data.varm[gene_trend_key + "_" + branch] = results[branch]["trends"].values
            data.uns[gene_trend_key + "_" + branch + "_pseudotime"] = results[branch][
                "trends"
            ].columns.values
        end = time.time()
        print("Time for processing {}: {} minutes".format(branch, (end - start) / 60))

    return results


def compute_gene_trends(
    ad: sc.AnnData,
    lineages: Optional[List[str]] = None,
    masks_key: str = "branch_masks",
    expression_key: str = None,
    pseudo_time_key: str = "palantir_pseudotime",
    gene_trend_key: str = "gene_trends",
    **kwargs,
) -> pd.DataFrame:
    """
    Compute gene expression trends along pseudotime in the given AnnData object.

    This function computes the gene expression trends for each branch of the
    pseudotime trajectory using mellon.FunctionEstimator. The computed gene
    trends are stored in the varm attribute of the AnnData object, with keys
    in the format '{gene_trend_key}_{branch}'. Each key maps to a 2D numpy
    array where rows correspond to genes and columns correspond to the
    pseudotime grid. The pseudotime grid for each branch is stored in the uns
    attribute of the AnnData object, with keys in the format
    '{gene_trend_key}_{branch}_pseudotime'.

    Parameters
    ----------
    ad : sc.AnnData
        AnnData object containing the gene expression data and pseudotime.
    lineages : List[str], optional
        Subset of lineages for which to compute the trends.
        If None uses all columns of the fate probability matrix.
        Default is None.
    masks_key : str, optional
        Key to access the branch cell selection masks from obsm of the AnnData object.
        Default is 'branch_masks'.
    expression_key : str, optional
        Key to access the gene expression data in the layers of the AnnData object.
        If None, uses raw expression data in .X. Default is None.
    pseudo_time_key : str, optional
        Key to access the pseudotime values in the AnnData object. Default is 'palantir_pseudotime'.
    gene_trend_key : str, optional
        Key base to store the gene expression trends in varm of the AnnData object.
        Default is 'palantir_gene_trends'.
    **kwargs
        Additional arguments to be passed to mellon.FunctionEstimator.

    Returns
    -------
    dict
        A dictionary containing gene expression trends for each branch. The keys of the dictionary
        are the branch names. The value for each branch is a sub-dictionary with a key 'trends' that
        maps to a DataFrame. The DataFrame contains the gene expression trends, indexed by gene names
        and columns representing pseudotime points.
    """
    # Check the AnnData object for the necessary keys
    if pseudo_time_key not in ad.obs_keys():
        raise ValueError(
            f"'{pseudo_time_key}' is not found in the AnnData object's obs."
        )

    assert masks_key in ad.obsm, f"{masks_key} not found in ad.obsm"
    assert masks_key + "_columns" in ad.uns, f"{masks_key}_columns not found in ad.uns"

    gene_exprs = ad.to_df(expression_key)
    pseudo_time = ad.obs[pseudo_time_key].values
    masks = ad.obsm[masks_key]
    branches = ad.uns[masks_key + "_columns"]

    if lineages is not None:
        for lin in lineages:
            if lin not in branches:
                raise ValueError(
                    f"Lineage '{lin}' does not seem to have a selection in obsm['{masks_key}']."
                )
    else:
        lineages = branches

    # Set the default arguments for mellon.FunctionEstimator
    mellon_args = dict(sigma=1, ls=5, n_landmarks=0)
    mellon_args.update(kwargs)

    lagacy_results = dict()
    # Compute the gene expression trends
    for i, branch in enumerate(branches):
        if branch not in lineages:
            continue
        print(branch)
        mask = masks[:, i]
        pt = pseudo_time[mask]
        pt_grid = np.linspace(pt.min(), pt.max(), PSEUDOTIME_RES)
        expr = gene_exprs.loc[mask, :]
        func_est = mellon.FunctionEstimator(**mellon_args)
        result = func_est.fit_predict(pt, expr, pt_grid).T
        result = np.asarray(result)

        lagacy_results[branch] = {
            "trends": pd.DataFrame(result, columns=pt_grid, index=ad.var_names)
        }
        # Store the trends in the AnnData object and return as a DataFrame
        ad.varm[gene_trend_key + "_" + branch] = result
        ad.uns[gene_trend_key + "_" + branch + "_pseudotime"] = pt_grid

    return lagacy_results


def gam_fit_predict(x, y, weights=None, pred_x=None, n_splines=4, spline_order=2):
    """
    Function to compute individual gene trends using pyGAM

    :param x: Pseudotime axis
    :param y: Magic imputed expression for one gene
    :param weights: Lineage branch weights
    :param pred_x: Pseudotime axis for predicted values
    :param n_splines: Number of splines to use. Must be non-negative.
    :param spline_order: Order of spline to use. Must be non-negative.
    """

    # Weights
    if weights is None:
        weights = np.repeat(1.0, len(x))

    # Construct dataframe
    use_inds = np.where(weights > 0)[0]

    # GAM fit
    gam = LinearGAM(s(0, n_splines=n_splines, spline_order=spline_order)).fit(
        x[use_inds], y[use_inds], weights=weights[use_inds]
    )

    # Predict
    if pred_x is None:
        pred_x = x
    y_pred = gam.predict(pred_x)

    # Standard deviations
    p = gam.predict(x[use_inds])
    n = len(use_inds)
    sigma = np.sqrt(((y[use_inds] - p) ** 2).sum() / (n - 2))
    stds = (
        np.sqrt(1 + 1 / n + (pred_x - np.mean(x)) ** 2 / ((x - np.mean(x)) ** 2).sum())
        * sigma
        / 2
    )

    return y_pred, stds


def _gam_fit_predict_rpy2(x, y, weights=None, pred_x=None):

    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri, Formula
    from rpy2.robjects.packages import importr

    pandas2ri.activate()

    # Weights
    if weights is None:
        weights = np.repeat(1.0, len(x))

    # Construct dataframe
    use_inds = np.where(weights > 0)[0]
    r_df = pandas2ri.py2rpy(
        pd.DataFrame(np.array([x, y]).T[use_inds, :], columns=["x", "y"])
    )

    # Fit the model
    rgam = importr("gam")
    model = rgam.gam(Formula("y~s(x)"), data=r_df, weights=pd.Series(weights[use_inds]))

    # Predictions
    if pred_x is None:
        pred_x = x
    y_pred = np.array(
        robjects.r.predict(
            model, newdata=pandas2ri.py2rpy(pd.DataFrame(pred_x, columns=["x"]))
        )
    )

    # Standard deviations
    p = np.array(
        robjects.r.predict(
            model, newdata=pandas2ri.py2rpy(pd.DataFrame(x[use_inds], columns=["x"]))
        )
    )
    n = len(use_inds)
    sigma = np.sqrt(((y[use_inds] - p) ** 2).sum() / (n - 2))
    stds = (
        np.sqrt(1 + 1 / n + (pred_x - np.mean(x)) ** 2 / ((x - np.mean(x)) ** 2).sum())
        * sigma
        / 2
    )

    return y_pred, stds


def cluster_gene_trends(
    data: Union[sc.AnnData, pd.DataFrame],
    branch_name: str,
    genes: Optional[List[str]] = None,
    gene_trend_key: Optional[str] = "gene_trends",
    n_neighbors: int = 150,
    **kwargs,
) -> pd.Series:
    """
    Cluster gene trends using the Leiden algorithm.

    This function applies the Leiden clustering algorithm to gene expression trends
    along the pseudotemporal trajectory. If the input is an AnnData object, it uses
    the gene trends stored in the `varm` attribute accessed using the `gene_trend_key`.
    If the input is a DataFrame, it directly uses the input data for clustering.

    Parameters
    ----------
    data : Union[sc.AnnData, pd.DataFrame]
        AnnData object or a DataFrame of gene expression trends.
    branch_name : str
        Name of the branch for which the gene trends are to be clustered.
    genes : list of str, optional
        List of genes to be considered for clustering. If None, all genes are considered. Default is None.
    gene_trend_key : str, optional
        Key to access gene trends in the AnnData object's varm. Default is 'palantir_gene_trends'.
    n_neighbors : int, optional
        The number of nearest neighbors to use for the k-NN graph construction. Default is 150.
    **kwargs
        Additional keyword arguments passed to `scanpy.tl.leiden`.

    Returns
    -------
    pd.Series
        A pandas series with the cluser lables for all passed genes.

    Raises
    ------
    KeyError
        If `gene_trend_key` is None when `data` is an AnnData object.
    """
    if isinstance(data, sc.AnnData):
        if gene_trend_key is None:
            raise KeyError(
                "Must provide a gene_trend_key when data is an AnnData object."
            )
        varm_name = gene_trend_key + "_" + branch_name
        if varm_name not in data.varm:
            raise ValueError(
                f"'gene_trend_key + \"_\" + branch_name' = '{varm_name}' not found in .varm. "
            )
        pt_grid_name = gene_trend_key + "_" + branch_name + "_pseudotime"
        if pt_grid_name not in data.uns.keys():
            raise ValueError(
                '\'gene_trend_key + "_" + branch_name + "_pseudotime"\' '
                f"= '{pt_grid_name}' not found in .uns. "
            )
        pseudotimes = data.uns[pt_grid_name]
        trends = pd.DataFrame(
            data.varm[varm_name], index=data.var_names, columns=pseudotimes
        )
    else:
        trends = data

    if genes is not None:
        trends = trends.loc[genes, :]

    # Standardize the trends
    trends = pd.DataFrame(
        StandardScaler().fit_transform(trends.T).T,
        index=trends.index,
        columns=trends.columns,
    )

    gt_ad = sc.AnnData(trends.values)
    sc.pp.neighbors(gt_ad, n_neighbors=n_neighbors, use_rep="X")
    sc.tl.leiden(gt_ad, **kwargs)

    communities = pd.Series(gt_ad.obs["leiden"].values, index=trends.index)

    if isinstance(data, sc.AnnData):
        col_name = gene_trend_key + "_clusters"
        if genes is None:
            data.var[col_name] = communities
        else:
            data.var[col_name] = communities
            data.var[col_name] = data.var[col_name].astype("category")

    return communities


def select_branch_cells(
    ad: sc.AnnData,
    pseudo_time_key: str = "palantir_pseudotime",
    fate_prob_key: str = "palantir_fate_probabilities",
    q: float = 1e-2,
    eps: float = 1e-2,
    masks_key: str = "branch_masks",
):
    """
    Selects cells along specific branches of pseudotime ordering.

    This function identifies cells that are most likely to follow a certain lineage or "fate" by looking at
    their pseudotime order and fate probabilities. These cells are expected to be along the path of differentiation
    towards that specific fate.

    Parameters
    ----------
    ad : sc.AnnData
        Annotated data matrix. The pseudotime and fate probabilities should be stored under the keys provided.
    pseudo_time_key : str, optional
        Key to access the pseudotime from obs of the AnnData object. Default is 'palantir_pseudotime'.
    fate_prob_key : str, optional
        Key to access the fate probabilities from obsm of the AnnData object. Default is 'palantir_fate_probabilities'.
    q : float, optional
        Quantile used to determine the threshold for the fate probability.
        This parameter should be between 0 and 1. Default is 1e-2.
    eps : float, optional
        A small constant substracted from the fate probability threshold. Default is 1e-2.
    masks_key : str, optional
        Key under which the resulting branch cell selection masks are stored in the obsm of the AnnData object.
        Default is 'branch_masks'.

    Returns
    -------
    masks : np.ndarray
        An array of boolean masks that indicates whether each cell is on the path to each fate.
    """

    # make sure that the necessary keys are in the AnnData object
    if pseudo_time_key not in ad.obs:
        raise KeyError(f"{pseudo_time_key} not found in ad.obs")
    if fate_prob_key not in ad.obsm:
        raise KeyError(f"{fate_prob_key} not found in ad.obsm")
    if fate_prob_key + "_columns" not in ad.uns:
        raise KeyError(f"{fate_prob_key}_columns not found in ad.uns")

    # retrieve fate probabilities, names, and pseudotime
    fate_probs = ad.obsm[fate_prob_key]
    fate_names = ad.uns[fate_prob_key + "_columns"]
    pseudotime = ad.obs[pseudo_time_key].values

    idx = np.argsort(pseudotime)
    sorted_fate_probs = fate_probs[idx, :]
    max_probs = np.empty_like(fate_probs)
    n = max_probs.shape[0]

    step = n // PSEUDOTIME_RES
    nsteps = n // step
    for i in range(nsteps):
        l, r = i * step, (i + 1) * step
        mprob = np.quantile(sorted_fate_probs[:r, :], 1 - eps, axis=0)
        max_probs[l:r, :] = mprob[None, :]
    mprob = np.quantile(sorted_fate_probs, 1 - q, axis=0)
    max_probs[r:, :] = mprob[None, :]

    masks = np.empty_like(fate_probs).astype(bool)
    masks[idx, :] = max_probs < sorted_fate_probs + eps

    ad.obsm[masks_key] = masks
    ad.uns[masks_key + "_columns"] = fate_names

    return masks
