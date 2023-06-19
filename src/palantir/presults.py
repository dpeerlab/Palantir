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


def compute_gene_trends(
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
        bins = np.linspace(0, pr_res.pseudotime[br_cells].max(), 500)

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
    gene_trend_key: Optional[str] = None,
    n_neighbors: int = 150,
    **kwargs,
) -> pd.Series:
    """Cluster gene trends.

    This function applies the Leiden clustering algorithm to gene expression trends
    along the pseudotemporal trajectory computed by Palantir.

    Parameters
    ----------
    data : Union[sc.AnnData, pd.DataFrame]
        Either a Scanpy AnnData object or a DataFrame of gene expression trends.
    gene_trend_key : str, optional
        Key to access gene trends from varm of the AnnData object. If data is a DataFrame,
        gene_trend_key should be None. If gene_trend_key is None when data is an AnnData,
        a KeyError will be raised. Default is None.
    n_neighbors : int, optional
        The number of nearest neighbors to use for the k-NN graph construction. Default is 150.
    **kwargs
        Additional arguments to be passed to `scanpy.pp.neighbors`.

    Returns
    -------
    pd.Series
        Clustering of gene trends, indexed by gene names.

    Raises
    ------
    KeyError
        If gene_trend_key is None when data is an AnnData object.
    """
    if isinstance(data, sc.AnnData):
        if gene_trend_key is None:
            raise KeyError(
                "Must provide a gene_trend_key when data is an AnnData object."
            )
        pseudotimes = data.uns[gene_trend_key + "_pseudotime"]
        trends = pd.DataFrame(
            data.varm[gene_trend_key], index=data.var_names, columns=pseudotimes
        )
    else:
        trends = data

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
        col_name = gene_trend_key + '_clusters'
        data.var[col_name] = communities

    return communities


def select_branch_cells(
    ad: sc.AnnData,
    pseudo_time_key: str = "palantir_pseudotime",
    fate_prob_key: str = "palantir_fate_probabilities",
    eps: float = 1e-2,
    selection_key: str = "branch_mask",
):
    """
    Selects cells along specific branches of pseudotime ordering.

    Parameters
    ----------
    ad : sc.AnnData
        Annotated data matrix. The pseudotime and fate probabilities should be stored under the keys provided.
    pseudo_time_key : str, optional
        Key to access the pseudotime from obs of the AnnData object. Default is 'palantir_pseudotime'.
    fate_prob_key : str, optional
        Key to access the fate probabilities from obsm of the AnnData object. Default is 'palantir_fate_probabilities'.
    eps : float, optional
        Epsilon value used to create a buffer around the fate probabilities. Default is 1e-2.
    selection_key : str, optional
        Key under which the resulting branch cell selection mask is stored in the AnnData object. Default is 'branch_mask'.

    Returns
    -------
    None
        The resulting selection mask is written to the obs of the AnnData object.
    """
    # make sure that the necessary keys are in the AnnData object
    assert pseudo_time_key in ad.obs, f"{pseudo_time_key} not found in ad.obs"
    assert fate_prob_key in ad.obsm, f"{fate_prob_key} not found in ad.obsm"
    assert (
        fate_prob_key + "_columns" in ad.uns
    ), f"{fate_prob_key}_columns not found in ad.uns"

    # retrieve fate probabilities and names
    fate_probs = ad.obsm[fate_prob_key]
    fate_names = ad.uns[fate_prob_key + "_columns"]

    # calculate the max probability along the pseudotime
    max_prob = np.max(fate_probs, axis=1)
    pt = ad.obs[pseudo_time_key] / ad.obs[pseudo_time_key].max()

    # find the early cell
    early_cell = np.argmin(pt)

    # calculate the mask for each branch
    for i, fate in enumerate(fate_names):
        out_column = selection_key + "_" + fate
        fate_prob = fate_probs[:, i]
        probability_buffer = max_prob[early_cell] - fate_prob[early_cell]
        ad.obs[out_column] = (max_prob - fate_prob) < probability_buffer * (
            1 - pt
        ) + eps

    return None
