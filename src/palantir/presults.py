import numpy as np
import pandas as pd
import pickle
import time
import shutil
import phenograph

from collections import OrderedDict
from joblib import delayed, Parallel
from sklearn.preprocessing import StandardScaler


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


def compute_gene_trends(pr_res, gene_exprs, lineages=None, n_jobs=-1):
    """Function for computing gene expression trends along Palantir pseudotime

    :param pr_res: Palantir results object
    :param gene_exprs: Magic imputed data [Cells X Genes]
    :param lineages: Subset of lineages for which to compute the trends
    :return: Dictionary of gene expression trends and standard deviations for each branch
    """

    # Error check
    try:
        import rpy2
        import rpy2.rinterface_lib.embedded as embedded
        from rpy2.robjects.packages import importr
    except ImportError:
        raise RuntimeError(
            'Cannot compute gene expression trends without installing rpy2. \
            \nPlease use "pip3 install rpy2" to install rpy2'
        )

    if not shutil.which("R"):
        raise RuntimeError(
            "R installation is necessary for computing gene expression trends. \
            \nPlease install R and try again"
        )

    try:
        rgam = importr("gam")
    except embedded.RRuntimeError:
        raise RuntimeError(
            'R package "gam" is necessary for computing gene expression trends. \
            \nPlease install gam from https://cran.r-project.org/web/packages/gam/ and try again'
        )

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
            delayed(_gam_fit_predict)(
                pr_res.pseudotime[gene_exprs.index].values,
                gene_exprs.loc[:, gene].values,
                weights,
                bins,
            )
            for gene in gene_exprs.columns
        )

        # Fill in the matrices
        for i, gene in enumerate(gene_exprs.columns):
            results[branch]["trends"].loc[gene, :] = res[i][0]
            results[branch]["std"].loc[gene, :] = res[i][1]
        end = time.time()
        print("Time for processing {}: {} minutes".format(branch, (end - start) / 60))

    return results


def _gam_fit_predict(x, y, weights=None, pred_x=None):

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


def cluster_gene_trends(trends, k=150, n_jobs=-1):
    """Function to cluster gene trends
    :param trends: Matrix of gene expression trends
    :param k: K for nearest neighbor construction
    :param n_jobs: Number of jobs for parallel processing
    :return: Clustering of gene trends
    """

    # Standardize the trends
    trends = pd.DataFrame(
        StandardScaler().fit_transform(trends.T).T,
        index=trends.index,
        columns=trends.columns,
    )

    # Cluster
    clusters, _, _ = phenograph.cluster(trends, k=k, n_jobs=n_jobs)
    clusters = pd.Series(clusters, index=trends.index)
    return clusters
