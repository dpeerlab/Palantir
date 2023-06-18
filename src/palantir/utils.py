from typing import Iterable, Union, Tuple
from warnings import warn
import pandas as pd
import numpy as np

import phenograph

from scipy.sparse import csr_matrix, find, issparse
from scipy.sparse.linalg import eigs
import scanpy as sc

from .core import run_palantir


class CellNotFoundException(Exception):
    """Exception raised when no valid component is found for the provided cell type."""

    pass


def run_pca(
    data: Union[pd.DataFrame, sc.AnnData],
    n_components: int = 300,
    use_hvg: bool = True,
    pca_key: str = "X_pca",
) -> Union[Tuple[pd.DataFrame, np.array], None]:
    """
    Run PCA on the data.

    Parameters
    ----------
    data : Union[pd.DataFrame, sc.AnnData]
        Dataframe of cells X genes or sc.AnnData object.
        Typically multi-scale space diffusion components.
    n_components : int, optional
        Number of principal components. Default is 300.
    use_hvg : bool, optional
        Whether to use highly variable genes only for PCA. Default is True.
    pca_key : str, optional
        Key to store the PCA projections in obsm of data if it is a sc.AnnData object. Default is 'X_pca'.

    Returns
    -------
    Union[Tuple[pd.DataFrame, np.array], None]
        Tuple of PCA projections of the data and the explained variance.
        If sc.AnnData is passed as data, the results are also written to the input object and None is returned.
    """
    if isinstance(data, pd.DataFrame):
        ad = sc.AnnData(data.values)
    else:
        ad = data
        if pca_key != "X_pca":
            old_pca = ad.obsm.get("X_pca", None)
        else:
            old_pca = None

    # Run PCA
    if not use_hvg:
        n_comps = n_components
    else:
        sc.pp.pca(ad, n_comps=1000, use_highly_variable=True, zero_center=False)
        try:
            n_comps = np.where(np.cumsum(ad.uns["pca"]["variance_ratio"]) > 0.85)[0][0]
        except IndexError:
            n_comps = n_components

    # Rerun with selection number of components
    sc.pp.pca(ad, n_comps=n_comps, use_highly_variable=use_hvg, zero_center=False)

    if isinstance(data, sc.AnnData):
        data.obsm[pca_key] = ad.obsm["X_pca"]
        if old_pca is None and pca_key != "X_pca":
            del data.obsm["X_pca"]
        else:
            data.obsm["X_pca"] = old_pca

    pca_projections = pd.DataFrame(ad.obsm["X_pca"], index=ad.obs_names)
    return pca_projections, ad.uns["pca"]["variance_ratio"]


def run_diffusion_maps(
    data: Union[pd.DataFrame, sc.AnnData],
    n_components: int = 10,
    knn: int = 30,
    alpha: float = 0,
    seed: Union[int, None] = None,
    pca_key: str = "X_pca",
    kernel_key: str = "DM_Kernel",
    eigval_key: str = "DM_EigenValues",
    eigvec_key: str = "DM_EigenVectors",
):
    """
    Run Diffusion maps using the adaptive anisotropic kernel.

    Parameters
    ----------
    data : Union[pd.DataFrame, sc.AnnData]
        PCA projections of the data or adjacency matrix.
        If sc.AnnData is passed, its obsm[pca_key] is used and the result is written to
        its obsp[kernel_key], obsm[eigvec_key], and uns[eigval_key].
    n_components : int, optional
        Number of diffusion components. Default is 10.
    knn : int, optional
        Number of nearest neighbors for graph construction. Default is 30.
    alpha : float, optional
        Normalization parameter for the diffusion operator. Default is 0.
    seed : Union[int, None], optional
        Numpy random seed, randomized if None, set to an arbitrary integer for reproducibility.
        Default is None.
    pca_key : str, optional
        Key to retrieve PCA projections from data if it is a sc.AnnData object. Default is 'X_pca'.
    kernel_key : str, optional
        Key to store the kernel in obsp of data if it is a sc.AnnData object. Default is 'DM_Kernel'.
    eigval_key : str, optional
        Key to store the EigenValues in uns of data if it is a sc.AnnData object. Default is 'DM_EigenValues'.
    eigvec_key : str, optional
        Key to store the EigenVectors in obsm of data if it is a sc.AnnData object. Default is 'DM_EigenVectors'.

    Returns
    -------
    dict
        Diffusion components, corresponding eigen values and the diffusion operator.
        If sc.AnnData is passed as data, these results are also written to the input object
        and returned.
    """

    if isinstance(data, sc.AnnData):
        data_df = pd.DataFrame(data.obsm["X_pca"], index=data.obs_names)
    else:
        data_df = data

    if not isinstance(data_df, pd.DataFrame):
        raise ValueError("'data_df' should be a pd.DataFrame or a sc.AnnData instance")

    # Determine the kernel
    N = data_df.shape[0]
    if not issparse(data_df):
        print("Determing nearest neighbor graph...")
        temp = sc.AnnData(data_df.values)
        sc.pp.neighbors(temp, n_pcs=0, n_neighbors=knn)
        kNN = temp.obsp["distances"]

        # Adaptive k
        adaptive_k = int(np.floor(knn / 3))
        adaptive_std = np.zeros(N)

        for i in np.arange(len(adaptive_std)):
            adaptive_std[i] = np.sort(kNN.data[kNN.indptr[i] : kNN.indptr[i + 1]])[
                adaptive_k - 1
            ]

        # Kernel
        x, y, dists = find(kNN)

        # X, y specific stds
        dists = dists / adaptive_std[x]
        W = csr_matrix((np.exp(-dists), (x, y)), shape=[N, N])

        # Diffusion components
        kernel = W + W.T
    else:
        kernel = data_df

    # Markov
    D = np.ravel(kernel.sum(axis=1))

    if alpha > 0:
        # L_alpha
        D[D != 0] = D[D != 0] ** (-alpha)
        mat = csr_matrix((D, (range(N), range(N))), shape=[N, N])
        kernel = mat.dot(kernel).dot(mat)
        D = np.ravel(kernel.sum(axis=1))

    D[D != 0] = 1 / D[D != 0]
    T = csr_matrix((D, (range(N), range(N))), shape=[N, N]).dot(kernel)
    # Eigen value dcomposition
    np.random.seed(seed)
    v0 = np.random.rand(min(T.shape))
    D, V = eigs(T, n_components, tol=1e-4, maxiter=1000, v0=v0)
    D = np.real(D)
    V = np.real(V)
    inds = np.argsort(D)[::-1]
    D = D[inds]
    V = V[:, inds]

    # Normalize
    for i in range(V.shape[1]):
        V[:, i] = V[:, i] / np.linalg.norm(V[:, i])

    # Create are results dictionary
    res = {"T": T, "EigenVectors": V, "EigenValues": D}
    res["EigenVectors"] = pd.DataFrame(res["EigenVectors"])
    if not issparse(data_df):
        res["EigenVectors"].index = data_df.index
    res["EigenValues"] = pd.Series(res["EigenValues"])
    res["kernel"] = kernel

    if isinstance(data, sc.AnnData):
        data.obsp["DM_Kernel"] = res["kernel"]
        data.obsm["DM_EigenVectors"] = res["EigenVectors"].values
        data.uns["DM_EigenValues"] = res["EigenValues"].values

    return res


def run_magic_imputation(
    data: Union[pd.DataFrame, sc.AnnData],
    dm_res: Union[dict, None] = None,
    n_steps: int = 3,
    kernel_key: str = "DM_Kernel",
    imputation_key: str = "MAGIC_imputed_data",
) -> Union[pd.DataFrame, None]:
    """
    Run MAGIC imputation on the data.

    Parameters
    ----------
    data : Union[pd.DataFrame, sc.AnnData]
        Dataframe of cells X genes or sc.AnnData object.
    dm_res : Union[dict, None], optional
        Diffusion map results from run_diffusion_maps.
        If None and data is a sc.AnnData object, its obsp[kernel_key] is used. Default is None.
    n_steps : int, optional
        Number of steps in the diffusion operator. Default is 3.
    kernel_key : str, optional
        Key to access the kernel in obsp of data if it is a sc.AnnData object. Default is 'DM_Kernel'.
    imputation_key : str, optional
        Key to store the imputed data in layers of data if it is a sc.AnnData object. Default is 'MAGIC_imputed_data'.

    Returns
    -------
    Union[pd.DataFrame, None]
        Imputed data matrix. If sc.AnnData is passed as data, the result is written to its layers[imputation_key]
        and None is returned.
    """
    if isinstance(data, sc.AnnData):
        data_df = pd.DataFrame(
            data.X.todense(), index=data.obs_names, columns=data.var_names
        )
        if dm_res is None:
            T = data.obsp[kernel_key]
    else:
        data_df = data
    if dm_res is not None:
        T = dm_res["T"]
    elif not isinstance(data, sc.AnnData):
        raise ValueError(
            "Diffusion map results (dm_res) must be provided if data is not sc.AnnData"
        )

    T_steps = T**n_steps
    imputed_data = pd.DataFrame(
        np.dot(T_steps.todense(), data_df), index=data_df.index, columns=data_df.columns
    )

    if isinstance(data, sc.AnnData):
        data.layers[imputation_key] = imputed_data.values

    return imputed_data


def determine_multiscale_space(
    dm_res: Union[dict, sc.AnnData],
    n_eigs: Union[int, None] = None,
    eigval_key: str = "DM_EigenValues",
    eigvec_key: str = "DM_EigenVectors",
    out_key: str = "DM_EigenVectors_multiscaled",
) -> Union[pd.DataFrame, None]:
    """
    Determine the multi-scale space of the data.

    Parameters
    ----------
    dm_res : Union[dict, sc.AnnData]
        Diffusion map results from run_diffusion_maps.
        If sc.AnnData is passed, its uns[eigval_key] and obsm[eigvec_key] are used.
    n_eigs : Union[int, None], optional
        Number of eigen vectors to use. If None is specified, the number
        of eigen vectors will be determined using the eigen gap. Default is None.
    eigval_key : str, optional
        Key to retrieve EigenValues from dm_res if it is a sc.AnnData object. Default is 'DM_EigenValues'.
    eigvec_key : str, optional
        Key to retrieve EigenVectors from dm_res if it is a sc.AnnData object. Default is 'DM_EigenVectors'.
    out_key : str, optional
        Key to store the result in obsm of dm_res if it is a sc.AnnData object. Default is 'DM_EigenVectors_multiscaled'.

    Returns
    -------
    Union[pd.DataFrame, None]
        Multi-scale data matrix. If sc.AnnData is passed as dm_res, the result
        is written to its obsm[out_key] and None is returned.
    """
    if isinstance(dm_res, sc.AnnData):
        eigenvectors = dm_res.obsm[eigvec_key]
        if not isinstance(eigenvectors, pd.DataFrame):
            eigenvectors = pd.DataFrame(eigenvectors, index=dm_res.obs_names)
        dm_res_dict = {
            "EigenValues": dm_res.uns[eigval_key],
            "EigenVectors": eigenvectors,
        }
    else:
        dm_res_dict = dm_res

    if not isinstance(dm_res_dict, dict):
        raise ValueError("'dm_res' should be a dict or a sc.AnnData instance")
    if n_eigs is None:
        vals = np.ravel(dm_res_dict["EigenValues"])
        n_eigs = np.argsort(vals[: (len(vals) - 1)] - vals[1:])[-1] + 1
        if n_eigs < 3:
            n_eigs = np.argsort(vals[: (len(vals) - 1)] - vals[1:])[-2] + 1

    # Scale the data
    use_eigs = list(range(1, n_eigs))
    eig_vals = np.ravel(dm_res_dict["EigenValues"][use_eigs])
    data = dm_res_dict["EigenVectors"].values[:, use_eigs] * (eig_vals / (1 - eig_vals))
    data = pd.DataFrame(data, index=dm_res_dict["EigenVectors"].index)

    if isinstance(dm_res, sc.AnnData):
        dm_res.obsm[out_key] = data

    return data


def run_tsne(
    data: Union[pd.DataFrame, sc.AnnData],
    n_dim: int = 2,
    perplexity: int = 150,
    tsne_key: str = "X_tsne",
    **kwargs,
) -> Union[pd.DataFrame, None]:
    """
    Run t-SNE on the data.

    Parameters
    ----------
    data : Union[pd.DataFrame, sc.AnnData]
        Dataframe of cells X genes or sc.AnnData object. Typically, multiscale space diffusion components.
    n_dim : int, optional
        Number of dimensions for t-SNE embedding. Default is 2.
    perplexity : int, optional
        The perplexity parameter for t-SNE. Default is 150.
    tsne_key : str, optional
        Key to store the t-SNE embedding in obsm of data if it is a sc.AnnData object. Default is 'X_tsne'.
    **kwargs : dict, optional
        Additional keyword arguments to pass to the t-SNE function.

    Returns
    -------
    Union[pd.DataFrame, None]
        t-SNE embedding of the data. If sc.AnnData is passed as data, the result is written to its obsm[tsne_key]
        and None is returned.
    """
    if isinstance(data, sc.AnnData):
        data_df = pd.DataFrame(
            data.X.todense(), index=data.obs_names, columns=data.var_names
        )
    else:
        data_df = data

    try:
        from MulticoreTSNE import MulticoreTSNE as TSNE

        print("Using the 'MulticoreTSNE' package by Ulyanov (2017)")
        tsne = TSNE(n_components=n_dim, perplexity=perplexity, **kwargs).fit_transform(
            data_df.values
        )
    except ImportError:
        from sklearn.manifold import TSNE

        print(
            "Could not import 'MulticoreTSNE'. Install for faster runtime. Falling back to scikit-learn."
        )
        tsne = TSNE(n_components=n_dim, perplexity=perplexity, **kwargs).fit_transform(
            data_df.values
        )

    tsne = pd.DataFrame(tsne, index=data_df.index)
    tsne.columns = ["x", "y"]

    if isinstance(data, sc.AnnData):
        data.obsm[tsne_key] = tsne.values

    return tsne


def determine_cell_clusters(
    data: Union[pd.DataFrame, sc.AnnData],
    k: int = 50,
    pca_key: str = "X_pca",
    cluster_key: str = "phenograph_clusters",
) -> Union[pd.Series, None]:
    """
    Run PhenoGraph for clustering cells.

    Parameters
    ----------
    data : Union[pd.DataFrame, sc.AnnData]
        Principal components of the data or a sc.AnnData object.
    k : int, optional
        Number of neighbors for k-NN graph construction. Default is 50.
    pca_key : str, optional
        Key to access PCA results from obsm of data if it is a sc.AnnData object. Default is 'X_pca'.
    cluster_key : str, optional
        Key to store the clusters in obs of data if it is a sc.AnnData object. Default is 'phenograph_clusters'.

    Returns
    -------
    Union[pd.Series, None]
        Cell clusters. If sc.AnnData is passed as data, the result is written to its obs[cluster_key]
        and None is returned.
    """
    if isinstance(data, sc.AnnData):
        if pca_key not in data.obsm.keys():
            raise ValueError(
                f"obsm[{pca_key}] not found in AnnData. "
                "Consider running 'palantir.utils.run_pca' first."
            )
        data_df = pd.DataFrame(data.obsm[pca_key], index=data.obs_names)
    else:
        data_df = data

    # Cluster and cluster centrolds
    communities, _, _ = phenograph.cluster(data_df.values, k=k)
    communities = pd.Series(communities, index=data_df.index)

    if isinstance(data, sc.AnnData):
        data.obs[cluster_key] = communities.values

    return communities


def _return_cell(ec, obs_names, celltype, mm, dcomp):
    """
    Helper function to print and return the early cell.

    Args:
        ec (int): Index of the early cell.
        obs_names (list): Names of cells.
        celltype (str): The cell type of interest.
        mm (str): Max/min status of the diffusion component.
        dcomp (int): Index of diffusion component.

    Returns:
        str: Name of the early cell.
    """
    early_cell = obs_names[ec]
    print(
        f"Using {early_cell} for cell type {celltype} which is {mm} in "
        f"diffusion component {dcomp}."
    )
    return early_cell


def early_cell(
    ad: sc.AnnData,
    celltype: str,
    celltype_column: str = "celltype",
    fallback_seed: int = None,
):
    """
    Helper to determine 'early_cell' for 'palantir.core.run_palantir'.
    Finds cell of 'celltype' at the extremes of the state space represented by diffusion maps.

    Args:
        ad (AnnData): Annotated data matrix.
        celltype (str): The cell type of interest.
        celltype_column (str): Column name in the data matrix where the cell
        type information is stored. Default is 'celltype'.
        fallback_seed (int): Seed for random number generator in fallback method.
        Default is None.

    Returns:
        str: Name of the terminal cell for the given cell type.

    Raises:
        CellNotFoundException: If no valid component is found for the provided cell type.
    """
    if not isinstance(ad, sc.AnnData):
        raise ValueError("'ad' should be an instance of sc.AnnData")

    if "DM_EigenVectors" not in ad.obsm:
        raise ValueError(
            "'DM_EigenVectors' not found in ad.obsm. "
            "Run `palantir.utils.run_diffusion_maps(ad)` first."
        )

    if not isinstance(celltype_column, str):
        raise ValueError("'celltype_column' should be a string")

    if celltype_column not in ad.obs.columns:
        raise ValueError("'celltype_column' should be a column of ad.obs.")

    if not isinstance(celltype, str):
        raise ValueError("'celltype' should be a string")

    if celltype not in ad.obs[celltype_column].values:
        raise ValueError(
            f"Celltype '{celltype}' not found in ad.obs['{celltype_column}']."
        )

    if fallback_seed is not None and not isinstance(fallback_seed, int):
        raise ValueError("'fallback_seed' should be an integer")

    for dcomp in range(ad.obsm["DM_EigenVectors"].shape[1]):
        ec = ad.obsm["DM_EigenVectors"][:, dcomp].argmax()
        if ad.obs[celltype_column][ec] == celltype:
            return _return_cell(ec, ad.obs_names, celltype, "max", dcomp)
        ec = ad.obsm["DM_EigenVectors"][:, dcomp].argmin()
        if ad.obs[celltype_column][ec] == celltype:
            return _return_cell(ec, ad.obs_names, celltype, "min", dcomp)

    if fallback_seed is not None:
        print("Falling back to slow early cell detection.")
        return fallback_terminal_cell(
            ad, celltype, celltype_column=celltype_column, seed=fallback_seed
        )

    raise CellNotFoundException(
        f"No valid component found: {celltype} "
        "Consider increasing the number of diffusion components "
        "('n_components' in palantir.utils.run_diffusion_maps) "
        "or specify a 'fallback_seed' to determine an early cell based on "
        f"reverse pseudotime starting from random non-{celltype} cell."
    )


def fallback_terminal_cell(ad, celltype, celltype_column="anno", seed=2353):
    """
    Fallback method to find terminal cells when no valid diffusion component
    is found for the provided cell type.

    Args:
        ad (AnnData): Annotated data matrix.
        celltype (str): The cell type of interest.
        celltype_column (str): Column name in the data matrix where the cell
        type information is stored. Default is 'anno'.
        seed (int): Seed for random number generator. Default is 2353.

    Returns:
        str: Name of the terminal cell for the given cell type.
    """
    other_cells = ad.obs_names[ad.obs[celltype_column] != celltype]
    fake_early_cell = other_cells.to_series().sample(1, random_state=seed)[0]
    pr_res = run_palantir(
        ad,
        fake_early_cell,
        terminal_states=None,
        use_early_cell_as_start=True,
    )
    idx = ad.obs[celltype_column] == celltype
    ec = pr_res.pseudotime[idx].argmax()
    early_cell = ad.obs_names[ec]
    print(
        f"Using {early_cell} for cell type {celltype} which is latest cell in "
        "{celltype} when starting from {fake_early_cell}."
    )
    return early_cell


def find_terminal_states(
    ad: sc.AnnData,
    celltypes: Iterable,
    celltype_column: str = "celltype",
    fallback_seed: int = None,
):
    """
    Identifies terminal states for a list of cell types in the AnnData object.

    This function iterates over the provided cell types, attempting to find a terminal cell for each one
    using the `palantir.utils.early_cell` function. In cases where no valid component is found for a cell type,
    it emits a warning and skips to the next cell type.

    Parameters
    ----------
    ad : AnnData
        Annotated data matrix from Scanpy. It should contain computed diffusion maps.
    celltypes : Iterable
        An iterable (like a list or tuple) of cell type names for which terminal states are to be identified.
    celltype_column : str, optional
        Column name in the AnnData object where the cell type information is stored. Default is 'celltype'.
    fallback_seed : int, optional
        Seed for the random number generator used in the fallback method of `palantir.utils.early_cell` function.
        Defaults to None, in which case the RNG will be seeded randomly.

    Returns
    -------
    pd.Series
        A pandas Series where the index are the cell types and the values are the names of the terminal cells.
        If no terminal cell is found for a cell type, it will not be included in the series.
    """
    terminal_states = pd.Series(dtype=str)
    for ct in celltypes:
        try:
            cell = early_cell(ad, ct, celltype_column, fallback_seed)
        except CellNotFoundException:
            warn(
                f"No valid component found: {ct} "
                "Consider increasing the number of diffusion components "
                "('n_components' in palantir.utils.run_diffusion_maps). "
                f"The cell type {ct} will be skipped."
            )
            continue
        terminal_states[ct] = cell
    return terminal_states
