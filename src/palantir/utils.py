import pandas as pd
import numpy as np
from MulticoreTSNE import MulticoreTSNE as TSNE
import phenograph

from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix, find, issparse
from scipy.sparse.linalg import eigs
import scanpy as sc


def run_pca(data, n_components=300):
    """Run PCA

    :param data: Dataframe of cells X genes. Typicaly multiscale space diffusion components
    :param n_components: Number of principal components
    :return: PCA projections of the data and the explained variance
    """
    pca = PCA(n_components=n_components, svd_solver="randomized")
    pca_projections = pca.fit_transform(data)
    pca_projections = pd.DataFrame(pca_projections, index=data.index)
    return pca_projections, pca.explained_variance_ratio_


def run_diffusion_maps(data_df, n_components=10, knn=30, alpha=0):
    """Run Diffusion maps using the adaptive anisotropic kernel

    :param data_df: PCA projections of the data or adjacency matrix
    :param n_components: Number of diffusion components
    :param knn: Number of nearest neighbors for graph construction
    :param alpha: Normalization parameter for the diffusion operator
    :return: Diffusion components, corresponding eigen values and the diffusion operator
    """

    # Determine the kernel
    N = data_df.shape[0]
    if not issparse(data_df):
        print("Determing nearest neighbor graph...")
        temp = sc.AnnData(data_df.values)
        sc.pp.neighbors(temp, n_pcs=0, n_neighbors=knn)
        # maintaining backwards compatibility to Scanpy `sc.pp.neighbors`
        try:
            kNN = temp.uns["neighbors"]["distances"]
        except KeyError:
            kNN = temp.obsp['distances']

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
    D, V = eigs(T, n_components, tol=1e-4, maxiter=1000)
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

    return res


def run_magic_imputation(data, dm_res, n_steps=3):
    """Run MAGIC imputation

    :param dm_res: Diffusion map results from run_diffusion_maps
    :param n_steps: Number of steps in the diffusion operator
    :return: Imputed data matrix
    """
    T_steps = dm_res["T"] ** n_steps
    imputed_data = pd.DataFrame(
        np.dot(T_steps.todense(), data), index=data.index, columns=data.columns
    )

    return imputed_data


def determine_multiscale_space(dm_res, n_eigs=None):
    """Determine multi scale space of the data

    :param dm_res: Diffusion map results from run_diffusion_maps
    :param n_eigs: Number of eigen vectors to use. If None specified, the number
            of eigen vectors will be determined using eigen gap
    :return: Multi scale data matrix
    """
    if n_eigs is None:
        vals = np.ravel(dm_res["EigenValues"])
        n_eigs = np.argsort(vals[: (len(vals) - 1)] - vals[1:])[-1] + 1
        if n_eigs < 3:
            n_eigs = np.argsort(vals[: (len(vals) - 1)] - vals[1:])[-2] + 1

    # Scale the data
    use_eigs = list(range(1, n_eigs))
    eig_vals = np.ravel(dm_res["EigenValues"][use_eigs])
    data = dm_res["EigenVectors"].values[:, use_eigs] * (eig_vals / (1 - eig_vals))
    data = pd.DataFrame(data, index=dm_res["EigenVectors"].index)

    return data


def run_tsne(data, n_dim=2, perplexity=150, **kwargs):
    """Run tSNE

    :param data: Dataframe of cells X genes. Typicaly multiscale space diffusion components
    :param n_dim: Number of dimensions for tSNE embedding
    :return: tSNE embedding of the data
    """
    tsne = TSNE(n_components=n_dim, perplexity=perplexity, **kwargs).fit_transform(
        data.values
    )
    tsne = pd.DataFrame(tsne, index=data.index)
    tsne.columns = ["x", "y"]
    return tsne


def determine_cell_clusters(data, k=50):
    """Run phenograph for clustering cells

    :param data: Principal components of the data.
    :param k: Number of neighbors for kNN graph construction
    :return: Clusters
    """
    # Cluster and cluster centrolds
    communities, _, _ = phenograph.cluster(data, k=k)
    communities = pd.Series(communities, index=data.index)
    return communities
