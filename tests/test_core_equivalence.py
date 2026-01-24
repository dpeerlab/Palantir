from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, csgraph, find
from sklearn.neighbors import NearestNeighbors

import scanpy as sc
from anndata import AnnData

from palantir.core import _compute_pseudotime, _connect_graph, _max_min_sampling
from palantir.utils import compute_kernel


def _max_min_sampling_reference(
    data: pd.DataFrame, num_waypoints: int, seed: int | None = None
) -> pd.Index:
    waypoint_set = []
    no_iterations = int(num_waypoints / data.shape[1])
    if seed is not None:
        np.random.seed(seed)
    N = data.shape[0]
    for ind in data.columns:
        vec = np.ravel(data[ind])
        iter_set = [np.random.randint(N)]
        dists = np.zeros([N, no_iterations])
        dists[:, 0] = abs(vec - data[ind].values[iter_set])
        for k in range(1, no_iterations):
            min_dists = dists[:, 0:k].min(axis=1)
            new_wp = np.where(min_dists == min_dists.max())[0][0]
            iter_set.append(new_wp)
            dists[:, k] = abs(vec - data[ind].values[new_wp])
        waypoint_set = waypoint_set + iter_set
    return data.index[waypoint_set].unique()


def _compute_pseudotime_reference(
    data: pd.DataFrame,
    start_cell: str,
    knn: int,
    waypoints: pd.Index,
    max_iterations: int = 25,
):
    nbrs = NearestNeighbors(n_neighbors=knn, metric="euclidean", n_jobs=1).fit(data)
    adj = nbrs.kneighbors_graph(data, mode="distance")
    adj = _connect_graph(adj, data, np.where(data.index == start_cell)[0][0])

    dists = []
    for cell in waypoints:
        idx = np.where(data.index == cell)[0][0]
        dists.append(csgraph.dijkstra(adj, False, idx))

    D = pd.DataFrame(0.0, index=waypoints, columns=data.index)
    for i, cell in enumerate(waypoints):
        D.loc[cell, :] = pd.Series(np.ravel(dists[i]), index=data.index)

    sdv = np.std(np.ravel(D)) * 1.06 * len(np.ravel(D)) ** (-1 / 5)
    W = np.exp(-0.5 * np.power((D / sdv), 2))
    W = W / W.sum()

    pseudotime = D.loc[start_cell, :].copy()
    converged = False
    iteration = 1
    while not converged and iteration < max_iterations:
        P = D.copy()
        for wp in waypoints[1:]:
            idx_val = pseudotime[wp]
            before_indices = pseudotime.index[pseudotime < idx_val]
            P.loc[wp, before_indices] = -D.loc[wp, before_indices]
            P.loc[wp, :] = P.loc[wp, :] + idx_val
        new_traj = P.multiply(W).sum()
        corr = np.corrcoef(pseudotime, new_traj)[0, 1]
        if corr > 0.9999:
            converged = True
        pseudotime = new_traj
        iteration += 1

    pseudotime -= np.min(pseudotime)
    pseudotime /= np.max(pseudotime)
    return pseudotime, W


def _compute_kernel_reference(data: pd.DataFrame, knn: int, alpha: float = 0) -> csr_matrix:
    N = data.shape[0]
    temp = AnnData(data.values)
    sc.pp.neighbors(temp, n_pcs=0, n_neighbors=knn)
    kNN = temp.obsp["distances"]

    adaptive_k = int(np.floor(knn / 3))
    adaptive_std = np.zeros(N)
    for i in np.arange(N):
        adaptive_std[i] = np.sort(kNN.data[kNN.indptr[i] : kNN.indptr[i + 1]])[
            adaptive_k - 1
        ]

    x, y, dists = find(kNN)
    dists /= adaptive_std[x]
    W = csr_matrix((np.exp(-dists), (x, y)), shape=[N, N])
    kernel = W + W.T

    if alpha > 0:
        D = np.ravel(kernel.sum(axis=1))
        D[D != 0] = D[D != 0] ** (-alpha)
        mat = csr_matrix((D, (range(N), range(N))), shape=[N, N])
        kernel = mat.dot(kernel).dot(mat)
    return kernel


def test_max_min_sampling_equivalence():
    rng = np.random.default_rng(0)
    data = pd.DataFrame(rng.normal(size=(30, 3)))
    waypoints_new = _max_min_sampling(data, num_waypoints=9, seed=0)
    waypoints_ref = _max_min_sampling_reference(data, num_waypoints=9, seed=0)
    assert list(waypoints_new) == list(waypoints_ref)


def test_compute_pseudotime_equivalence():
    rng = np.random.default_rng(1)
    data = pd.DataFrame(rng.normal(size=(25, 4)))
    data.index = pd.Index([f"cell_{i}" for i in range(data.shape[0])])
    start_cell = data.index[0]
    waypoints = pd.Index([start_cell, data.index[5], data.index[10], data.index[15]])

    pt_new, W_new = _compute_pseudotime(
        data, start_cell, knn=5, waypoints=waypoints, n_jobs=1, max_iterations=10
    )
    pt_ref, W_ref = _compute_pseudotime_reference(
        data, start_cell, knn=5, waypoints=waypoints, max_iterations=10
    )

    assert np.allclose(pt_new.values, pt_ref.values, atol=1e-10, rtol=1e-10)
    assert np.allclose(W_new.values, W_ref.values, atol=1e-10, rtol=1e-10)


def test_compute_kernel_scanpy_equivalence():
    rng = np.random.default_rng(2)
    data = pd.DataFrame(rng.normal(size=(20, 5)))
    kernel_new = compute_kernel(data, knn=6, alpha=0, backend="scanpy")
    kernel_ref = _compute_kernel_reference(data, knn=6, alpha=0)
    assert np.allclose(kernel_new.toarray(), kernel_ref.toarray(), atol=1e-10, rtol=1e-10)
