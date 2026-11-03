from __future__ import annotations

import warnings

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix, csgraph, find
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from numpy.random import BitGenerator, SeedSequence, RandomState
from numpy.typing import ArrayLike

import scanpy as sc
from anndata import AnnData

from palantir.core import _compute_pseudotime, _connect_graph, _max_min_sampling
from palantir.utils import compute_kernel


def _max_min_sampling_reference(
    data: pd.DataFrame,
    num_waypoints: int,
    seed: int | ArrayLike[int] | np.random.Generator | BitGenerator | SeedSequence | RandomState | None = None
) -> pd.Index:
    waypoint_set = []
    no_iterations = int(num_waypoints / data.shape[1])
    rng = np.random.default_rng(seed)
    N = data.shape[0]
    for ind in data.columns:
        vec = np.ravel(data[ind])
        iter_set = [rng.integers(N)]
        dists = np.zeros([N, no_iterations])
        dists[:, 0] = abs(vec - data[ind].values[iter_set])
        for k in range(1, no_iterations):
            min_dists = dists[:, 0:k].min(axis=1)
            new_wp = np.where(min_dists == min_dists.max())[0][0]
            iter_set.append(new_wp)
            dists[:, k] = abs(vec - data[ind].values[new_wp])
        waypoint_set = waypoint_set + iter_set
    return data.index[waypoint_set].unique()


def _connect_graph_reference(adj: csr_matrix, data: pd.DataFrame, start_cell: int) -> csr_matrix:
    graph = nx.Graph(adj)
    dists = pd.Series(nx.single_source_dijkstra_path_length(graph, start_cell))
    dists = pd.Series(dists.values, index=data.index[dists.index])
    unreachable_nodes = data.index.difference(dists.index)
    while len(unreachable_nodes) > 0:
        farthest_reachable = np.where(data.index == dists.idxmax())[0][0]
        unreachable_dists = pairwise_distances(
            data.iloc[farthest_reachable, :].values.reshape(1, -1),
            data.loc[unreachable_nodes, :],
        )
        unreachable_dists = pd.Series(np.ravel(unreachable_dists), index=unreachable_nodes)
        add_edge = np.where(data.index == unreachable_dists.idxmin())[0][0]
        adj[farthest_reachable, add_edge] = unreachable_dists.min()
        graph = nx.Graph(adj)
        dists = pd.Series(nx.single_source_dijkstra_path_length(graph, start_cell))
        dists = pd.Series(dists.values, index=data.index[dists.index])
        unreachable_nodes = data.index.difference(dists.index)
    return adj


def _compute_pseudotime_reference(
    data: pd.DataFrame,
    start_cell: str | list[str],
    knn: int,
    waypoints: pd.Index,
    max_iterations: int = 25,
):
    # several start cells: pseudotime starts from the nearest one,
    # and every start row keeps sign +1 and time 0
    start_cells = list(start_cell) if isinstance(start_cell, (list, pd.Index)) else [start_cell]
    nbrs = NearestNeighbors(n_neighbors=knn, metric="euclidean", n_jobs=1).fit(data)
    adj = nbrs.kneighbors_graph(data, mode="distance")
    adj = _connect_graph_reference(adj, data, np.where(data.index == start_cells[0])[0][0])

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

    pseudotime = D.loc[start_cells, :].min()
    converged = False
    iteration = 1
    while not converged and iteration < max_iterations:
        P = D.copy()
        for wp in waypoints.difference(start_cells, sort=False):
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


@pytest.mark.parametrize("n_jobs, cell_chunk", [(1, 10_000), (2, 7)])
def test_compute_pseudotime_equivalence(monkeypatch, n_jobs, cell_chunk):
    monkeypatch.setattr("palantir.core._CELL_CHUNK", cell_chunk)
    rng = np.random.default_rng(1)
    data = pd.DataFrame(rng.normal(size=(25, 4)))
    data.index = pd.Index([f"cell_{i}" for i in range(data.shape[0])])
    start_cell = data.index[0]
    waypoints = pd.Index([start_cell, data.index[5], data.index[10], data.index[15]])

    pt_new, W_new = _compute_pseudotime(
        data, start_cell, knn=5, waypoints=waypoints, n_jobs=n_jobs, max_iterations=10
    )
    pt_ref, W_ref = _compute_pseudotime_reference(
        data, start_cell, knn=5, waypoints=waypoints, max_iterations=10
    )

    assert np.allclose(pt_new.values, pt_ref.values, atol=1e-10, rtol=1e-10)
    assert np.allclose(W_new.values, W_ref.values, atol=1e-10, rtol=1e-10)


def test_compute_pseudotime_chunks_and_jobs_exact(monkeypatch):
    # chunks of cells and parallel shortest paths give results identical to one chunk, serially;
    # 81 cells in chunks of at most 8 would leave a one-cell chunk
    rng = np.random.default_rng(0)
    data = pd.DataFrame(rng.random((81, 1)))
    waypoints = data.index[:32]
    start_cell = waypoints[0]
    pt_ref, W_ref = _compute_pseudotime(data, start_cell, 30, waypoints, n_jobs=1)
    monkeypatch.setattr("palantir.core._CELL_CHUNK", 8)
    pt_new, W_new = _compute_pseudotime(data, start_cell, 30, waypoints, n_jobs=2)
    assert np.array_equal(pt_new.values, pt_ref.values)
    assert np.array_equal(W_new.values, W_ref.values)


def test_connect_graph_equivalence():
    # three separate clusters, so the 5-NN graph is disconnected
    rng = np.random.default_rng(3)
    data = pd.DataFrame(np.vstack([rng.normal(loc=10 * i, size=(15, 3)) for i in range(3)]))
    adj = NearestNeighbors(n_neighbors=5).fit(data).kneighbors_graph(data, mode="distance")
    assert csgraph.connected_components(adj, directed=False)[0] == 3

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adj_new = _connect_graph(adj.copy(), data, 0)
        adj_ref = _connect_graph_reference(adj.copy(), data, 0)
    assert csgraph.connected_components(adj_new, directed=False)[0] == 1
    assert (adj_new != adj_ref).nnz == 0


def test_compute_pseudotime_cell_far_from_waypoints():
    # The last cell is so far from every waypoint that all its Gaussian weights underflow in float64
    rng = np.random.default_rng(3)
    data = pd.DataFrame(rng.normal(size=(200, 2)))
    data.iloc[-1] = 1e3
    data.index = pd.Index([f"cell_{i}" for i in range(data.shape[0])])
    waypoints = data.index[:5]

    pt, W = _compute_pseudotime(data, waypoints[0], knn=5, waypoints=waypoints, n_jobs=1)

    assert np.isfinite(pt.values).all()
    assert np.allclose(W.sum(axis=0), 1)
    assert pt.idxmax() == data.index[-1]


def test_compute_pseudotime_several_start_cells_equivalence():
    # Two start cells close enough that each carries weight on the other's row, so the
    # refinement must hold every start row at sign +1 and time 0 (checked against the reference)
    rng = np.random.default_rng(0)
    x = np.linspace(0, 1, 60)
    data = pd.DataFrame(np.column_stack([x, 0.01 * rng.random(60)]))
    data.index = pd.Index([f"cell_{i}" for i in range(data.shape[0])])
    start_cells = [data.index[20], data.index[30]]
    waypoints = pd.Index(start_cells).append(data.index.difference(start_cells, sort=False)[::3])

    pt_new, W_new = _compute_pseudotime(
        data, start_cells, knn=5, waypoints=waypoints, n_jobs=1, max_iterations=10
    )
    pt_ref, W_ref = _compute_pseudotime_reference(
        data, start_cells, knn=5, waypoints=waypoints, max_iterations=10
    )

    assert np.allclose(pt_new.values, pt_ref.values, atol=1e-10, rtol=1e-10)
    assert np.allclose(W_new.values, W_ref.values, atol=1e-10, rtol=1e-10)


def test_compute_kernel_scanpy_equivalence():
    rng = np.random.default_rng(2)
    data = pd.DataFrame(rng.normal(size=(20, 5)))
    kernel_new = compute_kernel(data, knn=6, alpha=0, backend="scanpy")
    kernel_ref = _compute_kernel_reference(data, knn=6, alpha=0)
    assert np.allclose(kernel_new.toarray(), kernel_ref.toarray(), atol=1e-10, rtol=1e-10)
