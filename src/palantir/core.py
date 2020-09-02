"""
Core functions for running Palantir
"""
import numpy as np
import pandas as pd
import networkx as nx
import time
import random
import copy

from sklearn.metrics import pairwise_distances
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed
from scipy.sparse.linalg import eigs

from scipy.sparse import csr_matrix, find, csgraph
from scipy.stats import entropy, pearsonr, norm
from numpy.linalg import inv
from copy import deepcopy
from palantir.presults import PResults

import warnings

warnings.filterwarnings(action="ignore", message="scipy.cluster")
warnings.filterwarnings(
    action="ignore", module="scipy", message="Changing the sparsity"
)


def run_palantir(
    ms_data,
    early_cell,
    terminal_states=None,
    knn=30,
    num_waypoints=1200,
    n_jobs=-1,
    scale_components=True,
    use_early_cell_as_start=False,
    max_iterations: int = 25,
):
    """Function for max min sampling of waypoints

    :param ms_data: Multiscale space diffusion components
    :param early_cell: Start cell for pseudotime construction
    :param terminal_states: List/Series of user defined terminal states
    :param knn: Number of nearest neighbors for graph construction
    :param num_waypoints: Number of waypoints to sample
    :param n_jobs: Number of jobs for parallel processing
    :param scale_components:
    :param use_early_cell_as_start:
    :param max_iterations: Maximum number of iterations for pseudotime convergence
    :return: PResults object with pseudotime, entropy, branch probabilities and waypoints
    """

    if scale_components:
        data = pd.DataFrame(
            preprocessing.minmax_scale(ms_data),
            index=ms_data.index,
            columns=ms_data.columns,
        )
    else:
        data = copy.copy(ms_data)

    # ################################################
    # Determine the boundary cell closest to user defined early cell
    dm_boundaries = pd.Index(set(data.idxmax()).union(data.idxmin()))
    dists = pairwise_distances(
        data.loc[dm_boundaries, :], data.loc[early_cell, :].values.reshape(1, -1)
    )
    start_cell = pd.Series(np.ravel(dists), index=dm_boundaries).idxmin()
    if use_early_cell_as_start:
        start_cell = early_cell

    # Sample waypoints
    print("Sampling and flocking waypoints...")
    start = time.time()

    # Append start cell
    if isinstance(num_waypoints, int):
        waypoints = _max_min_sampling(data, num_waypoints)
    else:
        waypoints = num_waypoints
    waypoints = waypoints.union(dm_boundaries)
    if terminal_states is not None:
        waypoints = waypoints.union(terminal_states)
    waypoints = pd.Index(waypoints.difference([start_cell]).unique())

    # Append start cell
    waypoints = pd.Index([start_cell]).append(waypoints)
    end = time.time()
    print("Time for determining waypoints: {} minutes".format((end - start) / 60))

    # pseudotime and weighting matrix
    print("Determining pseudotime...")
    pseudotime, W = _compute_pseudotime(
        data, start_cell, knn, waypoints, n_jobs, max_iterations
    )

    # Entropy and branch probabilities
    print("Entropy and branch probabilities...")
    ent, branch_probs = _differentiation_entropy(
        data.loc[waypoints, :], terminal_states, knn, n_jobs, pseudotime
    )

    # Project results to all cells
    print("Project results to all cells...")
    branch_probs = pd.DataFrame(
        np.dot(W.T, branch_probs.loc[W.index, :]),
        index=W.columns,
        columns=branch_probs.columns,
    )
    ent = branch_probs.apply(entropy, axis=1)

    # UPdate results into PResults class object
    res = PResults(pseudotime, ent, branch_probs, waypoints)

    return res


def _max_min_sampling(data, num_waypoints):
    """Function for max min sampling of waypoints

    :param data: Data matrix along which to sample the waypoints,
                 usually diffusion components
    :param num_waypoints: Number of waypoints to sample
    :param num_jobs: Number of jobs for parallel processing
    :return: pandas Series reprenting the sampled waypoints
    """

    waypoint_set = list()
    no_iterations = int((num_waypoints) / data.shape[1])

    # Sample along each component
    N = data.shape[0]
    for ind in data.columns:
        # Data vector
        vec = np.ravel(data[ind])

        # Random initialzlation
        iter_set = random.sample(range(N), 1)

        # Distances along the component
        dists = np.zeros([N, no_iterations])
        dists[:, 0] = abs(vec - data[ind].values[iter_set])
        for k in range(1, no_iterations):
            # Minimum distances across the current set
            min_dists = dists[:, 0:k].min(axis=1)

            # Point with the maximum of the minimum distances is the new waypoint
            new_wp = np.where(min_dists == min_dists.max())[0][0]
            iter_set.append(new_wp)

            # Update distances
            dists[:, k] = abs(vec - data[ind].values[new_wp])

        # Update global set
        waypoint_set = waypoint_set + iter_set

    # Unique waypoints
    waypoints = data.index[waypoint_set].unique()

    return waypoints


def _compute_pseudotime(data, start_cell, knn, waypoints, n_jobs, max_iterations=25):
    """Function for compute the pseudotime

    :param data: Multiscale space diffusion components
    :param start_cell: Start cell for pseudotime construction
    :param knn: Number of nearest neighbors for graph construction
    :param waypoints: List of waypoints
    :param n_jobs: Number of jobs for parallel processing
    :param max_iterations: Maximum number of iterations for pseudotime convergence
    :return: pseudotime and weight matrix
    """

    # ################################################
    # Shortest path distances to determine trajectories
    print("Shortest path distances using {}-nearest neighbor graph...".format(knn))
    start = time.time()
    nbrs = NearestNeighbors(n_neighbors=knn, metric="euclidean", n_jobs=n_jobs).fit(
        data
    )
    adj = nbrs.kneighbors_graph(data, mode="distance")

    # Connect graph if it is disconnected
    adj = _connect_graph(adj, data, np.where(data.index == start_cell)[0][0])

    # Distances
    dists = Parallel(n_jobs=n_jobs, max_nbytes=None)(
        delayed(_shortest_path_helper)(np.where(data.index == cell)[0][0], adj)
        for cell in waypoints
    )

    # Convert to distance matrix
    D = pd.DataFrame(0.0, index=waypoints, columns=data.index)
    for i, cell in enumerate(waypoints):
        D.loc[cell, :] = pd.Series(
            np.ravel(dists[i]), index=data.index[dists[i].index]
        )[data.index]
    end = time.time()
    print("Time for shortest paths: {} minutes".format((end - start) / 60))

    # ###############################################
    # Determine the perspective matrix

    print("Iteratively refining the pseudotime...")
    # Waypoint weights
    sdv = np.std(np.ravel(D)) * 1.06 * len(np.ravel(D)) ** (-1 / 5)
    W = np.exp(-0.5 * np.power((D / sdv), 2))
    # Stochastize the matrix
    W = W / W.sum()

    # Initalize pseudotime to start cell distances
    pseudotime = D.loc[start_cell, :]
    converged = False

    # Iteratively update perspective and determine pseudotime
    iteration = 1
    while not converged and iteration < max_iterations:
        # Perspective matrix by alinging to start distances
        P = deepcopy(D)
        for wp in waypoints[1:]:
            # Position of waypoints relative to start
            idx_val = pseudotime[wp]

            # Convert all cells before starting point to the negative
            before_indices = pseudotime.index[pseudotime < idx_val]
            P.loc[wp, before_indices] = -D.loc[wp, before_indices]

            # Align to start
            P.loc[wp, :] = P.loc[wp, :] + idx_val

        # Weighted pseudotime
        new_traj = P.multiply(W).sum()

        # Check for convergence
        corr = pearsonr(pseudotime, new_traj)[0]
        print("Correlation at iteration %d: %.4f" % (iteration, corr))
        if corr > 0.9999:
            converged = True

        # If not converged, continue iteration
        pseudotime = new_traj
        iteration += 1

    return pseudotime, W


def identify_terminal_states(
    ms_data, early_cell, knn=30, num_waypoints=1200, n_jobs=-1, max_iterations=25,
):

    # Scale components
    data = pd.DataFrame(
        preprocessing.minmax_scale(ms_data),
        index=ms_data.index,
        columns=ms_data.columns,
    )

    #  Start cell as the nearest diffusion map boundary
    dm_boundaries = pd.Index(set(data.idxmax()).union(data.idxmin()))
    dists = pairwise_distances(
        data.loc[dm_boundaries, :], data.loc[early_cell, :].values.reshape(1, -1)
    )
    start_cell = pd.Series(np.ravel(dists), index=dm_boundaries).idxmin()

    # Sample waypoints
    # Append start cell
    waypoints = _max_min_sampling(data, num_waypoints)
    waypoints = waypoints.union(dm_boundaries)
    waypoints = pd.Index(waypoints.difference([start_cell]).unique())

    # Append start cell
    waypoints = pd.Index([start_cell]).append(waypoints)

    # Distance to start cell as pseudo pseudotime
    pseudotime, _ = _compute_pseudotime(
        data, start_cell, knn, waypoints, n_jobs, max_iterations
    )

    # Markov chain
    wp_data = data.loc[waypoints, :]
    T = _construct_markov_chain(wp_data, knn, pseudotime, n_jobs)

    # Terminal states
    terminal_states = _terminal_states_from_markov_chain(T, wp_data, pseudotime)

    # Excluded diffusion map boundaries
    dm_boundaries = pd.Index(set(wp_data.idxmax()).union(wp_data.idxmin()))
    excluded_boundaries = dm_boundaries.difference(terminal_states).difference(
        [start_cell]
    )
    return terminal_states, excluded_boundaries


def _construct_markov_chain(wp_data, knn, pseudotime, n_jobs):

    # Markov chain construction
    print("Markov chain construction...")
    waypoints = wp_data.index

    # kNN graph
    n_neighbors = knn
    nbrs = NearestNeighbors(
        n_neighbors=n_neighbors, metric="euclidean", n_jobs=n_jobs
    ).fit(wp_data)
    kNN = nbrs.kneighbors_graph(wp_data, mode="distance")
    dist, ind = nbrs.kneighbors(wp_data)

    # Standard deviation allowing for "back" edges
    adpative_k = np.min([int(np.floor(n_neighbors / 3)) - 1, 30])
    adaptive_std = np.ravel(dist[:, adpative_k])

    # Directed graph construction
    # pseudotime position of all the neighbors
    traj_nbrs = pd.DataFrame(
        pseudotime[np.ravel(waypoints.values[ind])].values.reshape(
            [len(waypoints), n_neighbors]
        ),
        index=waypoints,
    )

    # Remove edges that move backwards in pseudotime except for edges that are within
    # the computed standard deviation
    rem_edges = traj_nbrs.apply(
        lambda x: x < pseudotime[traj_nbrs.index] - adaptive_std
    )
    rem_edges = rem_edges.stack()[rem_edges.stack()]

    # Determine the indices and update adjacency matrix
    cell_mapping = pd.Series(range(len(waypoints)), index=waypoints)
    x = list(cell_mapping[rem_edges.index.get_level_values(0)])
    y = list(rem_edges.index.get_level_values(1))
    # Update adjacecy matrix
    kNN[x, ind[x, y]] = 0

    # Affinity matrix and markov chain
    x, y, z = find(kNN)
    aff = np.exp(
        -(z ** 2) / (adaptive_std[x] ** 2) * 0.5
        - (z ** 2) / (adaptive_std[y] ** 2) * 0.5
    )
    W = csr_matrix((aff, (x, y)), [len(waypoints), len(waypoints)])

    # Transition matrix
    D = np.ravel(W.sum(axis=1))
    x, y, z = find(W)
    T = csr_matrix((z / D[x], (x, y)), [len(waypoints), len(waypoints)])

    return T


def _terminal_states_from_markov_chain(T, wp_data, pseudotime):
    print("Identification of terminal states...")

    # Identify terminal statses
    waypoints = wp_data.index
    dm_boundaries = pd.Index(set(wp_data.idxmax()).union(wp_data.idxmin()))
    vals, vecs = eigs(T.T, 10)

    ranks = np.abs(np.real(vecs[:, np.argsort(vals)[-1]]))
    ranks = pd.Series(ranks, index=waypoints)

    # Cutoff and intersection with the boundary cells
    cutoff = norm.ppf(
        0.9999,
        loc=np.median(ranks),
        scale=np.median(np.abs((ranks - np.median(ranks)))),
    )

    # Connected components of cells beyond cutoff
    cells = ranks.index[ranks > cutoff]

    # Find connected components
    T_dense = pd.DataFrame(T.todense(), index=waypoints, columns=waypoints)
    graph = nx.from_pandas_adjacency(T_dense.loc[cells, cells])
    cells = [pseudotime[i].idxmax() for i in nx.connected_components(graph)]

    # Nearest diffusion map boundaries
    terminal_states = [
        pd.Series(
            np.ravel(
                pairwise_distances(
                    wp_data.loc[dm_boundaries, :],
                    wp_data.loc[i, :].values.reshape(1, -1),
                )
            ),
            index=dm_boundaries,
        ).idxmin()
        for i in cells
    ]

    terminal_states = np.unique(terminal_states)

    # excluded_boundaries = dm_boundaries.difference(terminal_states)
    return terminal_states


def _differentiation_entropy(wp_data, terminal_states, knn, n_jobs, pseudotime):
    """Function to compute entropy and branch probabilities

    :param wp_data: Multi scale data of the waypoints
    :param terminal_states: Terminal states
    :param knn: Number of nearest neighbors for graph construction
    :param n_jobs: Number of jobs for parallel processing
    :param pseudotime: Pseudo time ordering of cells
    :return: entropy and branch probabilities
    """

    T = _construct_markov_chain(wp_data, knn, pseudotime, n_jobs)

    # Identify terminal states if not specified
    if terminal_states is None:
        terminal_states = _terminal_states_from_markov_chain(T, wp_data, pseudotime)
    # Absorption states should not have outgoing edges
    waypoints = wp_data.index
    abs_states = np.where(waypoints.isin(terminal_states))[0]
    # Reset absorption state affinities by Removing neigbors
    T[abs_states, :] = 0
    # Diagnoals as 1s
    T[abs_states, abs_states] = 1

    # Fundamental matrix and absorption probabilities
    print("Computing fundamental matrix and absorption probabilities...")
    # Transition states
    trans_states = list(set(range(len(waypoints))).difference(abs_states))

    # Q matrix
    Q = T[trans_states, :][:, trans_states]
    # Fundamental matrix
    mat = np.eye(Q.shape[0]) - Q.todense()
    N = inv(mat)

    # Absorption probabilities
    branch_probs = np.dot(N, T[trans_states, :][:, abs_states].todense())
    branch_probs = pd.DataFrame(
        branch_probs, index=waypoints[trans_states], columns=waypoints[abs_states]
    )
    branch_probs[branch_probs < 0] = 0

    # Entropy
    ent = branch_probs.apply(entropy, axis=1)

    # Add terminal states
    ent = ent.append(pd.Series(0, index=terminal_states))
    bp = pd.DataFrame(0, index=terminal_states, columns=terminal_states)
    bp.values[range(len(terminal_states)), range(len(terminal_states))] = 1
    branch_probs = branch_probs.append(bp.loc[:, branch_probs.columns])

    return ent, branch_probs


def _shortest_path_helper(cell, adj):
    return pd.Series(csgraph.dijkstra(adj, False, cell))


def _connect_graph(adj, data, start_cell):

    # Create graph and compute distances
    graph = nx.Graph(adj)
    dists = pd.Series(nx.single_source_dijkstra_path_length(graph, start_cell))
    dists = pd.Series(dists.values, index=data.index[dists.index])

    # Idenfity unreachable nodes
    unreachable_nodes = data.index.difference(dists.index)
    if len(unreachable_nodes) > 0:
        print(
            "Warning: Some of the cells were unreachable. Consider increasing the k for \n \
            nearest neighbor graph construction."
        )

    # Connect unreachable nodes
    while len(unreachable_nodes) > 0:
        farthest_reachable = np.where(data.index == dists.idxmax())[0][0]

        # Compute distances to unreachable nodes
        unreachable_dists = pairwise_distances(
            data.iloc[farthest_reachable, :].values.reshape(1, -1),
            data.loc[unreachable_nodes, :],
        )
        unreachable_dists = pd.Series(
            np.ravel(unreachable_dists), index=unreachable_nodes
        )

        # Add edge between farthest reacheable and its nearest unreachable
        add_edge = np.where(data.index == unreachable_dists.idxmin())[0][0]
        adj[farthest_reachable, add_edge] = unreachable_dists.min()

        # Recompute distances to early cell
        graph = nx.Graph(adj)
        dists = pd.Series(nx.single_source_dijkstra_path_length(graph, start_cell))
        dists = pd.Series(dists.values, index=data.index[dists.index])

        # Idenfity unreachable nodes
        unreachable_nodes = data.index.difference(dists.index)

    return adj
