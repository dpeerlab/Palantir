"""
Core functions for running Palantir
"""

from typing import Union, Optional, List, Dict, Tuple
import numpy as np
import pandas as pd
import networkx as nx
import time
import copy

from sklearn.metrics import pairwise_distances
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed
from scipy.sparse.linalg import eigs, splu

from scipy.sparse import csr_matrix, find, csgraph, eye
from scipy.sparse.csgraph import connected_components
from scipy.stats import entropy, pearsonr, norm
from numpy.linalg import inv, pinv, LinAlgError
import warnings
from anndata import AnnData

from . import config
from .validation import normalize_cell_identifiers

warnings.filterwarnings(action="ignore", message="scipy.cluster")
warnings.filterwarnings(action="ignore", module="scipy", message="Changing the sparsity")

def _get_joblib_backend():
    """
    Determine the appropriate joblib backend based on configuration and environment.

    Returns
    -------
    str or None
        The backend to use, or None to use joblib's default.
    """
    # Check if user has explicitly set backend in config
    if config.JOBLIB_BACKEND is not None:
        return config.JOBLIB_BACKEND

    # Automatic selection for Python 3.12+ with old joblib
    # This avoids ResourceTracker ChildProcessError warnings
    # See: https://github.com/joblib/joblib/issues/1708
    import sys
    if sys.version_info >= (3, 12):
        try:
            import joblib
            joblib_version = tuple(map(int, joblib.__version__.split('.')[:2]))
            if joblib_version < (1, 5):
                # Use threading backend to avoid ResourceTracker errors
                return "threading"
        except Exception:
            pass  # If we can't determine version, use default

    return None  # Use joblib's default backend


def run_palantir(
    data: Union[pd.DataFrame, AnnData],
    early_cell,
    terminal_states: Optional[Union[List, Dict, pd.Series]] = None,
    knn: int = 30,
    num_waypoints: int = 1200,
    n_jobs: int = -1,
    scale_components: bool = True,
    use_early_cell_as_start: bool = False,
    max_iterations: int = 25,
    eigvec_key: str = "DM_EigenVectors_multiscaled",
    pseudo_time_key: str = "palantir_pseudotime",
    entropy_key: str = "palantir_entropy",
    fate_prob_key: str = "palantir_fate_probabilities",
    save_as_df: bool = None,
    waypoints_key: str = "palantir_waypoints",
    seed: int = 20,
) -> Optional[object]:
    """
    Executes the Palantir algorithm to derive pseudotemporal ordering of cells, their fate probabilities, and
    state entropy based on the multiscale diffusion map results.

    Parameters
    ----------
    data : Union[pd.DataFrame, AnnData]
        Either a DataFrame of multiscale space diffusion components or a Scanpy AnnData object.
    early_cell : str
        Start cell for pseudotime construction.
    terminal_states : List/Series/Dict, optional
        User-defined terminal states structure in the format {terminal_name:cell_name}. Default is None.
    knn : int, optional
        Number of nearest neighbors for graph construction. Default is 30.
    num_waypoints : int, optional
        Number of waypoints to sample. Default is 1200.
    n_jobs : int, optional
        Number of jobs for parallel processing. Default is -1.
    scale_components : bool, optional
        If True, components are scaled. Default is True.
    use_early_cell_as_start : bool, optional
        If True, the early cell is used as start. Default is False.
    max_iterations : int, optional
        Maximum number of iterations for pseudotime convergence. Default is 25.
    eigvec_key : str, optional
        Key to access multiscale space diffusion components from obsm of the AnnData object. Default is 'DM_EigenVectors_multiscaled'.
    pseudo_time_key : str, optional
        Key to store the pseudotime in obs of the AnnData object. Default is 'palantir_pseudotime'.
    entropy_key : str, optional
        Key to store the entropy in obs of the AnnData object. Default is 'palantir_entropy'.
    fate_prob_key : str, optional
        Key to store the fate probabilities in obsm of the AnnData object. Default is 'palantir_fate_probabilities'.
        If save_as_df is True, the fate probabilities are stored as pandas DataFrame with terminal state names as columns.
        If False, the fate probabilities are stored as numpy array and the terminal state names are stored in uns[fate_prob_key + "_columns"].
    save_as_df : bool, optional
        If True, the fate probabilities are saved as pandas DataFrame. If False, the data is saved as numpy array.
        The option to save as DataFrame is there due to some versions of AnnData not being able to
        write h5ad files with DataFrames in ad.obsm. Default is palantir.SAVE_AS_DF = True.
    waypoints_key : str, optional
        Key to store the waypoints in uns of the AnnData object. Default is 'palantir_waypoints'.
    seed : int, optional
        The seed for the random number generator used in waypoint sampling. Default is 20.

    Returns
    -------
    Optional[PResults]
        PResults object with pseudotime, entropy, branch probabilities, and waypoints.
        If an AnnData object is passed as data, the result is written to its obs, obsm, and uns attributes
        using the provided keys and None is returned.
    """
    if save_as_df is None:
        save_as_df = config.SAVE_AS_DF

    # Handle terminal_states with normalization
    if terminal_states is not None:
        if isinstance(data, AnnData):
            obs_names = data.obs_names
        else:
            obs_names = data.index

        if isinstance(terminal_states, dict):
            terminal_states = pd.Series(terminal_states)

        if isinstance(terminal_states, pd.Series):
            # Normalize terminal state cell identifiers
            terminal_dict, n_requested, n_found = normalize_cell_identifiers(
                terminal_states, obs_names, context="run_palantir terminal_states"
            )
            # Reconstruct Series with normalized keys
            terminal_states = pd.Series(terminal_dict)
            terminal_cells = list(terminal_dict.keys())
        else:
            # List or array of cell identifiers
            terminal_dict, n_requested, n_found = normalize_cell_identifiers(
                terminal_states, obs_names, context="run_palantir terminal_states"
            )
            terminal_cells = list(terminal_dict.keys())
    else:
        terminal_cells = None

    if isinstance(data, AnnData):
        ms_data = pd.DataFrame(data.obsm[eigvec_key], index=data.obs_names)
    else:
        ms_data = data

    # Normalize early_cell identifier
    early_cell_dict, _, n_early_found = normalize_cell_identifiers(
        [early_cell], ms_data.index, context="run_palantir early_cell"
    )
    if n_early_found == 0:
        raise ValueError(
            f"Early cell '{early_cell}' not found in data. "
            f"Please provide a valid cell identifier from data.obs_names (type: {type(ms_data.index[0]).__name__})"
        )
    early_cell = list(early_cell_dict.keys())[0]

    if scale_components:
        data_df = pd.DataFrame(
            preprocessing.minmax_scale(ms_data),
            index=ms_data.index,
            columns=ms_data.columns,
        )
    else:
        data_df = copy.copy(ms_data)

    # ################################################
    # Determine the boundary cell closest to user defined early cell
    dm_boundaries = pd.Index(set(data_df.idxmax()).union(data_df.idxmin()))
    dists = pairwise_distances(
        data_df.loc[dm_boundaries, :], data_df.loc[early_cell, :].values.reshape(1, -1)
    )
    start_cell = pd.Series(np.ravel(dists), index=dm_boundaries).idxmin()
    if use_early_cell_as_start:
        start_cell = early_cell

    # Sample waypoints
    print("Sampling and flocking waypoints...")
    start = time.time()

    # Append start cell
    if isinstance(num_waypoints, int):
        waypoints = _max_min_sampling(data_df, num_waypoints, seed)
    else:
        waypoints = num_waypoints
    waypoints = waypoints.union(dm_boundaries)
    if terminal_cells is not None:
        waypoints = waypoints.union(terminal_cells)
    waypoints = pd.Index(waypoints.difference([start_cell]).unique())

    # Append start cell
    waypoints = pd.Index([start_cell]).append(waypoints)
    end = time.time()
    print("Time for determining waypoints: {} minutes".format((end - start) / 60))

    # pseudotime and weighting matrix
    print("Determining pseudotime...")
    pseudotime, W = _compute_pseudotime(data_df, start_cell, knn, waypoints, n_jobs, max_iterations)

    # Entropy and branch probabilities
    print("Entropy and branch probabilities...")
    ent, branch_probs = _differentiation_entropy(
        data_df.loc[waypoints, :], terminal_cells, knn, n_jobs, pseudotime
    )

    # Project results to all cells
    print("Project results to all cells...")
    branch_probs = pd.DataFrame(
        np.dot(W.T, branch_probs.loc[W.index, :]),
        index=W.columns,
        columns=branch_probs.columns,
    )
    ent = branch_probs.apply(entropy, axis=1)

    # Import PResults class only when needed to avoid circular imports
    from .presults import PResults

    pr_res = PResults(pseudotime, ent, branch_probs, waypoints)

    if isinstance(data, AnnData):
        data.obs[pseudo_time_key] = pseudotime
        data.obs[entropy_key] = ent
        data.uns[waypoints_key] = waypoints.values
        if isinstance(terminal_states, pd.Series):
            branch_probs.columns = terminal_states[branch_probs.columns]
        if save_as_df:
            data.obsm[fate_prob_key] = branch_probs
        else:
            data.obsm[fate_prob_key] = branch_probs.values
            data.uns[fate_prob_key + "_columns"] = branch_probs.columns.values

    return pr_res


def _max_min_sampling(
    data: pd.DataFrame, num_waypoints: int, seed: Optional[int] = None
) -> pd.Index:
    """Function for max min sampling of waypoints.

    This function implements the maxmin sampling approach to select waypoints from the data.
    It iteratively selects points that are maximally distant from the already selected points.

    Parameters
    ----------
    data : pd.DataFrame
        Data matrix along which to sample the waypoints, usually diffusion components.
    num_waypoints : int
        Number of waypoints to sample.
    seed : Optional[int], default=None
        Random number generator seed for the initial point selection.

    Returns
    -------
    pd.Index
        Indices of the sampled waypoints.
    """

    waypoint_set = list()
    min_waypoints = max(3, data.shape[1])
    if num_waypoints < min_waypoints:
        warnings.warn(
            f"num_waypoints={num_waypoints} is too small for {data.shape[1]} components; "
            f"using {min_waypoints} instead for stable sampling.",
            UserWarning,
        )
        num_waypoints = min_waypoints
    no_iterations = int((num_waypoints) / data.shape[1])
    if seed is not None:
        np.random.seed(seed)

    N = data.shape[0]
    data_values = data.values

    for i, ind in enumerate(data.columns):
        vec = data_values[:, i]

        current_wp = np.random.randint(N)
        iter_set = [
            current_wp,
        ]

        min_dists = np.abs(vec - vec[current_wp])

        for k in range(1, no_iterations):
            new_wp = np.argmax(min_dists)
            iter_set.append(new_wp)

            new_dists = np.abs(vec - vec[new_wp])
            min_dists = np.minimum(min_dists, new_dists)

        waypoint_set.extend(iter_set)

    waypoints = data.index[waypoint_set].unique()

    return waypoints


def _compute_pseudotime(
    data: pd.DataFrame,
    start_cell: str,
    knn: int,
    waypoints: pd.Index,
    n_jobs: int,
    max_iterations: int = 25,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Compute pseudotime and weight matrix using shortest path distances.

    This function constructs a kNN graph and computes shortest path distances from
    the start cell to all other cells. It then iteratively refines the pseudotime
    by creating a perspective matrix and determining a weighted pseudotime.

    Parameters
    ----------
    data : pd.DataFrame
        Multiscale space diffusion components.
    start_cell : str
        Start cell for pseudotime construction.
    knn : int
        Number of nearest neighbors for graph construction.
    waypoints : pd.Index
        List of waypoints.
    n_jobs : int
        Number of jobs for parallel processing.
    max_iterations : int, default=25
        Maximum number of iterations for pseudotime convergence.

    Returns
    -------
    Tuple[pd.Series, pd.DataFrame]
        pseudotime : pd.Series
            Pseudotime ordering of cells.
        W : pd.DataFrame
            Weight matrix for each cell.
    """

    print("Shortest path distances using {}-nearest neighbor graph...".format(knn))
    start = time.time()
    nbrs = NearestNeighbors(n_neighbors=knn, metric="euclidean", n_jobs=n_jobs).fit(data)
    adj = nbrs.kneighbors_graph(data, mode="distance")

    adj = _connect_graph(adj, data, np.where(data.index == start_cell)[0][0])

    wp_indices = data.index.get_indexer(waypoints)

    dists = csgraph.dijkstra(adj, directed=False, indices=wp_indices)
    D_vals = np.asarray(dists, dtype=float)
    end = time.time()
    print("Time for shortest paths: {} minutes".format((end - start) / 60))

    print("Iteratively refining the pseudotime...")
    sdv = np.std(D_vals.ravel()) * 1.06 * len(D_vals.ravel()) ** (-1 / 5)
    W_vals = np.exp(-0.5 * np.power((D_vals / sdv), 2))
    W_vals = W_vals / W_vals.sum(axis=0, keepdims=True)

    start_row = waypoints.get_loc(start_cell)
    pseudotime = D_vals[start_row, :].copy()
    converged = False

    iteration = 1
    wp_cell_indices = data.index.get_indexer(waypoints)
    while not converged and iteration < max_iterations:
        t_wp = pseudotime[wp_cell_indices][:, None]
        t_cells = pseudotime[None, :]
        mask = t_cells < t_wp
        signs = np.where(mask, -1.0, 1.0)
        t_wp[start_row] = 0.0
        signs[start_row, :] = 1.0
        P_vals = D_vals * signs + t_wp
        new_traj_vals = np.sum(P_vals * W_vals, axis=0)

        corr = pearsonr(pseudotime, new_traj_vals)[0]
        print("Correlation at iteration %d: %.4f" % (iteration, corr))
        if corr > 0.9999:
            converged = True

        pseudotime = new_traj_vals
        iteration += 1

    pseudotime -= np.min(pseudotime)
    pseudotime /= np.max(pseudotime)

    pseudotime_series = pd.Series(pseudotime, index=data.index)
    W = pd.DataFrame(W_vals, index=waypoints, columns=data.index)
    return pseudotime_series, W


def identify_terminal_states(
    ms_data: pd.DataFrame,
    early_cell: str,
    knn: int = 30,
    num_waypoints: int = 1200,
    n_jobs: int = -1,
    max_iterations: int = 25,
    seed: int = 20,
) -> Tuple[np.ndarray, pd.Index]:
    """
    Identify terminal states from multi-scale data.
    
    This function identifies terminal states by sampling waypoints, constructing a 
    pseudotime ordering, building a Markov chain, and analyzing its properties.
    
    Parameters
    ----------
    ms_data : pd.DataFrame
        Multi-scale space diffusion components.
    early_cell : str
        Start cell for pseudotime construction.
    knn : int, optional
        Number of nearest neighbors for graph construction. Default is 30.
    num_waypoints : int, optional
        Number of waypoints to sample. Default is 1200.
    n_jobs : int, optional
        Number of jobs for parallel processing. Default is -1.
    max_iterations : int, optional
        Maximum number of iterations for pseudotime convergence. Default is 25.
    seed : int, optional
        Random seed for waypoint sampling. Default is 20.
        
    Returns
    -------
    Tuple[np.ndarray, pd.Index]
        terminal_states : np.ndarray
            Array of identified terminal state cells.
        excluded_boundaries : pd.Index
            Boundary cells that are not terminal states.
    """
    # Normalize early_cell identifier
    early_cell_dict, _, n_early_found = normalize_cell_identifiers(
        [early_cell], ms_data.index, context="identify_terminal_states early_cell"
    )
    if n_early_found == 0:
        raise ValueError(
            f"Early cell '{early_cell}' not found in data. "
            f"Please provide a valid cell identifier from data.index (type: {type(ms_data.index[0]).__name__})"
        )
    early_cell = list(early_cell_dict.keys())[0]

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
    waypoints = _max_min_sampling(data, num_waypoints, seed)
    waypoints = waypoints.union(dm_boundaries)
    waypoints = pd.Index(waypoints.difference([start_cell]).unique())

    # Append start cell
    waypoints = pd.Index([start_cell]).append(waypoints)

    # Distance to start cell as pseudo pseudotime
    pseudotime, _ = _compute_pseudotime(data, start_cell, knn, waypoints, n_jobs, max_iterations)

    # Markov chain
    wp_data = data.loc[waypoints, :]
    T = _construct_markov_chain(wp_data, knn, pseudotime, n_jobs)

    # Terminal states
    terminal_states = _terminal_states_from_markov_chain(T, wp_data, pseudotime)

    # Excluded diffusion map boundaries
    dm_boundaries = pd.Index(set(wp_data.idxmax()).union(wp_data.idxmin()))
    excluded_boundaries = dm_boundaries.difference(terminal_states).difference([start_cell])
    return terminal_states, excluded_boundaries


def _construct_markov_chain(
    wp_data: pd.DataFrame, knn: int, pseudotime: pd.Series, n_jobs: int
) -> csr_matrix:
    """Constructs a Markov chain from waypoints data.

    This function builds a directed graph based on pseudotime and computes
    a transition matrix representing a Markov chain.

    Parameters
    ----------
    wp_data : pd.DataFrame
        Multi-scale data of the waypoints.
    knn : int
        Number of nearest neighbors for graph construction.
    pseudotime : pd.Series
        Pseudotime ordering of cells.
    n_jobs : int
        Number of jobs for parallel processing.

    Returns
    -------
    csr_matrix
        Transition matrix of the Markov chain.
    """
    # Markov chain construction
    print("Markov chain construction...")
    waypoints = wp_data.index
    N = len(waypoints)

    # kNN graph
    n_neighbors = knn
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean", n_jobs=n_jobs).fit(wp_data)
    dist, ind = nbrs.kneighbors(wp_data)

    # Standard deviation allowing for "back" edges
    adpative_k = np.min([int(np.floor(n_neighbors / 3)) - 1, 30])
    adaptive_std = np.ravel(dist[:, adpative_k])

    # Directed graph construction
    # Pseudotime of waypoints (N,)
    pt = pseudotime[waypoints].values
    
    # Pseudotime of neighbors (N, knn)
    traj = pt[ind]
    
    # Threshold per waypoint (N, 1)
    cut = (pt - adaptive_std)[:, None]
    
    # Edges to keep (where neighbor is not too far back)
    keep = traj >= cut
    
    # Build sparse matrix directly with kept edges
    row_idx, nbr_idx = np.nonzero(keep)
    cols = ind[row_idx, nbr_idx]
    data = dist[row_idx, nbr_idx]
    
    # kNN distance matrix (directed)
    kNN = csr_matrix((data, (row_idx, cols)), shape=(N, N))

    # Affinity matrix and markov chain
    x, y, z = find(kNN)
    aff = np.exp(-(z**2) / (adaptive_std[x] ** 2) * 0.5 - (z**2) / (adaptive_std[y] ** 2) * 0.5)
    W = csr_matrix((aff, (x, y)), [len(waypoints), len(waypoints)])

    # Transition matrix
    D = np.ravel(W.sum(axis=1))
    x, y, z = find(W)
    T = csr_matrix((z / D[x], (x, y)), [len(waypoints), len(waypoints)])

    return T


def _terminal_states_from_markov_chain(
    T: csr_matrix, wp_data: pd.DataFrame, pseudotime: pd.Series
) -> np.ndarray:
    """Identifies terminal states from the Markov chain.

    This function identifies terminal states by examining the eigenvectors of the
    transition matrix and finding connected components of cells that have high ranks.

    Parameters
    ----------
    T : csr_matrix
        Transition matrix of the Markov chain.
    wp_data : pd.DataFrame
        Multi-scale data of the waypoints.
    pseudotime : pd.Series
        Pseudotime ordering of cells.

    Returns
    -------
    np.ndarray
        Array of terminal state identifiers.
    """
    print("Identification of terminal states...")

    # Identify terminal statses
    waypoints = wp_data.index
    dm_boundaries = pd.Index(set(wp_data.idxmax()).union(wp_data.idxmin()))
    n = min(*T.shape)
    if n <= 2:
        vals, vecs = np.linalg.eig(T.T.toarray())
        lead_idx = np.argsort(vals)[-1]
        ranks = np.abs(np.real(vecs[:, lead_idx]))
    else:
        vals, vecs = eigs(T.T, 1, maxiter=n * 50)
        ranks = np.abs(np.real(vecs[:, 0]))
    ranks = pd.Series(ranks, index=waypoints)

    # Cutoff and intersection with the boundary cells
    cutoff = norm.ppf(
        0.9999,
        loc=np.median(ranks),
        scale=np.median(np.abs((ranks - np.median(ranks)))),
    )

    # Connected components of cells beyond cutoff
    cells = ranks.index[ranks > cutoff]
    
    # Map cells to indices in T
    # waypoints are the index of T
    cells_indices = np.where(waypoints.isin(cells))[0]

    # Find connected components on subgraph
    # T is directed, but we want components irrespective of direction (weakly connected)
    # connected_components with directed=False treats the graph as undirected
    T_sub = T[cells_indices, :][:, cells_indices]
    n_comps, labels = connected_components(T_sub, directed=False)
    
    # Identify max pseudotime cell per component
    # Iterate over component labels
    terminal_candidates = []
    for i in range(n_comps):
        # Get indices of cells in this component (relative to T_sub)
        comp_indices = np.where(labels == i)[0]
        # Map back to global waypoints indices
        global_indices = cells_indices[comp_indices]
        # Get corresponding cell names
        comp_cells = waypoints[global_indices]
        # Find cell with max pseudotime in this component
        terminal_candidates.append(pseudotime[comp_cells].idxmax())
    
    cells = terminal_candidates

    # Nearest diffusion map boundaries
    if len(cells) == 0:
        return np.array([])

    dm_vecs = wp_data.loc[dm_boundaries, :].values
    cell_vecs = wp_data.loc[cells, :].values
    dists = pairwise_distances(dm_vecs, cell_vecs)
    nearest = np.argmin(dists, axis=0)
    terminal_states = np.unique(dm_boundaries[nearest])

    # excluded_boundaries = dm_boundaries.difference(terminal_states)
    return terminal_states


def _differentiation_entropy(
    wp_data: pd.DataFrame, 
    terminal_states: Optional[np.ndarray], 
    knn: int, 
    n_jobs: int, 
    pseudotime: pd.Series
) -> Tuple[pd.Series, pd.DataFrame]:
    """Compute entropy and branch probabilities from a Markov chain.

    This function constructs a Markov chain from waypoints data and computes 
    cell branch probabilities and differentiation entropy.

    Parameters
    ----------
    wp_data : pd.DataFrame
        Multi-scale data of the waypoints.
    terminal_states : Optional[np.ndarray]
        Terminal states to use for probability calculations. If None, they will be
        automatically detected.
    knn : int
        Number of nearest neighbors for graph construction.
    n_jobs : int
        Number of jobs for parallel processing.
    pseudotime : pd.Series
        Pseudotime ordering of cells.

    Returns
    -------
    Tuple[pd.Series, pd.DataFrame]
        entropy : pd.Series
            Differentiation entropy for each cell.
        branch_probs : pd.DataFrame
            Branch probabilities for each cell and terminal state.
    """

    T = _construct_markov_chain(wp_data, knn, pseudotime, n_jobs)

    # Identify terminal states if not specified
    if terminal_states is None:
        terminal_states = _terminal_states_from_markov_chain(T, wp_data, pseudotime)
    # Absorption states should not have outgoing edges
    waypoints = wp_data.index
    abs_states = np.where(waypoints.isin(terminal_states))[0]
    if len(abs_states) == 0:
        ent = pd.Series(0, index=waypoints)
        branch_probs = pd.DataFrame(index=waypoints, columns=[])
        return ent, branch_probs
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
    if len(trans_states) == 0:
        ent = pd.Series(0, index=terminal_states)
        bp = pd.DataFrame(0, index=terminal_states, columns=terminal_states)
        bp.values[range(len(terminal_states)), range(len(terminal_states))] = 1
        return ent, bp
    
    # Fundamental matrix solver
    # We want to solve (I - Q) * B = T_trans_abs
    # Instead of inverting (I - Q), we solve the linear system
    I_Q = eye(Q.shape[0], format="csc") - Q.tocsc()
    
    n_abs = len(abs_states)
    if n_abs == 0:
        branch_probs_vals = np.zeros((len(trans_states), 0))
    else:
        try:
            # Use sparse LU solver
            lu = splu(I_Q)
            if n_abs <= 256:
                R = T[trans_states, :][:, abs_states].toarray()
                branch_probs_vals = lu.solve(R)
            else:
                branch_probs_vals = np.empty((len(trans_states), n_abs), dtype=float)
                for start in range(0, n_abs, 256):
                    end = min(start + 256, n_abs)
                    R_block = T[trans_states, :][:, abs_states[start:end]].toarray()
                    branch_probs_vals[:, start:end] = lu.solve(R_block)
        except Exception:
            # Fallback if singular or other issues (though unlikely for I-Q in absorbing chain)
            warnings.warn("Sparse solver failed. Falling back to dense inverse.")
            mat = I_Q.todense()
            try:
                N = inv(mat)
            except LinAlgError:
                N = pinv(mat, hermitian=True)
            R = T[trans_states, :][:, abs_states].toarray()
            branch_probs_vals = np.dot(N, R)

    # Absorption probabilities
    branch_probs = pd.DataFrame(
        branch_probs_vals, index=waypoints[trans_states], columns=waypoints[abs_states]
    )
    branch_probs[branch_probs < 0] = 0

    # Entropy
    # entropy expects (N, k) array and axis
    ent = pd.Series(entropy(branch_probs.values, axis=1), index=branch_probs.index)

    # Add terminal states
    ent = pd.concat([ent, pd.Series(0, index=terminal_states)])
    bp = pd.DataFrame(0, index=terminal_states, columns=terminal_states)
    bp.values[range(len(terminal_states)), range(len(terminal_states))] = 1
    branch_probs = pd.concat([branch_probs, bp.loc[:, branch_probs.columns]])

    return ent, branch_probs


def _shortest_path_helper(cell: int, adj: csr_matrix) -> pd.Series:
    """Compute shortest path distances from a cell to all other cells.

    Parameters
    ----------
    cell : int
        Index of the source cell.
    adj : csr_matrix
        Adjacency matrix representing the graph.

    Returns
    -------
    pd.Series
        Series containing shortest path distances.
    """
    return pd.Series(csgraph.dijkstra(adj, False, cell))


def _connect_graph(adj: csr_matrix, data: pd.DataFrame, start_cell: int) -> csr_matrix:
    """Connect disconnected components in the graph to ensure all cells are reachable.

    This function identifies unreachable nodes in the graph and connects them
    to the nearest reachable node.

    Parameters
    ----------
    adj : csr_matrix
        Adjacency matrix representing the graph.
    data : pd.DataFrame
        Multiscale data matrix.
    start_cell : int
        Index of the start cell.

    Returns
    -------
    csr_matrix
        Updated adjacency matrix with all cells connected.
    """
    # Create graph and compute distances
    graph = nx.Graph(adj)
    dists = pd.Series(nx.single_source_dijkstra_path_length(graph, start_cell))
    dists = pd.Series(dists.values, index=data.index[dists.index])

    # Idenfity unreachable nodes
    unreachable_nodes = data.index.difference(dists.index)
    if len(unreachable_nodes) > 0:
        warnings.warn(
            "Some of the cells were unreachable. Consider increasing the k for \n \
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
        unreachable_dists = pd.Series(np.ravel(unreachable_dists), index=unreachable_nodes)

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
