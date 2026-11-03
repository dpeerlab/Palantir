import warnings

import pytest
import pandas as pd
import numpy as np

from palantir.presults import PResults
from palantir.core import _compute_pseudotime, identify_terminal_states, run_palantir


# Test with basic DataFrame input
@pytest.mark.filterwarnings("ignore:np.find_common_type is deprecated.")
@pytest.mark.filterwarnings("ignore:Changing the sparsity structure of a csr_matrix is expensive.")
def test_palantir_dataframe(mock_data):
    result = run_palantir(mock_data, "cell_0")
    assert isinstance(result, PResults), "Should return a PResults object"


# Test with basic AnnData input
@pytest.mark.filterwarnings("ignore:np.find_common_type is deprecated.")
@pytest.mark.filterwarnings("ignore:Changing the sparsity structure of a csr_matrix is expensive.")
def test_palantir_anndata(mock_anndata):
    run_palantir(mock_anndata, "cell_0")
    assert (
        "palantir_pseudotime" in mock_anndata.obs.keys()
    ), "Pseudotime key missing in AnnData object"
    assert "palantir_entropy" in mock_anndata.obs.keys(), "Entropy key missing in AnnData object"
    assert (
        "palantir_fate_probabilities" in mock_anndata.obsm.keys()
    ), "Fate probability key missing in AnnData object"
    assert "palantir_waypoints" in mock_anndata.uns.keys(), "Waypoint key missing in AnnData object"


# Test terminal states
@pytest.mark.filterwarnings("ignore:np.find_common_type is deprecated.")
@pytest.mark.filterwarnings("ignore:Changing the sparsity structure of a csr_matrix is expensive.")
def test_palantir_terminal_states(mock_data):
    result = run_palantir(mock_data, "cell_0", terminal_states=["cell_1", "cell_2"])
    assert "cell_1" in result.branch_probs.columns, "Terminal state cell_1 missing"
    assert "cell_2" in result.branch_probs.columns, "Terminal state cell_2 missing"


# Test scaling components
@pytest.mark.filterwarnings("ignore:np.find_common_type is deprecated.")
@pytest.mark.filterwarnings("ignore:Changing the sparsity structure of a csr_matrix is expensive.")
def test_scaling_components(mock_data):
    result1 = run_palantir(mock_data, "cell_0", scale_components=True)
    result2 = run_palantir(mock_data, "cell_0", scale_components=False)
    assert not np.array_equal(
        result1.pseudotime, result2.pseudotime
    ), "Scaling components should affect pseudotime"


# Test for invalid knn
def test_invalid_knn(mock_data):
    with pytest.raises(ValueError):
        run_palantir(mock_data, "cell_0", knn=0)


# Test a list of one early cell
@pytest.mark.filterwarnings("ignore:Changing the sparsity structure of a csr_matrix is expensive.")
def test_palantir_single_early_cell_list(mock_data):
    result1 = run_palantir(mock_data, "cell_0")
    result2 = run_palantir(mock_data, ["cell_0"])
    assert np.array_equal(result1.pseudotime, result2.pseudotime)
    assert result1.waypoints.equals(result2.waypoints)


# Test several early cells
@pytest.mark.filterwarnings("ignore:Changing the sparsity structure of a csr_matrix is expensive.")
def test_palantir_several_early_cells():
    # a line, with a start cell at each end: pseudotime peaks in the middle
    n_cells = 200
    rng = np.random.default_rng(0)
    x = np.linspace(0, 1, n_cells)
    data = pd.DataFrame(
        np.column_stack([x, 0.01 * rng.random(n_cells)]),
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    ends = ["cell_0", f"cell_{n_cells - 1}"]
    result = run_palantir(data, ends, num_waypoints=50, use_early_cell_as_start=True)
    assert result.pseudotime[ends].max() < 0.05
    assert abs(x[result.pseudotime.to_numpy().argmax()] - 0.5) < 0.1


# Test early cells missing from the data
@pytest.mark.filterwarnings("ignore:run_palantir early_cell")
def test_palantir_missing_early_cell(mock_data):
    with pytest.raises(ValueError):
        run_palantir(mock_data, ["cell_0", "not_a_cell"])


# Test start cells in separate components of the kNN graph
@pytest.mark.filterwarnings("ignore:Some of the cells were unreachable")
@pytest.mark.filterwarnings("ignore:Changing the sparsity structure of a csr_matrix is expensive.")
def test_compute_pseudotime_start_cells_disconnected():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.random(30), 10 + rng.random(30)])
    data = pd.DataFrame(
        np.column_stack([x, 0.01 * rng.random(60)]), index=[f"cell_{i}" for i in range(60)]
    )
    starts = ["cell_0", "cell_30"]
    waypoints = pd.Index(starts).append(data.index[1:30:3]).append(data.index[31::3])
    pseudotime, _ = _compute_pseudotime(data, starts, 5, waypoints, 1)
    assert np.isfinite(pseudotime).all()


# Test that the not-found error names only the missing early cells
@pytest.mark.filterwarnings("ignore:run_palantir early_cell")
def test_palantir_missing_early_cell_message(mock_data):
    with pytest.raises(ValueError) as excinfo:
        run_palantir(mock_data, ["cell_0", "not_a_cell"])
    assert "not_a_cell" in str(excinfo.value)
    assert "cell_0" not in str(excinfo.value)


# Test the warning when snapping to the boundaries merges early cells
@pytest.mark.filterwarnings("ignore:Changing the sparsity structure of a csr_matrix is expensive.")
def test_palantir_early_cells_merged_by_snapping():
    n_cells = 200
    rng = np.random.default_rng(0)
    x = np.linspace(0, 1, n_cells)
    data = pd.DataFrame(
        np.column_stack([x, 0.01 * rng.random(n_cells)]),
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    # two early cells near the same end both snap to its boundary cell
    starts = ["cell_1", "cell_2"]
    with pytest.warns(UserWarning, match="leaves only 1 start cell"):
        run_palantir(data, starts, num_waypoints=50, terminal_states=[f"cell_{n_cells - 1}"])
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        warnings.filterwarnings("ignore", message="Changing the sparsity")
        result = run_palantir(
            data, starts, num_waypoints=50, terminal_states=[f"cell_{n_cells - 1}"],
            use_early_cell_as_start=True,
        )
    assert list(result.waypoints[:2]) == starts


# Test the warning when automatic detection finds no terminal state
@pytest.mark.filterwarnings("ignore:Changing the sparsity structure of a csr_matrix is expensive.")
def test_palantir_no_terminal_state_warns():
    # a line with a start cell at each end: the latest cells sit in the middle, not at a boundary
    n_cells = 200
    rng = np.random.default_rng(0)
    x = np.linspace(0, 1, n_cells)
    data = pd.DataFrame(
        np.column_stack([x, 0.01 * rng.random(n_cells)]),
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    ends = ["cell_0", f"cell_{n_cells - 1}"]
    with pytest.warns(UserWarning, match="No terminal states were detected"):
        result = run_palantir(data, ends, num_waypoints=50, use_early_cell_as_start=True)
    assert result.branch_probs.shape[1] == 0


# Test identify_terminal_states with a list of one early cell and with several
@pytest.mark.filterwarnings("ignore:Changing the sparsity structure of a csr_matrix is expensive.")
def test_identify_terminal_states_early_cell_list():
    # two arms that merge into one stem: the arm tips are the start cells, the stem end is terminal
    rng = np.random.default_rng(0)
    t = np.linspace(0, 1, 150)
    arms = [
        np.column_stack([t, 1 - t]),
        np.column_stack([t, t - 1]),
        np.column_stack([1 + t, 0 * t]),
    ]
    data = pd.DataFrame(
        np.vstack(arms) + 0.01 * rng.normal(size=(450, 2)), index=[f"cell_{i}" for i in range(450)]
    )
    terms1, excl1 = identify_terminal_states(data, "cell_0", num_waypoints=60, n_jobs=1)
    terms2, excl2 = identify_terminal_states(data, ["cell_0"], num_waypoints=60, n_jobs=1)
    assert list(terms1) == list(terms2)
    assert list(excl1) == list(excl2)

    # the two arm tips, both diffusion-component boundaries, so both must leave excluded_boundaries
    starts = [data[1].idxmax(), data[1].idxmin()]
    terms, excl = identify_terminal_states(
        data, starts, num_waypoints=60, n_jobs=1, use_early_cell_as_start=True
    )
    assert not set(starts) & set(terms)
    assert not set(starts) & set(excl)
