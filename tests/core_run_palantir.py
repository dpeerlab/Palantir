import pytest
import pandas as pd
import scanpy as sc
import numpy as np

from palantir.presults import PResults
from palantir.core import run_palantir


@pytest.fixture
def mock_data():
    n_cells = 50
    n_genes = 10
    return pd.DataFrame(
        np.random.rand(n_cells, n_genes),
        columns=[f"gene_{i}" for i in range(n_genes)],
        index=[f"cell_{i}" for i in range(n_cells)],
    )


@pytest.fixture
def mock_anndata(mock_data):
    ad = sc.AnnData(X=mock_data)
    ad.obsm["DM_EigenVectors_multiscaled"] = mock_data
    return ad


# Test with basic DataFrame input
@pytest.mark.filterwarnings("ignore:np.find_common_type is deprecated.")
@pytest.mark.filterwarnings(
    "ignore:Changing the sparsity structure of a csr_matrix is expensive."
)
def test_palantir_dataframe(mock_data):
    result = run_palantir(mock_data, "cell_0")
    assert isinstance(result, PResults), "Should return a PResults object"


# Test with basic AnnData input
@pytest.mark.filterwarnings("ignore:np.find_common_type is deprecated.")
@pytest.mark.filterwarnings(
    "ignore:Changing the sparsity structure of a csr_matrix is expensive."
)
def test_palantir_anndata(mock_anndata):
    run_palantir(mock_anndata, "cell_0")
    assert (
        "palantir_pseudotime" in mock_anndata.obs.keys()
    ), "Pseudotime key missing in AnnData object"
    assert (
        "palantir_entropy" in mock_anndata.obs.keys()
    ), "Entropy key missing in AnnData object"
    assert (
        "palantir_fate_probabilities" in mock_anndata.obsm.keys()
    ), "Fate probability key missing in AnnData object"
    assert (
        "palantir_waypoints" in mock_anndata.uns.keys()
    ), "Waypoint key missing in AnnData object"


# Test terminal states
@pytest.mark.filterwarnings("ignore:np.find_common_type is deprecated.")
@pytest.mark.filterwarnings(
    "ignore:Changing the sparsity structure of a csr_matrix is expensive."
)
def test_palantir_terminal_states(mock_data):
    result = run_palantir(mock_data, "cell_0", terminal_states=["cell_1", "cell_2"])
    assert "cell_1" in result.branch_probs.columns, "Terminal state cell_1 missing"
    assert "cell_2" in result.branch_probs.columns, "Terminal state cell_2 missing"


# Test scaling components
@pytest.mark.filterwarnings("ignore:np.find_common_type is deprecated.")
@pytest.mark.filterwarnings(
    "ignore:Changing the sparsity structure of a csr_matrix is expensive."
)
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
