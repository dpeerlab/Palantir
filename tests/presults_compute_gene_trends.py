import pytest
import pandas as pd
import numpy as np
from anndata import AnnData
import palantir


@pytest.fixture
def mock_adata():
    n_cells = 10

    # Create mock data
    adata = AnnData(
        X=np.random.rand(n_cells, 3),
        obs=pd.DataFrame(
            {"palantir_pseudotime": np.random.rand(n_cells)},
            index=[f"cell_{i}" for i in range(n_cells)],
        ),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(3)]),
    )

    adata.obsm["branch_masks"] = pd.DataFrame(
        np.random.randint(2, size=(n_cells, 2)),
        columns=["branch_1", "branch_2"],
        index=adata.obs_names,
    ).astype(bool)

    return adata


@pytest.fixture
def custom_mock_adata():
    n_cells = 10

    # Create mock data
    adata = AnnData(
        X=np.random.rand(n_cells, 3),
        obs=pd.DataFrame(
            {"custom_time": np.random.rand(n_cells)},
            index=[f"cell_{i}" for i in range(n_cells)],
        ),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(3)]),
    )

    adata.obsm["custom_masks"] = pd.DataFrame(
        np.random.randint(2, size=(n_cells, 2)),
        columns=["branch_1", "branch_2"],
        index=adata.obs_names,
    ).astype(bool)

    return adata


@pytest.fixture
def mock_adata_old():
    n_cells = 10

    # Create mock data
    adata = AnnData(
        X=np.random.rand(n_cells, 3),
        obs=pd.DataFrame(
            {"palantir_pseudotime": np.random.rand(n_cells)},
            index=[f"cell_{i}" for i in range(n_cells)],
        ),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(3)]),
    )

    # Create mock branch_masks in obsm
    adata.obsm["branch_masks"] = np.random.randint(2, size=(n_cells, 2)).astype(bool)
    adata.uns["branch_masks_columns"] = ["branch_1", "branch_2"]

    return adata


@pytest.mark.parametrize("adata_fixture", ["mock_adata", "mock_adata_old"])
def test_compute_gene_trends(request, adata_fixture):
    adata = request.getfixturevalue(adata_fixture)

    # Call the function with default keys
    res = palantir.presults.compute_gene_trends(adata)

    # Asserts to check the output
    assert isinstance(res, dict)
    assert "branch_1" in res
    assert "branch_2" in res
    assert isinstance(res["branch_1"], dict)
    assert isinstance(res["branch_1"]["trends"], pd.DataFrame)
    assert "gene_0" in res["branch_1"]["trends"].index
    assert adata.varm["gene_trends_branch_1"].shape == (3, 500)


def test_compute_gene_trends_custom_anndata(custom_mock_adata):
    # Call the function with custom keys
    res = palantir.presults.compute_gene_trends(
        custom_mock_adata,
        masks_key="custom_masks",
        pseudo_time_key="custom_time",
        gene_trend_key="custom_trends",
    )

    # Asserts to check the output with custom keys
    assert isinstance(res, dict)
    assert "branch_1" in res
    assert "branch_2" in res
    assert isinstance(res["branch_1"], dict)
    assert isinstance(res["branch_1"]["trends"], pd.DataFrame)
    assert "gene_0" in res["branch_1"]["trends"].index
    assert custom_mock_adata.varm["custom_trends_branch_1"].shape == (3, 500)
