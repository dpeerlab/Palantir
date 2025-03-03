import pytest
import pandas as pd
import scanpy as sc
from anndata import AnnData
import numpy as np
from scipy.sparse import csr_matrix
from anndata import AnnData


@pytest.fixture
def example_dataframe():
    # Create an example dataframe for testing
    return pd.DataFrame(
        [[1, 2, 0, 4], [0, 0, 0, 0], [3, 0, 0, 0]],
        columns=["A", "B", "C", "D"],
        index=["X", "Y", "Z"],
    )


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
    ad = AnnData(X=mock_data)
    ad.obsm["X_pca"] = mock_data
    ad.obsm["DM_EigenVectors_multiscaled"] = mock_data
    return ad


@pytest.fixture
def mock_tsne():
    n_cells = 50
    return pd.DataFrame(
        np.random.rand(n_cells, 2),
        columns=["tSNE1", "tSNE2"],
        index=[f"cell_{i}" for i in range(n_cells)],
    )


@pytest.fixture
def mock_umap_df():
    n_cells = 50
    return pd.DataFrame(
        np.random.rand(n_cells, 2),
        columns=["UMAP1", "UMAP2"],
        index=[f"cell_{i}" for i in range(n_cells)],
    )


@pytest.fixture
def mock_gene_data():
    n_cells = 50
    n_genes = 5
    return pd.DataFrame(
        np.random.rand(n_cells, n_genes),
        columns=[f"gene_{i}" for i in range(n_genes)],
        index=[f"cell_{i}" for i in range(n_cells)],
    )


@pytest.fixture
def mock_dm_res():
    n_cells = 50
    n_components = 10
    return {
        "EigenVectors": pd.DataFrame(
            np.random.rand(n_cells, n_components),
            columns=[f"DC_{i}" for i in range(n_components)],
            index=[f"cell_{i}" for i in range(n_cells)],
        ),
        "EigenValues": np.random.rand(n_components),
    }


@pytest.fixture
def mock_clusters():
    n_cells = 50
    return pd.Series(
        np.random.randint(0, 5, n_cells),
        index=[f"cell_{i}" for i in range(n_cells)],
    )


@pytest.fixture
def mock_gene_trends():
    n_bins = 25
    n_genes = 5
    return pd.DataFrame(
        np.random.rand(n_bins, n_genes),
        columns=[f"gene_{i}" for i in range(n_genes)],
        index=np.linspace(0, 1, n_bins),
    )
