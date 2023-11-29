from anndata._core.anndata import AnnData
from pandas.core.frame import DataFrame
import pytest
import pandas as pd
import scanpy as sc
import numpy as np

from palantir.utils import (
    run_density,
    run_low_density_variability,
    run_density_evaluation,
)


@pytest.fixture
def mock_data():
    n_cells = 50
    n_genes = 500
    return pd.DataFrame(
        np.random.rand(n_cells, n_genes),
        columns=[f"gene_{i}" for i in range(n_genes)],
        index=[f"cell_{i}" for i in range(n_cells)],
    )


@pytest.fixture
def mock_anndata(mock_data: DataFrame):
    ad = sc.AnnData(X=mock_data)
    ad.obsm["DM_EigenVectors"] = mock_data.iloc[:, :10].copy()
    ad.obsm["branch_masks"] = pd.DataFrame(
        columns=["branch_0", "branch_1"],
        index=mock_data.index,
        data=np.random.choice([True, False], size=(mock_data.shape[0], 2)),
    )
    ad.obs["other_density"] = np.random.rand(mock_data.shape[0])
    ad.layers["local_variability"] = np.random.rand(*mock_data.shape)
    return ad


@pytest.fixture
def mock_anndata_custom(mock_data: DataFrame):
    ad = sc.AnnData(X=mock_data)
    ad.obsm["DM_EigenVectors_custom"] = mock_data.iloc[:, :10].copy()
    return ad


def test_run_density(mock_anndata: AnnData):
    run_density(mock_anndata)
    assert "mellon_log_density" in mock_anndata.obs.keys()
    assert "mellon_log_density_clipped" in mock_anndata.obs.keys()


def test_run_density_custom_keys(mock_anndata_custom: AnnData):
    run_density(
        mock_anndata_custom, repr_key="DM_EigenVectors_custom", density_key="custom_key"
    )
    assert "custom_key" in mock_anndata_custom.obs.keys()
    assert "custom_key_clipped" in mock_anndata_custom.obs.keys()


def test_run_low_density_variability(mock_anndata: AnnData):
    run_low_density_variability(mock_anndata, density_key="other_density")
    for branch in mock_anndata.obsm["branch_masks"].columns:
        assert f"low_density_gene_variability_{branch}" in mock_anndata.var.keys()


def test_run_density_evaluation(mock_anndata: AnnData, mock_anndata_custom: AnnData):
    run_density(mock_anndata)
    run_density_evaluation(
        mock_anndata, mock_anndata_custom, repr_key="DM_EigenVectors_custom"
    )
    assert "cross_log_density" in mock_anndata_custom.obs.keys()
    assert "cross_log_density_clipped" in mock_anndata_custom.obs.keys()
