import pytest
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix, issparse
import numpy as np

from palantir.utils import run_diffusion_maps

# Generate mock DataFrame data
def mock_dataframe(rows, cols):
    return pd.DataFrame(np.random.rand(rows, cols))


# Generate mock sc.AnnData object
def mock_anndata(rows, cols, keys):
    ad = sc.AnnData(np.random.rand(rows, cols))
    for key in keys:
        ad.obsm[key] = np.random.rand(rows, cols)
    return ad


def test_run_diffusion_maps_dataframe():
    df = mock_dataframe(50, 30)
    result = run_diffusion_maps(df)

    assert isinstance(result, dict)
    assert set(result.keys()) == {"T", "EigenVectors", "EigenValues", "kernel"}

    assert isinstance(result["kernel"], csr_matrix)
    assert isinstance(result["T"], csr_matrix)
    assert isinstance(result["EigenVectors"], pd.DataFrame)
    assert isinstance(result["EigenValues"], pd.Series)


def test_run_diffusion_maps_anndata():
    keys = ["X_pca"]
    ad = mock_anndata(50, 30, keys)
    result = run_diffusion_maps(ad)

    assert "DM_Kernel" in ad.obsp
    assert "DM_Similarity" in ad.obsp
    assert "DM_EigenVectors" in ad.obsm
    assert "DM_EigenValues" in ad.uns

    assert np.array_equal(ad.obsp["DM_Kernel"].toarray(), result["kernel"].toarray())
    assert np.array_equal(ad.obsp["DM_Similarity"].toarray(), result["T"].toarray())
    assert np.array_equal(ad.obsm["DM_EigenVectors"], result["EigenVectors"].values)
    assert np.array_equal(ad.uns["DM_EigenValues"], result["EigenValues"])


def test_run_diffusion_maps_exceptions():
    # Test with neither pd.DataFrame nor sc.AnnData
    with pytest.raises(ValueError):
        run_diffusion_maps("invalid_type")
