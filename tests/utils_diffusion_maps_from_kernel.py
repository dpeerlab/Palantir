import pytest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
from pytest import approx

from palantir.utils import diffusion_maps_from_kernel, determine_multiscale_space


@pytest.fixture
def mock_kernel():
    size = 50
    A = np.random.rand(size, size)
    return csr_matrix((A + A.T) / 2)


def test_diffusion_maps_basic(mock_kernel):
    result = diffusion_maps_from_kernel(mock_kernel)

    assert isinstance(result, dict)
    assert "T" in result and "EigenVectors" in result and "EigenValues" in result

    assert result["T"].shape == (50, 50)
    assert result["EigenVectors"].shape == (50, 10)
    assert result["EigenValues"].shape == (10,)


def test_diffusion_maps_n_components(mock_kernel):
    result = diffusion_maps_from_kernel(mock_kernel, n_components=5)

    assert result["EigenVectors"].shape == (50, 5)
    assert result["EigenValues"].shape == (5,)


def test_diffusion_maps_seed(mock_kernel):
    result1 = diffusion_maps_from_kernel(mock_kernel, seed=0)
    result2 = diffusion_maps_from_kernel(mock_kernel, seed=0)

    # Seed usage should yield the same result
    assert np.allclose(result1["EigenValues"], result2["EigenValues"])


def test_diffusion_maps_eigen(mock_kernel):
    result = diffusion_maps_from_kernel(mock_kernel)

    T = result["T"].toarray()
    e_values, e_vectors = eigs(T, 10, tol=1e-4, maxiter=1000)

    assert np.allclose(
        result["EigenValues"], np.real(sorted(e_values, reverse=True)[:10]), atol=1e-4
    )


def test_determine_multiscale_space(mock_kernel):
    result = diffusion_maps_from_kernel(mock_kernel)
    determine_multiscale_space(result)
    assert "EigenValues" in result.keys()
    assert "EigenValues" in result.keys()
    assert "T" in result.keys()
