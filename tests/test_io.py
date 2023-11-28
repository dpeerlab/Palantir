import pytest
import h5py
import pandas as pd
import numpy as np
import os.path
import fcsparser
import scanpy as sc
from scipy.io import mmread, mmwrite
from scipy.sparse import csr_matrix, csc_matrix
from palantir.io import (
    _clean_up,
    from_csv,
    from_mtx,
    from_10x,
    from_10x_HDF5,
    from_fcs,
)


@pytest.fixture
def example_dataframe():
    # Create an example dataframe for testing
    return pd.DataFrame(
        [[1, 2, 0, 4], [0, 0, 0, 0], [3, 0, 0, 0]],
        columns=["A", "B", "C", "D"],
        index=["X", "Y", "Z"],
    )


@pytest.fixture
def mock_10x_h5(tmp_path):
    # Number of genes and cells
    n_genes = 400
    n_cells = 300
    
    # Simulate a sparse gene expression matrix
    data = np.random.poisson(lam=0.3, size=(n_genes, n_cells))
    sparse_matrix = csc_matrix(data)
    
    # Create barcodes, gene names, etc.
    barcodes = np.array([f"Cell_{i:05d}-1" for i in range(n_cells)])
    gene_names = np.array([f"Gene_{i}" for i in range(n_genes)])
    feature_type = np.array(["Gene Expression" for i in range(n_genes)])
    features = np.array(["gene", ])
    genome = np.array([f"genome_{i%4}" for i in range(n_genes)])
    
    # Creating an HDF5 file
    hdf5_file = tmp_path / "mock_10x_v3_data.h5"
    with h5py.File(hdf5_file, "w") as f:
        f.create_group("matrix")
        f["matrix"].create_dataset("shape", data=np.array(sparse_matrix.shape))
        f["matrix"].create_dataset("data", data=sparse_matrix.data)
        f["matrix"].create_dataset("indices", data=sparse_matrix.indices)
        f["matrix"].create_dataset("indptr", data=sparse_matrix.indptr)
        f["matrix"].create_dataset("barcodes", data=barcodes.astype("S"))
        f["matrix"].create_dataset("name", data=gene_names.astype("S"))
        f["matrix"].create_dataset("id", data=gene_names.astype("S"))
        f["matrix"].create_dataset("feature_type", data=feature_type.astype("S"))
        f["matrix"].create_dataset("genome", data=genome.astype("S"))
    
        f["matrix"].create_group("features")
        f["matrix/features"].create_dataset("name", data=gene_names.astype("S"))
        f["matrix/features"].create_dataset("id", data=gene_names.astype("S"))
        f["matrix/features"].create_dataset("feature_type", data=feature_type.astype("S"))
        f["matrix/features"].create_dataset("genome", data=genome.astype("S"))
    
    return str(hdf5_file)


def test_clean_up(example_dataframe):
    # Test for the _clean_up function
    cleaned_df = _clean_up(example_dataframe)
    assert len(cleaned_df) == 2
    assert len(cleaned_df.columns) == 3


def test_from_csv(tmp_path, example_dataframe):
    # Test for the from_csv function
    csv_file = tmp_path / "test.csv"
    example_dataframe.to_csv(csv_file)

    clean_df = from_csv(csv_file)
    assert len(clean_df) == 2
    assert len(clean_df.columns) == 3


def test_from_mtx(tmp_path):
    # Test for the from_mtx function
    mtx_file = tmp_path / "test.mtx"
    gene_name_file = tmp_path / "gene_names.txt"

    # Create a mock mtx file
    mtx_data = [
        "%%MatrixMarket matrix coordinate integer general",
        "3 4 6",
        "1 1 1",
        "1 2 2",
        "2 4 3",
        "3 1 3",
        "3 2 4",
        "3 3 5",
    ]
    with open(mtx_file, "w") as f:
        f.write("\n".join(mtx_data))

    # Create gene names file
    gene_names = ["Gene1", "Gene2", "Gene3", "Gene4"]
    np.savetxt(gene_name_file, gene_names, fmt="%s")

    clean_df = from_mtx(mtx_file, gene_name_file)
    assert len(clean_df) == 3
    assert len(clean_df.columns) == 4


def test_from_10x(tmp_path):
    # Test for the from_10x function
    data_dir = tmp_path / "data"
    os.makedirs(data_dir, exist_ok=True)

    matrix_file = data_dir / "matrix.mtx"
    gene_file = data_dir / "genes.tsv"
    barcode_file = data_dir / "barcodes.tsv"

    mmwrite(str(matrix_file), csr_matrix([[1, 2], [3, 4]]))
    np.savetxt(str(gene_file), ["Gene1", "Gene2"], fmt="%s")
    np.savetxt(str(barcode_file), ["Cell1", "Cell2"], fmt="%s")

    clean_df = from_10x(str(data_dir))
    print(clean_df)
    assert len(clean_df) == 2
    assert len(clean_df.columns) == 2


def test_from_10x_HDF5(mock_10x_h5):
    clean_df = from_10x_HDF5(mock_10x_h5)
    assert len(clean_df) == 300
    assert len(clean_df.columns) == 400


def test_from_fcs():
    df = from_fcs(None, fcsparser.test_sample_path)
    assert len(df) == 14945
    assert len(df.columns) == 10
