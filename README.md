[![PyPI version](https://badge.fury.io/py/palantir.svg)](https://badge.fury.io/py/palantir)
[![codecov](https://codecov.io/github/settylab/Palantir/graph/badge.svg?token=KJTEY76FTK)](https://codecov.io/github/settylab/Palantir)

Palantir
------

Palantir is an algorithm to align cells along differentiation trajectories. Palantir models differentiation as a stochastic process where stem cells differentiate to terminally differentiated cells by a series of steps through a low dimensional phenotypic manifold. Palantir effectively captures the continuity in cell states and the stochasticity in cell fate determination. Palantir has been designed to work with multidimensional single cell data from diverse technologies such as Mass cytometry and single cell RNA-seq.

## Installation
Palantir has been implemented in Python3 and can be installed using:

### Using pip
```sh
pip install palantir
```

### Using conda, mamba, or micromamba from the bioconda channel
You can also install Palantir via conda, mamba, or micromamba from the bioconda channel:

#### Using conda
```sh
conda install -c conda-forge -c bioconda palantir
```

#### Using mamba
```sh
mamba install -c conda-forge -c bioconda palantir
```

#### Using micromamba
```sh
micromamba install -c conda-forge -c bioconda palantir
```

These methods ensure that all dependencies are resolved and installed efficiently.


## Usage

A tutorial on Palantir usage and results visualization for single cell RNA-seq
data can be found in this notebook:
https://github.com/dpeerlab/Palantir/blob/master/notebooks/Palantir_sample_notebook.ipynb

More tutorials and a documentation of all the Palantir components can be found
here: https://palantir.readthedocs.io

## Processed data and metadata

`scanpy anndata` objects are available for download for the three replicates generated in the manuscript:
- [Replicate 1 (Rep1)](https://s3.amazonaws.com/dp-lab-data-public/palantir/human_cd34_bm_rep1.h5ad)
- [Replicate 2 (Rep2)](https://s3.amazonaws.com/dp-lab-data-public/palantir/human_cd34_bm_rep2.h5ad)
- [Replicate 3 (Rep3)](https://s3.amazonaws.com/dp-lab-data-public/palantir/human_cd34_bm_rep3.h5ad)

This notebook details how to use the data in `Python` and `R`: http://nbviewer.jupyter.org/github/dpeerlab/Palantir/blob/master/notebooks/manuscript_data.ipynb

## Comparison to trajectory detection algorithms
Notebooks detailing the generation of results comparing Palantir to trajectory detection algorithms are available [here](https://github.com/dpeerlab/Palantir/blob/master/notebooks/comparisons)

## Citations
Palantir manuscript is available from [Nature Biotechnology](https://www.nature.com/articles/s41587-019-0068-4). If you use Palantir for your work, please cite our paper.

        @article{Palantir_2019,
                title = {Characterization of cell fate probabilities in single-cell data with Palantir},
                author = {Manu Setty and Vaidotas Kiseliovas and Jacob Levine and Adam Gayoso and Linas Mazutis and Dana Pe'er},
                journal = {Nature Biotechnology},
                year = {2019},
                month = {march},
                url = {https://doi.org/10.1038/s41587-019-0068-4},
                doi = {10.1038/s41587-019-0068-4}
        }
____

Release Notes
-------------
 ### Version 1.4.1
  * update `LICENSE` file to be consistent with MIT - license

 ### Version 1.4.0
 * Made pygam an optional dependency that can be installed with `pip install palantir[gam]` or `pip install palantir[full]`
 * Added proper conditional imports and improved error handling for pygam
 * Enhanced `run_magic_imputation` to return appropriate data types for different inputs
 * Updated code to use direct AnnData imports for newer compatibility
 * Improved version detection using `importlib.metadata` with graceful fallbacks
 * Fixed Series indexing deprecation warnings in early cell detection functions
 * Expanded and standardized documentation with NumPy-style docstrings throughout the codebase
 * Added comprehensive type hints to improve code quality and IDE support
 * Remove dependency from `_` methods in scanpy for plotting.
 * add `pseudotime_interval` argument to control path length in `palantir.plot.plot_trajectory`
 
 #### Testing and Quality Improvements
 * Added comprehensive tests for optional pygam dependency
 * Improved test coverage for run_magic_imputation with various input/output types
 * Added integration tests against expected results
 * Enhanced test infrastructure to work with newer library versions
 * Expanded test coverage to catch edge cases in data processing

 ### Version 1.3.6
 * `run_magic_imputation` now has a boolean parameter `sparse` to control output sparsity
 * **bugfix**: `run_local_variability` for dense expression arrays now runs much faster and more accurate

 ### Version 1.3.4
 * avoid devision by zero in `select_branch_cells` for very small datasets
 * make branch selection robust against NaNs
 * do not plot unclustered trends (NaN cluster) in `plot_gene_trend_clusters`

 ### Version 1.3.3
 * optional progress bar with `progress=True` in `palantir.utils.run_local_variability`
 * avoid NaN in local variablility output
 * compatibility with `scanpy>=1.10.0`

 ### Version 1.3.2
 * require `python>=3.9`
 * implement CI for testing
 * fixes for edge cases discovered through extended testing
 * implement `plot_trajectory` function to show trajectory on the umap
 * scale pseudotime to unit interval in anndata

 ### Version 1.3.1
 * implemented `palantir.plot.plot_stats` to plot arbitrary cell-wise statistics as x-/y-positions.
 * reduce memory usage of `palantir.presults.compute_gene_trends`
 * removed seaborn dependency
 * refactor `run_diffusion_maps` to split out `compute_kernel` and `diffusion_maps_from_kernel`
 * remove unused dependencies `tables`, `Cython`, `cmake`, and `tzlocal`.
 * fixes in `run_pca` (return correct projections and do not use too many components)

 ### Version 1.3.0

 #### New Features
 * Enable an AnnData-centric workflow for improved usability and interoperability with other single-cell analysis tools.
 * Introduced new utility functions
     * `palantir.utils.early_cell` To automate finding an early cell based on cell type and diffusion components.
     * `palantir.utils.find_terminal_states` To automate finding terminal cell states based on cell type and diffusion components.
     * `palantir.presults.select_branch_cells` To find cells associated to each branch based on fate probability.
     * `palantir.plot.plot_branch_selection` To inspect the cell to branch association.
     * `palantir.utils.run_local_variability` To compute local gene expression variability.
     * `palantir.utils.run_density` A wrapper for [mellon.DensityEstimator](https://mellon.readthedocs.io/en/latest/model.html#mellon.model.DensityEstimator).
     * `palantir.utils.run_density_evaluation` Evaluate computed density on a different dataset.
     * `palantir.utils.run_low_density_variability`. To aggregate local gene expression variability in low density.
     * `palantir.plot.plot_branch`. To plot branch-selected cells over pseudotime in arbitrary y-position and coloring.
     * `palantir.plot.plot_trend`. To plot the gene trend on top of `palantir.plot.plot_branch`.
 * Added input validation for better error handling and improved user experience.
 * Expanded documentation within docstrings, providing additional clarity for users and developers.

 #### Enhancements
 * Updated tutorial notebook to reflect the new workflow, guiding users through the updated processes.
 * Implemented gene trend computation using [Mellon](https://github.com/settylab/Mellon), providing more robust and efficient gene trend analysis.
 * Enable annotation in `palantir.plot.highlight_cells_on_umap`.

 #### Changes
 * Replaced PhenoGraph dependency with `scanpy.tl.leiden` for gene trend clustering.
 * Deprecated the `run_tsne`, `determine_cell_clusters`, and `plot_cell_clusters` functions. Use corresponding implementations from [Scanpy](https://scanpy.readthedocs.io/en/stable/), widely used single-cell analysis library and direct dependency of Palantir.
 * Rename `palantir.plot.highlight_cells_on_tsne` to `palantir.plot.highlight_cells_on_umap`
 * Depend on `anndata>=0.8.0` to avoid issues writing dataframes in `ad.obsm`.

 #### Fixes
 * Addressed the issue of variability when reproducing results ([issue#64](https://github.com/dpeerlab/Palantir/issues/64)), enhancing the reproducibility and reliability of Palantir.


### Version 1.1.0
 * Replaced rpy2 with pyGAM for computing gene expression trends. 
 * Updated tutorial and plotting functions 


### Version 1.0.0

 * A fix to [issue#41](https://github.com/dpeerlab/Palantir/issues/41) 
 * A fix to [issue#42](https://github.com/dpeerlab/Palantir/issues/42)
 * Revamped tutorial with support for Anndata and force directed layouts

### Version 0.2.6

 * A fix to [issue#33](https://github.com/dpeerlab/Palantir/issues/33) and [issue#31](https://github.com/dpeerlab/Palantir/issues/31)
 
### Version 0.2.5

 * A fix related to [issue#28](https://github.com/dpeerlab/Palantir/issues/28). When identifying terminal states, duplicate values were generated instead of unique ones.
