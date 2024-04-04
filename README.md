[![PyPI version](https://badge.fury.io/py/palantir.svg)](https://badge.fury.io/py/palantir)
[![codecov](https://codecov.io/github/settylab/Palantir/graph/badge.svg?token=KJTEY76FTK)](https://codecov.io/github/settylab/Palantir)

Palantir
------

Palantir is an algorithm to align cells along differentiation trajectories. Palantir models differentiation as a stochastic process where stem cells differentiate to terminally differentiated cells by a series of steps through a low dimensional phenotypic manifold. Palantir effectively captures the continuity in cell states and the stochasticity in cell fate determination. Palantir has been designed to work with multidimensional single cell data from diverse technologies such as Mass cytometry and single cell RNA-seq. 


## Installation and dependencies
Palantir has been implemented in Python3 and can be installed using:

        pip install palantir


## Usage

A tutorial on Palantir usage and results visualization for single cell RNA-seq data can be found in this notebook: http://nbviewer.jupyter.org/github/dpeerlab/Palantir/blob/master/notebooks/Palantir_sample_notebook.ipynb

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
 ### Version 1.3.3
 * optional progress bar with `progress=True` in `palantir.utils.run_local_variability`
 * avoid NaN in local variablility output
 * compatibility with `scanpy>=1.10.0`

 ### Version 1.3.2
 * require `python>=3.8`
 * implement CI for testing
 * fixes for edge cases discoverd through extended testing
 * implement `plot_trajectory` function to show trajectory on the umap
 * scale pseudotime to unit intervall in anndata

 ### Version 1.3.1
 * implemented `palantir.plot.plot_stats` to plot arbitray cell-wise statistics as x-/y-positions.
 * reduce memory usgae of `palantir.presults.compute_gene_trends`
 * removed seaborn dependency
 * refactor `run_diffusion_maps` to split out `compute_kernel` and `diffusion_maps_from_kernel`
 * remove unused dependencies `tables`, `Cython`, `cmake`, and `tzlocal`.
 * fixes in `run_pca` (return correct projections and do not use too many components)

 ### Version 1.3.0

 #### New Features
 * Enable an AnnData-centric workflow for improved usability and interoperability with other single-cell analysis tools.
 * Introduced new utility functions
     * `palantir.utils.early_cell` To automate fining an early cell based on cell type and diffusion components.
     * `palantir.utils.find_terminal_states` To automate finding terminal cell states based on cell type and diffusion components.
     * `palantir.presults.select_branch_cells` To find cells associated to each branch based on fate probability.
     * `palantir.plot.plot_branch_selection` To inspect the cell to branch association.
     * `palantir.utils.run_local_variability` To compute local gene expression variability.
     * `palantir.utils.run_density` A wrapper for [mellon.DensityEstimator](https://mellon.readthedocs.io/en/latest/model.html#mellon.model.DensityEstimator).
     * `palantir.utils.run_density_evaluation` Evaluate computed density on a different dataset.
     * `palantir.utils.run_low_density_variability`. To aggregate local gene expression variability in low density.
     * `palantir.plot.plot_branch`. To plot branch-selected cells over pseudotime in arbitrary y-postion and coloring.
     * `palantir.plot.plot_trend`. To plot the gene trend ontop of `palantir.plot.plot_branch`.
 * Added input validation for better error handling and improved user experience.
 * Expanded documentation within docstrings, providing additional clarity for users and developers.

 #### Enhancements
 * Updated tutorial notebook to reflect the new workflow, guiding users through the updated processes.
 * Implemented gene trend computation using [Mellon](https://github.com/settylab/Mellon), providing more robust and efficient gene trend analysis.
 * Enable annotation in `palantir.plot.highight_cells_on_umap`.

 #### Changes
 * Replaced PhenoGraph dependency with `scanpy.tl.leiden` for gene trend clustering.
 * Deprecated the `run_tsne`, `determine_cell_clusters`, and `plot_cell_clusters` functions. Use corresponding implementations from [Scanpy](https://scanpy.readthedocs.io/en/stable/), widely used single-cell analysis library and direct dependecy of Palantir.
 * Rename `palantir.plot.highight_cells_on_tsne` to `palantir.plot.highight_cells_on_umap`
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
