Palantir
------

Palantir is an algorithm to align cells along differentiation trajectories. Palantir models differentiation as a stochastic process where stem cells differentiate to terminally differentiated cells by a series of steps through a low dimensional phenotypic manifold. Palantir effectively captures the continuity in cell states and the stochasticity in cell fate determination. Palantir has been designed to work with multidimensional single cell data from diverse technologies such as Mass cytometry and single cell RNA-seq. 


#### Installation and dependencies
1. Palantir has been implemented in Python3 and can be installed using:

        $> pip install PhenoGraph
        $> pip install palantir

2. Palantir depends on a number of `python3` packages available on pypi and these dependencies are listed in `setup.py`

    All the dependencies will be automatically installed using the above commands

3. To uninstall:
		
		$> pip uninstall palantir

4. If you would like to determine gene expression trends, please install <a href="https://cran.r-project.org"> R <a> programming language and the R package <a href="https://cran.r-project.org/web/packages/gam/">GAM </a>. You will also need to install the rpy2 module using 
	
		$> pip install .['PLOT_GENE_TRENDS']
		    OR,
		$> pip install rpy2
	
    In case of compiler error during installation of `rpy2`, try to link your compiler in `env`. Example:
    
        $> env CC=/usr/local/Cellar/gcc/xxx/bin/gcc-x pip install .['PLOT_GENE_TRENDS']

    where `x` should be replaced with the version numbers
		
5. Palantir can also be used with [**Scanpy**](https://github.com/theislab/scanpy). It is fully integrated into Scanpy, and can be found under Scanpy's external modules ([link](https://scanpy.readthedocs.io/en/latest/api/scanpy.external.html#external-api))


#### Usage

A tutorial on Palantir usage and results visualization for single cell RNA-seq data can be found in this notebook: http://nbviewer.jupyter.org/github/dpeerlab/Palantir/blob/master/notebooks/Palantir_sample_notebook.ipynb


#### Processed data and metadata
```scanpy anndata``` objects are available for download for the three replicates generated in the manuscript: [Rep1](https://s3.amazonaws.com/dp-lab-data-public/palantir/human_cd34_bm_rep1.h5ad), [Rep2](https://s3.amazonaws.com/dp-lab-data-public/palantir/human_cd34_bm_rep2.h5ad), [Rep3](https://s3.amazonaws.com/dp-lab-data-public/palantir/human_cd34_bm_rep3.h5ad)

Each object has the following elements
* `.X`: Filtered, normalized and log transformed count matrix 
* `.raw`: Filtered raw count matrix
* `.obsm['MAGIC_imputed_data']`: Imputed count matrix using MAGIC
* `.obsm['tsne']`: tSNE maps presented in the manuscript generated using scaled diffusion components as inputs
* `.obs['clusters']`: Clustering of cells
* `.obs['palantir_pseudotime']`: Palantir pseudo-time ordering
* `.obs['palantir_diff_potential']`: Palantir differentation potential 
* `.obsm['palantir_branch_probs']`: Palantir branch probabilities
* `.uns['palantir_branch_probs_cell_types']`: Column names for branch probabilities
* `.uns['ct_colors']`: Cell type colors used in the manuscript
* `.uns['cluster_colors']`: Cluster colors used in the manuscript
* `.varm['mast_diff_res_pval']`: MAST p-values for differentially expression in each cluster compared to others
* `.varm['mast_diff_res_statistic']`: MAST statistic for differentially expression in each cluster compared to others
* `.uns['mast_diff_res_columns']`: Column names for the differential expression results


#### Comparison to trajectory detection algorithms
Notebooks detailing the generation of results comparing Palantir to trajectory detection algorithms are available [here](https://github.com/dpeerlab/Palantir/blob/master/notebooks/comparisons)


#### Convert to Seurat objects
Use the snippet below to convert `anndata` to `Seurat` objects 
```
library("SeuratDisk")
library("Seurat")
library("reticulate")
use_condaenv(<conda env>, required = T) # before, install "anndata" into <conda env>
anndata <- import('anndata')

#link to Anndata files
url_Rep1 <- "https://s3.amazonaws.com/dp-lab-data-public/palantir/human_cd34_bm_rep1.h5ad"
curl::curl_download(url_Rep1, basename(url_Rep1))
url_Rep2 <- "https://s3.amazonaws.com/dp-lab-data-public/palantir/human_cd34_bm_rep2.h5ad"
curl::curl_download(url_Rep2, basename(url_Rep2))
url_Rep3 <- "https://s3.amazonaws.com/dp-lab-data-public/palantir/human_cd34_bm_rep3.h5ad"
curl::curl_download(url_Rep3, basename(url_Rep3))

#H5AD files are compressed using the LZF filter. 
#This filter is Python-specific, and cannot easily be used in R. 
#To use this file with Seurat and SeuratDisk, you'll need to read it in Python and save it out using the gzip compression
#https://github.com/mojaveazure/seurat-disk/issues/7
adata_Rep1 = anndata$read("human_cd34_bm_rep1.h5ad")
adata_Rep2 = anndata$read("human_cd34_bm_rep2.h5ad")
adata_Rep3 = anndata$read("human_cd34_bm_rep3.h5ad")

adata_Rep1$write_h5ad("human_cd34_bm_rep1.gzip.h5ad", compression="gzip")
adata_Rep2$write_h5ad("human_cd34_bm_rep2.gzip.h5ad", compression="gzip")
adata_Rep3$write_h5ad("human_cd34_bm_rep3.gzip.h5ad", compression="gzip")


#convert gzip-compressed h5ad file to Seurat Object
Convert("human_cd34_bm_rep1.gzip.h5ad", dest = "h5seurat", overwrite = TRUE)
Convert("human_cd34_bm_rep2.gzip.h5ad", dest = "h5seurat", overwrite = TRUE)
Convert("human_cd34_bm_rep3.gzip.h5ad", dest = "h5seurat", overwrite = TRUE)

human_cd34_bm_Rep1 <- LoadH5Seurat("human_cd34_bm_rep1.gzip.h5seurat")
human_cd34_bm_Rep2 <- LoadH5Seurat("human_cd34_bm_rep2.gzip.h5seurat")
human_cd34_bm_Rep3 <- LoadH5Seurat("human_cd34_bm_rep3.gzip.h5seurat")
```
Thanks to Anne Ludwig from University Hospital Heidelberg for the tip!


#### Citations
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

### Version 1.0.0

 * A fix to [issue#41](https://github.com/dpeerlab/Palantir/issues/41) 
 * A fix to [issue#42](https://github.com/dpeerlab/Palantir/issues/42)
 * Revamped tutorial with support for Anndata and force directed layouts

### Version 0.2.6

 * A fix to [issue#33](https://github.com/dpeerlab/Palantir/issues/33) and [issue#31](https://github.com/dpeerlab/Palantir/issues/31)
 
### Version 0.2.5

 * A fix related to [issue#28](https://github.com/dpeerlab/Palantir/issues/28). When identifying terminal states, duplicate values were generated instead of unique ones.
