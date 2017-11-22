import pandas as pd
import numpy as np
import bhtsne
import GraphDiffusion
import phenograph

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, find


def run_pca(data, n_components=300):
	"""Run PCA

	:param data: Dataframe of cells X genes. Typicaly multiscale space diffusion components
	:param n_components: Number of principal components
	:return: PCA projections of the data and the explained variance
	"""
	pca = PCA(n_components=n_components, svd_solver='randomized')    
	pca_projections = pca.fit_transform(data)    
	pca_projections = pd.DataFrame(pca_projections, index=data.index)
	return pca_projections, pca.explained_variance_ratio_


def run_diffusion_maps(pca_projections, n_components=10, knn=30, n_jobs=-1):
	"""Run Diffusion maps using the adaptive anisotropic kernel

	:param pca_projections: PCA projections of the data
	:param n_components: Number of diffusion components
	:return: Diffusion components, corresponding eigen values and the diffusion operator
	"""

	# Determine the kernel
	print('Determing nearest neighbor graph...')
	nbrs = NearestNeighbors(n_neighbors=int(knn), metric='euclidean', 
		n_jobs=n_jobs).fit(pca_projections.values)
	kNN = nbrs.kneighbors_graph(pca_projections.values, mode='distance' ) 

	# Adaptive k
	adaptive_k = int(np.floor(knn / 3))
	nbrs = NearestNeighbors(n_neighbors=int(adaptive_k), 
		metric='euclidean', n_jobs=n_jobs).fit(pca_projections.values)
	adaptive_std = nbrs.kneighbors_graph(pca_projections.values, mode='distance' ).max(axis=1)	
	adaptive_std = np.ravel(adaptive_std.todense())

	# Kernel
	N = pca_projections.shape[0]
	x, y, dists = find(kNN)

	# X, y specific stds
	sigmas = (adaptive_std[x] ** 2 + adaptive_std[y] ** 2) / 2
	dists = dists/adaptive_std[x]
	W = csr_matrix( (np.exp(-dists), (x, y)), shape=[N, N] )

	
	# Diffusion components
	kernel = W + W.T
	res = GraphDiffusion.graph_diffusion.run_diffusion_map(kernel, 
	    normalization='markov', n_diffusion_components=n_components)

	# Convert to dataframe
	res['EigenVectors'] = pd.DataFrame(res['EigenVectors'], index=pca_projections.index)
	res['EigenValues'] = pd.Series(res['EigenValues'])

	return res


def run_magic_imputation(data, dm_res, n_steps=3):
	"""Run MAGIC imputation

	:param dm_res: Diffusion map results from run_diffusion_maps
	:param n_steps: Number of steps in the diffusion operator
	:return: Imputed data matrix	
	"""
	T_steps = dm_res['T'] ** n_steps
	imputed_data = pd.DataFrame(np.dot(T_steps.todense(), data), 
                index=data.index, columns=data.columns)

	return imputed_data


def determine_multiscale_space(dm_res, n_eigs = None):
	"""Determine multi scale space of the data

	:param dm_res: Diffusion map results from run_diffusion_maps
	:param n_eigs: Number of eigen vectors to use. If None specified, the number 
		of eigen vectors will be determined using eigen gap
	:return: Multi scale data matrix
	"""
	if n_eigs is None:
		vals = np.ravel(dm_res['EigenValues'])
		n_eigs = np.argsort(vals[:(len(vals)-1)] - vals[1:])[-1] + 1

	# Scale the data
	use_eigs = list(range(1, n_eigs))
	eig_vals = np.ravel(dm_res['EigenValues'][use_eigs])
	data = dm_res['EigenVectors'].values[:, use_eigs] * (eig_vals / (1-eig_vals))
	data = pd.DataFrame(data, index=dm_res['EigenVectors'].index)

	return data


def run_tsne(data, n_dim=2, perplexity=150, **kwargs):
	"""Run tSNE

	:param data: Dataframe of cells X genes. Typicaly multiscale space diffusion components
	:param n_dim: Number of dimensions for tSNE embedding
	:return: tSNE embedding of the data
	"""
	tsne = bhtsne.tsne(data.values.astype(float), 
		dimensions=n_dim, perplexity=perplexity, **kwargs)
	tsne = pd.DataFrame(tsne, index=data.index)
	tsne.columns = ['x', 'y']
	return tsne


def determine_cell_clusters(data, k=50):
	"""Run phenograph for clustering cells

	:param data: Principal components of the data.
	:param k: Number of neighbors for kNN graph construction
	:return: Clusters
	"""
	# Cluster and cluster centrolds
	communities, _, _ = phenograph.cluster(data, k=k)
	communities = pd.Series(communities, index=data.index)	
	return communities
	


