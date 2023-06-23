from typing import Union, List, Dict
import pandas as pd
import scanpy as sc


def _validate_obsm_key(ad, key, as_df=True):
    """
    Validates and retrieves the data associated with a specified key from the provided AnnData object.

    Parameters
    ----------
    ad : sc.AnnData
        The annotated data matrix from which the data is to be retrieved.
    key : str
        The key for accessing the data from the AnnData object's obsm.
    as_df : bool, optional
        If True, the data will be returned as pandas DataFrame with pseudotime as column names.
        If False, the data will be returned as numpy array.
        Default is True.

    Returns
    -------
    data : pd.DataFrame
        A DataFrame containing the data associated with the specified key.
    data_names : List[str]
        A list of column names for the DataFrame.

    Raises
    ------
    KeyError
        If the key or its corresponding columns are not found in the AnnData object.
    """
    if key not in ad.obsm:
        raise KeyError(f"{key} not found in ad.obsm")
    data = ad.obsm[key]
    if not isinstance(data, pd.DataFrame):
        if key + "_columns" not in ad.uns:
            raise KeyError(
                f"{key}_columns not found in ad.uns and ad.obsm[key] is not a DataFrame."
            )
        data_names = list(ad.uns[key + "_columns"])
        if as_df:
            data = pd.DataFrame(data, columns=data_names, index=ad.obs_names)
    else:
        data_names = list(data.columns)
        if not as_df:
            data = data.values
    return data, data_names


def _validate_varm_key(ad, key, as_df=True):
    """
    Validates and retrieves the data associated with a specified key from the provided AnnData object's varm attribute.

    Parameters
    ----------
    ad : sc.AnnData
        The annotated data matrix from which the data is to be retrieved.
    key : str
        The key for accessing the data from the AnnData object's varm.
    as_df : bool, optional
        If True, the trends will be returned as pandas DataFrame with pseudotime as column names.
        If False, the trends will be returned as numpy array.
        Default is True.

    Returns
    -------
    data : Union[pd.DataFrame, np.ndarray]
        A DataFrame or numpy array containing the data associated with the specified key.
    data_names : List[str]
        A list of column names for the DataFrame.

    Raises
    ------
    KeyError
        If the key or its corresponding columns are not found in the AnnData object.
    """
    if key not in ad.varm:
        raise KeyError(f"{key} not found in ad.varm")
    data = ad.varm[key]
    if not isinstance(data, pd.DataFrame):
        if key + "_pseudotime" not in ad.uns:
            raise KeyError(
                f"{key}_pseudotime not found in ad.uns and ad.varm[key] is not a DataFrame."
            )
        data_names = list(ad.uns[key + "_pseudotime"])
        if as_df:
            data = pd.DataFrame(data, columns=data_names, index=ad.var_names)
    else:
        data_names = list(data.columns)
        if not as_df:
            data = data.values
    return data, data_names


def _validate_gene_trend_input(
    data: Union[sc.AnnData, Dict],
    gene_trend_key: str = "gene_trends",
    branch_names: Union[str, List] = "branch_masks_columns",
) -> Dict:
    """
    Validates the input for gene trend plots, and converts it into a dictionary of gene trends.

    Parameters
    ----------
    data : Union[sc.AnnData, Dict]
        AnnData object or dictionary of gene trends.
    gene_trend_key : str, optional
        Key to access gene trends in the AnnData object's varm. Default is 'gene_trends'.
    branch_names : Union[str, List], optional
        Key to access branch names from AnnData object or list of branch names. If a string is provided,
        it is assumed to be a key in AnnData.uns. Default is 'branch_masks_columns'.

    Returns
    -------
    gene_trends : Dict
        Dictionary of gene trends.
    """
    if isinstance(data, sc.AnnData):
        if isinstance(branch_names, str):
            if branch_names not in data.uns.keys():
                raise KeyError(
                    f"'{branch_names}' not found in .uns. "
                    "'branch_names' must either be in .uns or a list of branch names."
                )
            branch_names = data.uns[branch_names]

        gene_trends = dict()
        for branch in branch_names:
            gene_trends[branch], pt_grid = _validate_varm_key(
                data, gene_trend_key + "_" + branch
            )
    elif isinstance(data, Dict):
        gene_trends = data
    else:
        raise ValueError("Input should be an AnnData object or a dictionary.")

    return gene_trends
