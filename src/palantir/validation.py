from typing import Union, List, Dict, Tuple
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
import warnings

from . import config


def _validate_obsm_key(ad, key, as_df=True):
    """
    Validates and retrieves the data associated with a specified key from the provided AnnData object.

    Parameters
    ----------
    ad : AnnData
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
    ad : AnnData
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
    data_names : np.ndarray
        A an array of pseudotimes.

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
        data_names = np.array(ad.uns[key + "_pseudotime"])
        if as_df:
            data = pd.DataFrame(data, columns=data_names, index=ad.var_names)
    else:
        data_names = np.array(data.columns.astype(float))
        if not as_df:
            data = data.values
    return data, data_names


def _validate_gene_trend_input(
    data: Union[AnnData, Dict],
    gene_trend_key: str = "gene_trends",
    branch_names: Union[str, List[str]] = "branch_masks",
) -> Dict:
    """
    Validates the input for gene trend plots, and converts it into a dictionary of gene trends.

    Parameters
    ----------
    data : Union[AnnData, Dict]
        An AnnData object or a dictionary containing gene trends.
    gene_trend_key : str, optional
        Key to access gene trends in the varm of the AnnData object. Default is 'gene_trends'.
    branch_names : Union[str, List[str]], optional
        Key to retrieve branch names from the AnnData object or a list of branch names. If a string is provided,
        it is assumed to be a key in AnnData.uns. Default is 'branch_masks'.

    Returns
    -------
    gene_trends : Dict
        A dictionary containing gene trends.

    Raises
    ------
    KeyError
        If 'branch_names' is a string that is not found in .uns, or if 'gene_trend_key + "_" + branch_name'
        is not found in .varm.
    ValueError
        If 'data' is neither an AnnData object nor a dictionary.
    """
    if isinstance(data, AnnData):
        if isinstance(branch_names, str):
            if branch_names in data.uns.keys():
                branch_names = data.uns[branch_names]
            elif branch_names in data.obsm.keys() and isinstance(
                data.obsm[branch_names], pd.DataFrame
            ):
                branch_names = list(data.obsm[branch_names].columns)
            elif branch_names + "_columns" in data.uns.keys():
                branch_names = data.uns[branch_names + "_columns"]
            else:
                raise KeyError(
                    f"The provided key '{branch_names}' is not found in AnnData.uns or as a DataFrame in AnnData.obsm. "
                    "Please ensure the 'branch_names' either exists in AnnData.uns or is a list of branch names."
                )

        gene_trends = dict()
        for branch in branch_names:
            trends, pt_grid = _validate_varm_key(data, gene_trend_key + "_" + branch)
            gene_trends[branch] = {"trends": trends}
    elif isinstance(data, Dict):
        gene_trends = data
    else:
        raise ValueError(
            "The input 'data' must be an instance of either AnnData object or dictionary."
        )

    return gene_trends


def normalize_cell_identifiers(
    cells: Union[List, Dict, pd.Series, pd.Index, np.ndarray],
    obs_names: pd.Index,
    context: str = "cell selection",
) -> Tuple[Dict[str, str], int, int]:
    """
    Normalize cell identifiers to match the format of obs_names.

    This function handles type mismatches between integer and string cell identifiers,
    which commonly occur in spatial data where cells have integer indices but AnnData
    may store obs_names as strings.

    Parameters
    ----------
    cells : Union[List, Dict, pd.Series, pd.Index, np.ndarray]
        Cell identifiers to normalize. Can be:
        - List: List of cell identifiers
        - Dict: {cell_id: annotation} mapping
        - pd.Series: Cell identifiers as index, annotations as values
        - pd.Index: Cell identifiers
        - np.ndarray: Cell identifiers (1D array)
    obs_names : pd.Index
        The cell identifiers from the data (typically data.obs_names)
    context : str, optional
        Description of where this function is called from, for better error messages.
        Default is "cell selection".

    Returns
    -------
    normalized_cells : Dict[str, str]
        Dictionary mapping normalized cell identifiers to annotations (empty string if no annotation)
    n_requested : int
        Number of cells requested
    n_found : int
        Number of cells found in obs_names after normalization

    Raises
    ------
    ValueError
        If cells format is invalid or if no cells are found after normalization
    """
    # Convert input to dict format
    if isinstance(cells, dict):
        cell_dict = cells.copy()
    elif isinstance(cells, pd.Series):
        cell_dict = dict(cells)
    elif isinstance(cells, (list, pd.Index, np.ndarray)):
        cell_dict = {cell: "" for cell in cells}
    else:
        raise ValueError(
            f"Invalid cells format for {context}. Expected list, dict, pd.Series, "
            f"pd.Index, or np.ndarray, got {type(cells).__name__}"
        )

    if len(cell_dict) == 0:
        raise ValueError(f"No cells provided for {context}")

    n_requested = len(cell_dict)

    # Check if any cells match as-is
    n_matched_original = sum(1 for cell in cell_dict.keys() if cell in obs_names)

    # Detect type mismatch
    first_cell = next(iter(cell_dict.keys()))
    first_obs_name = obs_names[0]

    cell_type = type(first_cell).__name__
    obs_type = type(first_obs_name).__name__

    # Check if there's a type mismatch (int vs str)
    has_type_mismatch = False
    if isinstance(first_cell, (int, np.integer)) and isinstance(first_obs_name, str):
        has_type_mismatch = True
        mismatch_desc = f"Cell identifiers are integers but data.obs_names contains strings"
    elif isinstance(first_cell, str) and isinstance(first_obs_name, (int, np.integer)):
        has_type_mismatch = True
        mismatch_desc = f"Cell identifiers are strings but data.obs_names contains integers"

    # Try conversion if there's a mismatch and auto-conversion is enabled
    converted = False
    if has_type_mismatch and config.AUTO_CONVERT_CELL_IDS_TO_STR:
        try:
            # Convert both to strings for consistent matching
            if isinstance(first_cell, (int, np.integer)):
                cell_dict = {str(k): v for k, v in cell_dict.items()}
                converted = True
                conversion_desc = "Converting cell identifiers from int to str"
            elif isinstance(first_obs_name, (int, np.integer)):
                obs_names_str = obs_names.astype(str)
                # Check matches with converted obs_names
                n_matched_converted = sum(1 for cell in cell_dict.keys() if cell in obs_names_str)
                if n_matched_converted > n_matched_original:
                    # Use string version for matching
                    obs_names = obs_names_str
                    converted = True
                    conversion_desc = "Treating data.obs_names as strings for matching"
        except (ValueError, TypeError):
            pass

    # Count matches after potential conversion
    n_found = sum(1 for cell in cell_dict.keys() if cell in obs_names)

    # Generate warnings if configured
    if config.WARN_ON_CELL_ID_CONVERSION:
        if has_type_mismatch:
            if converted and n_found > 0:
                warnings.warn(
                    f"{context}: {mismatch_desc}. "
                    f"{conversion_desc} (config.AUTO_CONVERT_CELL_IDS_TO_STR=True). "
                    f"Found {n_found}/{n_requested} cells after conversion. "
                    f"To disable this warning, set config.WARN_ON_CELL_ID_CONVERSION=False. "
                    f"To fix this issue, ensure your cell identifiers match the type of data.obs_names: "
                    f"use pd.Series(values, index=data.obs_names[your_indices]) or ensure consistent types.",
                    UserWarning,
                    stacklevel=3
                )
            elif not converted:
                warnings.warn(
                    f"{context}: {mismatch_desc}. "
                    f"Automatic conversion is disabled (config.AUTO_CONVERT_CELL_IDS_TO_STR=False). "
                    f"Only {n_found}/{n_requested} cells matched. "
                    f"To enable automatic conversion, set config.AUTO_CONVERT_CELL_IDS_TO_STR=True. "
                    f"To fix this issue manually, ensure your cell identifiers match the type of data.obs_names.",
                    UserWarning,
                    stacklevel=3
                )

        # Warn about cells not found
        if n_found == 0:
            warnings.warn(
                f"{context}: None of the {n_requested} requested cells were found in data.obs_names. "
                f"Your cells have type {cell_type}, data.obs_names has type {obs_type}. "
                f"When using pd.Series, the index should contain actual cell identifiers from data.obs_names, "
                f"not positional indices. Example: pd.Series(['label1', 'label2'], index=['cell_A', 'cell_B'])",
                UserWarning,
                stacklevel=3
            )
        elif n_found < n_requested:
            n_missing = n_requested - n_found
            warnings.warn(
                f"{context}: Only {n_found}/{n_requested} requested cells were found in data.obs_names "
                f"({n_missing} cells missing). Check that cell identifiers match data.obs_names exactly.",
                UserWarning,
                stacklevel=3
            )

    return cell_dict, n_requested, n_found
