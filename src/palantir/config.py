import matplotlib

matplotlib.rcParams["figure.dpi"] = 100
matplotlib.rcParams["image.cmap"] = "Spectral_r"
matplotlib.rcParams["axes.spines.bottom"] = "on"
matplotlib.rcParams["axes.spines.top"] = "off"
matplotlib.rcParams["axes.spines.left"] = "on"
matplotlib.rcParams["axes.spines.right"] = "off"
matplotlib.rcParams["figure.figsize"] = [4, 4]

SELECTED_COLOR = "#377EB8"
DESELECTED_COLOR = "#CFD5E2"

# This global variable sets the default behaviour for saving pandas.DataFrames
# in AnnData.obsm and AnnData.varm. When set to True, the data is saved as pandas.DataFrame.
SAVE_AS_DF = True

# Cell identifier handling configuration
# When True, automatically convert integer cell identifiers to strings to match obs_names format.
# This is useful for spatial data where cells often have integer indices.
# When False, cell identifiers must exactly match the data's obs_names format.
AUTO_CONVERT_CELL_IDS_TO_STR = True

# When True, emit warnings when cell identifier format mismatches are detected and converted,
# or when requested cells are not found in the data.
WARN_ON_CELL_ID_CONVERSION = True

# Joblib parallel processing backend configuration
# Options: None (automatic selection), 'loky', 'threading', 'multiprocessing'
# When None, automatically selects:
#   - 'threading' for Python 3.12+ with joblib < 1.5 to avoid ResourceTracker errors
#   - Default backend otherwise
# See: https://github.com/joblib/joblib/issues/1708
JOBLIB_BACKEND = None
