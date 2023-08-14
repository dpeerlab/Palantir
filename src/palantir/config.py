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
