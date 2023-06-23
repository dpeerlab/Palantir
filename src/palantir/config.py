import warnings
import matplotlib
from matplotlib import font_manager

# set plotting defaults
with warnings.catch_warnings():
    # catch warnings that system can't find fonts
    warnings.simplefilter("ignore")
    import seaborn as sns

    sns.set(context="paper", style="ticks", font_scale=1.5, font="Bitstream Vera Sans")
    fm = font_manager.fontManager
    fm.findfont("Raleway")
    fm.findfont("Lato")

matplotlib.rcParams["figure.dpi"] = 100
matplotlib.rcParams["image.cmap"] = "viridis"
matplotlib.rcParams["axes.spines.bottom"] = "on"
matplotlib.rcParams["axes.spines.top"] = "off"
matplotlib.rcParams["axes.spines.left"] = "on"
matplotlib.rcParams["axes.spines.right"] = "off"
matplotlib.rcParams["figure.figsize"] = [4, 4]

SELECTED_COLOR = "#377eb8"
DESELECTED_COLOR = "#CFD5E2"

# This global variable sets the default behaviour for saving pandas.DataFrames
# in AnnData.obsm and AnnData.varm. When set to True, the data is saved as pandas.DataFrame.
SAVE_AS_DF = True