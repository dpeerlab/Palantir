"""
Utility functions for plotting in Palantir
"""

from typing import Optional, Union, Dict, List, Tuple, Any, Callable
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable


def _scatter_with_colorbar(
    ax: Axes,
    x: np.ndarray,
    y: np.ndarray,
    c: np.ndarray,
    colorbar_label: Optional[str] = None,
    s: float = 5,
    cmap: Union[str, matplotlib.colors.Colormap] = "viridis",
    norm: Optional[Normalize] = None,
    alpha: float = 1.0,
    **kwargs,
) -> Tuple[Axes, matplotlib.colorbar.Colorbar]:
    """Helper function to create scatter plot with colorbar.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object to plot on.
    x : np.ndarray
        X-coordinates for scatter plot.
    y : np.ndarray
        Y-coordinates for scatter plot.
    c : np.ndarray
        Values for color mapping.
    colorbar_label : str, optional
        Label for the colorbar. Default is None.
    s : float, optional
        Size of scatter points. Default is 5.
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap for the scatter plot. Default is 'viridis'.
    norm : Normalize, optional
        Normalization for colormap. Default is None.
    alpha : float, optional
        Transparency of scatter points. Default is 1.0.
    **kwargs : dict
        Additional keyword arguments to pass to plt.scatter.

    Returns
    -------
    Tuple[Axes, matplotlib.colorbar.Colorbar]
        The axes object and the colorbar object.
    """
    sc = ax.scatter(x, y, c=c, s=s, cmap=cmap, norm=norm, alpha=alpha, **kwargs)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(sc, cax=cax, orientation="vertical")
    if colorbar_label:
        cbar.set_label(colorbar_label)
    return ax, cbar


def _highlight_cells(
    ax: Axes,
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    deselected_color: str = "lightgray",
    selected_color: str = "crimson",
    s_selected: float = 10,
    s_deselected: float = 3,
    alpha_deselected: float = 0.5,
    alpha_selected: float = 1.0,
    **kwargs,
) -> Axes:
    """Helper function to highlight cells in scatter plot based on mask.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object to plot on.
    x : np.ndarray
        X-coordinates for scatter plot.
    y : np.ndarray
        Y-coordinates for scatter plot.
    mask : np.ndarray
        Boolean mask for selecting cells to highlight.
    deselected_color : str, optional
        Color for non-highlighted cells. Default is "lightgray".
    selected_color : str, optional
        Color for highlighted cells. Default is "crimson".
    s_selected : float, optional
        Size of highlighted scatter points. Default is 10.
    s_deselected : float, optional
        Size of non-highlighted scatter points. Default is 3.
    alpha_deselected : float, optional
        Transparency of non-highlighted cells. Default is 0.5.
    alpha_selected : float, optional
        Transparency of highlighted cells. Default is 1.0.
    **kwargs : dict
        Additional keyword arguments to pass to plt.scatter.

    Returns
    -------
    Axes
        The modified axes object.
    """
    ax.scatter(
        x[~mask],
        y[~mask],
        c=deselected_color,
        s=s_deselected,
        alpha=alpha_deselected,
        label="Other Cells",
        **kwargs,
    )
    ax.scatter(
        x[mask],
        y[mask],
        c=selected_color,
        s=s_selected,
        alpha=alpha_selected,
        label="Selected Cells",
        **kwargs,
    )
    return ax


def _add_legend(
    ax: Axes,
    handles: Optional[List] = None,
    labels: Optional[List[str]] = None,
    loc: str = "best",
    title: Optional[str] = None,
    **kwargs,
) -> matplotlib.legend.Legend:
    """Helper function to add legend to plot.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object to add legend to.
    handles : List, optional
        List of artists (lines, patches) to be added to the legend. Default is None.
    labels : List[str], optional
        List of labels for the legend. Default is None.
    loc : str, optional
        Location of the legend. Default is "best".
    title : str, optional
        Title for the legend. Default is None.
    **kwargs : dict
        Additional keyword arguments to pass to ax.legend().

    Returns
    -------
    matplotlib.legend.Legend
        The legend object.
    """
    if handles is not None and labels is not None:
        legend = ax.legend(handles, labels, loc=loc, title=title, **kwargs)
    else:
        legend = ax.legend(loc=loc, title=title, **kwargs)
    return legend


def _setup_axes(
    figsize: Tuple[float, float] = (6, 6),
    ax: Optional[Axes] = None,
    fig: Optional[plt.Figure] = None,
    **kwargs,
) -> Tuple[plt.Figure, Axes]:
    """Helper function to set up figure and axes for plotting.

    Parameters
    ----------
    figsize : Tuple[float, float], optional
        Size of the figure (width, height) in inches. Default is (6, 6).
    ax : Axes, optional
        Existing axes to plot on. Default is None.
    fig : Figure, optional
        Existing figure to plot on. Default is None.
    **kwargs : dict
        Additional keyword arguments to pass to plt.subplots().

    Returns
    -------
    Tuple[plt.Figure, Axes]
        The figure and axes objects.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, **kwargs)
    elif fig is None:
        fig = ax.figure
    return fig, ax
