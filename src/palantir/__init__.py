"""
Palantir - Modeling continuous cell state and cell fate choices in single cell data.

Palantir is an algorithm to align cells along differentiation trajectories, identify
differentiation endpoints, and estimate cell-fate probabilities in single-cell data.
The package provides functions for preprocessing, visualization, trajectory analysis,
and gene expression modeling along the trajectories.

Modules
-------
config : Configuration settings for Palantir
core : Core functions for running the Palantir algorithm
presults : Class for storing and accessing Palantir results
io : Input/output functions for loading and saving data
preprocess : Preprocessing functions for single-cell data
utils : Utility functions for analysis
plot : Visualization functions
"""

import importlib.metadata

from . import config

# Import modules in a specific order to avoid circular imports
from . import presults
from . import core
from . import io
from . import preprocess
from . import utils
from . import plot

__version__ = importlib.metadata.version("palantir")

__all__ = [
    "config",
    "core",
    "presults",
    "io",
    "preprocess",
    "utils",
    "plot",
    "__version__",
]
