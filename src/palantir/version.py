"""Version information."""
import importlib.metadata

try:
    # Get version from pyproject.toml via package metadata
    __version__ = importlib.metadata.version("palantir")
except importlib.metadata.PackageNotFoundError:
    # Package is not installed, fall back to hardcoded version
    __version__ = "1.4.1"  # Should match pyproject.toml

__author__ = "Palantir development team"
__author_email__ = "manu.talanki@gmail.com"