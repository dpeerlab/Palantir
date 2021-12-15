import sys
from setuptools import setup
from warnings import warn

if sys.version_info.major != 3:
    raise RuntimeError("Palantir requires Python 3")
if sys.version_info.minor < 6:
    warn("Analysis methods were developed using Python 3.6")

# get version
with open("src/palantir/version.py") as f:
    exec(f.read())
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="palantir",
    version=__version__,  # read in from the exec of version.py; ignore error
    description="Palantir for modeling continuous cell state and cell fate choices in single cell data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dpeerlab/palantir",
    author=__author__,
    author_email=__author_email__,
    package_dir={"": "src"},
    packages=["palantir"],
    install_requires=[
        "numpy>=1.14.2",
        "pandas>=0.22.0",
        "scipy>=1.3",
        "networkx>=2.1",
        "scikit-learn",
        "joblib",
        "fcsparser>=0.1.2",
        "PhenoGraph>=1.5.3",
        "tables>=3.4.2",
        "Cython",
        "cmake",
        "matplotlib>=2.2.2",
        "seaborn>=0.8.1",
        "tzlocal",
        "scanpy>=1.6.0",
    ],
    extras_require={
        'PLOT_GENE_TRENDS': ["rpy2>=3.0.2"]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Operating System :: POSIX :: Linux",
        "Development Status :: 5 - Production/Stable",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    python_requires=">=3.6",
)
