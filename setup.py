import sys
from setuptools import setup
from warnings import warn

# get version and other attributes
version_info = {}
with open("src/palantir/version.py") as f:
    exec(f.read(), version_info)
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="palantir",
    version=version_info['__version__'],
    author=version_info['__author__'],
    author_email=version_info['__author_email__'],
    description="Palantir for modeling continuous cell state and cell fate choices in single cell data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dpeerlab/palantir",
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
        "leidenalg>=0.9.1",
        "matplotlib>=2.2.2",
        "anndata>=0.8.0",
        "scanpy>=1.6.0",
        "mellon>=1.3.0",
        "pygam",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Operating System :: POSIX :: Linux",
        "Development Status :: 5 - Production/Stable",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    python_requires=">=3.8",
)
