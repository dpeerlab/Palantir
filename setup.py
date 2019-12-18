import sys
import shutil
from subprocess import call
from setuptools import setup
from warnings import warn

if sys.version_info.major != 3:
    raise RuntimeError('Palantir requires Python 3')
if sys.version_info.minor < 6:
    warn('Analysis methods were developed using Python 3.6')

# get version
with open('src/palantir/version.py') as f:
    exec(f.read())


# install GraphDiffusion
if shutil.which('pip3'):
    call(['pip3', 'install', 'git+https://github.com/jacoblevine/PhenoGraph.git'])


setup(name='palantir',
      version=__version__,# read in from the exec of version.py; ignore error
      description='Palantir for modeling continuous cell state and cell fate choices in single cell data',
      url='https://github.com/dpeerlab/palantir',
      author='Manu Setty',
      author_email='manu.talanki@gmail.com',
      package_dir={'': 'src'},
      packages=['palantir'],
      install_requires=[
          'numpy>=1.14.2',
          'pandas>=0.22.0',
          'scipy>=1.3',
          'networkx>=2.1',
          'scikit-learn',
          'joblib',
          'fcsparser>=0.1.2',
          'phenograph',
          'tables>=3.4.2',
          'Cython',
          'cmake',
          'MulticoreTSNE',
          'matplotlib>=2.2.2',
          'seaborn>=0.8.1',
          'tzlocal',
          'rpy2>=3.0.2',
          'scanpy'
      ],
      )
