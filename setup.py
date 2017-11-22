import os
import sys
import shutil
from subprocess import call
from setuptools import setup
from warnings import warn

if sys.version_info.major != 3:
    raise RuntimeError('Palantir requires Python 3')
if sys.version_info.minor < 6:
    warn('Analysis methods were developed using Python 3.6')


# install GraphDiffusion
if shutil.which('pip3'):
    call(['pip3', 'install', 'git+https://github.com/pkathail/GraphDiffusion.git'])
    call(['pip3', 'install', 'git+https://github.com/jacoblevine/PhenoGraph.git'])


setup(name='palantir',
      description='Palantir for modeling continuous cell state and cell fate choices in single cell data',
      url='https://github.com/manusetty/palantir',
      author='Manu Setty',
      author_email='manu.talanki@gmail.com',
      package_dir={'': 'src'},
      packages=['palantir'],
      install_requires=[
          'numpy>=1.13.0',
          'pandas>=0.21.0',
          'scipy>=1.0.0',
          'sklearn',
          'networkx>=2.0',
          'joblib',
          'fcsparser',
          'GraphDiffusion', 
          'phenograph'
      ],
      )
