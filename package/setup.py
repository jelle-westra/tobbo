from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy as np

def read_readme() -> str:
    with open('README.md', 'r') as handle : 
        long_description = handle.read()
    return long_description

setup(
    name='tobbo',
    version = '0.1.0',
    description='Topology Optimization with Black-Box Optimization methods',
    packages=find_packages(where='.'),
    package_dir={'tobbo': 'tobbo'},
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/jelle-westra/tobbo',
    author='jelle-westra, olarterodriguezivan, elenaraponi',
    author_email='jelwestra@gmail.com',
    license='MIT',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Scientific/Engineering',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',
    install_requires=[
        'numpy>=1.24.4',
        'scipy>=1.12.0',
        'matplotlib>=3.10',
        'tqdm>=4.67',
        'rasterio>=1.4',
        'shapely>=2.1',
        'cma>=3.2.2',
        'hebo==0.3.6',
        'GPy>=1.13.2',
    ],
    ext_modules=cythonize(
        Extension('tobbo.models._membrane_cython', ['tobbo/models/_membrane_cython.pyx']),
        language_level='3',
    ),
    include_dirs=[np.get_include()] 
)