from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# build : ~ python setup.py build_ext --inplace
setup(
    ext_modules=cythonize(
        Extension('models._membrane_cython', ['models/_membrane_cython.pyx']),
        language_level='3',
    ),
    script_args=['build_ext', '--inplace'],  # local build
    include_dirs=[np.get_include()] 
)