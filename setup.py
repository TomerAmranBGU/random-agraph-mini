from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("longest_path_c.pyx"),
    include_dirs=[np.get_include()]
)
setup(
    ext_modules=cythonize("minimizer_c.pyx"),
    include_dirs=[np.get_include()]
)