from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = 'Speed up solve_geometry.',
    ext_modules = cythonize('solve_geometry_core.pyx'))

