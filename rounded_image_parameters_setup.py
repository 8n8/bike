from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = 'Speed up solve_geometry.',
    ext_modules = cythonize('rounded_image_parameters.pyx'))

