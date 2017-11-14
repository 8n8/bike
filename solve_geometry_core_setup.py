""" It builds the solve_geometry_core module. """

from distutils.core import setup  # noqa # pylint: disable=import-error, no-name-in-module
from Cython.Build import cythonize  # type: ignore

setup(
    name='Speed up solve_geometry.',
    ext_modules=cythonize('solve_geometry_core.pyx'))
