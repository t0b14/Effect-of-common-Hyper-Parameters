from setuptools import setup
from Cython.Build import cythonize

#setup cython 
setup(
    ext_modules = cythonize(["c_layer.pyx"])
)