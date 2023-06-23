from setuptools import setup
from Cython.Build import cythonize
import numpy
#build command -> $ python cythonExtension.py build_ext --inplace -> Cria um modulo em cython
#import script_Rocha_2020

# Rocha 2020
setup(
    ext_modules = cythonize("script_Rocha_2020.pyx", annotate=True),
    include_dirs=[numpy.get_include()]
)

# Pinto 2020
# setup(
#     ext_modules = cythonize("script_Pinto.pyx", annotate=True),
#     include_dirs=[numpy.get_include()]
# )

# Newtoniano
# setup(
#     ext_modules = cythonize("script_GL_92.pyx", annotate=True),
#     include_dirs=[numpy.get_include()]
# )