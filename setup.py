from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import platform

if platform.system() == 'Linux':
    compilation_args = ['-fopenmp']
elif platform.system() == 'Darwin':
    compilation_args = ['-Xpreprocessor', '-fopenmp', '-lomp']

ext_modules = [
    Extension(
        "Fields",
        ["Fields.pyx"],
        extra_compile_args=compilation_args,
        include_dirs=[numpy.get_include()],
    )
]


setup(
    ext_modules=cythonize(ext_modules)
)
