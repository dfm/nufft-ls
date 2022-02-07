from glob import glob

from setuptools import setup
import pybind11.setup_helpers
from pybind11.setup_helpers import Pybind11Extension, build_ext

compile_link_args = ['-march=native',
                     '-Ofast',
                     '-fopenmp',
                     #'-unroll',
                     
                     # icc
                     #'-march=core-avx2',  # for rome on icc
                     #'-fp-model', 'strict',

                     # clang
                     #'-stdlib=libc++',
                     #'-fveclib=SVML',
                     #'-lsvml'

                     # aocc
                     #'-std=c++11',
                     #'-ffast-math',
                     #'-lamdlibm',
                     #'-lm',
                     ]

ext_modules = pybind11.setup_helpers.intree_extensions(["src/nufft_ls/cpu.cpp"])

ext_modules[0].include_dirs = ['include/']
ext_modules[0].extra_compile_args = compile_link_args
ext_modules[0].extra_link_args = compile_link_args
ext_modules[0].language = 'c++'

setup(name="nufft_ls",
    version="0.0.1",
    url="https://github.com/dfm/nufft-ls",
    zip_safe=False,
    python_requires=">=3.6",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    package_dir={'':'src'},
)
