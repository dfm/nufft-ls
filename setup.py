from glob import glob

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

compile_link_args = ['-march=native',
                     '-Ofast',
                     '-fopenmp',
                     #'-funroll-loops',
                     
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

ext_modules = [
    Pybind11Extension(
        "nufft_ls",
        sorted(glob("src/*.cpp")),  # Sort source files for reproducibility
        include_dirs=['include/'],
        extra_compile_args=compile_link_args,
        extra_link_args=compile_link_args,
        language='c++',
    ),
]

setup(name="nufft_ls",
    version="0.0.1",
    url="https://github.com/dfm/nufft-ls",
    zip_safe=False,
    python_requires=">=3.6",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
