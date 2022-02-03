from glob import glob

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext, MACOS, WIN, has_flag


ext_modules = [
    Pybind11Extension(
        "nufft_ls",
        sorted(glob("src/*.cpp")),  # Sort source files for reproducibility
        include_dirs=["include/"],
        language="c++",
    ),
]


class custom_build_ext(build_ext):
    def build_extensions(self):
        # Handle platform specific optimization flags
        if WIN:
            # I don't know what optimization flags would be right on Windows
            pass
        else:
            cflags = []
            ldflags = []
            for flag in [
                "-march=native",
                "-Ofast",
                "-fopenmp",
                "-funroll-loops",
                "-ffast-math",
            ]:
                if has_flag(self.compiler, flag):
                    cflags.append(flag)

            # None of the rest of these will work on Mac
            if not MACOS:
                cc = self.compiler.compiler_so[:2].join(" ")
                if "icc" in cc:
                    if has_flag(self.compiler, "-march=core-avx2"):
                        cflags += ["-march=core-avx2"]
                    cflags += ["-fp-model", "strict"]
                elif "clang" in cc:
                    cflags += ["-fveclib=SVML"]
                    ldflags += ["-lsvml"]
                elif "aocc" in cc:
                    ldflags += ["-lamdlibm", "-lm"]

            for ext in self.extensions:
                ext.extra_compile_args += cflags
                ext.extra_link_args += ldflags

        build_ext.build_extensions(self)


setup(
    name="nufft_ls",
    version="0.0.1",
    url="https://github.com/dfm/nufft-ls",
    zip_safe=False,
    python_requires=">=3.6",
    install_requires=["numpy", "astropy", "threadpoolctl"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": custom_build_ext},
)
