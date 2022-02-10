from glob import glob

from setuptools import setup

from pybind11.setup_helpers import Pybind11Extension, build_ext, MACOS, WIN, has_flag, intree_extensions


ext_modules = intree_extensions(["src/nufft_ls/cpu.cpp"])
ext_modules[0].include_dirs += ['include/']
ext_modules[0].language = 'c++'

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
                cc = " ".join(self.compiler.compiler_so[:2])
                
                # TODO: this can give a huge speedup when using the Intel compiler on an AMD CPU
                # but if we're on an Intel CPU, we want -march=native instead. How to detect this?
                #if "icc" in cc:
                #    if has_flag(self.compiler, "-march=core-avx2"):
                #        cflags += ["-march=core-avx2"]
                
                if "clang" in cc:
                    cflags += ["-stdlib=libc++"]
                    ldflags += ["-stdlib=libc++"]
                    # These are optimistic/untested commands that might work if one has
                    # an SVML lib available for Clang to link against
                    #cflags += ["-fveclib=SVML"]
                    #ldflags += ["-lsvml"]

                # TODO: AOCC on Clang 13 seems to only support C++11
                # but the Pybind11 auto-detection doesn't seem to notice
                #if "aocc" in cc:  # TODO: the AOCC executable is actually just named `clang`...
                #    cflags += ["-std=c++11"]
                #    ldflags += ["-lamdlibm", "-lm"]  # not sure if these help...

            for ext in self.extensions:
                ext.extra_compile_args += cflags
                ext.extra_link_args += ldflags

        build_ext.build_extensions(self)


setup(
    name="nufft_ls",
    version="0.0.2",
    url="https://github.com/dfm/nufft-ls",
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=["numpy", "astropy", "threadpoolctl"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": custom_build_ext},
    package_dir={'':'src'},
)
