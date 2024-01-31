ml modules/2.3-alpha1
ml cmake
ml intel-oneapi-compilers
ml python
ml cuda/12
ml gdb

export CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=75"

. venv/bin/activate

export CC=icx
export CXX=icpx
