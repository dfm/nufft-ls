ml modules/2.3-alpha1
ml cmake
ml gcc
ml python
ml cuda/12
ml gdb

export CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=75"

. venv/bin/activate
