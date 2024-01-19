ml modules/2.2
ml cmake
ml gcc
ml python
ml cuda

export CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=75"

. venv/bin/activate
