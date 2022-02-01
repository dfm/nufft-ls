module load cmake/3
module load llvm/13
module load python/3.9

. venv/bin/activate

export CC=clang
export CXX=clang++

# only required for 'python setup.py develop'
# because `pip install -e .` doesn't seem to use these vars...?
export LDSHARED="$CC -L$(python3-config --prefix)/lib -shared"
export LDCXXSHARED="$CXX -L$(python3-config --prefix)/lib -shared"
