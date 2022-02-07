module load cmake/3
module load gcc/11
module load python/3.9

. venv/bin/activate

export CC=gcc
export CXX=g++

# only required for 'python setup.py develop'
# because `pip install -e .` doesn't seem to use these vars...?
export LDSHARED="$CC -L$(python3-config --prefix)/lib -shared"
export LDCXXSHARED="$CXX -L$(python3-config --prefix)/lib -shared"
