module load cmake/3
module load intel-oneapi-compilers/2022.0.1
module load python/3.9

. venv/bin/activate

export CC=icc
export CXX=icc

# only required for 'python setup.py develop'
# because `pip install -e .` doesn't seem to use these vars...?
export LDSHARED="$CC -L$(python3-config --prefix)/lib -shared"
export LDCXXSHARED="$CXX -L$(python3-config --prefix)/lib -shared"
