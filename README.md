# nufft-ls
Benchmarking LS periodogram implementations

## Running
### Benchmarking
```console
pip install -e .
./bench/bench.py
```

### Testing (C++)
```console
mkdir build
cd build
cmake ..
cmake --build .
./tests/test_[baseline|trig_sum]
```

## Environments
The `env/` directory contains the module commands and other environment config
that @lgarrison used at Flatiron for testing on various compiler stacks. Some other
hand-tuned compiler flags are left as comments in `setup.py`.
