"""Core modules for the z02 visual value baseline: data, contracts, model, cache, evaluation."""
import os

# This env's BLAS/OpenMP threading races during import and corrupts the interpreter
# (segfaults or bogus TypeErrors from scipy, transformers and matplotlib import-time
# docstring parsing). Single-threaded math makes those imports deterministic. This must
# run before any of them load, so it sits in the package the entry point imports first.
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'
