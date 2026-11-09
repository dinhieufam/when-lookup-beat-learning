import os

# Shared machine: cap BLAS threads (64 by default) before numpy is imported. Override via the environment.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "8")
