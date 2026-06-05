#!/bin/bash
# Run stress test with controlled threading
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

cd /home/gabuf/projects/ctc-detect
source .venv/bin/activate
python stress_test_v3.py > results/stress_test_v3.log 2>&1
