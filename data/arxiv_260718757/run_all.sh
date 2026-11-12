#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
MPI_RANKS="${MPI_RANKS:-2}"
MAX_MEMORY="${MAX_MEMORY:-2000}"
PAPER_DATA=data/arxiv_260718757

figures=(fig1_scales fig2_hierarchy fig3_mechanism fig4_projection fig5_tc)
if [ "$#" -gt 0 ]; then
    figures=("$@")
fi
for name in "${figures[@]}"; do
    folder="$PAPER_DATA/$name"
    mkdir -p "$folder/output"
    log="$folder/output/production.log"
    if [ "$name" = fig1_scales ]; then
        python -u "$folder/run.py" > "$log" 2>&1
    else
        mpirun -n "$MPI_RANKS" python -u "$folder/run.py" \
            --max-memory "$MAX_MEMORY" > "$log" 2>&1
    fi
    python "$folder/plot.py" --data-dir "$folder/output"
done
