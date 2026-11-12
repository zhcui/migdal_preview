# arXiv:2607.18757

Data and calculations for
[Self-consistent GW theory for superconductivity in SrTiO3 models](https://arxiv.org/abs/2607.18757).

| Directory | Calculation | 
| --- | --- | 
| `fig1_scales` | Energy scales and lattice dielectric factor | 
| `fig2_hierarchy` | G0W0, GW0, scGW eigenvalues and Z |
| `fig3_mechanism` | Screening/phonon controls and gap functions |
| `fig4_projection` | Full kernel, frequency truncation, Takada projection |
| `fig5_tc` | Pairing-onset temperatures |

Each figure has its own `run.py`, `plot.py`, and `reference/`. Parameters and
plotting styles are local to each folder. Calculations write to `output/`.

## Shared IR input

`ir.h5` is the only shared calculation input (6.7 MiB). It contains the
sampling points, transform matrices, and inverse matrices at T=1 K and
Lambda=1e9.

The default calculations use this file to prepare temperature-specific caches
in each figure's `output/cache/`.
A different `--lamb` builds a new local basis. Figs. 2–4 also accept
`--ir-file` for an existing cache at the requested temperature and Lambda.

## Plot

From the repository root, redraw the published figures with NumPy and Matplotlib:

```bash
for folder in data/arxiv_260718757/fig*; do
    python "$folder/plot.py"
done
```

Reference plots go into each `output/reference/`. To plot a new calculation:

```bash
python data/arxiv_260718757/fig2_hierarchy/plot.py \
    --data-dir data/arxiv_260718757/fig2_hierarchy/output
```

These plots go into `output/`. Use `--out-dir` to choose another location.

## Recompute

Install Migdal as described in the repository [README](../../README.md), then:

```bash
bash data/arxiv_260718757/run_all.sh
```

This runs all five figures at their full default grids and draws the results.
The default is two MPI ranks, one thread per rank, and a 2000 MB cache budget
per rank. Set `MPI_RANKS`,
`OMP_NUM_THREADS`, and `MAX_MEMORY` to change these. Run settings and
`production.log` are saved in each `output/`.

Pass folder names to run a subset, or run a script directly:

```bash
bash data/arxiv_260718757/run_all.sh fig2_hierarchy fig4_projection
OMP_NUM_THREADS=1 mpirun -n 2 python data/arxiv_260718757/fig2_hierarchy/run.py \
    --max-memory 2000
```
