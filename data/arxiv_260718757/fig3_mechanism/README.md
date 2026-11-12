# Fig. 3: pairing mechanism

Panel A compares screening and phonon controls. Panel B shows the
momentum-averaged scGW gap eigenfunction; solid and dashed ticks mark E_F
and the plasma frequency.

```bash
mpirun -n 2 python data/arxiv_260718757/fig3_mechanism/run.py --max-memory 2000
python data/arxiv_260718757/fig3_mechanism/plot.py \
    --data-dir data/arxiv_260718757/fig3_mechanism/output
```

Omit `--data-dir` to draw the reference data. Use `--part controls` or
`--part gap` to calculate one panel. Defaults are 11 control densities,
three gap densities, T=1 K, Nq=500, and Lambda=1e9.

The scGW control continues from the previous converged density. Other controls
and gap calculations start cold. `G0V` and `GV` set electronic polarization
to zero while retaining the lattice factor. The no-phonon control sets
`eps_0=eps_inf` and clears the phonon poles.

Gap CSVs divide `phi_tilde` by k, average with dk weights, rotate the largest
component to positive real, and normalize by the maximum real amplitude.
