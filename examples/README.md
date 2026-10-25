# Migdal examples

The examples are meant to be read in order. They move from sparse-IR basics to
the radial numerical treatment, then to G0W0 density curves, self-consistency,
screening/phonon controls, frequency-resolved diagnostics, Tc interpolation,
and the explicit multi-phonon-pole lattice factor.

Run MPI examples from the repository root, for example:

```bash
mpirun -n 2 python examples/03_g0w0_takada_curve/run.py
```

The first example does not use MPI:

```bash
python examples/01_ir_noninteracting_gf/run.py
```

Each example lives in its own directory. By default the script writes CSV data,
a short text report, and figures into the same directory as `run.py`. Use
`--out-dir` to write elsewhere.

The radial examples default to production-style sparse-IR and radial grids.
They reuse an existing `examples/ir.h5`, per-example `ir.h5`, or per-example
`ir.5` cache when one matches the requested temperature/lambda. If you
deliberately use very coarse k grids for a quick smoke run, avoid the highest
density points or relax `--n0-check-atol` while treating the output as a
control-flow check rather than a production calculation.

## Example map

1. `01_ir_noninteracting_gf`: verifies sparse-IR round-trip accuracy for
   `G0(k, iw) = 1 / (iw - (e_k - mu))`.
2. `02_cell_average_convergence`: shows why diagonal cell averages matter for
   no-truncation radial G0W0.
3. `03_g0w0_takada_curve`: compares truncated G0W0, no-truncation G0W0, and
   the Takada projected solver on the density curve.
4. `04_g0w0_gw0_scgw_curve`: compares no-truncation G0W0, GW0, and scGW with
   tuned SCF defaults.
5. `05_screening_phonon_controls`: adds compact GV and no-phonon controls to
   isolate electronic screening and phonon enhancement.
6. `06_g0w0_w_gap_diagnostics`: plots fixed G0W0 q-averaged `W(i nu_n)`,
   bare `V_tot`, and no-truncation/truncated leading gap functions.
7. `07_gw0_scgw_w_gap_diagnostics`: reconstructs q-averaged GW0/scGW
   interactions and compares their gap functions to G0W0 overlays when
   matching G0W0 CSV data is available.
8. `08_tc_vs_density`: runs each density as an independent high-to-low
   temperature chain so GW0/scGW can reuse same-density SCF restart states while
   forming a Tc(n) curve from `lambda(T)=1`.
9. `09_multi_phonon_modes`: repeats the G0W0/GW0/scGW density-curve comparison
   with an explicit three-pole phonon dielectric factor.

Examples 4, 5, 7, 8, and 9 write GW0/scGW rows even when the SCF loop is not
converged, because the point of those scripts is to expose convergence status
on a full curve. Use `--fail-unconverged` to make non-convergence fatal.

Example 5 keeps the default control plot compact:

```bash
mpirun -n 2 python examples/05_screening_phonon_controls/run.py
```

Use `--all-controls` there only when you want the extra one-shot G0V and
fixed-Pi/no-phonon combinations.

Example 8 estimates Tc from the lambda(T)=1 crossing for G0W0, GW0, and scGW.
For each method and density it cools from high temperature to low temperature;
GW0/scGW reuse only the previous converged same-density SCF state. The default
temperature grid is a 20-point log grid from `150 K` to `0.1 K`, and the scan
stops once a usable `lambda=1` bracket is found:

```bash
OMP_NUM_THREADS=2 mpirun -n 2 -bind-to core:2 -map-by core:2 \
  python examples/08_tc_vs_density/run.py
```

To redraw the figures from saved CSV data without rerunning any physics
calculations:

```bash
python examples/08_tc_vs_density/plot.py
```

Example 9 uses the multipole fields in `STOParams`:

```bash
mpirun -n 2 python examples/09_multi_phonon_modes/run.py
```

Its default run uses six log-spaced densities from `1e23` to `1e16 cm^-3` and
writes `multi_phonon_modes.csv`, `multi_phonon_modes.txt`, and
`multi_phonon_modes.png`. The built-in phonon parameters are the TO-anchored
three-mode STO set from the paper `calc/3_mode` inputs; the text report prints
the TO/LO pole energies and the generalized LST check.
