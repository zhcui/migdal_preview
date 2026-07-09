# Workflows

This page gives a runnable path through the code. Start small, inspect the
output, and only then spend time on dense curves.

A useful mental model is:

| Step | Question | Artifact to trust first |
| --- | --- | --- |
| Environment check | Do the IR transforms and kernels run? | Small serial output directory. |
| Smoke radial point | Does the radial control flow complete? | CSV plus convergence flags. |
| Density curve | Is the physical trend stable across density? | CSV; figures for visual inspection. |
| SCF comparison | Does self-consistency change the result? | `scf_converged`, `scf_res_sigma`, then `lambda`. |
| Tc scan | Where does $\lambda(T)$ cross 1? | `lambda_vs_temperature.csv`, then `tc_vs_density.csv`. |

(workflows-choose-method)=
## Choose a method

| Method | Use it when | First example |
| --- | --- | --- |
| `G0W0` | You want a fixed noninteracting `G0` and fixed screened `W0`. | {ref}`03_g0w0_takada_curve <example-03>` |
| `GW0` | You want a self-consistent normal-state `G` with fixed `W0`. | {ref}`04_g0w0_gw0_scgw_curve <example-04>` |
| `ScGW` | You want both `G` and `W` updated in the normal-state loop. | {ref}`04_g0w0_gw0_scgw_curve <example-04>` |
| `TakadaProjection` | You want the dense projected comparison in eV/Angstrom units. | {ref}`03_g0w0_takada_curve <example-03>` |

For a new setup, verify `G0W0` first. It exercises the radial interaction and
gap solver without adding the normal-state SCF loop.

(workflows-check-environment)=
## 1. Check the environment

Run the serial sparse-IR round trip:

```bash
python examples/01_ir_noninteracting_gf/run.py --out-dir /tmp/migdal-example01
```

Success means the sparse-IR transforms work for a known

$$
G_0(k,\mathrm{i}\omega_n)=
\frac{1}{\mathrm{i}\omega_n-(\varepsilon_k-\mu)}.
$$

It does not validate a production radial calculation.

(workflows-small-radial-point)=
## 2. Run a small radial point

Use the quickstart smoke command:

```bash
mpirun -n 2 python examples/03_g0w0_takada_curve/run.py \
  --n-density 2 \
  --nq 40 \
  --k-left 20 \
  --k-right 20 \
  --k-shell-min 8 \
  --k-shell-max 80 \
  --out-dir /tmp/migdal-example03-smoke
```

This is a control-flow check. If it writes CSV files and figures, move to the
default grids before interpreting the numbers.

(workflows-compare-pairing-curves)=
## 3. Compare pairing-eigenvalue curves

Run {ref}`Example 03 <example-03>` for the first full comparison:

```bash
mpirun -n 2 python examples/03_g0w0_takada_curve/run.py \
  --out-dir /tmp/migdal-example03
```

Read the text report first, then the CSV files. The figure gives the trend; the
CSV gives the data you should cite.

(workflows-add-self-consistency)=
## 4. Add self-consistency

Run {ref}`Example 04 <example-04>` when the fixed-W0 curve is understood:

```bash
mpirun -n 2 python examples/04_g0w0_gw0_scgw_curve/run.py \
  --out-dir /tmp/migdal-example04
```

For every GW0 or scGW row, check `scf_converged` before using `lambda`. Some
tutorial scripts keep unconverged rows so the plot can show where the method
becomes difficult.

(workflows-inspect-diagnostics)=
## 5. Inspect interactions and gap functions

Use {ref}`Example 06 <example-06>` and {ref}`Example 07 <example-07>` when a
curve looks surprising:

```bash
mpirun -n 2 python examples/06_g0w0_w_gap_diagnostics/run.py \
  --out-dir /tmp/migdal-example06

mpirun -n 2 python examples/07_gw0_scgw_w_gap_diagnostics/run.py \
  --out-dir /tmp/migdal-example07
```

These scripts write q-averaged interaction data and gap-function lines. They
are better for diagnosis than for a first tutorial.

(workflows-estimate-tc)=
## 6. Estimate Tc

{ref}`Example 08 <example-08>` uses same-density cooling restarts for GW0/scGW
and interpolates the crossing:

```bash
OMP_NUM_THREADS=2 mpirun -n 2 -bind-to core:2 -map-by core:2 \
  python examples/08_tc_vs_density/run.py \
  --out-dir /tmp/migdal-example08
```

Use `lambda_vs_temperature.csv` to audit which rows entered the Tc
interpolation, then use `tc_vs_density.csv` for the final summary.

The cooling restart is deliberately more careful than copying a previous
self-energy array. A converged GW0/scGW row stores the gauge-invariant
combination $X=\Sigma-\mu$, remaps its sparse-IR coefficients to the next
temperature, rescales the IR normalization, interpolates the k grid when needed,
and then re-applies the Fermi-surface pin if that gauge is enabled. This requires
the same sparse-IR `lambda`, the same fermionic `nw`, and matching reduced
Matsubara points. If a restarted row fails and retry is enabled, the script tries
that temperature once more from a cold start and records the event in
`cooling_restart_log.csv`.

For interrupted same-temperature GW0/scGW jobs, the Python driver keywords
`scf_chkfile` and `scf_restart_chkfile` provide a lighter disk checkpoint for the
normal-state SCF seed. This is separate from the temperature-remapped restart
above: the disk checkpoint is intended to resume the same method, temperature,
and grid, not to bridge a cooling step.

(workflows-reading-order)=
## Reading order after a run

1. Text report: what was asked and which settings were used.
2. CSV: machine-readable data and convergence flags.
3. Figure: visual trend and obvious outliers.
4. {ref}`CSV columns <output-csv-columns>` and
   {ref}`quick trust checklist <output-trust-checklist>`: field meanings and
   row-level sanity checks.
5. {doc}`troubleshooting`: convergence, grid, cache, and denominator diagnostics
   when a row fails the checklist.
