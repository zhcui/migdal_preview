# Command Line

Migdal has two runnable surfaces:

- Scripts in the {ref}`recommended example order <examples-recommended-order>`
  are guided tutorials. They write figures, reports, and example-specific CSV
  files for a particular question.
- `python -m migdal.run` is the package command-line entry point. Use it when
  you want a compact density scan from one of the public drivers.

For a first physics run, follow the {ref}`workflow path
<workflows-check-environment>` through the example scripts. Use the package CLI
once you already know which method, density range, temperature, and grid
settings you want.

## Driver commands

The package CLI has four subcommands:

```bash
python -m migdal.run g0w0 --densities 1e20 --output /tmp/migdal-g0w0.csv

python -m migdal.run takada --densities 1e20 --output /tmp/migdal-takada.csv
```

`GW0` and `ScGW` are usually run under MPI because the radial convolutions are
the expensive part:

```bash
mpirun -n 2 python -m migdal.run gw0 \
  --densities 1e20 \
  --output /tmp/migdal-gw0.csv

mpirun -n 2 python -m migdal.run scgw \
  --densities 1e20 \
  --output /tmp/migdal-scgw.csv
```

Use a comma-separated list for a density scan:

```bash
mpirun -n 2 python -m migdal.run g0w0 \
  --densities 1e23,1e22,1e21,1e20 \
  --temperature 1.0 \
  --output /tmp/migdal-g0w0-scan.csv
```

The string `default` selects the built-in logarithmic density grid from
$10^{23}$ down to $10^{16}\,\mathrm{cm}^{-3}$.

## Important options

All methods share the density, temperature, output, and restart options. The
sparse-IR drivers are `G0W0`, `GW0`, and `ScGW`; `TakadaProjection` has its own
dense projected controls.

| Option | How to use it |
| --- | --- |
| `--densities` | Density list in $\mathrm{cm}^{-3}$, or `default`. |
| `--output` | CSV path. Parent directories are created automatically. |
| `--temperature` | Temperature in K. Sparse-IR drivers convert this to $\beta = 1/(k_{\mathrm{B}} T)$. |
| `--restart` | Passes the restart flag to `scan()`. It defaults to `0` for all methods. |
| `--ir-file`, `--save-ir`, `--no-save-ir` | Sparse-IR cache path and cache-writing flag for sparse-IR drivers; cache writing defaults on. |
| `--allow-unconverged` | For `GW0` and `ScGW`, continue to output rows even when normal-state SCF did not converge. |
| `--max-memory` | Per-rank memory target in MB for sparse-IR radial cache decisions. |

For method-specific knobs, run:

```bash
python -m migdal.run g0w0 --help
python -m migdal.run gw0 --help
python -m migdal.run scgw --help
python -m migdal.run takada --help
```

Keep the defaults for `q_interp_mode`, `diag_average`, denominator handling,
and sparse-IR settings unless you are doing a convergence or stability audit.
Those defaults encode the current {ref}`RPA denominator sign convention
<theory-rpa-sign-convention>`, {ref}`Coulomb-head diagonal-cell convention
<theory-q-singular-diagonal>`, and {ref}`Sparse-IR sampling convention
<theory-sparse-ir>`.

## Sparse-IR cache command

Low-temperature runs can spend noticeable time constructing sparse-IR sampling
matrices. The drivers create a cache automatically with `--ir-file`; use
`--no-save-ir` to disable writing:

```bash
python -m migdal.run g0w0 \
  --densities 1e20 \
  --temperature 1.0 \
  --ir-file /tmp/migdal-ir-T1K.h5 \
  --output /tmp/migdal-g0w0.csv
```

You can also build or validate a cache directly:

```bash
python -m migdal.ir_cache \
  --beta 315775.022182 \
  --lamb 1e9 \
  --output /tmp/migdal-ir-T1K.h5 \
  --overwrite
```

A sparse-IR cache is tied to `beta`, `lamb`, and tolerance. Do not reuse one
cache across different temperatures unless it was deliberately built as a
compatible template. For temperature scans, keep separate cache files under the
output directory, as {ref}`Example 08 <example-08>` does.

The HDF5 cache is also a provenance record. It stores the requested `_beta`,
`lamb`, `wmax`, `tol`, and `tau_beta_rtol`; sampled tau and Matsubara grids;
forward transform matrices and, when written by the cache builder, pseudoinverse
matrices. File attributes record the sparse-ir and pylibsparseir versions, the
sampling algorithm, basis sizes, whether the basis was built at a scaled
reference beta, template-scaling metadata when applicable, build timings, and
matrix condition diagnostics. Use `python -m migdal.ir_cache --validate-only`
when you want to audit a cache without writing a new file.

The cache parameter

$$
\Lambda = \beta \omega_{\mathrm{max}}
$$

controls the IR basis scale. Larger `lamb` values cover a wider frequency
window but also increase the number of sampled points.

## Before trusting a run

Treat the CSV as the source of truth. A figure or smooth density trend is only
a quick visual check.

Before using a CLI row in a report:

1. `lambda` is finite.
2. `gap_converged == 1`.
3. For `GW0` and `ScGW`, `scf_converged == 1`, unless the row is being kept as
   a diagnostic.
4. `n_re` is close to `n_bohr3`.
5. Denominator diagnostics do not show an unexplained outlier.

See {ref}`CSV column meanings <output-csv-columns>` for field meanings and the
{ref}`quick trust checklist <output-trust-checklist>` before diagnosing
{ref}`gap <troubleshooting-gap-solver>`, {ref}`SCF <troubleshooting-scf>`, or
{ref}`denominator <troubleshooting-rpa-denominator>` failures.

## Environment notes

The optimized Cython/OpenMP radial kernels are the intended path for
production-style radial calculations. The pure-Python fallback is useful for
debugging and CI fallback checks, but it is slow.

Useful environment variables:

- `MIGDAL_REQUIRE_OPENMP=1`: fail installation if optimized radial kernels
  cannot be built.
- `MIGDAL_DISABLE_OPENMP=1`: install without OpenMP extensions for fallback
  testing.
- `MIGDAL_DISABLE_CYTHON_RADIAL=1`: force the Python fallback at runtime.
- `MIGDAL_MPIEXEC`: override the MPI launcher used by MPI tests.
- `OMP_NUM_THREADS`: control OpenMP threads per MPI rank. Start with `1`
  unless you are benchmarking.

When scaling a calculation, increase problem quality first through grid and
convergence checks, then tune MPI ranks and OpenMP threads. Parallel speed is
not a substitute for a converged row.
