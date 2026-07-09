# Troubleshooting

Most failures are useful signals: a grid is too coarse, an SCF loop did not
settle, an IR cache does not match the requested temperature, or the RPA
denominator entered a delicate regime. Do not hide those signals by changing a
tolerance unless the run is explicitly a smoke test.

(troubleshooting-kernel-checks)=
## Installation or kernel checks fail

First confirm the compiled radial kernels:

```bash
python - <<'PY'
from migdal import radial

print("radial Cython:", radial._cython_radial_available())
print("qW Cython:", radial._qw_cython_available())
PY
```

If either value is false, rebuild with the compiler, OpenMP, Cython, and MPI
development headers installed. The pure-Python fallback is useful for debugging
and CI fallback tests, but it is not the intended path for production-style
radial scripts in the {ref}`recommended example order
<examples-recommended-order>`.

(troubleshooting-mpi-start)=
## MPI run does not start

Check the serial tutorial first:

```bash
python examples/01_ir_noninteracting_gf/run.py --out-dir /tmp/migdal-serial
```

Then try the {ref}`small radial smoke check <quickstart-small-radial-smoke>`.
If your MPI
launcher rejects the run because of local policy, set `MIGDAL_MPIEXEC` as shown
in the README and rerun the targeted MPI tests before trusting long
{ref}`example runs <examples-recommended-order>`.

(troubleshooting-ir-cache-mismatch)=
## IR cache mismatch

Sparse-IR caches are tied to `beta`, `lamb`, and tolerance. A cache from a
different temperature is not interchangeable.

Use distinct output directories for temperature scans, or pass an `ir_file`
template as {ref}`Example 08 <example-08>` does. If a cache mismatch appears,
regenerate the cache for the requested settings rather than forcing the file to
load.

For a provenance audit, inspect the HDF5 datasets `_beta`, `lamb`, `wmax`, `tol`,
and `tau_beta_rtol`, plus the file attributes describing sparse-ir versions,
sampling algorithm, basis scaling, template scaling, and matrix conditioning.
`python -m migdal.ir_cache --validate-only --beta ... --lamb ...` performs the
same construction and validation path without requiring an output file.

(troubleshooting-particle-number)=
## Particle-number check fails

The initial density check compares the requested density with the radial-grid
reconstruction. A failure usually means the quick grid is too coarse for that
density or temperature.

Prefer:

1. increase `nq`;
2. increase `kgrid_n_left` and `kgrid_n_right`;
3. increase `kgrid_n_shell_min`;
4. avoid interpreting very coarse smoke-run data.

Relax `n0_check_atol` only when the point is a control-flow check.

(troubleshooting-q-singular-diagonal)=
## q_singular requires diagonal averages

The default `q_interp_mode="q_singular"` separates the Coulomb-like

$$
W(q) = \frac{A}{q^2} + W_{\mathrm{reg}}(q)
$$

head. In this mode, the diagonal radial block requires a cell average. Keep
`diag_average="auto"` unless you are intentionally comparing legacy behavior.

(troubleshooting-gap-solver)=
## Gap solver does not converge

First inspect `gap_cycles`, `gap_matvecs`, `gap_n_active_w`, and
`gap_used_min_pair`. A nonconverged gap solve can come from an undersized active
frequency window, a difficult interaction, or a Davidson subspace that is too
small.

Reasonable follow-ups are:

- increase `davidson_max_cycle`;
- increase `davidson_max_space`;
- avoid aggressive frequency truncation;
- compare the gap-function diagnostics in {ref}`Example 06 <example-06>` or
  {ref}`Example 07 <example-07>`.

(troubleshooting-scf)=
## Normal-state SCF does not converge

For GW0/scGW, inspect `scf_res_sigma`, `scf_dn`, and `scf_cycles`. The residual
is measured for the gauge-invariant combination

$$
(\Sigma_{\mathrm{new}}-\mu_{\mathrm{new}})
-
(\Sigma_{\mathrm{old}}-\mu_{\mathrm{old}}).
$$

Then vary the SCF mixing controls one at a time. `scf_damp=0.98` means each
cycle keeps 98% of the old state and mixes in 2% of the new state. If the
residual decreases monotonically but too slowly, try a lower `scf_damp`. If the
residual oscillates or DIIS appears unstable, compare against linear mixing with
`scf_diis_space=0`, reduce `scf_diis_space`, or delay acceleration with a larger
`scf_diis_start_cycle`.

Use `--fail-unconverged` when you want scripts to stop at the first failed SCF
point. Use {ref}`Example 04 <example-04>` and the
{ref}`Tc workflow <workflows-estimate-tc>` to compare density continuation and
cooling-restart behavior.

(troubleshooting-rpa-denominator)=
## RPA denominator warnings

The denominator is

$$
1 - \Pi(q,\mathrm{i}\nu_n)V_{\mathrm{tot}}(q,\mathrm{i}\nu_n).
$$

If denominator diagnostics are large or regularization counts are nonzero,
inspect the worst q/frequency and compare nearby density points. A regularized
row can still be a useful diagnostic, but it should not be treated as a clean
reference without review.

(troubleshooting-smooth-figure-failed-flags)=
## Figure looks smooth but flags fail

Trust the flags before the picture. A smooth $\lambda(n)$ or $\lambda(T)$ curve
can include unconverged SCF rows, failed gap solves, or excluded Tc points. Read
the CSV and summary text before using the figure in a report.

(troubleshooting-pre-submit-checks)=
## Pre-submit checks

Before publishing a branch or relying on a documentation update, run the checks
that match the change.

For documentation-only edits:

```bash
python -m sphinx -E -W --keep-going -b html docs docs/_build/html
python -m sphinx -E -W --keep-going -b linkcheck docs docs/_build/linkcheck
git diff --check
```

For code or driver changes:

```bash
python -m pytest -q migdal/test
```

MPI checks are opt-in locally:

```bash
MIGDAL_RUN_MPI_TESTS=1 python -m pytest -q -m mpi \
  migdal/test/test_mpi.py migdal/test/test_ir_cache.py
```

For package changes, also build and check the distributions:

```bash
python -m pip install build twine
MIGDAL_REQUIRE_OPENMP=1 python -m build
python -m twine check --strict dist/*
```

These checks mirror the project split: serial correctness, MPI behavior,
documentation rendering, and package metadata are separate failure modes.
