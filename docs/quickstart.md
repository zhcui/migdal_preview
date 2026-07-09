# Quickstart

This page gets you from a clean checkout to one working tutorial. The commands
are meant to check the workflow, not to prove that a production calculation
is converged.

(quickstart-install)=
## Install

Migdal supports Python 3.10 and newer on Linux. Install a compiler,
OpenMP, and MPI development headers first:

```bash
sudo apt-get update
sudo apt-get install -y build-essential libgomp1 openmpi-bin libopenmpi-dev
```

Then install the Python dependencies and build the package in editable mode:

```bash
python -m pip install --upgrade pip setuptools wheel Cython
python -m pip install -r requirements.txt
MIGDAL_REQUIRE_OPENMP=1 python -m pip install -e .
```

Check that both optimized radial kernels are available:

```bash
python - <<'PY'
from migdal import radial

assert radial._cython_radial_available()
assert radial._qw_cython_available()
print("compiled radial kernels available")
PY
```

(quickstart-run-first-tutorial)=
## Run the first tutorial

The first tutorial, {ref}`Example 01 <example-01>`, is serial. It checks a
sparse-IR round trip for a known
noninteracting Green's function,

$$
G_0(k, \mathrm{i}\omega_n) =
\frac{1}{\mathrm{i}\omega_n - (\varepsilon_k - \mu)}.
$$

Run it from the repository root:

```bash
python examples/01_ir_noninteracting_gf/run.py --out-dir /tmp/migdal-example01
```

The output directory contains a short text report, a CSV file, and a figure of
the reconstruction error. Passing this tutorial means the sparse-IR transforms
are usable for the toy case.

(quickstart-small-radial-smoke)=
## Run a small radial smoke check

Most radial scripts in the {ref}`recommended example order
<examples-recommended-order>` are intended to run under MPI. For a
quick flow check, use a deliberately small grid and write outputs outside the
source tree:

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

Treat this as a control-flow check. The grid is too small for production
physics, especially at difficult densities. Use the {ref}`example practical
tips <examples-practical-tips>`, convergence flags, and independent checks
before interpreting a curve.

(quickstart-build-docs)=
## Build this website locally

Install the documentation dependencies:

```bash
python -m pip install -r docs/requirements.txt
```

Build the HTML pages:

```bash
python -m sphinx -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` in a browser.

Next steps:

- {ref}`Choose a method <workflows-choose-method>` shows which example to run
  next.
- {doc}`parameters` explains the common driver settings and controls.
- {ref}`Quick trust checklist <output-trust-checklist>` explains how to decide
  whether an output row is usable.
