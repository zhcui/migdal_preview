# Public API

This page lists the public driver surface. It is intentionally short: the
drivers are the supported user API, while internal radial kernels are
implementation modules. The main convolution formulas are summarized in
{ref}`Radial convolution <theory-radial-convolution>` and source docstrings.

The stable user-facing surface is the driver layer: the top-level drivers
`G0W0`, `GW0`, `ScGW`, and `TakadaProjection`; the model parameter object
`migdal.model.STOParams`; and the result rows returned by those drivers. The
radial kernels, SCF helpers, and sparse-IR cache builders are implementation
modules unless a page explicitly describes them as a command-line tool.

(api-driver-pattern)=
## Driver pattern

The main drivers follow a PySCF-like pattern:

```python
from migdal import G0W0

drv = G0W0(densities=[1.0e20], temperature=1.0, verbose=4)
row = drv.kernel()
print(row.lam, row.gap_converged)
```

Use `kernel()` for one density point, `scan()` for a density list,
`scan_temp()` for a fixed-density temperature list, `run()` for
`set(...).scan(...)` style chaining, and `write(path)` for CSV output after a
run.

For keyword meanings and defaults, see {ref}`Physical inputs
<parameters-physical-inputs>` and the later parameter tables. For result fields
and CSV interpretation, see {ref}`In-memory rows <output-in-memory-rows>` and
{ref}`CSV columns <output-csv-columns>`.

(api-main-drivers)=
## Main drivers

- `G0W0`: fixed `G0` and fixed `W0`. It supports `pi0_mode="finite_T"`,
  `"0K"`, or `"GG"`.
- `GW0`: self-consistent normal-state `G` with fixed `W0`. It adds SCF
  controls such as `scf_conv_tol`, `scf_damp`, and `allow_unconverged`.
- `ScGW`: self-consistent normal-state `G` and `W`. It adds polarization and
  tail controls such as `n_tail`, `pi_imag_policy`, and `sigma_tail_policy`.
- `TakadaProjection`: dense projected Takada comparison. It uses eV/Angstrom
  units internally, unlike the sparse-IR radial drivers.

Common sparse-IR driver keywords include:

```text
densities        density list in cm^-3, or "default"
temperature      temperature in K
params           STOParams object, or the built-in STO-like defaults
lamb             sparse-IR lambda cutoff parameter
ir_file          optional sparse-IR cache path
nq               q-grid size target
q_interp_mode    "q_singular" or "legacy"
diag_average     "auto", 0, or 1
max_memory       per-rank memory limit in MB
verbose          PySCF-style logging level
```

The sparse-IR drivers also accept gap-solver keywords through the shared
`GapConfig`, such as `davidson_tol`, `davidson_max_cycle`, `precond`, and
`return_gap_function`.

Common methods:

```text
set(**kwargs)    update driver attributes and return self
build()          build or load sparse-IR data
kernel(n_cm3)    run one density point
scan(densities)  run a density scan
scan_temp(temperatures, n_cm3)  run a fixed-density temperature scan
run(...)         apply optional updates, then scan
write(path)      write CSV output after kernel/scan
```

(api-model-parameters)=
## Model parameters

`STOParams` stores the SrTiO3-like one-band model constants used by the public
drivers.

```python
from migdal.model import STOParams
```

| Field | Meaning |
| --- | --- |
| `a_ang` | Cubic lattice constant in Angstrom. |
| `m_eff` | Effective mass in electron-mass units. |
| `eps_inf`, `eps_0` | High-frequency and static dielectric constants. |
| `omega_t0_cm` | Zone-center transverse optical phonon energy in $\mathrm{cm}^{-1}$. |
| `omega_t_disp_cm_a2` | Optional single-pole linear-frequency transverse-mode dispersion coefficient. |
| `phonon_to_poles_cm`, `phonon_lo_zeros_cm` | Optional multipole phonon representation in $\mathrm{cm}^{-1}$. |
| `phonon_to_omega2_disp_cm2_a2` | Optional multipole TO squared-frequency dispersion coefficients. |
| `phonon_to_omega2_cap_cm2` | Optional multipole TO squared-frequency caps; use `math.inf` for uncapped modes. |

Derived properties such as `a_bohr`, `band_a_ha_bohr2`, `omega_t_ha`, and
`omega_l_ha` handle unit conversion for the radial drivers.  Grid sizing uses
`phonon_grid_ha(params)`, which equals `omega_l_ha` for the single-pole model
and the largest supplied TO/LO feature for a multipole model.

When both multipole tuples are provided, they must have the same positive
length and satisfy the generalized Lyddane-Sachs-Teller endpoint condition
$\prod_j(\omega_{\mathrm{LO},j}/\omega_{\mathrm{TO},j})^2 = \epsilon_0/\epsilon_\infty$.
The multipole representation is q-independent unless
`phonon_to_omega2_disp_cm2_a2` is supplied.  Finite
`phonon_to_omega2_cap_cm2` entries replace the uncapped parabola by a smooth
large-q saturation that keeps the same small-q slope and stays below the
matching LO zero.  The legacy `omega_t_disp_cm_a2` field cannot be combined with
multipole phonons.

(api-result-conventions)=
## Result conventions

`G0W0`, `GW0`, and `ScGW` return `PointResult` rows. The key fields are:

| Field | Meaning |
| --- | --- |
| `n_cm3` | Requested spin-summed density in $\mathrm{cm}^{-3}$. |
| `temperature_K` | Temperature used for this row in K. |
| `lam` | Reported pairing eigenvalue. |
| `gap_converged` | Davidson gap-solver convergence flag. |
| `scf_converged` | Normal-state SCF flag, or `None` for `G0W0`. |
| `mu`, `mu0` | Chemical potentials in Hartree; CSV writes `mu_Ha` and `mu0_Ha`. |
| `n_bohr3`, `n_re` | Requested and reconstructed densities in $\mathrm{Bohr}^{-3}$. |
| `nk`, `nq` | Active momentum-grid sizes. |
| `gap` | `GapResult` with eigenvalue, convergence, cycles, matvecs, and active-frequency diagnostics. |
| `denom` | RPA denominator diagnostics for the screened interaction. |

`TakadaProjection` returns `TakadaResult` rows with `eps_F_eV`, `k_F_A`
($\mathrm{Angstrom}^{-1}$), and a kernel symmetry diagnostic.
Its text output writes the same momentum as `k_F_Angstrom_inv`.

| Field | Meaning |
| --- | --- |
| `n_cm3` | Requested density in $\mathrm{cm}^{-3}$. |
| `lam` | Largest dense projected pairing eigenvalue. |
| `eps_F_eV` | Fermi energy in eV. |
| `k_F_A` | Fermi momentum in $\mathrm{Angstrom}^{-1}$. |
| `symmetry_error` | Maximum dense-kernel symmetry error after symmetrization checks. |

(api-minimal-program)=
## Minimal program

```python
from migdal import G0W0

drv = G0W0(
    densities=[1.0e20, 3.0e19],
    temperature=1.0,
    verbose=4,
)
rows = drv.scan()
drv.write("/tmp/migdal-g0w0.csv")

for row in rows:
    print(row.n_cm3, row.lam, row.gap_converged)
```

For GW0 and scGW, also inspect `row.scf_converged` before interpreting
`row.lam`.
