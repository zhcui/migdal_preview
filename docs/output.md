# Output

Migdal writes compact result rows on purpose. A smooth curve is useful, but the
convergence and diagnostic columns decide whether the curve should be
interpreted.

Use figures to see the shape of a calculation. Use CSV rows, settings, and
convergence flags when you quote a number or compare methods.

(output-pairing-eigenvalue)=
## Pairing eigenvalue

The sparse-IR radial drivers report `lam`, written as `lambda` in CSV files.
It is the selected gap-kernel eigenvalue

$$
\lambda = -\operatorname{Re}(\eta_{\mathrm{selected}}).
$$

In {ref}`Example 08 <example-08>`, the crossing condition is $\lambda(T)=1$.
This convention does not mean that every positive `lambda` is physically usable;
first check the gap and normal-state convergence flags.

(output-in-memory-rows)=
## In-memory rows

`G0W0`, `GW0`, and `ScGW` return `PointResult` rows.

| Field | Meaning |
| --- | --- |
| `n_cm3` | Requested spin-summed density in $\mathrm{cm}^{-3}$. |
| `temperature_K` | Temperature for this result row in K. |
| `lam` | Reported pairing eigenvalue. |
| `gap_converged` | Davidson gap-solver convergence flag. |
| `scf_converged` | Normal-state SCF flag; `None` for `G0W0`. |
| `mu`, `mu0` | Finite-temperature and zero-temperature chemical potentials in Hartree. |
| `n_bohr3`, `n_re` | Target density and reconstructed density in $\mathrm{Bohr}^{-3}$. |
| `nk`, `nq` | Active k-grid and q-grid sizes. |
| `gap` | `GapResult` with root, raw eigenvalue, cycles, matvecs, and active frequency count. |
| `denom` | RPA denominator diagnostics. |

`TakadaProjection` returns `TakadaResult` rows with `n_cm3`, `lam`,
`eps_F_eV`, `k_F_A`, and `symmetry_error`. The `k_F_A` field is a Fermi
momentum in $\mathrm{Angstrom}^{-1}$; text output writes the same quantity in
the column `k_F_Angstrom_inv`.

(output-csv-columns)=
## CSV columns

The sparse-IR driver `write(path)` method writes these core columns:

| Column | Meaning |
| --- | --- |
| `method` | Driver class name. |
| `temperature_K` | Temperature used for this row. |
| `n_cm3` | Requested density in $\mathrm{cm}^{-3}$. |
| `lambda` | Same value as `PointResult.lam`. |
| `gap_converged` | `1` when the Davidson gap solve converged. |
| `scf_converged` | `1`/`0` for GW0/scGW, `-1` for G0W0. |
| `scf_cycles`, `scf_res_sigma`, `scf_dn` | Normal-state SCF diagnostics. |
| `mu_Ha`, `mu0_Ha` | Chemical potentials in Hartree. |
| `n_bohr3`, `n_re` | Density target and reconstructed density in internal units. |
| `gap_cycles`, `gap_matvecs`, `gap_n_active_w`, `gap_used_min_pair` | Gap-solver diagnostics. |
| `nk`, `nq` | Momentum-grid sizes. |

Denominator columns report the RPA denominator policy, regularization count,
minimum denominator magnitude, maximum $|W|$, and the q/frequency location of
the worst denominator. Nonzero regularization is not automatically a failed
calculation, but it is a flag to inspect the row before using it in a physics
claim.

Optional columns appear when a driver records Fermi-surface pinning or
polarization diagnostics, for example `fs_pin_shift`, `mu_abs`,
`pi_w_imag`, and `pi_positive_count`.

(output-singular-columns)=
When `q_interp_mode="q_singular"` records low-q head diagnostics, CSV output also
adds these `singular_*` columns:

| Column | Meaning |
| --- | --- |
| `singular_head_source` | `pi_head` when the analytic polarization head sets the screened singular coefficient; `fit_w` when the direct low-q fit is used instead. |
| `singular_direct_fit_rel_residual`, `singular_direct_fit_abs_residual`, `singular_direct_fit_worst_v` | How well the direct low-q polynomial represents $q^2 W$. With the default `pi_head` policy this is an unused cross-check; with `fit_w` it controls the screened singular coefficient. |
| `singular_head_min_abs_denom`, `singular_head_reg_count` | Minimum magnitude and regularization count for the analytic low-q head denominator. |
| `singular_max_abs_A_W`, `singular_direct_fit_max_abs_A_W`, `singular_max_abs_A_dyn` | Maximum magnitudes over the bosonic grid of the screened singular coefficient, the direct-fit screened coefficient, and the dynamic residual coefficient $A_{\mathrm{dyn}}=A_W-A_c$. |

(output-tc-example-outputs)=
## Tc example outputs

{ref}`Example 08 <example-08>` writes two levels of data:

| File | How to read it |
| --- | --- |
| `lambda_vs_temperature.csv` | Raw rows at each method, density, and temperature. Check convergence here first. |
| `tc_vs_density.csv` | Interpolated Tc summary from usable $\lambda(T)$ rows. |
| `lambda_vs_temperature.png` | Visual check of the crossing and excluded points. |
| `tc_vs_density.png` | Final compact Tc curve. |

The interpolation is linear in $\log(T)$ and uses the highest-temperature
$\lambda=1$ crossing. Rows can be excluded when `lambda` is nonfinite,
nonpositive, or unconverged, depending on the script options.

(output-trust-checklist)=
## Quick trust checklist

Before interpreting a row:

1. `lambda` is finite.
2. `gap_converged == 1`.
3. For `GW0` and `ScGW`, `scf_converged == 1`, unless you are following an
   example script that intentionally keeps convergence failures; see
   {ref}`How to read outputs <examples-read-outputs>`.
4. The reconstructed density is close to the target density.
5. Denominator and polarization diagnostics do not show a suspicious outlier.

If any item fails, keep the row as a diagnostic rather than a final result.
