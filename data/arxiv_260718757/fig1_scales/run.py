#!/usr/bin/env python
"""Figure 1: analytic electronic scales and the three-phonon lattice factor.

    python data/arxiv_260718757/fig1_scales/run.py
    python data/arxiv_260718757/fig1_scales/plot.py --data-dir data/arxiv_260718757/fig1_scales/output
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from migdal import model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=HERE / "output")
    parser.add_argument("--n-density", type=int, default=600)
    parser.add_argument("--n-frequency", type=int, default=800)
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    params = model.STOParams(
        m_eff=2.1, eps_inf=6.163085088, eps_0=23000.0,
        phonon_to_poles_cm=(9.98977881246349, 180.5746693840041, 590.7326992132146),
        phonon_lo_zeros_cm=(175.83894684689574, 449.4013216886174, 823.7949569635838),
    )
    n = np.geomspace(1.0e14, 1.0e23, args.n_density)
    n_bohr3 = n * (model.BOHR * 1.0e-8) ** 3
    k_F = (3.0 * np.pi**2 * n_bohr3) ** (1.0 / 3.0)
    ha_to_mev = 1000.0 / model.EV_TO_HA
    e_F = k_F**2 / (2.0 * params.m_eff) * ha_to_mev
    omega_pl = np.sqrt(4.0 * np.pi * n_bohr3 / (params.eps_inf * params.m_eff)) * ha_to_mev
    e_TF = 2.0 * k_F / (np.pi * params.eps_inf) * ha_to_mev
    np.savetxt(
        args.out_dir / "scales.csv", np.column_stack((n, e_F, omega_pl, e_TF)),
        delimiter=",", header="n_cm3,E_F_meV,omega_pl_meV,E_TF_meV", comments="",
    )

    nu_mev = np.geomspace(1.0e-2, 1.0e4, args.n_frequency)
    factor = model.get_phonon_factor_iw(nu_mev / ha_to_mev, params)
    np.savetxt(
        args.out_dir / "interaction_factor.csv", np.column_stack((nu_mev, factor)),
        delimiter=",", header="nu_meV,F3", comments="",
    )
    settings = {"n_density": args.n_density, "n_frequency": args.n_frequency,
                "m_eff": params.m_eff, "eps_inf": params.eps_inf, "eps_0": params.eps_0,
                "phonon_to_poles_cm": params.phonon_to_poles_cm,
                "phonon_lo_zeros_cm": params.phonon_lo_zeros_cm}
    (args.out_dir / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    print("wrote", args.out_dir)


if __name__ == "__main__":
    main()
