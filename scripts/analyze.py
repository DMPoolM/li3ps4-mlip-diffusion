#!/usr/bin/env python
"""Li MSD -> diffusion coefficients -> Arrhenius fit for a directory of MD runs.

    python scripts/analyze.py runs/beta_li3ps4
Writes results.json, msd.png and (with >= 2 temperatures) arrhenius.png into that directory.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from li3ps4md.analysis import (arrhenius, fit_diffusion, load_positions, msd,  # noqa: E402
                               nernst_einstein)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir")
    p.add_argument("--fit-start", type=float, default=0.2)
    p.add_argument("--fit-end", type=float, default=0.8)
    a = p.parse_args()
    run = Path(a.run_dir)

    trajs = sorted(run.glob("traj_*K.extxyz"), key=lambda f: int(re.findall(r"(\d+)K", f.name)[0]))
    if not trajs:
        sys.exit(f"no traj_*K.extxyz in {run}")

    results, fig, ax = [], *plt.subplots(figsize=(5, 3.6))
    for f in trajs:
        T = int(re.findall(r"(\d+)K", f.name)[0])
        times, pos, frames = load_positions(f)
        m = msd(pos)
        fit = fit_diffusion(times, m, a.fit_start, a.fit_end)
        vol = frames[0].get_volume()
        sigma = nernst_einstein(fit.D_total, pos.shape[1], vol, T)
        results.append({"T_K": T, "sim_time_ps": float(times[-1]), "n_Li": pos.shape[1],
                        "D_cm2s": fit.D_total, "D_xyz_cm2s": fit.D_xyz.tolist(),
                        "sigma_NE_Scm": sigma, "fit_range_ps": fit.fit_range_ps})
        ax.plot(times[: len(m)], m[:, 3], label=f"{T} K")
        print(f"{T} K: D = {fit.D_total:.2e} cm²/s  (x,y,z = "
              + ", ".join(f"{d:.1e}" for d in fit.D_xyz) + f"),  sim {times[-1]:.1f} ps")
    ax.set_xlabel("lag time (ps)")
    ax.set_ylabel("Li MSD (Å²)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(run / "msd.png", dpi=200)

    summary = {"runs": results}
    if len(results) >= 2:
        T = np.array([r["T_K"] for r in results])
        D = np.array([r["D_cm2s"] for r in results])
        ea, d0 = arrhenius(T, D)
        d300 = d0 * np.exp(-ea / (8.617333262e-5 * 300))
        vol300 = frames[0].get_volume()
        summary |= {"Ea_eV": ea, "D0_cm2s": d0, "D_300K_extrapolated_cm2s": d300,
                    "sigma_300K_extrapolated_Scm": nernst_einstein(d300, pos.shape[1], vol300, 300)}
        fig2, ax2 = plt.subplots(figsize=(4.5, 3.6))
        ax2.semilogy(1000 / T, D, "o")
        tt = np.linspace(T.min(), T.max(), 50)
        ax2.semilogy(1000 / tt, d0 * np.exp(-ea / (8.617333262e-5 * tt)), "-", label=f"$E_a$ = {ea:.2f} eV")
        ax2.set_xlabel("1000 / T (K$^{-1}$)")
        ax2.set_ylabel("$D_{Li}$ (cm$^2$/s)")
        ax2.legend(frameon=False)
        fig2.tight_layout()
        fig2.savefig(run / "arrhenius.png", dpi=200)
        print(f"Ea = {ea:.3f} eV; extrapolated sigma(300 K) = "
              f"{summary['sigma_300K_extrapolated_Scm']:.1e} S/cm (Nernst–Einstein, Haven ratio 1)")
    (run / "results.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
