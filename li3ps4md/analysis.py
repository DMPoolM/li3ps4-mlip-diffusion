"""Li mean-squared displacement, diffusion coefficients and Arrhenius analysis."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import units
from ase.io import read

KB_EV = units.kB  # eV/K
E_CHARGE = 1.602176634e-19  # C
KB_SI = 1.380649e-23  # J/K


def load_positions(traj_path, species: str = "Li"):
    """Return (times_ps, positions[frame, atom, 3] of ``species``, frames).

    Positions must be unwrapped (as written by ``li3ps4md.md.run_nvt``). The centre-of-mass
    drift of the whole cell is removed frame by frame.
    """
    frames = read(traj_path, index=":")
    times = np.array([f.info["time_ps"] for f in frames], dtype=float)
    pos = np.stack([f.get_positions() for f in frames])
    masses = frames[0].get_masses()
    com = (pos * masses[None, :, None]).sum(1) / masses.sum()
    pos = pos - (com - com[0])[:, None, :]
    idx = [i for i, s in enumerate(frames[0].get_chemical_symbols()) if s == species]
    return times - times[0], pos[:, idx, :], frames


def msd(positions: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    """Windowed MSD averaged over atoms and all time origins.

    Returns array [lag, 4]: columns x, y, z, total (Å^2).
    """
    n = positions.shape[0]
    max_lag = max_lag or n // 2
    out = np.zeros((max_lag, 4))
    for lag in range(1, max_lag):
        d = positions[lag:] - positions[:-lag]
        sq = (d ** 2).mean(axis=(0, 1))
        out[lag, :3] = sq
        out[lag, 3] = sq.sum()
    return out


@dataclass
class DiffusionFit:
    D_total: float        # cm^2/s
    D_xyz: np.ndarray     # cm^2/s, per Cartesian direction
    fit_range_ps: tuple


def fit_diffusion(times_ps: np.ndarray, msd_arr: np.ndarray,
                  start_frac: float = 0.2, end_frac: float = 0.8) -> DiffusionFit:
    """Linear fit of MSD(t) in [start_frac, end_frac] of the lag range.

    D = slope / (2 d); 1 Å^2/ps = 1e-4 cm^2/s.
    """
    lags = times_ps[: len(msd_arr)]
    i0 = max(1, int(start_frac * len(lags)))
    i1 = max(i0 + 2, int(end_frac * len(lags)))
    t = lags[i0:i1]
    slopes = [np.polyfit(t, msd_arr[i0:i1, k], 1)[0] for k in range(4)]
    d_xyz = np.array(slopes[:3]) / 2 * 1e-4
    d_tot = slopes[3] / 6 * 1e-4
    return DiffusionFit(d_tot, d_xyz, (float(t[0]), float(t[-1])))


def arrhenius(temperatures_K, D_cm2s):
    """Fit D = D0 exp(-Ea / kB T). Returns (Ea_eV, D0_cm2s)."""
    T = np.asarray(temperatures_K, float)
    D = np.asarray(D_cm2s, float)
    slope, intercept = np.polyfit(1.0 / T, np.log(D), 1)
    return -slope * KB_EV, float(np.exp(intercept))


def nernst_einstein(D_cm2s: float, n_carriers: int, volume_A3: float, T: float,
                    charge: int = 1) -> float:
    """Ionic conductivity (S/cm) assuming a Haven ratio of 1."""
    n = n_carriers / (volume_A3 * 1e-24)  # cm^-3
    return n * (charge * E_CHARGE) ** 2 * D_cm2s / (KB_SI * T)
