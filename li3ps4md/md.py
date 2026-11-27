"""Relaxation and NVT molecular dynamics with a universal MLIP (MACE-MP) in ASE."""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
from ase import Atoms, units
from ase.filters import FrechetCellFilter
from ase.io import write
from ase.md.bussi import Bussi
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.optimize import FIRE


def mace_calculator(model: str = "small", device: str = "cpu", dtype: str = "float32"):
    """MACE-MP foundation model (MIT licence). ``model``: small / medium / large or a path."""
    from mace.calculators import mace_mp

    return mace_mp(model=model, device=device, default_dtype=dtype)


def relax(atoms: Atoms, calc, fmax: float = 0.05, steps: int = 300,
          log: str | None = "-", relax_cell: bool = False) -> Atoms:
    """Relax atomic positions with FIRE; with ``relax_cell`` also the cell (zero pressure).

    By default the experimental lattice is kept, so that MD runs at the measured density
    rather than at the MLIP's equilibrium volume.
    """
    atoms = atoms.copy()
    atoms.calc = calc
    target = FrechetCellFilter(atoms) if relax_cell else atoms
    opt = FIRE(target, logfile=log)
    opt.run(fmax=fmax, steps=steps)
    return atoms


def run_nvt(atoms: Atoms, calc, temperature_K: float, steps: int, out_dir: str | Path,
            timestep_fs: float = 2.0, taut_fs: float = 100.0, interval: int = 50,
            equil_steps: int = 0, seed: int = 0) -> Path:
    """Bussi (stochastic velocity rescaling) NVT MD.

    The Bussi thermostat keeps the dynamics close to Newtonian, which matters for
    transport properties. Positions are written *unwrapped* (ASE does not wrap them during
    MD) every ``interval`` steps to ``traj_<T>K.extxyz``; the thermo log goes to CSV.
    Frames from the first ``equil_steps`` are not written.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    atoms = atoms.copy()
    atoms.calc = calc
    rng = np.random.default_rng(seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=rng)
    Stationary(atoms)

    dyn = Bussi(atoms, timestep_fs * units.fs, temperature_K=temperature_K,
                taut=taut_fs * units.fs, rng=rng)
    tag = f"{int(round(temperature_K))}K"
    traj = out / f"traj_{tag}.extxyz"
    if traj.exists():
        traj.unlink()
    log_path = out / f"thermo_{tag}.csv"
    t0 = time.time()

    with open(log_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "time_ps", "T_K", "E_pot_eV", "E_tot_eV", "wall_s"])

        def record():
            step = dyn.nsteps
            if step % interval:
                return
            epot = atoms.get_potential_energy()
            ekin = atoms.get_kinetic_energy()
            w.writerow([step, step * timestep_fs / 1000, atoms.get_temperature(),
                        epot, epot + ekin, round(time.time() - t0, 1)])
            fh.flush()
            if step >= equil_steps:
                frame = atoms.copy()
                frame.calc = None
                frame.info.update(step=step, time_ps=step * timestep_fs / 1000,
                                  temperature_K=temperature_K)
                write(traj, frame, format="extxyz", append=True)

        dyn.attach(record, interval=1)  # also called once at step 0 by run()
        dyn.run(steps)
    return traj
