#!/usr/bin/env python
"""Relax beta-Li3PS4 with MACE-MP, then run NVT MD at one or more temperatures.

Example (GPU, production):
    python scripts/run_md.py --temps 600 700 800 900 --steps 50000 --device cuda
Example (CPU smoke test, a few ps):
    python scripts/run_md.py --temps 900 --steps 2000 --supercell 1 1 2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ase.io import write  # noqa: E402

from li3ps4md.md import mace_calculator, relax, run_nvt  # noqa: E402
from li3ps4md.structure import beta_li3ps4, check_structure  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--temps", type=float, nargs="+", default=[600, 700, 800, 900])
    p.add_argument("--steps", type=int, default=50000, help="MD steps per temperature")
    p.add_argument("--equil-steps", type=int, default=2500, help="steps discarded before writing frames")
    p.add_argument("--timestep", type=float, default=2.0, help="fs")
    p.add_argument("--interval", type=int, default=50, help="write a frame every N steps")
    p.add_argument("--supercell", type=int, nargs=3, default=[1, 2, 2])
    p.add_argument("--model", default="small", help="MACE-MP model: small / medium / large / path")
    p.add_argument("--device", default="cpu")
    p.add_argument("--relax-cell", action="store_true",
                   help="also relax the lattice (default: keep the experimental cell)")
    p.add_argument("--cif", default=None, help="start from a CIF instead of the built-in model")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/beta_li3ps4")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    calc = mace_calculator(a.model, a.device)

    cell = beta_li3ps4(a.cif)
    check_structure(cell)
    relaxed_cell = relax(cell, calc, log=str(out / "relax.log"), relax_cell=a.relax_cell)
    atoms = relaxed_cell.repeat(a.supercell)
    check_structure(atoms)
    write(out / "relaxed_supercell.extxyz", atoms)

    meta = vars(a) | {"n_atoms": len(atoms),
                      "relaxed_cellpar": [round(float(x), 4) for x in relaxed_cell.cell.cellpar()]}
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"relaxed cell: {meta['relaxed_cellpar']}  ->  {len(atoms)} atoms")

    for T in a.temps:
        traj = run_nvt(atoms, calc, T, a.steps, out, timestep_fs=a.timestep,
                       interval=a.interval, equil_steps=a.equil_steps, seed=a.seed)
        print(f"{T:.0f} K done -> {traj}")


if __name__ == "__main__":
    main()
