"""Build an ordered beta-Li3PS4 (Pnma) starting structure.

beta-Li3PS4 is Li-disordered: Li partially occupies 8d, 4b and 4c sites. MD needs an
ordered cell, so we use the common DFT model with Li on the full 8d site and the 4b
site (12 Li per cell = Li12P4S16) and the 4c site left empty.

Fractional coordinates follow the beta-Li3PS4 model of Stöffler et al. as tabulated in
the supporting information of J. Mater. Chem. A 2020, DOI 10.1039/D0TA02805C (Table S3).
The experimental high-temperature refinement by Homma et al., Solid State Ionics 182,
53 (2011) is available as COD entry 1570310; you can start from that CIF instead with
``--cif``. The cell is relaxed with the MLIP before MD in either case.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read
from ase.spacegroup import crystal

# Stöffler et al., via DOI 10.1039/D0TA02805C Table S3; Li2 (4b) promoted to full occupancy
_BASIS = [
    ("Li", (0.318, 0.017, 0.139)),   # Li1, 8d
    ("Li", (0.0, 0.0, 0.5)),         # Li2, 4b
    ("P", (0.088, 0.25, 0.167)),     # 4c
    ("S", (0.155, 0.0402, 0.267)),   # S1, 8d
    ("S", (0.941, 0.25, 0.254)),     # S2, 4c
    ("S", (0.101, 0.25, 0.801)),     # S3, 4c
]
_CELL = (12.993, 8.0458, 6.1377, 90.0, 90.0, 90.0)


def beta_li3ps4(cif: str | Path | None = None) -> Atoms:
    """Return one ordered beta-Li3PS4 unit cell (32 atoms)."""
    if cif is not None:
        atoms = read(cif)
        if np.any(atoms.arrays.get("occupancy", 1.0) != 1.0):  # pragma: no cover
            raise ValueError("CIF has partial occupancies; order it first")
        return atoms
    symbols, basis = zip(*_BASIS)
    atoms = crystal(symbols, basis=basis, spacegroup=62, cellpar=_CELL,
                    onduplicates="error")
    return atoms


def check_structure(atoms: Atoms, min_dist: float = 1.7) -> float:
    """Return the shortest interatomic distance; raise if atoms overlap."""
    d = atoms.get_all_distances(mic=True)
    np.fill_diagonal(d, np.inf)
    dmin = float(d.min())
    if dmin < min_dist:
        raise ValueError(f"atoms closer than {min_dist} Å (min {dmin:.2f} Å)")
    return dmin
