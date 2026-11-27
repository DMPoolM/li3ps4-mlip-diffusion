import numpy as np
import pytest

from li3ps4md.analysis import arrhenius, fit_diffusion, msd, nernst_einstein
from li3ps4md.structure import beta_li3ps4, check_structure


def test_msd_recovers_random_walk_diffusion():
    rng = np.random.default_rng(0)
    D = 0.5           # Å^2/ps per direction
    dt = 0.1          # ps
    steps = rng.normal(scale=np.sqrt(2 * D * dt), size=(4000, 64, 3))
    pos = np.cumsum(steps, axis=0)
    times = np.arange(4000) * dt
    fit = fit_diffusion(times, msd(pos, max_lag=400))
    assert fit.D_total == pytest.approx(D * 1e-4, rel=0.05)
    np.testing.assert_allclose(fit.D_xyz, D * 1e-4, rtol=0.08)


def test_arrhenius_recovers_activation_energy():
    T = np.array([600.0, 700.0, 800.0, 900.0])
    ea, d0 = 0.25, 1e-3
    D = d0 * np.exp(-ea / (8.617333262e-5 * T))
    ea_fit, d0_fit = arrhenius(T, D)
    assert ea_fit == pytest.approx(ea, rel=1e-6)
    assert d0_fit == pytest.approx(d0, rel=1e-6)


def test_nernst_einstein_order_of_magnitude():
    # 12 Li in 645 Å^3, D = 1e-6 cm^2/s at 600 K -> ~5e-2 S/cm
    s = nernst_einstein(1e-6, 12, 645.0, 600.0)
    assert 1e-2 < s < 1e-1


def test_beta_li3ps4_structure():
    atoms = beta_li3ps4()
    assert atoms.get_chemical_formula() == "Li12P4S16"
    assert check_structure(atoms) > 1.9
