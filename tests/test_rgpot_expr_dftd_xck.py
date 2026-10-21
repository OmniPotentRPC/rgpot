"""Published-wheel surface: ExprPot, D3Pot, D4Pot, XcKernel."""

from __future__ import annotations

import math

import numpy as np
import pytest

rgpot = pytest.importorskip("rgpot")


def two_atom_lj():
    positions = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=np.float64)
    atom_types = np.array([18, 18], dtype=np.int32)
    box = np.eye(3, dtype=np.float64) * 40.0
    return positions, atom_types, box


def water():
    positions = np.array(
        [
            [0.0, 0.0, 0.11779],
            [0.0, 0.75545, -0.47116],
            [0.0, -0.75545, -0.47116],
        ],
        dtype=np.float64,
    )
    atom_types = np.array([8, 1, 1], dtype=np.int32)
    box = np.eye(3, dtype=np.float64) * 100.0
    return positions, atom_types, box


@pytest.mark.skipif(not rgpot.has_expr, reason="built without -Dwith_expr")
def test_exprpot_identity_matches_lj():
    pos, z, box = two_atom_lj()
    e_lj, f_lj, _ = rgpot.LJPot()(pos, z, box)
    pot = rgpot.ExprPot("lj", {"lj": "lj"})
    e, f, _ = pot(pos, z, box)
    assert abs(e - e_lj) < 1e-12
    assert np.allclose(f, f_lj)


@pytest.mark.skipif(not rgpot.has_expr, reason="built without -Dwith_expr")
def test_exprpot_half_lj_plus_morse_finite():
    pos, z, box = two_atom_lj()
    pot = rgpot.ExprPot("0.5*lj + morse", {"lj": "lj", "morse": "morse"})
    e, f, _ = pot(pos, z, box)
    assert math.isfinite(float(e))
    assert f.shape == (2, 3)
    assert np.all(np.isfinite(f))


@pytest.mark.skipif(not rgpot.has_dftd3, reason="built without -Dwith_dftd3")
def test_d3pot_water_finite():
    pos, z, box = water()
    e, f, _ = rgpot.D3Pot()(pos, z, box)
    assert math.isfinite(float(e))
    assert f.shape == (3, 3)
    assert np.all(np.isfinite(f))
    e_off, _, _ = rgpot.D3Pot(atm=False)(pos, z, box)
    assert e != e_off


@pytest.mark.skipif(not rgpot.has_dftd4, reason="built without -Dwith_dftd4")
def test_d4pot_water_finite():
    pos, z, box = water()
    e, f, _ = rgpot.D4Pot()(pos, z, box)
    assert math.isfinite(float(e))
    assert f.shape == (3, 3)
    assert np.all(np.isfinite(f))


@pytest.mark.skipif(
    not (rgpot.has_expr and rgpot.has_dftd3),
    reason="needs ExprPot and D3Pot",
)
def test_exprpot_lj_plus_d3():
    pos, z, box = water()
    pot = rgpot.ExprPot(
        "0.5*lj + d3",
        {"lj": "lj", "d3": {"kind": "d3", "functional": "pbe", "atm": True}},
    )
    e, f, _ = pot(pos, z, box)
    assert math.isfinite(float(e))
    assert f.shape == (3, 3)


@pytest.mark.skipif(not rgpot.has_xckernel, reason="built without -Dwith_xckernel")
def test_xckernel_catalog_and_lda_contract():
    names = rgpot.XcKernel.catalog()
    assert "xck_lda_r_o1" in names
    k = rgpot.XcKernel("xck_lda_r_o1")
    nbf, npts = 2, 4
    chi = np.zeros((nbf, npts), dtype=np.float64)
    chi[0, :] = 0.1
    chi[1, :] = 0.2
    scal = {
        name: np.ones(npts, dtype=np.float64) * 0.01 for name in k.scal_names()
    }
    out = k.contract(chi, None, scal)
    assert out.shape == (nbf, nbf)
    assert np.all(np.isfinite(out))
    # dchi defaults to None, so an LDA call can omit it.
    omitted = k.contract(chi, scal=scal)
    np.testing.assert_array_equal(omitted, out)
