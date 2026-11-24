# MIT License
# Copyright 2023--present rgpot developers

"""QSCPot against generated references and the independent oracle.

* ``eon_pure.dat``: energies and forces of the real eOn QSC class from the
  pinned source snapshot, pure elements only (eOn reuses pair parameters
  across species there, so it is not a reference for mixtures).
* ``oracle_mixtures.dat``: scripts/qsc_reference/qsc_oracle.py, 50-digit
  arithmetic on Eqs. (1a)-(1c) and (2) of Qi, Cagin, Kimura and Goddard,
  Phys. Rev. B 59, 3527 (1999).

The oracle is also run live here, so a stale or edited data file fails.
"""

from __future__ import annotations

import sys
from decimal import Context, Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

import rgpot

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts" / "qsc_reference"))

import qsc_common as common  # noqa: E402
import qsc_oracle as oracle  # noqa: E402

# pytest-run-parallel on a free-threaded build (3.14t) reports a TSan data
# race inside CPython's _decimal while this module runs. Each oracle call
# builds its own 50-digit context, so the values do not depend on the
# thread, but the race is in the interpreter. The marker runs these tests
# on one thread. It does not skip them.
pytestmark = pytest.mark.thread_unsafe

DATA = _ROOT / "CppCore" / "tests" / "data" / "qsc"
REL_TOL = 1.0e-12

# Reference rows are 50-digit decimals. Parse them in that precision so the
# stored values do not follow the importing thread's context.
with localcontext(Context(prec=50)):
    _, EON_CASES = common.read_cases(DATA / "eon_pure.dat")
    _, MIX_CASES = common.read_cases(DATA / "oracle_mixtures.dat")

# Live oracle runs are 50-digit arithmetic; keep to the cheap cases.
LIVE_MIXTURES = [c for c in MIX_CASES if len(c.species) < 50]
LIVE_PURE = [c for c in EON_CASES if len(c.species) < 50]


def _evaluate(case):
    pot = rgpot.QSCPot(
        cutoff=case.cutoff, periodic=case.periodic, stress=case.stress is not None
    )
    pos = np.array(case.pos, dtype=np.float64)
    box = np.array(case.cell, dtype=np.float64)
    return pot.evaluate(pos, list(case.species), box)


def _relative_max(got, want):
    want = np.array([float(x) for x in np.ravel(want)])
    scale = np.max(np.abs(want))
    return float(np.max(np.abs(np.ravel(got) - want)) / scale)


def _check(case, out):
    assert abs(out["energy"] - float(case.energy)) <= REL_TOL * abs(float(case.energy))
    assert _relative_max(out["forces"], case.forces) <= REL_TOL
    if case.stress is not None:
        s = out["stress"]
        voigt = [s[a, b] for a, b in oracle.VOIGT]
        assert _relative_max(voigt, case.stress) <= REL_TOL


def test_data_files_are_complete():
    assert len(EON_CASES) == 20
    assert len(MIX_CASES) == 9
    for case in EON_CASES:
        assert len(set(case.species)) == 1
    for case in MIX_CASES:
        assert len(set(case.species)) >= 2


@pytest.mark.parametrize("case", EON_CASES, ids=lambda c: c.name)
def test_pure_elements_match_real_eon(case):
    _check(case, _evaluate(case))


@pytest.mark.parametrize("case", MIX_CASES, ids=lambda c: c.name)
def test_mixtures_match_the_oracle_data(case):
    _check(case, _evaluate(case))


def test_oracle_gradients_match_high_precision_differences():
    assert oracle.selfcheck(verbose=False)


@pytest.mark.parametrize("case", LIVE_MIXTURES, ids=lambda c: c.name)
def test_mixture_data_is_what_the_oracle_computes(case):
    pos = [[Decimal(x) for x in p] for p in case.pos]
    cell = [[Decimal(x) for x in row] for row in case.cell]
    energy, forces, _ = oracle.evaluate(
        case.species, pos, cell, case.periodic, case.cutoff
    )
    assert abs(energy - case.energy) <= Decimal("1e-17") * abs(energy)
    scale = max(abs(x) for row in case.forces for x in row)
    for got, want in zip(forces, case.forces):
        for g, w in zip(got, want):
            assert abs(g - w) <= Decimal("1e-17") * scale


@pytest.mark.parametrize("case", LIVE_PURE, ids=lambda c: c.name)
def test_real_eon_pure_data_agrees_with_the_published_equations(case):
    pos = [[Decimal(x) for x in p] for p in case.pos]
    cell = [[Decimal(x) for x in row] for row in case.cell]
    energy, forces, _ = oracle.evaluate(
        case.species, pos, cell, case.periodic, case.cutoff
    )
    assert abs(energy - case.energy) <= Decimal("1e-14") * abs(energy)
    scale = max(abs(x) for row in forces for x in row)
    for got, want in zip(forces, case.forces):
        for g, w in zip(got, want):
            assert abs(g - w) <= Decimal("1e-13") * scale


def test_density_uses_the_mixed_pair_parameters_of_eq_1c():
    """A Cu-Ag dimer follows (a_ij / r)^m_ij, not the neighbour's own (a_j / r)^m_j.

    The two density rules differ by 0.19 eV at 2.65 Angstrom, far above every
    tolerance in this file, so the rule in use is pinned.
    """
    pos = [[Decimal(0)] * 3, [Decimal(2.65), Decimal(0), Decimal(0)]]
    cell = [[Decimal(40) if i == j else Decimal(0) for j in range(3)] for i in range(3)]
    mixed = oracle.energy_only([29, 47], pos, cell, False, 10.0, density="mixed")
    source = oracle.energy_only([29, 47], pos, cell, False, 10.0, density="source")
    assert abs(mixed - source) > Decimal("0.1")
    pot = rgpot.QSCPot(cutoff=10.0, periodic=False)
    energy, _, _ = pot(
        np.array([[0.0, 0.0, 0.0], [2.65, 0.0, 0.0]]), [29, 47], np.eye(3) * 40.0
    )
    assert abs(energy - float(mixed)) <= 1e-12 * abs(float(mixed))
    assert abs(energy - float(source)) > 0.1
