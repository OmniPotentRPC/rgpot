"""librgpot_nblib against GROMACS mdrun -rerun on the same inputs.

Each system is written as a .gro frame and a GROMACS topology that says the
same as the nblib topology text; grompp and mdrun -rerun give the potential
energy and the forces of that frame, and a zero-step mdrun gives the virial.
The nblib side is then evaluated on the coordinates GROMACS read back from the
.gro file, so both see the same doubles.

The settings are those nblib's kernels implement: Verlet cut-off scheme,
potential-shifted Lennard-Jones, reaction-field Coulomb with epsilon-rf = 1
and a potential shift, no dispersion correction and an exact pair list.

Forces come from the .trr in double precision. gmx dump prints energies and
the virial with six significant digits, which sets the tolerance of those two.

Environment: RGPOT_NBLIB_ENGINE and RGPOT_GMX (a double-precision gmx).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nblib_support as nb  # noqa: E402

GMX = os.environ.get("RGPOT_GMX", "")
pytestmark = [
    pytest.mark.skipif(not nb.ENGINE, reason="needs RGPOT_NBLIB_ENGINE"),
    pytest.mark.skipif(not GMX, reason="needs RGPOT_GMX"),
]

# Six printed significant digits.
PRINTED = 2.0**-17
# The two programs evaluate the same sums in different orders.
FORCE = 2.0**-30


@pytest.mark.parametrize("make", [nb.argon, nb.water, nb.chain], ids=lambda f: f.__name__)
def test_energy_forces_virial_match_rerun(make, tmp_path):
    system = make()
    reference = nb.gmx_reference(GMX, system, tmp_path)
    positions = reference["positions"] * nb.ANGSTROM_PER_NM
    energy, forces, virial = nb.Potential(system.topology).evaluate(
        positions, system.box
    )
    assert abs(energy * nb.KJ_PER_EV - reference["energy"]) <= PRINTED * abs(
        reference["energy"]
    )
    kj_per_nm = nb.KJ_PER_EV * nb.ANGSTROM_PER_NM
    assert np.abs(forces * kj_per_nm - reference["forces"]).max() <= FORCE * np.abs(
        reference["forces"]
    ).max()
    assert np.abs(virial * nb.KJ_PER_EV - reference["virial"]).max() <= PRINTED * np.abs(
        reference["virial"]
    ).max()
