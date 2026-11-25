# MIT License
# Copyright 2023--present rgpot developers

"""SCC rescue flag and switch of the dlopen GFN-xTB frontend.

Needs an engine library: set RGPOT_XTB_ENGINE to libxtb_engine.so.
"""

import os
import subprocess
import sys
import threading

import numpy as np
import pytest

rgpot = pytest.importorskip("rgpot")

ENGINE = os.environ.get("RGPOT_XTB_ENGINE", "")
pytestmark = pytest.mark.skipif(not ENGINE, reason="RGPOT_XTB_ENGINE is not set")

# A dissociated ammonia tetramer whose GFN2 SCC does not converge at 300 K,
# even in 500 iterations; the same geometry as CppCore/tests/XTBPotTest.cc.
AMMONIA_Z = [7, 1, 1, 1] * 4
AMMONIA_XYZ = np.array(
    [
        [-1.66532428, 0.54889027, -2.03940010],
        [0.11165024, 2.04946110, 1.43904553],
        [-0.42884806, 2.72524649, 0.30972045],
        [1.43338889, 2.19082068, 0.90531718],
        [0.53411760, 1.59058476, -4.48817656],
        [-0.62302497, 2.69900828, -3.26741719],
        [1.34429373, 3.32407404, -1.22950979],
        [-0.35527480, 1.88804157, -4.87298828],
        [1.47996713, 0.55484625, 3.71752121],
        [1.28956128, -0.72444169, 2.95951158],
        [-0.20031872, 0.12475055, 3.35919321],
        [1.70295493, -0.38386526, 4.03334277],
        [-2.71341888, -5.98849237, 1.22333256],
        [-1.72758892, -6.21046903, 1.63826395],
        [-1.69113306, -6.08127583, -0.32892162],
        [-2.20633610, -6.61433582, 0.61469530],
    ],
    dtype=np.float64,
)
WATER_Z = [8, 1, 1]
WATER_XYZ = np.array(
    [[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]],
    dtype=np.float64,
)
BOX = np.diag([60.0, 60.0, 60.0])


def make(scc_rescue=True):
    return rgpot.XTBDlopen(
        engine_path=ENGINE,
        accuracy=0.01,
        electronic_temperature=300.0,
        max_iterations=500,
        scc_rescue=scc_rescue,
    )


def test_rescued_scc_is_flagged():
    pot = make()
    assert pot.reports_eval_flags
    assert pot.scc_rescue
    energy, forces, _ = pot(AMMONIA_XYZ, AMMONIA_Z, BOX)
    assert np.isfinite(energy)
    assert forces.shape == (16, 3)
    assert pot.last_scc_rescued
    assert pot.last_eval_flags & rgpot.EVAL_SCC_RESCUED

    pot(WATER_XYZ, WATER_Z, BOX)
    assert not pot.last_scc_rescued
    assert pot.last_eval_flags == 0


def test_rescue_off_raises_scc_not_converged():
    pot = make(scc_rescue=False)
    assert not pot.scc_rescue
    with pytest.raises(rgpot.SccNotConverged, match="SCC rescue disabled"):
        pot(AMMONIA_XYZ, AMMONIA_Z, BOX)
    assert pot.last_eval_flags == 0
    energy, _, _ = pot(WATER_XYZ, WATER_Z, BOX)
    assert np.isfinite(energy)

    pot.scc_rescue = True
    pot(AMMONIA_XYZ, AMMONIA_Z, BOX)
    assert pot.last_scc_rescued


def test_engine_names_its_revision():
    revision = make().engine_revision
    assert isinstance(revision, str)
    assert revision.isalnum()


def share_one_handle(rounds=20):
    """Two threads call one handle and read its flags and switch."""
    pot = make()
    errors = []

    def work(first):
        try:
            for i in range(rounds):
                if i == 0 and first:
                    pot(AMMONIA_XYZ, AMMONIA_Z, BOX)
                else:
                    pot(WATER_XYZ, WATER_Z, BOX)
                for _ in range(200):
                    _ = (pot.last_eval_flags, pot.last_scc_rescued)
                    pot.scc_rescue = pot.scc_rescue
        except BaseException as err:  # noqa: BLE001
            errors.append(err)

    threads = [threading.Thread(target=work, args=(k == 0,)) for k in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]


def test_two_threads_share_one_handle():
    # A thread that waited for the handle while holding the GIL used to
    # deadlock the call that needed the GIL back. A hung interpreter cannot
    # be joined from inside, so the threads run in a child process.
    child = subprocess.run(
        [sys.executable, __file__, "share"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert child.returncode == 0, child.stdout + child.stderr


if __name__ == "__main__" and sys.argv[1:] == ["share"]:
    share_one_handle()
