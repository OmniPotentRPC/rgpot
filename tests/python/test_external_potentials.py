"""Public LAMMPS, GROMACS and OpenMM constructors.

The dimer energy is -0.3203213355 eV. GROMACS and OpenMM store
nanometres and kJ/mol once. The LAMMPS constructor needs the engine
plugin; without it the test says why it skipped. The lammpc overload
needs an RPC build, and its dimer runs against the library that
RGPOT_LAMMPC_ENGINE names, such as lammpc_fake_engine.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import rgpot


DIMER = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
BOX = np.diag([20.0, 20.0, 20.0])
KJ_PER_EV = 96.48533212331002
SCHEMA = Path(__file__).resolve().parents[2] / "CppCore" / "rgpot" / "rpc" / "Potentials.capnp"
LJ_EPS = 0.2
LJ_SIGMA = 2.5
LAMMPC_DIMER = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
# A segment table for one segment of four words, with no segment after it.
TRUNCATED_CONFIG = bytes([0, 0, 0, 0, 4, 0, 0, 0])


def test_gromacs_and_openmm_store_units_once():
    for factory in (rgpot.rgpot_gromacs_potential, rgpot.rgpot_openmm_potential):
        pot = factory(1.0, 8.0, 1.0)
        energy, forces, variance = pot(DIMER, [1, 1], BOX)
        assert f"{energy:.10f}" == "-0.3203213355"
        assert variance == 0.0
        assert forces.shape == (2, 3)
        assert pot.native_u0() == pytest.approx(KJ_PER_EV, rel=1e-12)
        assert pot.native_cutoff() == pytest.approx(0.8)
        assert pot.native_psi() == pytest.approx(0.1)


def test_lammps_missing_plugin_names_the_library():
    with pytest.raises(RuntimeError, match="liblj_engine_missing"):
        rgpot.rgpot_lammps_potential(
            1.0, 8.0, 1.0, engine_path="./liblj_engine_missing.so"
        )


def test_lammps_plugin_dimer():
    path = os.environ.get("RGPOT_LJ_ENGINE", "")
    if not path:
        pytest.skip(
            "liblj_engine.so was not provided (RGPOT_LJ_ENGINE); "
            "the LAMMPS constructor keeps that plugin as its backend"
        )
    pot = rgpot.rgpot_lammps_potential(1.0, 8.0, 1.0, engine_path=path)
    energy, forces, _variance = pot(DIMER, [1, 1], BOX)
    assert f"{energy:.10f}" == "-0.3203213355"
    assert forces.shape == (2, 3)


def test_openmm_engine_dimer():
    path = os.environ.get("RGPOT_OPENMM_ENGINE", "")
    if not path:
        pytest.skip(
            "OpenMM was not found at configure time, so "
            "libopenmm_engine.so was not built"
        )
    pot = rgpot.rgpot_openmm_potential(1.0, 8.0, 1.0, engine_path=path)
    energy, forces, _variance = pot(DIMER, [1, 1], BOX)
    ref_e, ref_f, _ = rgpot.rgpot_gromacs_potential(1.0, 8.0, 1.0)(
        DIMER, [1, 1], BOX
    )
    # OpenMM's expression matches the in-process value to 1e-6 eV and
    # the force to 1e-5 eV/Angstrom, the same window as openmm_lj.
    assert f"{ref_e:.10f}" == "-0.3203213355"
    assert energy == pytest.approx(ref_e, abs=1e-6)
    assert forces.shape == (2, 3)
    assert forces[1, 0] == pytest.approx(ref_f[1, 0], abs=1e-5)


def lammps_config_bytes() -> bytes:
    capnp = pytest.importorskip("capnp")
    schema = capnp.load(str(SCHEMA))
    config = schema.PotentialConfig.new_message()
    lammps = config.init("lammps")
    lammps.pairStyle = "lj/cut 10.0"
    lammps.pairCoeffs = [f"1 1 {LJ_EPS} {LJ_SIGMA}"]
    lammps.typeToAtomicNumber = [29]
    lammps.boundary = "f f f"
    return config.to_bytes()


def skip_without_lammpc_profile(error: Exception) -> None:
    if "RPC-capable" in str(error):
        pytest.skip("this rgpot was built without the RPC stack lammpc needs")


def test_lammpc_config_bytes_round_trip():
    path = os.environ.get("RGPOT_LAMMPC_ENGINE", "")
    if not path:
        pytest.skip("no lammpc library was provided (RGPOT_LAMMPC_ENGINE)")
    try:
        pot = rgpot.rgpot_lammps_potential(lammps_config_bytes(), engine_path=path)
    except RuntimeError as error:
        skip_without_lammpc_profile(error)
        raise
    assert b"\x00" in lammps_config_bytes()
    energy, forces, variance = pot(LAMMPC_DIMER, [29, 29], BOX)
    sr6 = (LJ_SIGMA / 3.0) ** 6
    assert energy == pytest.approx(4.0 * LJ_EPS * (sr6 * sr6 - sr6), abs=1e-12)
    f = 24.0 * LJ_EPS * (2.0 * sr6 * sr6 - sr6) / 3.0
    np.testing.assert_allclose(forces, [[-f, 0.0, 0.0], [f, 0.0, 0.0]], atol=1e-12)
    assert variance == 0.0


@pytest.mark.parametrize(
    ("blob", "match"),
    [
        pytest.param(bytes(13), "word-aligned", id="misaligned"),
        pytest.param(TRUNCATED_CONFIG, "not a flat PotentialConfig", id="truncated"),
    ],
)
def test_lammpc_bad_config_bytes_raise_value_error(blob, match):
    with pytest.raises((ValueError, RuntimeError)) as info:
        rgpot.rgpot_lammps_potential(blob)
    if info.type is RuntimeError:
        skip_without_lammpc_profile(info.value)
    assert info.type is ValueError
    assert match in str(info.value)


@pytest.mark.parametrize(
    "blob",
    [
        pytest.param(bytes(16), id="leading-nul"),
        pytest.param(b"abc\x00" + bytes(12), id="nul-inside-first-word"),
    ],
)
def test_lammpc_config_bytes_keep_embedded_nuls(blob):
    # Cut at the first NUL, these would be empty or misaligned instead.
    with pytest.raises((ValueError, RuntimeError)) as info:
        rgpot.rgpot_lammps_potential(blob)
    if info.type is RuntimeError:
        skip_without_lammpc_profile(info.value)
    assert info.type is ValueError
    message = str(info.value)
    assert "PotentialConfig" in message
    assert "required" not in message
    assert "word-aligned" not in message


def test_lammpc_explicit_missing_engine_names_the_path():
    missing = "./liblammpc_missing_for_test.so"
    with pytest.raises(RuntimeError) as info:
        rgpot.rgpot_lammps_potential(lammps_config_bytes(), engine_path=missing)
    skip_without_lammpc_profile(info.value)
    assert missing in str(info.value)
