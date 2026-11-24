# Copyright 2023--present rgpot developers

"""Python bindings for ``rgpot.MdSession``."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import rgpot
from repo_data import source_root


_OPENMM_DIMER_XML = """<?xml version="1.0" ?>
<System openmmVersion="8.6.1" type="System" version="1">
  <PeriodicBoxVectors>
    <A x="3" y="0" z="0"/>
    <B x="0" y="3" z="0"/>
    <C x="0" y="0" z="3"/>
  </PeriodicBoxVectors>
  <Particles>
    <Particle mass="12"/>
    <Particle mass="16"/>
  </Particles>
  <Constraints/>
  <Forces>
    <Force forceGroup="0" name="HarmonicBondForce" type="HarmonicBondForce" usesPeriodic="0" version="2">
      <Bonds>
        <Bond d=".15" k="1000" p1="0" p2="1"/>
      </Bonds>
    </Force>
  </Forces>
</System>
"""


def _compile_fake_engine(tmp_path: Path) -> Path:
    root = source_root()
    if root is None:
        pytest.skip("rgpot source tree is not available")
    source = root / "CppCore" / "tests" / "md_session_fake_engine.c"
    if not source.is_file():
        pytest.skip("md_session_fake_engine.c not found")
    cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if not cc:
        pytest.skip("C compiler not available")
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    out = tmp_path / f"libmd_session_fake_py{suffix}"
    flags = ["-dynamiclib"] if sys.platform == "darwin" else ["-shared", "-fPIC"]
    subprocess.run(
        [
            cc,
            *flags,
            f"-I{root / 'CppCore'}",
            "-DFIXTURE_UNLOAD_POLICY=RGPOT_ENGINE_UNLOAD_SAFE",
            str(source),
            "-o",
            str(out),
        ],
        check=True,
    )
    return out


def test_md_session_missing_plugin_raises():
    with pytest.raises(RuntimeError, match="libmissing_md_engine"):
        rgpot.MdSession(
            "missing",
            engine_path="./libmissing_md_engine_does_not_exist.so",
        )


def test_md_session_fake_engine_lifecycle_and_validation(tmp_path: Path):
    lib = _compile_fake_engine(tmp_path)

    with rgpot.MdSession("fake", spec=b"", engine_path=str(lib), options="current") as md:
        assert md.n_atoms == 2
        masses = md.masses()
        assert masses.shape == (2,)
        assert masses.dtype == np.float64
        assert masses[0] > 0.0

        box = np.diag([12.0, 13.0, 14.0])
        md.set_periodic_box(box)
        np.testing.assert_allclose(md.periodic_box(), box)

        pos = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
        vel = np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
        md.set_positions(pos)
        md.set_velocities(vel)
        md.initialize()

        plane_pt = np.zeros((2, 3))
        plane_n = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        wall_pts = np.array([[[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]]])
        wall_ns = np.array([[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]])

        hits = md.step_langevin(
            0.02585,
            1.0,
            0.001,
            plane_pt,
            plane_n,
            steps=3,
            seed=7,
            wall_points=wall_pts,
            wall_normals=wall_ns,
        )
        assert hits == 0

        res = md.run_langevin(
            k_t=0.02585,
            friction=1.0,
            time_step=0.001,
            steps=4,
            seed=9,
            plane_point=plane_pt,
            plane_normal=plane_n,
            wall_points=wall_pts,
            wall_normals=wall_ns,
            wall_mode="reject",
            record_constraint_force=True,
        )
        assert res["wall_hits"] == 0
        assert res["constraint_force"].shape == (4,)

        res_plain = md.run_langevin(steps=2)
        assert res_plain["wall_hits"] == 0
        assert res_plain["constraint_force"] is None

        md.step_nve(0.001, steps=2)
        md.reseed_velocities(0.02585, seed=11, plane_normal=plane_n)

        st = md.state()
        assert set(st) == {
            "positions",
            "velocities",
            "forces",
            "potential_energy",
            "kinetic_energy",
        }
        assert st["positions"].shape == (2, 3)
        assert st["velocities"].shape == (2, 3)
        assert st["forces"].shape == (2, 3)
        assert md.positions().shape == (2, 3)
        assert md.velocities().shape == (2, 3)
        forces, pe = md.forces()
        assert forces.shape == (2, 3)
        assert isinstance(pe, float)
        e_pe, e_ke = md.energy()
        assert isinstance(e_pe, float)
        assert isinstance(e_ke, float)

        # Validation errors: dtype/contiguity -> TypeError, shape/mode -> ValueError.
        with pytest.raises(TypeError):
            md.set_positions(np.zeros((2, 3), dtype=np.float32))
        with pytest.raises(TypeError):
            md.set_positions(np.asfortranarray(np.zeros((2, 3))))
        with pytest.raises(ValueError, match="positions"):
            md.set_positions(np.zeros((3, 3)))
        with pytest.raises(ValueError, match="velocities"):
            md.set_velocities(np.zeros((2, 2)))
        with pytest.raises(ValueError, match="box"):
            md.set_periodic_box(np.eye(2))
        with pytest.raises(ValueError, match="wall_mode"):
            md.run_langevin(wall_mode="bounce")
        with pytest.raises(ValueError, match="plane_point"):
            md.run_langevin(plane_point=plane_pt)
        with pytest.raises(ValueError, match="wall_points"):
            md.run_langevin(wall_points=wall_pts)

    # Context exit closed the session; repeated close is a no-op, use raises.
    md.close()
    with pytest.raises(RuntimeError, match="closed"):
        _ = md.n_atoms
    with pytest.raises(RuntimeError, match="closed"):
        md.positions()


def test_md_session_openmm():
    path = os.environ.get("RGPOT_OPENMM_ENGINE", "")
    if not path:
        pytest.skip("RGPOT_OPENMM_ENGINE is not set")

    with rgpot.MdSession(
        "openmm",
        spec=_OPENMM_DIMER_XML.encode("utf-8"),
        engine_path=path,
        options="Reference",
    ) as md:
        assert md.n_atoms == 2
        np.testing.assert_allclose(md.masses(), [12.0, 16.0])

        box = np.diag([30.0, 30.0, 30.0])
        md.set_periodic_box(box)
        np.testing.assert_allclose(md.periodic_box(), box)

        pos = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        md.set_positions(pos)
        md.set_velocities(np.zeros((2, 3)))
        md.initialize()

        forces, pe = md.forces()
        assert forces.shape == (2, 3)
        assert pe > 0.0
        assert forces[0, 0] > 0.0
        assert forces[1, 0] == pytest.approx(-forces[0, 0], rel=1e-12)

        md.reseed_velocities(0.02585, seed=42)
        pe_before, ke_before = md.energy()
        assert ke_before > 0.0

        md.step_nve(0.0005, steps=10)
        pe_after, ke_after = md.energy()
        assert (pe_after + ke_after) == pytest.approx(pe_before + ke_before, rel=1e-3)

        plane_pt = pos.copy()
        plane_n = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]) / np.sqrt(2.0)
        md.set_positions(plane_pt)
        md.reseed_velocities(0.02585, seed=5, plane_normal=plane_n)
        out = md.run_langevin(
            k_t=0.02585,
            friction=1.0,
            time_step=0.0005,
            steps=5,
            seed=5,
            plane_point=plane_pt,
            plane_normal=plane_n,
            record_constraint_force=True,
        )
        assert out["wall_hits"] == 0
        assert out["constraint_force"].shape == (5,)
        assert np.all(np.isfinite(out["constraint_force"]))
