"""librgpot_nblib through the rgpot MD session ABI and the potential ABI.

Every session test drives the library through the engine-generic binding
(python/rgpot/mdsession.py), so the same assertions hold for any engine that
exports <prefix>_md_*; only the fixtures here are nblib topologies.

Environment: RGPOT_NBLIB_ENGINE (librgpot_nblib), optionally RGPOT_EVAL.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nblib_support as nb  # noqa: E402

pytestmark = pytest.mark.skipif(not nb.ENGINE, reason="needs RGPOT_NBLIB_ENGINE")

EPS = nb.EPS
# Roundoff of a force sum over a few hundred terms, and of a step chain.
SUM = 2.0**-30
ROUNDTRIP = 2.0**-28
# Relative energy fluctuation of potential-shifted Lennard-Jones, whose force
# is cut at rc, over a few hundred steps.
FLUCTUATION = 2.0**-10

# Relative energy fluctuation of a velocity Verlet run at the given step: the
# stiff bonds of the molecular systems set it, and it falls as the square of
# the step.
SYSTEMS = {
    "argon": (nb.argon, 0.004, 60.0, 2.0**-10),
    "water": (nb.water, 0.00025, 100.0, 2.0**-8),
    "chain": (nb.chain, 0.00025, 100.0, 2.0**-8),
}


@pytest.fixture(scope="module")
def engine():
    return nb.engine()


RELAXED = {}


def relaxed(engine, system, temperature):
    """Positions and velocities of a molecular system after a thermostatted
    relaxation of its lattice start, so energy checks begin near equilibrium."""
    if system.name not in RELAXED:
        md = engine.create(system.topology, SYSTEMS[system.name][1])
        md.set_box(system.box)
        md.set_positions(system.positions)
        md.set_velocities(nb.maxwell(system.masses, temperature, 7))
        md.set_seed(1)
        kt = nb.BOLTZMANN_EV * temperature
        md.step_projected(4000, kt_ev=kt, friction_per_ps=40.0)
        state = md.state()
        RELAXED[system.name] = (state.positions, state.velocities)
    return RELAXED[system.name]


def started(engine, system, dt, temperature, seed=7):
    md = engine.create(system.topology, dt)
    md.set_box(system.box)
    if system.name == "argon":
        md.set_positions(system.positions)
        md.set_velocities(nb.maxwell(system.masses, temperature, seed))
    else:
        positions, velocities = relaxed(engine, system, temperature)
        md.set_positions(positions)
        md.set_velocities(velocities)
    return md


def total(state):
    return state.potential_ev + state.kinetic_ev


# --------------------------------------------------------------------- probes


def test_probes(engine):
    wanted = (
        nb.mdsession.FEATURE_NVE
        | nb.mdsession.FEATURE_PROJECTED
        | nb.mdsession.FEATURE_HOOK
        | nb.mdsession.FEATURE_BOUNDARY
        | nb.mdsession.FEATURE_VIRIAL
    )
    assert engine.features & wanted == wanted
    assert engine.name == "nblib"
    assert engine.integrator == "leap-frog"
    for call in ("set_time_step", "set_seed", "initialize", "step_projected"):
        assert engine.has(call)
    lib = nb.Potential(nb.argon().topology).lib
    lib.rgpot_nblib_real_bits.restype = nb.C.c_int
    lib.rgpot_nblib_gromacs_version.restype = nb.C.c_char_p
    assert lib.rgpot_nblib_real_bits() == 64
    assert lib.rgpot_nblib_gromacs_version().decode().startswith("2025.")


@pytest.mark.parametrize(
    "text, fragment",
    [
        ("nblib 2\n", "format version"),
        ("nblib 1\natomtype A 1.0 0 0\nmolecule m 1\n atom a B 0\nend\ncutoff 1\n",
         "undefined type"),
        ("nblib 1\ncutoff 1\nbogus\n", "unknown directive"),
        ("nblib 1\ncutoff 1\natomtype A 1.0 0 0\nmolecule m 1\n atom a A 0\n", "not closed"),
        ("nblib 1\ncutoff -1\n", "positive"),
        ("nblib 1\ncutoff 1\natomtype A 0 0 0\n", "mass"),
        ("nblib 1\ncutoff 1\natomtype A 1 0 0\nmolecule m 1\n atom a A 0\n"
         " bond harmonic a a 1 1\nend\n", "twice"),
    ],
)
def test_bad_topology_names_the_line(engine, text, fragment):
    with pytest.raises(nb.mdsession.MDError, match=fragment):
        engine.create(text)


def test_box_and_positions_are_checked(engine):
    system = nb.argon()
    with engine.create(system.topology) as md:
        skewed = system.box.copy()
        skewed[0, 1] = 0.5
        with pytest.raises(nb.mdsession.MDError, match="rectangular"):
            md.set_box(skewed)
        with pytest.raises(nb.mdsession.MDError, match="box and the positions"):
            md.step_nve(1)
        md.set_box(system.box / 4.0)
        md.set_positions(system.positions)
        with pytest.raises(nb.mdsession.MDError, match="twice the cutoff"):
            md.step_nve(1)
        with pytest.raises(ValueError):
            md.set_positions(system.positions[:-1])


# ------------------------------------------------------------------ potential


@pytest.mark.parametrize("name", SYSTEMS)
def test_potential_invariances(name):
    system = SYSTEMS[name][0]()
    pot = nb.Potential(system.topology)
    energy, forces, virial = pot.evaluate(system.positions, system.box)
    scale = np.abs(forces).max()
    assert np.abs(forces.sum(axis=0)).max() <= SUM * scale * len(forces)
    assert np.abs(virial - virial.T).max() <= SUM * np.abs(virial).max()
    # The energy does not see the image the coordinates are given in.
    edge = system.box[0, 0]
    shift = np.array([edge, -2.0 * edge, 0.0])
    moved = system.positions.copy()
    moved[::2] += shift
    e2, f2, v2 = pot.evaluate(moved, system.box)
    assert abs(e2 - energy) <= SUM * abs(energy)
    assert np.abs(f2 - forces).max() <= SUM * scale
    assert np.abs(v2 - virial).max() <= SUM * np.abs(virial).max()


def test_potential_matches_session_state(engine):
    system = nb.water()
    md = started(engine, system, 0.00025, 100.0)
    md.step_nve(20)
    state = md.state()
    energy, forces, virial = nb.Potential(system.topology).evaluate(
        state.positions, system.box
    )
    assert abs(state.potential_ev - energy) <= SUM * abs(energy)
    assert np.abs(state.forces - forces.reshape(-1)).max() <= SUM * np.abs(forces).max()
    assert np.abs(state.virial_ev - virial).max() <= SUM * np.abs(virial).max()
    assert np.abs(state.box - system.box).max() <= SUM * system.box[0, 0]


# -------------------------------------------------------------- NVE and state


@pytest.mark.parametrize("name", SYSTEMS)
def test_nve_conserves_energy(engine, name):
    make, dt, temperature, tolerance = SYSTEMS[name]
    system = make()
    md = started(engine, system, dt, temperature)
    first = md.state()
    reference = total(first)
    worst = 0.0
    for _ in range(8):
        md.step_nve(50)
        worst = max(worst, abs(total(md.state()) - reference))
    assert worst <= tolerance * first.kinetic_ev


def test_velocities_read_back_and_step_is_velocity_verlet(engine):
    system = nb.argon()
    dt = 0.004
    md = started(engine, system, dt, 60.0)
    v0 = nb.maxwell(system.masses, 60.0, 7)
    before = md.state()
    assert np.abs(before.velocities - v0).max() <= SUM * np.abs(v0).max()
    mass = np.repeat(system.masses, 3)
    acceleration = before.forces / mass * nb.ACCELERATION
    md.step_nve(1)
    after = md.state()
    expected = before.positions + dt * before.velocities + 0.5 * dt * dt * acceleration
    assert np.abs(after.positions - expected).max() <= SUM * np.abs(expected).max()
    kicked = before.velocities + 0.5 * dt * (
        acceleration + after.forces / mass * nb.ACCELERATION
    )
    assert np.abs(after.velocities - kicked).max() <= SUM * np.abs(kicked).max()
    assert after.steps == 1 and after.time_ps == dt


@pytest.mark.parametrize("name", SYSTEMS)
def test_time_reversal(engine, name):
    make, dt, temperature, tolerance = SYSTEMS[name]
    system = make()
    md = started(engine, system, dt, temperature)
    start = md.state()
    md.step_nve(300)
    turned = md.state()
    md.set_velocities(-turned.velocities)
    md.step_nve(300)
    back = md.state()
    assert np.abs(back.positions - start.positions).max() <= ROUNDTRIP * system.box[0, 0]
    assert np.abs(back.velocities + start.velocities).max() <= ROUNDTRIP * np.abs(
        start.velocities
    ).max()


def test_trajectory_is_continuous_across_the_box(engine):
    system = nb.argon()
    md = started(engine, system, 0.004, 60.0)
    md.set_velocities(np.tile([40.0, 0.0, 0.0], system.positions.shape[0]))
    start = md.state().positions.copy()
    md.step_nve(400)
    drift = md.state().positions - start
    # 400 steps at about 40 Angstrom/ps cover more than a box edge.
    assert np.abs(drift).max() > system.box[0, 0] / 2.0
    assert np.abs(md.state().positions).max() > system.box[0, 0]


def test_time_step_change_keeps_the_state(engine):
    system = nb.argon()
    md = started(engine, system, 0.004, 60.0)
    md.step_nve(5)
    before = md.state()
    md.set_time_step(0.002)
    after = md.state()
    assert np.abs(after.velocities - before.velocities).max() <= SUM * np.abs(
        before.velocities
    ).max()
    assert after.force_evaluations == before.force_evaluations


# ---------------------------------------------------------------------- hooks


def test_hook_sees_both_phases_and_edits(engine):
    system = nb.argon()
    n = 3 * len(system.positions)
    md = started(engine, system, 0.004, 60.0)
    md.initialize()
    seen = []

    def hook(phase, step, x, v, f, m):
        seen.append((phase, step))
        assert x.size == n and m.size == n
        assert np.allclose(m, np.repeat(system.masses, 3))
        if phase == nb.mdsession.PHASE_FORCES:
            v[:] = 0.0
            f[:] = 0.0
        return 0

    md.set_hook(hook)
    start = md.state().positions.copy()
    md.step_nve(3)
    assert seen == [
        (nb.mdsession.PHASE_POSITIONS, 1), (nb.mdsession.PHASE_FORCES, 1),
        (nb.mdsession.PHASE_POSITIONS, 2), (nb.mdsession.PHASE_FORCES, 2),
        (nb.mdsession.PHASE_POSITIONS, 3), (nb.mdsession.PHASE_FORCES, 3),
    ]
    # The hook zeroed the velocities and forces driving the next step, so the
    # system stood still from the second step on and the first plain step
    # after removing it does not move it either.
    moved = md.state().positions - start
    assert np.abs(moved).max() > 0.0
    # state reports the potential's force, not the one the hook zeroed.
    state = md.state()
    _, forces, _ = nb.Potential(system.topology).evaluate(state.positions, system.box)
    assert np.abs(state.forces - forces.reshape(-1)).max() <= SUM * np.abs(forces).max()
    md.set_hook(None)
    frozen = md.state().positions.copy()
    md.step_nve(1)
    assert np.abs(md.state().positions - frozen).max() <= SUM * system.box[0, 0]


def test_hook_failure_stops_the_step(engine):
    system = nb.argon()
    md = started(engine, system, 0.004, 60.0)
    md.set_hook(lambda *args: 7)
    with pytest.raises(nb.mdsession.MDError, match="hook returned 7"):
        md.step_nve(1)


def test_boundary_reflects_and_conserves_energy(engine):
    system = nb.argon()
    md = started(engine, system, 0.004, 60.0)
    wall = system.positions[:, 0].max() + 0.4
    hits = []

    def boundary(x, normal):
        atom = int(np.argmax(x[0::3]))
        if x[3 * atom] <= wall:
            return 0
        normal[:] = 0.0
        normal[3 * atom] = 1.0
        hits.append(atom)
        return 1

    md.set_boundary(boundary)
    push = np.zeros(3 * len(system.positions))
    top = int(np.argmax(system.positions[:, 0]))
    push[3 * top] = 15.0
    md.set_velocities(nb.maxwell(system.masses, 30.0, 5) + push)
    reference = total(md.state())
    worst = 0.0
    highest = -np.inf
    for _ in range(120):
        md.step_nve(1)
        state = md.state()
        highest = max(highest, state.positions[0::3].max())
        worst = max(worst, abs(total(state) - reference))
    assert hits, "the atom never reached the wall"
    assert highest <= wall
    assert worst <= FLUCTUATION * state.kinetic_ev


def test_boundary_without_a_normal_reverses_the_velocities(engine):
    system = nb.argon()
    md = started(engine, system, 0.004, 60.0)
    before = md.state()
    calls = []

    def boundary(x, normal):
        calls.append(1)
        return 1 if len(calls) == 1 else 0

    md.set_boundary(boundary)
    md.step_nve(1)
    after = md.state()
    assert np.abs(after.positions - before.positions).max() <= SUM
    assert np.abs(after.velocities + before.velocities).max() <= SUM * np.abs(
        before.velocities
    ).max()
    assert after.force_evaluations == before.force_evaluations


# ------------------------------------------------- projected Langevin dynamics


def random_plane(md, seed=11):
    rng = np.random.default_rng(seed)
    normal = rng.normal(size=md.dimension)
    return md.state().positions.copy(), normal


def test_projection_moves_positions_along_inverse_mass(engine):
    system = nb.water()
    md = started(engine, system, 0.00025, 100.0)
    point, normal = random_plane(md)
    shifted = point + 0.05 * normal
    md.set_positions(shifted)
    md.step_projected(0, plane=(point, normal))
    state = md.state()
    mass = np.repeat(system.masses, 3)
    moved = (state.positions - shifted) * mass
    assert abs(normal @ (state.positions - point)) <= SUM * np.abs(normal).max()
    along = normal * (moved @ normal) / (normal @ normal)
    assert np.abs(moved - along).max() <= ROUNDTRIP * np.abs(along).max()
    assert abs(normal @ state.velocities) <= SUM * np.abs(normal).max() * np.abs(
        state.velocities
    ).max() * len(normal)


@pytest.mark.parametrize("name", ["argon", "water"])
def test_constrained_nve_stays_on_the_plane_and_conserves_energy(engine, name):
    make, dt, temperature, tolerance = SYSTEMS[name]
    system = make()
    md = started(engine, system, dt, temperature)
    point, normal = random_plane(md)
    md.set_positions(point)
    md.step_projected(0, plane=(point, normal))
    reference = total(md.state())
    kinetic = md.state().kinetic_ev
    worst_height = 0.0
    worst_speed = 0.0
    worst_energy = 0.0
    for _ in range(6):
        md.step_projected(40, plane=(point, normal))
        state = md.state()
        worst_height = max(worst_height, abs(normal @ (state.positions - point)))
        worst_speed = max(worst_speed, abs(normal @ state.velocities))
        worst_energy = max(worst_energy, abs(total(state) - reference))
    scale = np.linalg.norm(normal) * system.box[0, 0]
    assert worst_height <= SUM * scale
    assert worst_speed <= SUM * np.linalg.norm(normal) * np.abs(state.velocities).max() * 8
    assert worst_energy <= 2.0 * tolerance * kinetic


def test_projected_steps_leave_plain_steps_unprojected(engine):
    system = nb.argon()
    md = started(engine, system, 0.004, 60.0)
    point, normal = random_plane(md)
    md.step_projected(5, plane=(point, normal))
    before = md.state()
    md.step_nve(20)
    height = abs(normal @ (md.state().positions - point))
    assert height > abs(normal @ (before.positions - point))
    # Forces read back are the engine's, not the constrained ones.
    energy, forces, _ = nb.Potential(system.topology).evaluate(
        before.positions, system.box
    )
    assert np.abs(before.forces - forces.reshape(-1)).max() <= SUM * np.abs(forces).max()


def test_langevin_thermostat_reaches_the_target_temperature(engine):
    system = nb.argon()
    target = 90.0
    md = started(engine, system, 0.004, 20.0)
    md.set_seed(5)
    kt = nb.BOLTZMANN_EV * target
    md.step_projected(500, kt_ev=kt, friction_per_ps=5.0)
    samples = []
    for _ in range(60):
        md.step_projected(20, kt_ev=kt, friction_per_ps=5.0)
        samples.append(md.state().kinetic_ev)
    dof = 3 * len(system.positions)
    temperature = 2.0 * np.mean(samples) / (dof * nb.BOLTZMANN_EV)
    assert abs(temperature - target) <= target / 2**4


def test_thermostat_is_reproducible_from_the_seed(engine):
    system = nb.argon()
    kt = nb.BOLTZMANN_EV * 90.0

    def run(seed):
        md = started(engine, system, 0.004, 20.0)
        md.set_seed(seed)
        md.step_projected(30, kt_ev=kt, friction_per_ps=5.0)
        return md.state().positions

    assert np.array_equal(run(1), run(1))
    assert not np.array_equal(run(1), run(2))


def test_walls_hold_on_the_plane(engine):
    system = nb.argon()
    kt = nb.BOLTZMANN_EV * 90.0
    probe = started(engine, system, 0.004, 90.0)
    point, normal = random_plane(probe)
    mass = np.repeat(system.masses, 3)
    rng = np.random.default_rng(13)
    wall_normal = rng.normal(size=normal.size)
    # The wall coordinate is taken along the plane, so the walk can reach it.
    wall_normal -= normal * (normal @ (wall_normal / mass)) / (normal @ (normal / mass))

    def walk(walls):
        md = started(engine, system, 0.004, 90.0)
        md.set_seed(3)
        heights = []
        for _ in range(60):
            md.step_projected(
                10, kt_ev=kt, friction_per_ps=2.0, plane=(point, normal), walls=walls
            )
            state = md.state()
            assert abs(normal @ (state.positions - point)) <= SUM * np.linalg.norm(
                normal
            ) * system.box[0, 0]
            heights.append(wall_normal @ (state.positions - point))
        return np.array(heights)

    free = walk([])
    limit = 0.5 * free.max()
    assert limit > 0.0, "the free walk never left the start"
    held = walk([(wall_normal, float(wall_normal @ point) + limit)])
    assert held.max() <= limit
    assert free.max() > limit


def test_start_outside_a_wall_is_an_error(engine):
    system = nb.argon()
    md = started(engine, system, 0.004, 60.0)
    wall_normal = np.zeros(3 * len(system.positions))
    wall_normal[0] = 1.0
    with pytest.raises(nb.mdsession.MDError, match="outside a wall"):
        md.step_projected(1, walls=[(wall_normal, system.positions[0, 0] - 0.1)])


def test_missing_calls_are_reported(engine):
    class Bare(nb.mdsession.MDEngine):
        def __init__(self, parent):
            self.prefix = parent.prefix
            self.name = parent.name
            self._calls = {
                k: v for k, v in parent._calls.items() if k != "step_projected"
            }

    with Bare(engine).create(nb.argon().topology) as md:
        with pytest.raises(nb.mdsession.MDError, match="does not implement"):
            md.step_projected(1, kt_ev=0.01)


# ------------------------------------------------------------------- frontend

EVAL = os.environ.get("RGPOT_EVAL", "")


@pytest.mark.skipif(not EVAL, reason="needs RGPOT_EVAL built with -Dwith_nblib")
@pytest.mark.parametrize("name", ["argon", "water", "chain"])
def test_frontend_matches_the_c_abi(tmp_path, name):
    system = SYSTEMS[name][0]()
    topology = tmp_path / "topology.txt"
    topology.write_text(system.topology)
    box = system.box.reshape(-1)
    lines = [str(len(system.positions)), " ".join(f"{b:.17g}" for b in box)]
    lines += ["1 " + " ".join(f"{c:.17g}" for c in p) for p in system.positions]
    done = subprocess.run(
        [EVAL, "Nblib", "topology", str(topology)],
        input="\n".join(lines) + "\n",
        capture_output=True,
        text=True,
        env={**os.environ, "RGPOT_NBLIB_ENGINE": nb.ENGINE},
    )
    assert done.returncode == 0, done.stderr
    energy = None
    forces = []
    stress = None
    for line in done.stdout.splitlines():
        words = line.split()
        if words[0] == "energy":
            energy = float(words[1])
        elif words[0] == "force":
            forces.append([float(w) for w in words[2:5]])
        elif words[0] == "stress":
            stress = np.array([float(w) for w in words[1:]]).reshape(3, 3)
    ref_energy, ref_forces, ref_virial = nb.Potential(system.topology).evaluate(
        system.positions, system.box
    )
    volume = np.linalg.det(system.box)
    assert energy == ref_energy
    assert np.abs(np.array(forces) - ref_forces).max() <= SUM * np.abs(ref_forces).max()
    assert np.abs(stress - 2.0 * ref_virial / volume).max() <= SUM * np.abs(stress).max()
