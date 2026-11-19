#!/usr/bin/env python3
import asyncio
import argparse
import math
import subprocess
import traceback
import sys
import os
from pathlib import Path
import capnp
import numpy as np

# Load schema relative to this script unless a built effective schema is available.
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SOURCE_SCHEMA_PATH = Path(SCRIPT_DIR) / "../CppCore/rgpot/rpc/Potentials.capnp"
pot_capnp = None


def load_schema(server_bin):
    candidates = []
    if server_bin:
        server_path = Path(server_bin).resolve()
        if server_path.parent.name == "CppCore":
            build_schema = (
                server_path.parent.parent / "CppCore/rgpot/rpc/Potentials.capnp"
            )
            candidates.append(build_schema)
    candidates.append(SOURCE_SCHEMA_PATH.resolve())
    for candidate in candidates:
        if candidate.exists():
            print(f"Loading RPC schema: {candidate}")
            parser = capnp.SchemaParser()
            return parser.load(str(candidate))
    raise FileNotFoundError(
        f"could not find Potentials.capnp (checked {', '.join(str(c) for c in candidates)})"
    )

try:
    from cpmd_params import configure_cpmd
    from nwchem_params import configure_nwchem, make_potential_config_none
except ImportError:
    sys.path.insert(0, SCRIPT_DIR)
    from cpmd_params import configure_cpmd
    from nwchem_params import configure_nwchem, make_potential_config_none


def potential_config_arm(name):
    """A PotentialConfig with arm `name` set to its schema defaults."""
    cfg = pot_capnp.PotentialConfig.new_message()
    field = pot_capnp.PotentialConfig.schema.fields[name]
    if field.proto.slot.type.which() == "void":
        setattr(cfg, name, None)
    else:
        cfg.init(name)
    return cfg


async def refuse_every_arm(pot, label, applied):
    """configure() with each arm outside `applied` is refused by name, not dropped."""
    arms = list(pot_capnp.PotentialConfig.schema.union_fields)
    for name in arms:
        if name == "none" or name in applied:
            continue
        reply = await pot.configure(potential_config_arm(name))
        assert not reply.ok, f"{label} accepted configure({name}): {reply.message}"
        assert name in reply.message, f"{label} configure({name}) message: {reply.message}"
    none = await pot.configure(potential_config_arm("none"))
    assert none.ok, f"{label} configure(none): {none.message}"
    print(f"{label}: configure refused {len(arms) - 1 - len(applied)} arms by name, none is a no-op")


async def run_client(port):
    # Retry connection a few times to allow server startup
    for _ in range(10):
        try:
            # Create connection (this requires the KJ loop to be active)
            connection = await capnp.AsyncIoStream.create_connection(
                host="localhost", port=port
            )
            break
        except OSError:
            await asyncio.sleep(0.5)
    else:
        print("Failed to connect to server.")
        return False

    client = capnp.TwoPartyClient(connection)
    pot = client.bootstrap().cast_as(pot_capnp.Potential)

    # Test Case: CuH2 minimal system
    fip = pot_capnp.ForceInput.new_message()

    # Cu at 0,0,0 and H at 1.5, 0, 0 (Approx distance)
    # Using semantic types from your "Pro" schema if you applied it,
    # but based on your recent "simple" checks, we'll stick to the flat list
    # unless you updated the schema.
    # Assuming you kept the struct/flat hybrid from previous steps:
    pos_data = [0.0, 0.0, 0.0, 1.5, 0.0, 0.0]
    fip.init("pos", len(pos_data))
    for i, p in enumerate(pos_data):
        fip.pos[i] = p

    fip.init("atmnrs", 2)
    fip.atmnrs[0] = 29  # Cu
    fip.atmnrs[1] = 1  # H

    box_data = [10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0]
    fip.init("box", len(box_data))
    for i, b in enumerate(box_data):
        fip.box[i] = b

    print("Sending calculation request...")
    # This await requires the KJ loop
    result = await pot.calculate(fip)

    print(f"Received Energy: {result.result.energy}")
    print(f"Received Forces: {list(result.result.forces)}")
    # CuH2 is the Fortran 2018 kernel now, and it agrees with the kernel it
    # replaced to 1e-6 rather than bit for bit -- the same tolerance
    # CuH2PotTest pins. Exact equality here was asserting a guarantee the
    # rewrite never claimed.
    assert math.isclose(result.result.energy, -0.678808, rel_tol=1e-6)
    assert math.isclose(result.result.forces[0], -7.5565221, rel_tol=1e-6)
    assert math.isclose(result.result.forces[3], 7.5565221, rel_tol=1e-6)

    # Basic physical sanity check
    if result.result.energy == 0.0 or np.isnan(result.result.energy):
        print("Error: Energy is zero or NaN")
        return False

    await refuse_every_arm(pot, "CuH2", applied=())
    again = await pot.calculate(fip)
    assert again.result.energy == result.result.energy
    return True


async def run_morse_configure(port):
    """configure(morse) on a Morse server changes the energy it returns."""
    for _ in range(10):
        try:
            connection = await capnp.AsyncIoStream.create_connection(
                host="localhost", port=port
            )
            break
        except OSError:
            await asyncio.sleep(0.5)
    else:
        print("Morse configure: failed to connect to server.")
        return False

    client = capnp.TwoPartyClient(connection)
    pot = client.bootstrap().cast_as(pot_capnp.Potential)

    defaults = pot_capnp.MorseParams.new_message()
    fip = pot_capnp.ForceInput.new_message()
    fip.init("pos", 6)
    for i, p in enumerate([10.0, 10.0, 10.0, 10.0 + defaults.re, 10.0, 10.0]):
        fip.pos[i] = p
    fip.init("atmnrs", 2)
    fip.atmnrs[0] = 78
    fip.atmnrs[1] = 78
    fip.init("box", 9)
    for i, b in enumerate([40.0, 0.0, 0.0, 0.0, 40.0, 0.0, 0.0, 0.0, 40.0]):
        fip.box[i] = b

    before = (await pot.calculate(fip)).result.energy

    cfg = pot_capnp.PotentialConfig.new_message()
    cfg.init("morse").de = 2.0 * defaults.de
    reply = await pot.configure(cfg)
    print(f"Morse configure(morse): ok={reply.ok} msg={reply.message}")
    assert reply.ok, reply.message

    after = (await pot.calculate(fip)).result.energy
    print(f"Morse energy at re: default de {before}, doubled de {after}")
    # At r = re the shifted pair energy is -de d^2, linear in de.
    d = 1.0 - math.exp(-defaults.a * (defaults.cutoff - defaults.re))
    assert math.isclose(after, -2.0 * defaults.de * d * d, rel_tol=1e-12)
    assert math.isclose(after, 2.0 * before, rel_tol=1e-12)

    await refuse_every_arm(pot, "Morse", applied=("morse",))
    assert (await pot.calculate(fip)).result.energy == after
    return True


async def run_nwchem_smoke(port):
    """Optional smoke: start NWChem server, configure(), skip calculate if no engine."""
    for _ in range(10):
        try:
            connection = await capnp.AsyncIoStream.create_connection(
                host="localhost", port=port
            )
            break
        except OSError:
            await asyncio.sleep(0.5)
    else:
        print("NWChem smoke: failed to connect to server.")
        return False

    client = capnp.TwoPartyClient(connection)
    pot = client.bootstrap().cast_as(pot_capnp.Potential)

    # configure(none) should succeed
    cfg_none = make_potential_config_none(pot_capnp)
    r0 = await pot.configure(cfg_none)
    print(f"NWChem configure(none): ok={r0.ok} msg={r0.message}")

    ok, msg = await configure_nwchem(
        pot, pot_capnp, basis="sto-3g", theory="scf", scf_type="rhf"
    )
    print(f"NWChem configure(nwchem): ok={ok} msg={msg}")
    # ok may be False without engine; still a valid RPC round-trip
    await refuse_every_arm(pot, "NWChem", applied=("nwchem",))
    print("NWChem smoke: configure RPC completed.")
    return True


async def configure_cpmd_smoke(pot, pot_capnp):
    """Configure a CPMD backend over RPC; calculate may require an engine."""
    cfg_none = make_potential_config_none(pot_capnp)
    r0 = await pot.configure(cfg_none)
    print(f"CPMD configure(none): ok={r0.ok} msg={r0.message}")

    ok, msg = await configure_cpmd(pot, pot_capnp, functional="BLYP", task="gradient")
    print(f"CPMD configure(cpmd): ok={ok} msg={msg}")
    ok_sections, msg_sections = await configure_cpmd(
        pot,
        pot_capnp,
        functional="PBE0",
        task="gradient",
        input_sections=[
            {
                "kind": "generic",
                "name": "PIMD",
                "directives": [{"keyword": "TEMP", "args": ["300"]}],
            },
            {
                "kind": "cpmd",
                "optimizeGeometry": True,
                "molecularDynamics": True,
                "molecularDynamicsCp": True,
                "molecularDynamicsBo": True,
                "molecularDynamicsEh": True,
                "molecularDynamicsPt": True,
                "molecularDynamicsClassical": True,
                "molecularDynamicsFile": "TRAJECTORY.in",
                "convergenceGeometry": 1.0e-4,
                "maxStep": 8,
                "maxIter": 12,
                "electronMass": 450.0,
                "nose": True,
                "noseIons": True,
                "noseElectrons": True,
                "berendsen": "300 100",
                "langevin": True,
                "annealing": "IONS 300 50",
                "quench": True,
                "rattle": True,
                "shake": True,
                "constraint": "FIX COM",
                "trotter": "8",
                "restart": True,
                "printOptions": "FORCES ON",
                "storeOptions": "WAVEFUNCTION",
                "centerMoleculeOff": True,
                "centerMoleculeOn": True,
                "diis": True,
                "odiis": True,
                "pcg": True,
                "diagonalization": True,
                "freeEnergy": True,
                "interface": True,
                "qmmm": True,
                "bicanonicalEnsemble": True,
                "cdft": True,
                "properties": True,
                "directives": [{"keyword": "PRINT", "args": ["FORCES", "ON"]}],
            },
            {
                "kind": "dft",
                "functional": "PBE0",
                "lsd": True,
                "gcCutoff": 1.0e-8,
                "xcDriver": "LIBXC",
                "libxc": "GGA_X_PBE GGA_C_PBE",
                "lrKernel": "PBE",
                "refunct": "PBE",
                "mtsHighFunc": "PBE0",
                "mtsLowFunc": "PBE",
                "hfx": True,
                "hfxScreening": "0.2",
                "hubbard": "U 1 4.0",
                "alpha": 0.25,
                "beta": 0.75,
                "oldCode": True,
                "newCode": True,
                "correlation": "LYP",
                "exchange": "B88",
                "becke88": True,
                "directives": [{"keyword": "HFX", "args": ["SCREENING"]}],
            },
            {"kind": "raw", "text": "&VDW\n  DISPERSION\n&END"},
        ],
    )
    print(f"CPMD configure(cpmd sections): ok={ok_sections} msg={msg_sections}")
    print("CPMD smoke: configure RPC completed.")
    return True


async def run_cpmd_smoke(port):
    """Optional smoke: start CPMD server, configure(), skip calculate if no engine."""
    for _ in range(10):
        try:
            connection = await capnp.AsyncIoStream.create_connection(
                host="localhost", port=port
            )
            break
        except OSError:
            await asyncio.sleep(0.5)
    else:
        print("CPMD smoke: failed to connect to server.")
        return False

    client = capnp.TwoPartyClient(connection)
    pot = client.bootstrap().cast_as(pot_capnp.Potential)
    if not await configure_cpmd_smoke(pot, pot_capnp):
        return False
    await refuse_every_arm(pot, "CPMD", applied=("cpmd",))
    return True


def _force_input(point, unit):
    fip = pot_capnp.ForceInput.new_message()
    fip.init("pos", 3)
    fip.pos[0] = float(point)
    fip.pos[1] = 0.0
    fip.pos[2] = 0.0
    fip.init("atmnrs", 1)
    fip.atmnrs[0] = 1
    fip.init("box", 9)
    for i, b in enumerate([40.0, 0.0, 0.0, 0.0, 40.0, 0.0, 0.0, 0.0, 40.0]):
        fip.box[i] = b
    fip.lengthUnit = unit
    fip.energyUnit = "eV"
    return fip


async def run_hvp_unit_smoke(port):
    for _ in range(10):
        try:
            connection = await capnp.AsyncIoStream.create_connection(
                host="localhost", port=port
            )
            break
        except OSError:
            await asyncio.sleep(0.5)
    else:
        print("HVP unit smoke: failed to connect to server.")
        return False

    client = capnp.TwoPartyClient(connection)
    pot = client.bootstrap().cast_as(pot_capnp.Potential)
    bohr_to_angstrom = 0.529177210903
    cases = [
        ("bohr", 0.75 / bohr_to_angstrom, 0.28 / bohr_to_angstrom),
        ("nm", 0.075, 0.028),
    ]
    for unit, point, direction in cases:
        fip = _force_input(point, unit)
        req = pot.calculateHvp_request()
        req.fip = fip
        req.init("direction", 3)
        req.direction[0] = float(direction)
        req.direction[1] = 0.0
        req.direction[2] = 0.0
        hvp = np.array((await req.send()).product, dtype=np.float64)
        delta = np.sqrt(np.finfo(np.float64).eps) * (
            1.0 + abs(point) + abs(direction)
        )
        fp = np.array(
            (await pot.calculate(_force_input(point + delta * direction, unit))).result.forces,
            dtype=np.float64,
        )
        fm = np.array(
            (await pot.calculate(_force_input(point - delta * direction, unit))).result.forces,
            dtype=np.float64,
        )
        fd = -(fp - fm) / (2.0 * delta)
        # calculate() and calculateHvp() both return values in caller units.
        scale = np.max(np.abs(np.concatenate([hvp, fd])))
        bar = np.cbrt(np.finfo(np.float64).eps) * (1.0 + scale)
        assert np.max(np.abs(hvp - fd)) <= bar, (
            f"{unit} HVP mismatch: got={hvp[0]} fd={fd[0]} bar={bar}"
        )
    print("HVP unit smoke: bohr and nm finite differences agree.")
    return True


def _local_neighbours(pos, box, cutoff, skin, full, loose, periodic, shifts=True, vectors=True, distances=True):
    """Reference neighbour list using rgpot._core when available, else exact lattice walk."""
    pos_arr = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
    box_arr = np.asarray(box, dtype=np.float64).reshape(3, 3)
    if all(periodic) and not loose and skin == 0.0:
        try:
            import rgpot
            quantities = "ij"
            if shifts:
                quantities += "S"
            if vectors:
                quantities += "D"
            if distances:
                quantities += "d"
            out = rgpot.neighbor_list(pos_arr, box_arr, cutoff, quantities=quantities, full=full)
            if not isinstance(out, tuple):
                out = (out,)
            idx = 2
            res = {"first": np.asarray(out[0], dtype=np.int32), "second": np.asarray(out[1], dtype=np.int32)}
            if shifts:
                res["shift"] = np.asarray(out[idx], dtype=np.int32).reshape(-1, 3)
                idx += 1
            if vectors:
                res["vector"] = np.asarray(out[idx], dtype=np.float64).reshape(-1, 3)
                idx += 1
            if distances:
                res["distance"] = np.asarray(out[idx], dtype=np.float64)
            return res
        except ImportError:
            pass
    n = pos_arr.shape[0]
    search = cutoff + skin if loose else cutoff
    cut2 = cutoff * cutoff
    reach2 = search * search
    a, b, c = box_arr[0], box_arr[1], box_arr[2]
    vol = abs(float(np.dot(a, np.cross(b, c))))
    heights = [
        vol / float(np.linalg.norm(np.cross(b, c))) if np.linalg.norm(np.cross(b, c)) > 0 else 0.0,
        vol / float(np.linalg.norm(np.cross(c, a))) if np.linalg.norm(np.cross(c, a)) > 0 else 0.0,
        vol / float(np.linalg.norm(np.cross(a, b))) if np.linalg.norm(np.cross(a, b)) > 0 else 0.0,
    ]
    nmax = [
        int(math.ceil(search / h)) if p and h > 0 and search > 0 else 0
        for p, h in zip(periodic, heights)
    ]
    first, second, shift_list, vec_list, dist_list = [], [], [], [], []
    for i in range(n):
        for j in range(n):
            for sa in range(-nmax[0], nmax[0] + 1):
                for sb in range(-nmax[1], nmax[1] + 1):
                    for sc in range(-nmax[2], nmax[2] + 1):
                        if not full:
                            if i > j:
                                continue
                            if i == j:
                                s_sum = sa + sb + sc
                                if s_sum < 0 or (s_sum == 0 and (sc < 0 or (sc == 0 and sb < 0))) or (sa == 0 and sb == 0 and sc == 0):
                                    continue
                        vec = pos_arr[j] - pos_arr[i] + sa * a + sb * b + sc * c
                        r2 = float(np.dot(vec, vec))
                        if not (r2 > 0.0):
                            continue
                        if not loose:
                            ok = r2 < cut2
                        elif skin > 0.0:
                            ok = r2 < reach2
                        else:
                            ok = r2 <= cut2
                        if not ok:
                            continue
                        first.append(i)
                        second.append(j)
                        shift_list.append([sa, sb, sc])
                        vec_list.append(vec)
                        dist_list.append(math.sqrt(r2))
    res = {
        "first": np.asarray(first, dtype=np.int32),
        "second": np.asarray(second, dtype=np.int32),
        "shift": np.asarray(shift_list, dtype=np.int32).reshape(-1, 3),
        "vector": np.asarray(vec_list, dtype=np.float64).reshape(-1, 3),
        "distance": np.asarray(dist_list, dtype=np.float64),
    }
    return res


def _compare_nlist(got_msg, ref, shifts=True, vectors=True, distances=True):
    assert got_msg.status == 0, f"expected status 0, got {got_msg.status}: {got_msg.message}"
    first = np.asarray(got_msg.first, dtype=np.int32)
    second = np.asarray(got_msg.second, dtype=np.int32)
    assert len(first) == len(ref["first"]), f"pair count {len(first)} != {len(ref['first'])}"
    if shifts:
        shift = np.asarray(got_msg.shift, dtype=np.int32).reshape(-1, 3)
        got_keys = sorted(
            (int(first[k]), int(second[k]), int(shift[k, 0]), int(shift[k, 1]), int(shift[k, 2]), k)
            for k in range(len(first))
        )
        ref_keys = sorted(
            (int(ref["first"][k]), int(ref["second"][k]), int(ref["shift"][k, 0]), int(ref["shift"][k, 1]), int(ref["shift"][k, 2]), k)
            for k in range(len(ref["first"]))
        )
        assert [g[:5] for g in got_keys] == [r[:5] for r in ref_keys]
        got_order = [g[5] for g in got_keys]
        ref_order = [r[5] for r in ref_keys]
    else:
        assert len(got_msg.shift) == 0
        got_order = list(np.lexsort((second, first)))
        ref_order = list(np.lexsort((ref["second"], ref["first"])))
        assert list(first[got_order]) == list(ref["first"][ref_order])
        assert list(second[got_order]) == list(ref["second"][ref_order])
    nlist_atol = float(np.finfo(np.float64).eps) * 16.0
    if vectors:
        vec = np.asarray(got_msg.vector, dtype=np.float64).reshape(-1, 3)
        np.testing.assert_allclose(vec[got_order], ref["vector"][ref_order], rtol=0.0, atol=nlist_atol)
    else:
        assert len(got_msg.vector) == 0
    if distances:
        dist = np.asarray(got_msg.distance, dtype=np.float64)
        np.testing.assert_allclose(dist[got_order], ref["distance"][ref_order], rtol=0.0, atol=nlist_atol)
    else:
        assert len(got_msg.distance) == 0


async def run_neighbours_smoke(port, label, default_cutoff):
    for _ in range(10):
        try:
            connection = await capnp.AsyncIoStream.create_connection(
                host="localhost", port=port
            )
            break
        except OSError:
            await asyncio.sleep(0.5)
    else:
        print(f"{label} neighbours: failed to connect to server.")
        return False

    client = capnp.TwoPartyClient(connection)
    pot = client.bootstrap().cast_as(pot_capnp.Potential)

    geometries = [
        (
            "open_cluster",
            [0.0, 0.0, 0.0, 1.3, 0.1, -0.2, -0.4, 1.5, 0.3, 0.8, 0.9, 1.4],
            [20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 20.0],
            [False, False, False],
            2.4,
        ),
        (
            "cubic",
            [0.2, 0.2, 0.2, 1.4, 0.3, 0.4, 3.8, 0.2, 0.1, 0.3, 3.7, 3.9],
            [4.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 4.0],
            [True, True, True],
            2.2,
        ),
        (
            "triclinic",
            [0.2, 0.3, 0.4, 3.6, 0.5, 0.2, 1.0, 3.8, 0.7, 2.2, 2.1, 3.9],
            [4.0, 0.2, 0.1, 0.3, 4.5, -0.2, 0.05, 0.15, 4.2],
            [True, True, True],
            3.2,
        ),
    ]

    for geom_name, pos, box, pbc, cutoff in geometries:
        for full in (False, True):
            for skin, cut in ((0.0, 0), (0.3, 1)):
                fip = pot_capnp.ForceInput.new_message()
                fip.pos = pos
                fip.atmnrs = [1] * (len(pos) // 3)
                fip.box = box
                req = pot_capnp.NeighborRequest.new_message()
                req.cutoff = cutoff
                req.skin = skin
                req.format = 1 if full else 0
                req.cut = cut
                req.periodic = pbc
                req.shifts = True
                req.vectors = True
                req.distances = True
                req.mode = 1
                reply = (await pot.neighbours(fip, req)).neighbors
                ref = _local_neighbours(pos, box, cutoff, skin, full, cut == 1, pbc)
                _compare_nlist(reply, ref)

    # Also verify cutoff = 0 uses the potential's own locality cutoff when finite.
    if default_cutoff > 0.0:
        pos = [0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1.8, 0.0]
        box = [30.0, 0.0, 0.0, 0.0, 30.0, 0.0, 0.0, 0.0, 30.0]
        fip = pot_capnp.ForceInput.new_message()
        fip.pos = pos
        fip.atmnrs = [1, 1, 1]
        fip.box = box
        req = pot_capnp.NeighborRequest.new_message()
        req.cutoff = 0.0
        req.format = 0
        req.periodic = [False, False, False]
        reply = (await pot.neighbours(fip, req)).neighbors
        ref = _local_neighbours(pos, box, default_cutoff, 0.0, False, False, [False, False, False])
        _compare_nlist(reply, ref)

    print(f"{label} neighbours: open cluster, cubic and triclinic half/full shells match local nlist.")
    return True


async def run_model_surface_smoke(port, name):
    for _ in range(10):
        try:
            connection = await capnp.AsyncIoStream.create_connection(
                host="localhost", port=port
            )
            break
        except OSError:
            await asyncio.sleep(0.5)
    else:
        print(f"{name} smoke: failed to connect to server.")
        return False
    client = capnp.TwoPartyClient(connection)
    pot = client.bootstrap().cast_as(pot_capnp.Potential)
    fip = pot_capnp.ForceInput.new_message()
    fip.init("box", 9)
    for i, b in enumerate([40.0, 0.0, 0.0, 0.0, 40.0, 0.0, 0.0, 0.0, 40.0]):
        fip.box[i] = b
    fip.lengthUnit = "angstrom"
    fip.energyUnit = "eV"
    if name == "DoubleWell":
        pos = [0.7, -0.2, 0.0]
    else:
        pos = [0.7, -0.2, 0.0]
    fip.init("pos", len(pos))
    for i, value in enumerate(pos):
        fip.pos[i] = value
    fip.init("atmnrs", 1)
    fip.atmnrs[0] = 1
    result = await pot.calculate(fip)
    energy = float(result.result.energy)
    forces = np.array(result.result.forces, dtype=np.float64)
    assert np.isfinite(energy)
    assert np.all(np.isfinite(forces))
    assert np.max(np.abs(forces)) > np.finfo(np.float64).eps
    print(f"{name} smoke: energy={energy} force0={forces[0]}")
    return True


def main():
    global pot_capnp
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server-bin", required=True, help="Path to potserv executable"
    )
    parser.add_argument("--port", type=int, default=12345)
    parser.add_argument(
        "--nwchem-smoke",
        action="store_true",
        help="Also run NWChem configure() smoke on a second server port",
    )
    parser.add_argument(
        "--cpmd-smoke",
        action="store_true",
        help="Also run CPMD configure() smoke on a separate server port",
    )
    args = parser.parse_args()
    pot_capnp = load_schema(args.server_bin)

    # 1. Start Server
    print(f"Starting server: {args.server_bin} {args.port} CuH2")
    server_proc = subprocess.Popen(
        [args.server_bin, str(args.port), "CuH2"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        #  Run Client
        success = asyncio.run(capnp.run(run_client(args.port)))
    except BaseException as e:
        # Kill first, then drain: reading a live server's stderr blocks until
        # it exits, which turned an assertion failure into a job that sat
        # until the six-hour CI timeout. Print the type and traceback too --
        # `{e}` alone is empty for AssertionError.
        print(f"Exception during test: {type(e).__name__}: {e}")
        traceback.print_exc()
        server_proc.kill()
        out, err = server_proc.communicate(timeout=30)
        print("Server stdout:", out.decode(errors="replace")[-4000:])
        print("Server stderr:", err.decode(errors="replace")[-4000:])
        success = False
    finally:
        # Cleanup
        print("Killing server...")
        server_proc.kill()
        server_proc.wait()

    if not success:
        sys.exit(1)
    print("Integration test passed.")

    morse_port = args.port + 3
    print(f"Starting Morse server: {args.server_bin} {morse_port} Morse")
    morse_proc = subprocess.Popen(
        [args.server_bin, str(morse_port), "Morse"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        morse_ok = asyncio.run(capnp.run(run_morse_configure(morse_port)))
    except BaseException as e:
        print(f"Morse configure exception: {type(e).__name__}: {e}")
        traceback.print_exc()
        morse_proc.kill()
        out, err = morse_proc.communicate(timeout=30)
        print("Server stdout:", out.decode(errors="replace")[-4000:])
        print("Server stderr:", err.decode(errors="replace")[-4000:])
        morse_ok = False
    finally:
        morse_proc.kill()
        morse_proc.wait()
    if not morse_ok:
        sys.exit(1)
    print("Morse configure passed.")

    model_port = args.port + 6
    for name in ("DoubleWell", "CoupledProtonWell"):
        print(f"Starting model smoke server: {args.server_bin} {model_port} {name}")
        model_proc = subprocess.Popen(
            [args.server_bin, str(model_port), name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            model_ok = asyncio.run(capnp.run(run_model_surface_smoke(model_port, name)))
        except BaseException as e:
            print(f"{name} smoke exception: {type(e).__name__}: {e}")
            traceback.print_exc()
            model_proc.kill()
            out, err = model_proc.communicate(timeout=30)
            print("Server stdout:", out.decode(errors="replace")[-4000:])
            print("Server stderr:", err.decode(errors="replace")[-4000:])
            model_ok = False
        finally:
            model_proc.kill()
            model_proc.wait()
        if not model_ok:
            sys.exit(1)
        model_port += 1

    hvp_port = args.port + 9
    print(f"Starting HVP unit server: {args.server_bin} {hvp_port} Eckart")
    hvp_proc = subprocess.Popen(
        [args.server_bin, str(hvp_port), "Eckart"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        hvp_ok = asyncio.run(capnp.run(run_hvp_unit_smoke(hvp_port)))
    except BaseException as e:
        print(f"HVP unit smoke exception: {type(e).__name__}: {e}")
        traceback.print_exc()
        hvp_proc.kill()
        out, err = hvp_proc.communicate(timeout=30)
        print("Server stdout:", out.decode(errors="replace")[-4000:])
        print("Server stderr:", err.decode(errors="replace")[-4000:])
        hvp_ok = False
    finally:
        hvp_proc.kill()
        hvp_proc.wait()
    if not hvp_ok:
        sys.exit(1)

    # Run the neighbour list RPC checks (open cluster, cubic, triclinic, half/full).
    nlist_port = args.port + 12
    for pot_name, cutoff in (("LJ", 15.0), ("Morse", 9.5)):
        print(f"Starting {pot_name} server for neighbours: {args.server_bin} {nlist_port} {pot_name}")
        nlist_proc = subprocess.Popen(
            [args.server_bin, str(nlist_port), pot_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            nlist_ok = asyncio.run(capnp.run(run_neighbours_smoke(nlist_port, pot_name, cutoff)))
        except BaseException as e:
            print(f"{pot_name} neighbours exception: {type(e).__name__}: {e}")
            traceback.print_exc()
            nlist_proc.kill()
            out, err = nlist_proc.communicate(timeout=30)
            print("Server stdout:", out.decode(errors="replace")[-4000:])
            print("Server stderr:", err.decode(errors="replace")[-4000:])
            nlist_ok = False
        finally:
            nlist_proc.kill()
            nlist_proc.wait()
        if not nlist_ok:
            sys.exit(1)
        print(f"{pot_name} neighbours passed.")
        nlist_port += 1

    if args.nwchem_smoke:
        nw_port = args.port + 1
        print(f"Starting NWChem smoke server: {args.server_bin} {nw_port} NWChem")
        nw_proc = subprocess.Popen(
            [args.server_bin, str(nw_port), "NWChem"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            nw_ok = asyncio.run(capnp.run(run_nwchem_smoke(nw_port)))
        except Exception as e:
            print(f"NWChem smoke exception: {e}")
            print("Server stderr:")
            print(nw_proc.stderr.read().decode())
            nw_ok = False
        finally:
            nw_proc.kill()
            nw_proc.wait()
        if not nw_ok:
            sys.exit(1)
        print("NWChem smoke passed.")

    if args.cpmd_smoke:
        cp_port = args.port + 1 + int(args.nwchem_smoke)
        print(f"Starting CPMD smoke server: {args.server_bin} {cp_port} CPMD")
        cp_proc = subprocess.Popen(
            [args.server_bin, str(cp_port), "CPMD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            cp_ok = asyncio.run(capnp.run(run_cpmd_smoke(cp_port)))
        except Exception as e:
            print(f"CPMD smoke exception: {e}")
            print("Server stderr:")
            print(cp_proc.stderr.read().decode())
            cp_ok = False
        finally:
            cp_proc.kill()
            cp_proc.wait()
        if not cp_ok:
            sys.exit(1)
        print("CPMD smoke passed.")


if __name__ == "__main__":
    main()
