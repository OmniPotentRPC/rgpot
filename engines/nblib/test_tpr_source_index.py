#!/usr/bin/env python3
# MIT License
# Copyright 2023--present rgpot developers
"""Source TPR index of each packed nblib session site.

usage: test_tpr_source_index.py ENGINE.so GMX_PREFIX librgpot.so

Regenerate the system with the same commands the test runs. It writes a
topology, a coordinate file and a run-input file under a temporary
directory, then calls grompp from GMX_PREFIX:

    gmx_d grompp -f mix.mdp -c mix.gro -p mix.top -o mix.tpr -maxwarn 3

The system is one sodium, one TIP4P water, one chloride, another TIP4P
water and another sodium, in that order. Each TIP4P M site is a virtual
site, so virtual sites sit between massive atoms. The packed session
drops those M sites. The reported index of each packed site must be the
0-based TPR atom index, and that atom's coordinate and mass must be the
ones in the file.
"""

import ctypes as C
import math
import os
import re
import subprocess
import sys
import tempfile


ANGSTROM_PER_NM = 10.0


def fail(message):
    print("FAIL:", message)
    return 1


def run(cmd, cwd):
    done = subprocess.run(
        cmd, cwd=cwd, check=False, text=True, capture_output=True
    )
    if done.returncode != 0:
        raise RuntimeError(
            "command failed: "
            + " ".join(cmd)
            + "\n"
            + done.stdout
            + done.stderr
        )
    return done.stdout + done.stderr


def water(origin):
    """TIP4P sites in nm. M is the virtual site. Offsets are gro decimals."""
    ox, oy, oz = origin
    return [
        ("OW", "SOL", 15.9994, (ox, oy, oz), False),
        ("HW1", "SOL", 1.008, (ox + 0.076, oy + 0.059, oz), False),
        ("HW2", "SOL", 1.008, (ox - 0.076, oy + 0.059, oz), False),
        ("MW", "SOL", 0.0, (ox, oy, oz), True),
    ]


def atoms():
    """TPR order: Na, TIP4P, Cl, TIP4P, Na. Virtual M sites are intermixed."""
    sites = [("NA", "NA", 22.990, (0.300, 0.300, 0.300), False)]
    sites.extend(water((0.800, 0.800, 0.800)))
    sites.append(("CL", "CL", 35.450, (1.500, 0.400, 1.200), False))
    sites.extend(water((1.800, 1.600, 1.200)))
    sites.append(("NA", "NA", 22.990, (2.400, 2.400, 0.400), False))
    return sites


def write_inputs(directory):
    sites = atoms()
    top = """[ defaults ]
1 2 yes 0.5 0.8333

[ atomtypes ]
Na  22.990   0.0  A  0.0  0.0
Cl  35.450   0.0  A  0.0  0.0
OW  15.9994  0.0  A  0.0  0.0
HW   1.008   0.0  A  0.0  0.0
MW   0.000   0.0  A  0.0  0.0

[ moleculetype ]
NA 1
[ atoms ]
1 Na 1 NA NA 1 1.0 22.990

[ moleculetype ]
CL 1
[ atoms ]
1 Cl 1 CL CL 1 -1.0 35.450

[ moleculetype ]
SOL 2
[ atoms ]
1 OW 1 SOL OW  1  0.0   15.9994
2 HW 1 SOL HW1 1  0.52   1.008
3 HW 1 SOL HW2 1  0.52   1.008
4 MW 1 SOL MW  1 -1.04   0.000
[ settles ]
1 1 0.09572 0.15139
[ exclusions ]
1 2 3 4
2 1 3 4
3 1 2 4
4 1 2 3
[ virtual_sites3 ]
4 1 2 3 1 0.128012065 0.128012065

[ system ]
intermixed virtual sites
[ molecules ]
NA 1
SOL 1
CL 1
SOL 1
NA 1
"""
    gro = ["intermixed TIP4P and ions", f"{len(sites):5d}"]
    for number, (name, residue, _mass, (x, y, z), _virtual) in enumerate(
        sites, start=1
    ):
        gro.append(
            f"{1:5d}{residue:<5}{name:>5}{number:5d}"
            f"{x:8.3f}{y:8.3f}{z:8.3f}"
        )
    gro.append(f"{3.0:10.5f}{3.0:10.5f}{3.0:10.5f}")
    mdp = """integrator = md
nsteps = 0
dt = 0.001
cutoff-scheme = Verlet
nstlist = 10
pbc = xyz
coulombtype = Cut-off
rcoulomb = 0.9
rvdw = 0.9
rlist = 0.9
DispCorr = no
constraints = none
continuation = yes
"""
    for name, text in (
        ("mix.top", top),
        ("mix.gro", "\n".join(gro) + "\n"),
        ("mix.mdp", mdp),
    ):
        with open(os.path.join(directory, name), "w", encoding="utf-8") as handle:
            handle.write(text)


def gmx(prefix):
    for name in ("gmx_d", "gmx"):
        path = os.path.join(prefix, "bin", name)
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    raise RuntimeError(f"no gmx binary under {prefix}/bin")


def build_tpr(prefix, directory):
    binary = gmx(prefix)
    run(
        [
            binary,
            "grompp",
            "-f",
            "mix.mdp",
            "-c",
            "mix.gro",
            "-p",
            "mix.top",
            "-o",
            "mix.tpr",
            "-maxwarn",
            "3",
        ],
        directory,
    )
    dumped = run([binary, "dump", "-s", "mix.tpr"], directory)
    rows = dumped_coordinates(dumped)
    sites = atoms()
    if len(rows) != len(sites):
        raise RuntimeError(
            f"gmx dump reported {len(rows)} coordinates, expected {len(sites)}\n"
            + dumped[:4000]
        )
    # grompp rebuilds a virtual site from its constructors, so only the
    # massive atoms are required to keep the coordinate file.
    for index, site in enumerate(sites):
        if site[4]:
            continue
        for axis, value in enumerate(rows[index]):
            if abs(value - site[3][axis]) > 1.0e-4:
                raise RuntimeError(
                    f"TPR coordinate {index} is {rows[index]}, "
                    f"gro has {site[3]}"
                )
    return os.path.join(directory, "mix.tpr"), rows


def dumped_coordinates(text):
    """Coordinate rows from `gmx dump -s`, in TPR order.

    The printer writes one row per atom, for example
    x[    0]={ 3.00000e-01,  3.00000e-01,  3.00000e-01}.
    """
    rows = []
    # The name is exactly x. box[i] also ends in x and must not match.
    pattern = re.compile(
        r"(?:^|[^A-Za-z0-9_])x\[\s*(\d+)\]=\{\s*([^,]+),\s*([^,]+),\s*([^}]+)\}"
    )
    for line in text.splitlines():
        match = pattern.search(line.strip())
        if match is None:
            continue
        index = int(match.group(1))
        row = tuple(float(match.group(axis)) for axis in (2, 3, 4))
        if index != len(rows):
            raise RuntimeError(
                f"gmx dump x index {index} is not the next coordinate"
            )
        rows.append(row)
    return rows


class Ops(C.Structure):
    _fields_ = [
        ("struct_size", C.c_size_t),
        ("set_positions", C.c_void_p),
        ("set_velocities", C.c_void_p),
        ("step_langevin", C.c_void_p),
        ("step_nve", C.c_void_p),
        ("get_state", C.c_void_p),
        ("get_masses", C.c_void_p),
        ("destroy", C.c_void_p),
        ("set_periodic_box", C.c_void_p),
        ("get_periodic_box", C.c_void_p),
        ("initialize", C.c_void_p),
        ("run_langevin", C.c_void_p),
        ("get_neighbors", C.c_void_p),
        ("set_neighbors", C.c_void_p),
        ("get_source_indices", C.c_void_p),
    ]


STATE = C.CFUNCTYPE(
    C.c_int,
    C.c_void_p,
    C.POINTER(C.c_double),
    C.POINTER(C.c_double),
    C.POINTER(C.c_double),
    C.c_size_t,
    C.POINTER(C.c_double),
    C.POINTER(C.c_double),
    C.c_char_p,
    C.c_size_t,
)
MASSES = C.CFUNCTYPE(
    C.c_int,
    C.c_void_p,
    C.POINTER(C.c_double),
    C.c_size_t,
    C.c_char_p,
    C.c_size_t,
)
INDICES = C.CFUNCTYPE(
    C.c_int,
    C.c_void_p,
    C.POINTER(C.c_int),
    C.c_size_t,
    C.c_char_p,
    C.c_size_t,
)
DESTROY = C.CFUNCTYPE(None, C.c_void_p)


def load_engine(path):
    lib = C.CDLL(path)
    create = lib.rgpot_md_session_engine_create
    create.argtypes = [
        C.c_void_p,
        C.c_size_t,
        C.c_char_p,
        C.POINTER(Ops),
        C.POINTER(C.c_void_p),
        C.POINTER(C.c_size_t),
        C.c_char_p,
        C.c_size_t,
    ]
    create.restype = C.c_int
    return lib, create


def open_session(create, spec, options=b""):
    ops = Ops()
    ctx = C.c_void_p()
    count = C.c_size_t()
    err = C.create_string_buffer(1024)
    code = create(
        spec,
        C.c_size_t(len(spec)),
        options,
        C.byref(ops),
        C.byref(ctx),
        C.byref(count),
        err,
        C.c_size_t(len(err)),
    )
    if code != 0:
        raise RuntimeError(err.value.decode(errors="replace"))
    return ops, ctx, count.value, err


def close_session(ops, ctx):
    if ops.destroy and ctx:
        DESTROY(ops.destroy)(ctx)


def callback_ready(ops):
    offset = Ops.get_source_indices.offset
    width = C.sizeof(C.c_void_p)
    return ops.struct_size >= offset + width and bool(ops.get_source_indices)


def read_masses(ops, ctx, count, err):
    masses = (C.c_double * count)()
    if MASSES(ops.get_masses)(ctx, masses, count, err, C.c_size_t(len(err))):
        raise RuntimeError(err.value.decode(errors="replace"))
    return masses


def read_packed(ops, ctx, count, err):
    n3 = 3 * count
    positions = (C.c_double * n3)()
    velocities = (C.c_double * n3)()
    forces = (C.c_double * n3)()
    masses = (C.c_double * count)()
    potential = C.c_double()
    kinetic = C.c_double()
    state = STATE(ops.get_state)
    mass_fn = MASSES(ops.get_masses)
    if state(
        ctx,
        positions,
        velocities,
        forces,
        n3,
        C.byref(potential),
        C.byref(kinetic),
        err,
        C.c_size_t(len(err)),
    ):
        raise RuntimeError(err.value.decode(errors="replace"))
    if mass_fn(ctx, masses, count, err, C.c_size_t(len(err))):
        raise RuntimeError(err.value.decode(errors="replace"))
    return positions, masses


def read_indices(ops, ctx, count, err):
    if not callback_ready(ops):
        return None
    indices = (C.c_int * count)()
    if INDICES(ops.get_source_indices)(
        ctx, indices, count, err, C.c_size_t(len(err))
    ):
        raise RuntimeError(err.value.decode(errors="replace"))
    return [indices[k] for k in range(count)]


def check_tpr(create, path, coordinates):
    spec = ("tpr " + path).encode()
    ops, ctx, count, err = open_session(create, spec, b"threads=1")
    try:
        sites = atoms()
        massive = [i for i, site in enumerate(sites) if not site[4]]
        virtual = [i for i, site in enumerate(sites) if site[4]]
        indices = read_indices(ops, ctx, count, err)
        if indices is None:
            return (
                "the session vtable has no get_source_indices "
                f"(struct_size={ops.struct_size})"
            )
        if count != len(massive) or indices != massive:
            return f"packed indices {indices} for {count} sites, expected {massive}"
        masses = read_masses(ops, ctx, count, err)
        for packed, source in enumerate(indices):
            if source < 0 or source >= len(sites):
                return f"packed site {packed} names index {source} outside the TPR"
            if source in virtual:
                return f"packed site {packed} names virtual site {source}"
            mass = sites[source][2]
            if abs(masses[packed] - mass) > 1.0e-6:
                return (
                    f"site {packed} source {source} mass {masses[packed]} "
                    f"expected {mass}"
                )
        if coordinates is not None:
            positions, _masses = read_packed(ops, ctx, count, err)
            for packed, source in enumerate(indices):
                xyz = coordinates[source]
                for axis in range(3):
                    got = positions[3 * packed + axis] / ANGSTROM_PER_NM
                    if abs(got - xyz[axis]) > 1.0e-4:
                        return (
                            f"site {packed} source {source} axis {axis} "
                            f"is {got} nm, TPR has {xyz[axis]}"
                        )
        reported = set(indices)
        for source in virtual:
            if source in reported:
                return f"virtual site {source} was given a packed slot"
        print(
            "tpr packed sites",
            count,
            "indices",
            indices,
            "virtual sites absent",
            virtual,
        )
        return None
    finally:
        close_session(ops, ctx)


def check_text(create):
    spec = (
        b"nblib 1\ncutoff 0.8\ncoulomb none\n"
        b"atomtype Ar 39.948 0.001 0.000001\n"
        b"molecule argon 2\natom Ar Ar 0\nend\n"
    )
    ops, ctx, count, err = open_session(create, spec)
    try:
        indices = read_indices(ops, ctx, count, err)
        if indices is None:
            return "text session vtable has no get_source_indices"
        if count != 2 or indices != [0, 1]:
            return f"text topology indices {indices} count {count}"
        print("text topology indices", indices)
        return None
    finally:
        close_session(ops, ctx)


def check_library(librgpot, engine, path):
    os.environ["RGPOT_NBLIB_ENGINE"] = engine
    lib = C.CDLL(librgpot)
    symbol = getattr(lib, "rgpot_md_session_get_source_indices", None)
    if symbol is None:
        return "librgpot does not export rgpot_md_session_get_source_indices"
    symbol.argtypes = [
        C.c_void_p,
        C.POINTER(C.c_int),
        C.c_size_t,
        C.c_char_p,
        C.c_size_t,
    ]
    symbol.restype = C.c_int
    create = lib.rgpot_md_session_create
    create.argtypes = [
        C.c_char_p,
        C.c_char_p,
        C.c_void_p,
        C.c_size_t,
        C.c_char_p,
        C.c_char_p,
        C.c_size_t,
    ]
    create.restype = C.c_void_p
    destroy = lib.rgpot_md_session_destroy
    destroy.argtypes = [C.c_void_p]
    count_of = lib.rgpot_md_session_n_atoms
    count_of.argtypes = [C.c_void_p]
    count_of.restype = C.c_size_t
    spec = ("tpr " + path).encode()
    err = C.create_string_buffer(1024)
    session = create(
        b"nblib",
        None,
        spec,
        len(spec),
        b"threads=1",
        err,
        len(err),
    )
    if not session:
        return "rgpot_md_session_create failed: " + err.value.decode(
            errors="replace"
        )
    try:
        count = count_of(session)
        indices = (C.c_int * count)()
        if symbol(session, indices, count, err, len(err)) != 0:
            return err.value.decode(errors="replace")
        got = [indices[k] for k in range(count)]
        expected = [i for i, site in enumerate(atoms()) if not site[4]]
        if got != expected:
            return f"C API indices {got}, expected {expected}"
        print("C API indices", got)
        return None
    finally:
        destroy(session)


def check_coordinates(engine, path, coordinates, directory):
    """Compare packed positions with the TPR in a fresh process.

    The first force evaluation reads a nonbonded kernel type that is not a
    kernel the dispatcher accepts when glibc fills new heap allocations.
    The project test harness sets MALLOC_PERTURB_. A child started without
    that variable still checks the coordinates.
    """
    listing = os.path.join(directory, "coordinates.txt")
    with open(listing, "w", encoding="utf-8") as handle:
        for row in coordinates:
            handle.write(" ".join(f"{value:.8e}" for value in row) + "\n")
    env = os.environ.copy()
    env.pop("MALLOC_PERTURB_", None)
    env["RGPOT_TPR_INDEX_COORDS"] = "1"
    done = subprocess.run(
        [sys.executable, os.path.abspath(__file__), engine, path, listing],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if done.returncode != 0:
        detail = (done.stdout + done.stderr).strip()
        return "coordinate check failed\n" + detail
    print(done.stdout.strip())
    return None


def coordinate_main():
    engine, path, listing = sys.argv[1:]
    coordinates = []
    with open(listing, encoding="utf-8") as handle:
        for line in handle:
            coordinates.append(tuple(float(part) for part in line.split()))
    _lib, create = load_engine(engine)
    problem = check_tpr(create, path, coordinates)
    if problem:
        print(problem)
        return 1
    return 0


def main():
    if os.environ.get("RGPOT_TPR_INDEX_COORDS") == "1":
        return coordinate_main()
    if len(sys.argv) != 4:
        print(
            "usage: test_tpr_source_index.py ENGINE.so GMX_PREFIX librgpot.so",
            file=sys.stderr,
        )
        return 2
    engine, prefix, librgpot = sys.argv[1:]
    problems = []
    with tempfile.TemporaryDirectory(prefix="rgpot-tpr-index-") as directory:
        write_inputs(directory)
        # grompp and dump must run before the engine is loaded. Loading
        # libgromacs starts threads, and a later fork leaves that process
        # unable to dispatch the nonbonded kernel.
        path, coordinates = build_tpr(prefix, directory)
        _lib, create = load_engine(engine)
        tpr = check_tpr(create, path, None)
        if tpr:
            problems.append(tpr)
        else:
            coordinates_problem = check_coordinates(
                engine, path, coordinates, directory
            )
            if coordinates_problem:
                problems.append(coordinates_problem)
        text = check_text(create)
        if text:
            problems.append(text)
        library = check_library(librgpot, engine, path)
        if library:
            problems.append(library)
    if problems:
        for problem in problems:
            print("FAIL:", problem)
        return 1
    # The geometry used above is a real TIP4P O-H / H-H pair at gro precision.
    doh = math.hypot(0.076, 0.059)
    if abs(doh - 0.09572) > 0.01 * 0.09572:
        print("FAIL: water geometry is far from the SETTLE length")
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
