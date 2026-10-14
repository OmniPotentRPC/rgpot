#!/usr/bin/env python3
"""Compare classical dumps across threads and ranks.

With --strict, any moved entry, missing reference or throw exits 1.
"""

import pathlib
import statistics
import struct
import sys

MAGIC = 0x52504749
FAILURES: list[str] = []


def fail(msg: str) -> None:
    FAILURES.append(msg)


def load_bin(path: pathlib.Path):
    data = path.read_bytes()
    magic, n, has, _pad = struct.unpack_from("<IIII", data, 0)
    if magic != MAGIC:
        raise SystemExit(f"bad magic {path}")
    energy = struct.unpack_from("<d", data, 16)[0]
    stress = struct.unpack_from("<9d", data, 24)
    forces = struct.unpack_from(f"<{3 * n}d", data, 96)
    return energy, stress, forces, has


def pack_f(value: float) -> bytes:
    # Signed zero and NaN payloads stay distinct. Python != does not.
    return struct.pack("<d", value)


def n_diff(a, b) -> int:
    if len(a) != len(b):
        return max(len(a), len(b))
    return sum(pack_f(x) != pack_f(y) for x, y in zip(a, b))


def one_diff(a: float, b: float) -> int:
    return int(pack_f(a) != pack_f(b))


def ranks(directory: pathlib.Path, pot: str):
    files = sorted(directory.glob(f"{pot}.r*.bin")) + sorted(
        directory.glob(f"{pot}.r*.err")
    )
    return files


def report_inside(directory: pathlib.Path, label: str) -> None:
    for pot in ("lj", "morse", "eam"):
        files = sorted(p for p in directory.glob(f"{pot}.r*") if p.suffix in {".bin", ".err"})
        bins = [p for p in files if p.suffix == ".bin"]
        if len(bins) < 2:
            continue
        energy, stress, forces, _has = load_bin(bins[0])
        for other in bins[1:]:
            e2, s2, f2, _h2 = load_bin(other)
            de, df, ds = one_diff(energy, e2), n_diff(forces, f2), n_diff(stress, s2)
            print(
                f"{label} {pot} {bins[0].name} vs {other.name}: "
                f"energy_entries {de} force_entries {df} stress_entries {ds}"
            )
            if de or df or ds:
                fail(f"{label} {pot} {other.name}")


def report_group(label: str, directories: list[pathlib.Path]) -> None:
    pots = ("lj", "morse", "eam")
    base = directories[0]
    for pot in pots:
        ref_files = ranks(base, pot)
        if not ref_files:
            print(f"{label} {pot}: missing")
            fail(f"{label} {pot} missing")
            continue
        ref = ref_files[0]
        if ref.suffix == ".err":
            print(f"{label} {pot}: threw ({ref.read_text()[:80]})")
            fail(f"{label} {pot} threw")
            continue
        energy, stress, forces, has = load_bin(ref)
        print(
            f"{label} {pot}: n_atoms {len(forces) // 3} has_stress {has} "
            f"energy {energy:.8g}"
        )
        for other_dir in directories[1:]:
            got_files = ranks(other_dir, pot)
            if not got_files:
                print(f"  vs {other_dir.name}: not run")
                continue
            got = got_files[0]
            if got.suffix == ".err":
                print(f"  vs {other_dir.name}: threw")
                fail(f"{label} {pot} {other_dir.name} threw")
                continue
            e2, s2, f2, _h2 = load_bin(got)
            de, df, ds = one_diff(energy, e2), n_diff(forces, f2), n_diff(stress, s2)
            print(
                f"  vs {other_dir.name}: energy_entries {de} "
                f"force_entries {df} stress_entries {ds}"
            )
            if de or df or ds:
                fail(f"{label} {pot} {other_dir.name}")


def medians(path: pathlib.Path) -> None:
    rows: dict[tuple[str, str], list[float]] = {}
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 4 or parts[0] != "bench":
            continue
        _b, pot, threads, sec = parts
        rows.setdefault((pot, threads), []).append(float(sec))
    print("scaling")
    for (pot, threads), samples in sorted(rows.items()):
        print(
            f"  {pot} threads {threads} n {len(samples)} "
            f"median {statistics.median(samples):.6f} "
            f"min {min(samples):.6f} max {max(samples):.6f}"
        )


def main() -> None:
    args = sys.argv[1:]
    strict = "--strict" in args
    args = [a for a in args if a != "--strict"]
    if len(args) != 1:
        raise SystemExit("usage: compare_hashes.py [--strict] DUMP_ROOT")
    root = pathlib.Path(args[0])
    for n in (4000, 32000):
        thread_dirs = [root / f"n{n}_t{t}" for t in (1, 2, 3, 8)]
        thread_dirs = [d for d in thread_dirs if d.is_dir()]
        if thread_dirs:
            report_group(f"n {n} threads", thread_dirs)
        rank_dirs = [root / f"n{n}_np{np}" for np in (1, 2, 3)]
        rank_dirs = [d for d in rank_dirs if d.is_dir()]
        if rank_dirs:
            report_group(f"n {n} ranks", rank_dirs)
            for d in rank_dirs:
                report_inside(d, f"n {n} inside {d.name}")
    edge_dirs = [root / f"edge_t{t}" for t in (1, 2, 3, 8)]
    edge_dirs = [d for d in edge_dirs if d.is_dir()]
    if not edge_dirs and (root / "edge").is_dir():
        edge_dirs = [root / "edge"]
    if len(edge_dirs) > 1:
        report_group("edge threads", edge_dirs)
    if edge_dirs:
        edge = edge_dirs[0]
        for pot in ("lj", "morse", "eam"):
            err = list(edge.glob(f"{pot}.r*.err"))
            bins = list(edge.glob(f"{pot}.r*.bin"))
            if err:
                print(f"edge {pot}: threw ({err[0].read_text()[:80]})")
            elif bins:
                energy, _s, forces, has = load_bin(bins[0])
                nan_f = sum(x != x for x in forces)
                print(
                    f"edge {pot}: has_stress {has} energy_nan {int(energy != energy)} "
                    f"force_nan {nan_f}"
                )
    medians(root / "bench.txt")
    print(f"mismatches {len(FAILURES)}")
    for msg in FAILURES:
        print(f"  {msg}")
    if strict and FAILURES:
        sys.exit(1)


if __name__ == "__main__":
    main()
