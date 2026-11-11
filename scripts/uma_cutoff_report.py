#!/usr/bin/env python3
# Copyright 2023--present rgpot developers
"""Build the cutoff bench table from raw per-call samples.

Each file carries its own seed. The bootstrap for that file uses the
seed, 10000 resamples, and percentiles 250 and 9749. Recorded seeds
that have no sample file stay in the table. The script does not drop
one of them.

A by-size series keeps one thread and varies the atom count. Its
exponent is the slope of log(median time) against log(N). The 1.05
target is that slope. A fixed-atoms series grows the thread count
with N. Its slope is not the 1.05 target. A linear builder with
perfect parallel efficiency has slope 0 on that series.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

BOOTS = 10000


def load(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        item = {}
        for tok in line.split():
            key, value = tok.split("=", 1)
            item[key] = value
        item["us"] = float(item["us"])
        rows.append(item)
    return rows


def median(xs):
    ys = sorted(xs)
    n = len(ys)
    if n % 2:
        return ys[n // 2]
    return 0.5 * (ys[n // 2 - 1] + ys[n // 2])


def iqr(xs):
    ys = sorted(xs)
    n = len(ys)
    return ys[(3 * n) // 4] - ys[n // 4]


def lcg(state):
    return (state * 1664525 + 1013904223) & 0xFFFFFFFF


def ols(xs, ys):
    n = float(len(xs))
    sx = sy = sxx = sxy = 0.0
    for x, y in zip(xs, ys):
        lx = math.log(x)
        ly = math.log(y)
        sx += lx
        sy += ly
        sxx += lx * lx
        sxy += lx * ly
    return (n * sxy - sx * sy) / (n * sxx - sx * sx)


def slope(groups, seed):
    if len(groups) < 2:
        return None
    point = ols([g[0] for g in groups], [median(g[1]) for g in groups])
    boots = []
    state = seed
    for _ in range(BOOTS):
        meds = []
        for _n, samples in groups:
            draw = []
            m = len(samples)
            for _k in range(m):
                state = lcg(state)
                draw.append(samples[(state >> 8) % m])
            meds.append(median(draw))
        boots.append(ols([g[0] for g in groups], meds))
    boots.sort()
    return point, boots[250], boots[9749]


def grouped(rows, series):
    buckets = {}
    order = []
    for row in rows:
        if row["series"] != series:
            continue
        key = (int(row["n"]), int(row["threads"]))
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(row["us"])
    return [(n, buckets[(n, threads)]) for n, threads in order]


def report_file(rows, role: str) -> tuple[float, float]:
    seed = int(rows[0]["seed"])
    print(f"{role} seed={seed} boots {BOOTS} percentile 250 9749")
    reused = [r["us"] for r in rows if r["series"] == "reused65536"]
    if reused:
        print(
            f"{role} reused n=65536 threads=8 "
            f"median={median(reused):.3f} iqr={iqr(reused):.3f} n={len(reused)}"
        )
    weak = grouped(rows, "weak_atoms")
    for n, samples in weak:
        threads = n // 8192
        print(
            f"{role} weak_atoms n={n} threads={threads} "
            f"median={median(samples):.3f} iqr={iqr(samples):.3f}"
        )
    fitted = slope(weak, seed)
    point = 0.0
    if fitted is not None:
        point, lo, hi = fitted
        print(
            f"{role} weak_atoms seed={seed} exponent={point:.6f} "
            f"lo={lo:.6f} hi={hi:.6f}"
        )
    by_size = grouped(rows, "by_size")
    bpoint = 0.0
    if by_size:
        for n, samples in by_size:
            print(
                f"{role} by_size n={n} threads=1 "
                f"median={median(samples):.3f} iqr={iqr(samples):.3f}"
            )
        bpoint, blo, bhi = slope(by_size, seed)
        print(
            f"{role} by_size seed={seed} exponent={bpoint:.6f} "
            f"lo={blo:.6f} hi={bhi:.6f}"
        )
    known = {"reused65536", "weak_atoms", "by_size", "abba_v17", "abba_now"}
    seen = []
    for row in rows:
        key = (row["series"], row.get("n", ""), row.get("threads", ""), row.get("ver", ""))
        if row["series"] in known or key in seen:
            continue
        seen.append(key)
        samples = [
            r["us"]
            for r in rows
            if r["series"] == key[0]
            and r.get("n") == key[1]
            and r.get("threads") == key[2]
            and r.get("ver", "") == key[3]
        ]
        ver = f" ver={key[3]}" if key[3] else ""
        print(
            f"{role} {key[0]} n={key[1]} threads={key[2]}{ver} "
            f"median={median(samples):.3f} iqr={iqr(samples):.3f} n={len(samples)}"
        )
    for series in ("abba_v17", "abba_now"):
        samples = [r["us"] for r in rows if r["series"] == series]
        if not samples:
            continue
        print(
            f"{role} {series} n=65536 threads=8 "
            f"median={median(samples):.3f} iqr={iqr(samples):.3f} "
            f"n={len(samples)}"
        )
    return point, bpoint


def pair_report(rows) -> None:
    if not rows or "ver" not in rows[0]:
        return
    buckets = {}
    order = []
    for row in rows:
        key = (row["series"], int(row["n"]), int(row["threads"]), int(row["call"]))
        slot = buckets.setdefault(key, {})
        if key not in order:
            order.append(key)
        slot[row["ver"]] = row["us"]
    cells = []
    for key in order:
        cell = (key[0], key[1], key[2])
        if cell not in cells:
            cells.append(cell)
    print("pairs same call index, later version over v183")
    for series, n, threads in cells:
        for other in ("v184", "v186"):
            ratios = []
            slower = 0
            for key, slot in buckets.items():
                if (key[0], key[1], key[2]) != (series, n, threads):
                    continue
                if "v183" not in slot or other not in slot:
                    continue
                ratio = slot[other] / slot["v183"]
                ratios.append(ratio)
                if slot[other] > slot["v183"]:
                    slower += 1
            if len(ratios) < 2:
                continue
            seed = int(rows[0]["seed"])
            state = seed
            boots = []
            count = len(ratios)
            for _draw in range(BOOTS):
                sample = []
                for _k in range(count):
                    state = lcg(state)
                    sample.append(ratios[(state >> 8) % count])
                boots.append(median(sample))
            boots.sort()
            lo, hi = boots[250], boots[9749]
            print(
                f"pair {series} n={n} threads={threads} {other}/v183 "
                f"median={median(ratios):.3f} iqr={iqr(ratios):.3f} "
                f"n={len(ratios)} slower={slower} lo={lo:.3f} hi={hi:.3f}"
            )


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: uma_cutoff_report.py calls.txt ...", file=sys.stderr)
        return 2
    print("fixed_atoms_per_thread")
    print(
        "fixed_atoms_meaning 8192 atoms per thread. "
        "Exponent 0 is flat. This is not the 1.05 target."
    )
    points = []
    for arg in sys.argv[1:]:
        rows = load(Path(arg))
        point, _by = report_file(rows, "run")
        pair_report(rows)
        if len(grouped(rows, "weak_atoms")) >= 2:
            points.append(point)
    if len(points) >= 2:
        print(
            "fixed_atoms_spread "
            f"min={min(points):.6f} max={max(points):.6f} "
            f"spread={max(points) - min(points):.6f} n_seeds={len(points)}"
        )
    print(
        "by_size_meaning slope of log(median time) versus log(N) "
        "at one thread. The 1.05 target is this slope."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
