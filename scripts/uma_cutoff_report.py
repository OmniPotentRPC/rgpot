#!/usr/bin/env python3
# Copyright 2023--present rgpot developers
"""Build the cutoff bench table from raw per-call samples.

Seed rule, fixed before the 20261123 collection: bootstrap uses
20261123, 10000 resamples, percentiles 250 and 9749. The two earlier
seed runs stay in the table. The script does not drop one of them.

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

SEED = 20261123
BOOTS = 10000
# Recorded fixed-atoms exponents from the two earlier seed runs.
# Each value is the slope printed for that seed. They stay together.
PRIOR = (
    (20261121, 0.564866, 0.542017, 0.576587),
    (20261122, 0.516200, 0.511053, 0.521105),
)


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


def main() -> int:
    rows = load(Path(sys.argv[1]))
    print(f"seed_rule {SEED} boots {BOOTS} percentile 250 9749")
    reused = [r["us"] for r in rows if r["series"] == "reused65536"]
    print(
        "reused n=65536 threads=8 "
        f"median={median(reused):.3f} iqr={iqr(reused):.3f} n={len(reused)}"
    )
    print("fixed_atoms_per_thread")
    for seed, point, lo, hi in PRIOR:
        print(
            f"prior seed={seed} exponent={point:.6f} "
            f"lo={lo:.6f} hi={hi:.6f}"
        )
    weak = grouped(rows, "weak_atoms")
    for n, samples in weak:
        threads = n // 8192
        print(
            f"weak_atoms n={n} threads={threads} "
            f"median={median(samples):.3f} iqr={iqr(samples):.3f}"
        )
    point, lo, hi = slope(weak, SEED)
    print(
        f"weak_atoms seed={SEED} exponent={point:.6f} "
        f"lo={lo:.6f} hi={hi:.6f}"
    )
    exponents = [item[1] for item in PRIOR] + [point]
    spread = max(exponents) - min(exponents)
    print(
        "fixed_atoms_spread "
        f"min={min(exponents):.6f} max={max(exponents):.6f} "
        f"spread={spread:.6f}"
    )
    print("by_size_one_thread")
    by_size = grouped(rows, "by_size")
    for n, samples in by_size:
        print(
            f"by_size n={n} threads=1 "
            f"median={median(samples):.3f} iqr={iqr(samples):.3f}"
        )
    bpoint, blo, bhi = slope(by_size, SEED)
    print(
        f"by_size seed={SEED} exponent={bpoint:.6f} "
        f"lo={blo:.6f} hi={bhi:.6f}"
    )
    print(
        "by_size_meaning slope of log(median time) versus log(N) "
        "at one thread. The 1.05 target is this slope."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
