#!/usr/bin/env python3
# Copyright 2023--present rgpot developers
"""Write a provenance sidecar for raw bench files.

The raw files are read and hashed. This script does not modify them.
Each row names the raw file by its sha256. head and tree here are
attributions. The original driver did not print them.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

V171 = "79e7d5d884e2509bea9ba62464bffb3faa746f1a"
V171_TREE = "38c171eb8751d0f7f583c2515563190d0b2cef31"
RECYCLE = "74475a12625bb738f3630a8e49e1d663f82ed520"
RECYCLE_TREE = "a65af381df733b5eb8db65b94022c5eef365f120"
OLD = "a98fbfeb4c2c82d8ac45010c47122169e4d9ab34"
OLD_TREE = "75e1579170f79fe05734013548f6171207965226"
# Pre-replay commit to the commit this branch replays it as.
REPLAY = {
    V171: "f9d686a4d2b816074bc0a02e7a73a45e91c1da70",
    RECYCLE: "b95a7041c82c3307b8ce18b4a7af4448c829193f",
    OLD: "2eb9f61dd3c68e9adbf74b5b145af38cd4040cc1",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def data_rows(path: Path):
    for index, line in enumerate(path.read_text().splitlines()):
        if not line or line.startswith("#"):
            continue
        item = {"_index": str(index)}
        for tok in line.split():
            key, value = tok.split("=", 1)
            item[key] = value
        yield item


def attribution(name: str, series: str):
    if name == "uma_cutoff_abba_20261126.txt" and series == "abba_v17":
        return V171, V171_TREE, "v1.7.1 binary"
    if name == "uma_cutoff_abba_20261126.txt":
        return "pre-commit", "pre-commit", "recorded at 16:40 CST before f3c8ef56"
    if name in (
        "uma_cutoff_calls_20261121.txt",
        "uma_cutoff_calls_20261122.txt",
        "uma_cutoff_calls_20261125.txt",
    ):
        return RECYCLE, RECYCLE_TREE, "recycle builder"
    if name in ("uma_cutoff_calls.txt", "uma_cutoff_calls_20261124.txt"):
        return OLD, OLD_TREE, "builder before page recycle"
    return "unprinted", "unprinted", "no attribution"


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: uma_cutoff_sidecar.py raw.txt ...", file=sys.stderr)
        return 2
    lines = []
    lines.append("corrections abba_v17 rows are the v1.7.1 binary")
    lines.append(f"corrections abba_v17 head {V171} tree {V171_TREE}")
    lines.append("corrections abba_now and the other 20261126 rows are pre-commit")
    lines.append("corrections those rows were taken at 16:40 CST before f3c8ef56")
    lines.append("corrections they are not the recycle commit and not f3c8ef56")
    lines.append(
        "replay 79e7d5d884e2509bea9ba62464bffb3faa746f1a "
        + REPLAY[V171]
    )
    lines.append(
        "replay 74475a12625bb738f3630a8e49e1d663f82ed520 "
        + REPLAY[RECYCLE]
    )
    lines.append(
        "replay a98fbfeb4c2c82d8ac45010c47122169e4d9ab34 "
        + REPLAY[OLD]
    )
    lines.append(
        "corrections seeds 20261123 and 20261124 name a98fbfeb by code"
    )
    lines.append(
        "corrections those seeds were committed at edafc405 and 74475a12"
    )
    lines.append(
        "corrections their tree does not match a98fbfeb"
    )
    for arg in sys.argv[1:]:
        path = Path(arg)
        digest = sha256(path)
        lines.append(f"file {path.name} sha256 {digest}")
        for item in data_rows(path):
            head, tree, note = attribution(path.name, item.get("series", ""))
            replayed = REPLAY.get(head, "none")
            lines.append(
                "row file_sha256={digest} index={index} series={series} "
                "seed={seed} reuse={reuse} n={n} threads={threads} call={call} "
                "head={head} tree={tree} replayed={replayed}".format(
                    digest=digest,
                    index=item["_index"],
                    series=item.get("series", ""),
                    seed=item.get("seed", ""),
                    reuse=item.get("reuse", ""),
                    n=item.get("n", ""),
                    threads=item.get("threads", ""),
                    call=item.get("call", ""),
                    head=head,
                    tree=tree,
                    replayed=replayed,
                )
            )
            lines.append(f"note {note}")
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
