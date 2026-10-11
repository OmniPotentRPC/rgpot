"""One invariance measurement. Prints a SHA-256 of the public outputs.

Python thread count is argv[1]. OMP_NUM_THREADS is already in the environment.
"""

from __future__ import annotations

import hashlib
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from support import lj_dimer, pt7, raw_bytes


def main() -> None:
    threads = int(sys.argv[1])
    import rgpot
    from rgpot.ase import RgpotCalculator

    lj_frames = [lj_dimer(distance) for distance in (1.15, 1.6, 2.05)]
    chunks: list[bytes] = []

    lj = RgpotCalculator("lj")
    serial = []
    for atoms in lj_frames:
        row = lj.__rg_evaluate__(
            atoms.get_positions(),
            atoms.get_atomic_numbers(),
            np.asarray(atoms.cell.array),
            atoms.pbc,
        )
        serial.append(raw_bytes(row["energy"], row["forces"]))
    batched = lj.calculate_many(lj_frames, threads=threads)
    batch_bytes = [
        raw_bytes(row["energy"], row["forces"]) for row in batched
    ]
    if batch_bytes != serial:
        raise SystemExit("LJ calculate_many bytes differ from serial calls")

    def one(atoms) -> bytes:
        calc = RgpotCalculator("lj")
        row = calc.__rg_evaluate__(
            atoms.get_positions(),
            atoms.get_atomic_numbers(),
            np.asarray(atoms.cell.array),
            atoms.pbc,
        )
        return raw_bytes(row["energy"], row["forces"])

    if threads == 1:
        concurrent = [one(atoms) for atoms in lj_frames]
    else:
        with ThreadPoolExecutor(max_workers=threads) as pool:
            concurrent = list(pool.map(one, lj_frames))
    if concurrent != serial:
        raise SystemExit("LJ concurrent calculate bytes differ from serial calls")
    chunks.extend(serial)

    if rgpot.has_fortran_pots:
        eam_frames = [pt7(jitter=0.02, seed=index) for index in range(4)]
        eam = RgpotCalculator("eam_al")
        eam_serial = []
        for atoms in eam_frames:
            row = eam.__rg_evaluate__(
                atoms.get_positions(),
                atoms.get_atomic_numbers(),
                np.asarray(atoms.cell.array),
                atoms.pbc,
            )
            eam_serial.append(raw_bytes(row["energy"], row["forces"]))
        eam_batch = [
            raw_bytes(row["energy"], row["forces"])
            for row in eam.calculate_many(eam_frames, threads=threads)
        ]
        if eam_batch != eam_serial:
            raise SystemExit("EAM calculate_many bytes differ from serial calls")

        def one_eam(atoms) -> bytes:
            calc = RgpotCalculator("eam_al")
            row = calc.__rg_evaluate__(
                atoms.get_positions(),
                atoms.get_atomic_numbers(),
                np.asarray(atoms.cell.array),
                atoms.pbc,
            )
            return raw_bytes(row["energy"], row["forces"])

        if threads == 1:
            eam_concurrent = [one_eam(atoms) for atoms in eam_frames]
        else:
            with ThreadPoolExecutor(max_workers=threads) as pool:
                eam_concurrent = list(pool.map(one_eam, eam_frames))
        if eam_concurrent != eam_serial:
            raise SystemExit("EAM concurrent calculate bytes differ")
        chunks.extend(eam_serial)

    digest = hashlib.sha256(b"".join(chunks)).hexdigest()
    print(digest)


if __name__ == "__main__":
    main()
