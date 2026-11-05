"""Scan rgpot potentials over RPC and draw the tutorial figures.

    python docs/orgmode/tutorials/pes_scan/scan_pes.py --server-bin ./bbdir/CppCore/potserv

Writes dimer_curves.png and cuh2_elbow.png next to this file and prints
every number the tutorial quotes.
"""

import argparse
import asyncio
import socket
import subprocess
from pathlib import Path

import capnp
import matplotlib
import numpy as np
from evaluate_once import ROOT, calculate, connect
from scipy import ndimage

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
STEP = 1e-5


def scan(binary, name, frames, numbers, box):
    """Start potserv with one potential, evaluate every frame, stop it."""
    with socket.socket() as s:
        s.bind(("localhost", 0))
        port = s.getsockname()[1]
    server = subprocess.Popen([binary, str(port), name], stdout=subprocess.DEVNULL)

    async def run():
        _client, potential = await connect(port)
        return [await calculate(potential, f, numbers, box) for f in frames]

    try:
        results = asyncio.run(capnp.run(run()))
    finally:
        server.terminate()
        server.wait()
    return np.array([e for e, _ in results]), np.array([f for _, f in results])


def dimer(binary, name, r):
    """Energy, force on atom 1 along x, and -dE/dr by central differences."""
    radii = np.concatenate([r, r + STEP, r - STEP])
    frames = [[[0.0, 0.0, 0.0], [d, 0.0, 0.0]] for d in radii]
    energy, forces = scan(binary, name, frames, [0, 0], 30.0 * np.eye(3))
    n = len(r)
    return energy[:n], forces[:n, 1, 0], -(energy[n : 2 * n] - energy[2 * n :]) / (2 * STEP)


def read_con(path):
    """Cell, atomic numbers and positions of an orthogonal Cu/H .con file."""
    lines = Path(path).read_text().splitlines()
    cell = np.diag([float(x) for x in lines[2].split()])
    numbers, positions, row = [], [], 9
    for count in map(int, lines[7].split()):
        z = {"Cu": 29, "H": 1}[lines[row].strip()]
        positions += [[float(x) for x in line.split()[:3]] for line in lines[row + 2 : row + 2 + count]]
        numbers += [z] * count
        row += 2 + count
    return cell, np.array(numbers), np.array(positions)


def barrier(relative, start, end):
    """Lowest energy level at which start and end share one connected region."""
    low, high = relative[start], relative.max()
    for _ in range(40):
        level = 0.5 * (low + high)
        labels, _ = ndimage.label(relative <= level)
        low, high = (low, level) if labels[start] == labels[end] else (level, high)
    return high


def dimer_figure(binary):
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex="col")
    for col, (name, r, fmax) in enumerate(
        [("LJ", np.linspace(0.95, 3.0, 200), 3.0), ("Morse", np.linspace(2.3, 7.0, 200), 1.0)]
    ):
        energy, force, slope = dimer(binary, name, r)
        i = np.argmin(energy)
        print(f"{name} dimer: {len(r)} distances from {r[0]:.2f} to {r[-1]:.2f} Angstrom")
        print(f"  lowest energy {energy[i]:.6f} eV at r = {r[i]:.4f} Angstrom")
        print(f"  largest |force - (-dE/dr)| = {np.max(np.abs(force - slope)):.2e} eV/Angstrom")
        top, bottom = axes[:, col]
        top.plot(r, energy)
        top.plot(r[i], energy[i], "o", color="C3", label=f"lowest: r = {r[i]:.3f}")
        top.axhline(0.0, color="0.6", lw=0.8)
        top.set(title=f"{name} dimer through potserv", ylabel="energy (eV)", ylim=(1.3 * energy[i], -1.5 * energy[i]))
        bottom.plot(r, force, label="server force on atom 1")
        bottom.plot(r[::8], slope[::8], "x", label="-dE/dr, finite difference")
        bottom.set(xlabel="distance r (Angstrom)", ylabel="force (eV/Angstrom)", ylim=(-fmax, fmax))
        top.legend(frameon=False)
        bottom.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(HERE / "dimer_curves.png", dpi=150)


def elbow_figure(binary):
    cell, numbers, positions = read_con(ROOT / "CppCore/rgpot/CuH2/tmp.con")
    copper, hydrogen = positions[numbers == 29], positions[numbers == 1]
    centre, top = hydrogen.mean(axis=0), copper[:, 2].max()
    print(f"fixture H2: bond {np.linalg.norm(hydrogen[1] - hydrogen[0]):.4f} Angstrom, height {centre[2] - top:.4f} Angstrom")

    bonds, heights = np.linspace(0.4, 3.0, 66), np.linspace(0.4, 3.5, 63)

    def frame(bond, height):
        mid, half = [centre[0], centre[1], top + height], [bond / 2, 0.0, 0.0]
        return np.vstack([copper, np.subtract(mid, half), np.add(mid, half)])

    frames = [frame(b, h) for h in heights for b in bonds] + [frame(b, 50.0) for b in bonds]
    energy, _ = scan(binary, "CuH2", frames, [29] * len(copper) + [1, 1], cell)
    far = energy[-len(bonds) :]
    relative = energy[: -len(bonds)].reshape(len(heights), len(bonds)) - far.min()

    molecule = (len(heights) - 1, int(np.argmin(relative[-1])))
    split = int(np.argmax(bonds >= 2.0))
    j, k = np.unravel_index(np.argmin(relative[:, split:]), relative[:, split:].shape)
    apart = (int(j), int(k) + split)
    ridge = barrier(relative, molecule, apart)
    print(f"CuH2 elbow: {relative.size} geometries of {len(copper)} Cu + 2 H")
    print(f"  top Cu layer z = {top:.4f} Angstrom, H2 centre x, y = {centre[0]:.4f}, {centre[1]:.4f}")
    print(f"  gas-phase H2, 50 Angstrom up: {far.min():.6f} eV at bond {bonds[np.argmin(far)]:.4f} Angstrom")
    for label, (j, k) in [("molecule", molecule), ("atoms apart", apart)]:
        print(f"  {label}: {relative[j, k]:+.4f} eV at bond {bonds[k]:.4f}, height {heights[j]:.4f} Angstrom")
    print(f"  lowest ridge between them on this grid: {ridge:+.4f} eV")

    fig, ax = plt.subplots(figsize=(6, 5))
    filled = ax.contourf(bonds, heights, np.clip(relative, 0.0, 2.0), levels=np.linspace(0.0, 2.0, 21))
    ax.contour(bonds, heights, relative, levels=[ridge], colors="w", linewidths=1.0)
    for (j, k), marker, label in [(molecule, "o", "molecule"), (apart, "s", "atoms apart")]:
        ax.plot(bonds[k], heights[j], marker, color="C3", mec="w", ms=9, label=label)
    ax.plot(np.linalg.norm(hydrogen[1] - hydrogen[0]), centre[2] - top, "x", color="w", ms=9, label="H2 in tmp.con")
    ax.set(xlabel="H-H bond length (Angstrom)", ylabel="H2 height above top Cu layer (Angstrom)")
    ax.set_title("H2 over Cu, CuH2 EAM through potserv")
    ax.legend(loc="upper right")
    fig.colorbar(filled, label="energy above slab + gas-phase H2 (eV), clipped at 2")
    fig.tight_layout()
    fig.savefig(HERE / "cuh2_elbow.png", dpi=150)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--server-bin", required=True, help="path to potserv")
    args = parser.parse_args()
    dimer_figure(args.server_bin)
    elbow_figure(args.server_bin)
