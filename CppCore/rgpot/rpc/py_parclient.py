#!/usr/bin/env python3

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pycapnp",
#   "ase",
# ]
# ///

# Usage:
# 1. Start servers (e.g. in tmux)
# ./bbdir/CppCore/rgpot/rpc/potserv 12345 LJ
# ./bbdir/CppCore/rgpot/rpc/potserv 12346 LJ
# ./bbdir/CppCore/rgpot/rpc/potserv 12347 LJ
# 2. Run client
# uv run CppCore/rgpot/rpc/py_parclient.py \
#        localhost:12345 \
#        localhost:12346 \
#        localhost:12347

#!/usr/bin/env python3

import argparse
import asyncio
import time

import capnp
import matplotlib.pyplot as plt
import numpy as np
import Potentials_capnp
from ase.build import bulk, molecule
from ase.calculators.lj import LennardJones


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Connects to one or more Potential servers and runs calculations in parallel."
    )
    parser.add_argument(
        "hosts",
        nargs="+",
        help="List of server addresses in HOST:PORT format (e.g., localhost:12345 localhost:12346)",
    )
    parser.add_argument(
        "--sleep-ms",
        type=int,
        default=0,
        help="Milliseconds to sleep in the local ASE calculation to simulate a heavy workload. "
        "Set this to match the sleep duration in your C++ server for a fair comparison.",
    )
    return parser.parse_args()


def atoms_to_force_input(atoms):
    """Converts an ASE Atoms object to a Cap'n Proto ForceInput message."""
    force_input = Potentials_capnp.ForceInput.new_message()
    force_input.natm = len(atoms)

    positions = atoms.get_positions().flatten()
    pos_list = force_input.init("pos", len(positions))
    for i, pos in enumerate(positions):
        pos_list[i] = float(pos)

    atom_types = atoms.get_atomic_numbers()
    atom_list = force_input.init("atmnrs", len(atom_types))
    for i, atom_type in enumerate(atom_types):
        atom_list[i] = int(atom_type)

    box_matrix = atoms.get_cell().flatten()
    box_list = force_input.init("box", len(box_matrix))
    for i, val in enumerate(box_matrix):
        box_list[i] = float(val)

    return force_input


async def run_single_calculation(host, port, atoms_obj, structure_name):
    """Connects to a single server, runs one calculation, and returns the result."""
    try:
        connection = await capnp.AsyncIoStream.create_connection(host=host, port=port)
        client = capnp.TwoPartyClient(connection)
        calculator = client.bootstrap().cast_as(Potentials_capnp.Potential)

        force_input = atoms_to_force_input(atoms_obj)
        response = await calculator.calculate(force_input)

        energy = response.result.energy
        print(
            f"  => RPC Result from {host}:{port} for {structure_name}: Energy = {energy:.4f}"
        )
        return response.to_dict()
    except Exception as e:
        print(f"  => RPC Error for {structure_name} on {host}:{port}: {e}")
        return None


def create_timing_plot(results_data):
    """Generates a plot comparing the per-task execution times."""
    labels = list(results_data.keys())
    local_times = [r["local_time_ms"] for r in results_data.values()]
    rpc_times = [r["rpc_time_ms"] for r in results_data.values()]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 8))
    rects1 = ax.bar(
        x - width / 2,
        local_times,
        width,
        label=f"Local ASE Call (+{args.sleep_ms}ms sleep)",
    )
    rects2 = ax.bar(
        x + width / 2, rpc_times, width, label="C++ RPC Call (includes network)"
    )

    ax.set_ylabel("Time (milliseconds)", fontsize=12)
    ax.set_title("Per-Task Performance: Local ASE vs. C++ RPC", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)

    fig.tight_layout()
    plt.savefig("per_task_timing_comparison.png")
    print("\nSaved detailed timing comparison plot to 'per_task_timing_comparison.png'")


async def main(hosts, sleep_ms):
    """Main coroutine to orchestrate sequential and parallel calculations."""
    server_addresses = [host.split(":") for host in hosts]
    results_data = {}

    # 1. Create a workload using ASE
    print("--- Creating ASE structures for workload ---")
    structures = {
        "Cu_bulk": bulk("Cu", "fcc", a=3.6),
        "H2O_molecule": molecule("H2O"),
        "Au_bulk": bulk("Au", "fcc", a=4.07),
        "CH4_molecule": molecule("CH4"),
        "Si_bulk": bulk("Si", "diamond", a=5.43),
        "O2_molecule": molecule("O2"),
        "NaCl_crystal": bulk("NaCl", "rocksalt", a=5.64),
        "C6H6_benzene": molecule("C6H6"),
    }
    print("--- Ensuring all structures have a valid cell ---")
    for name, atoms in structures.items():
        atoms.set_cell([90, 90, 90])
        atoms.center()

    print(f"Generated {len(structures)} structures to calculate.\n")

    # 2. Run sequentially for detailed timing and correctness check
    print(
        f"--- Running {len(structures)} calculations sequentially on {hosts[0]} for detailed timing ---"
    )
    ase_lj_calc = LennardJones()
    total_sequential_rpc_time = 0
    for name, atoms in structures.items():
        print(f"\nProcessing {name}...")
        atoms.calc = ase_lj_calc

        # Time the local ASE call including the simulated workload
        t0 = time.monotonic()
        ref_energy = atoms.get_potential_energy()
        t1 = time.monotonic()
        local_time = t1 - t0

        # Time the remote RPC call
        host, port = server_addresses[0]
        t2 = time.monotonic()
        rpc_result = await run_single_calculation(host, int(port), atoms, name)
        t3 = time.monotonic()
        rpc_time = t3 - t2
        total_sequential_rpc_time += rpc_time

        print(f"  Local ASE call took: {local_time * 1000:.2f} ms")
        print(f"  Remote RPC call took: {rpc_time * 1000:.2f} ms")

        results_data[name] = {
            "local_time_ms": local_time * 1000,
            "rpc_time_ms": rpc_time * 1000,
        }
    print(
        f"\nTotal sequential RPC execution took: {total_sequential_rpc_time:.4f} seconds\n"
    )

    # 3. Run calculations in parallel for speed-up measurement
    print(
        f"--- Running {len(structures)} calculations in parallel across {len(hosts)} servers ---"
    )
    start_time_para = time.monotonic()
    tasks = []
    for i, (name, atoms) in enumerate(structures.items()):
        host, port = server_addresses[i % len(server_addresses)]
        task = asyncio.create_task(run_single_calculation(host, int(port), atoms, name))
        tasks.append(task)
    await asyncio.gather(*tasks)
    end_time_para = time.monotonic()
    total_time_para = end_time_para - start_time_para
    print(f"\nParallel execution took: {total_time_para:.4f} seconds\n")

    # 4. Show the results and create plots
    print("--- Summary ---")
    print(f"Total Sequential RPC time: {total_sequential_rpc_time:.4f} s")
    print(f"Parallel time:             {total_time_para:.4f} s")
    if total_time_para > 0:
        speedup = total_sequential_rpc_time / total_time_para
        print(f"Speed-up:                  {speedup:.2f}x")

    create_timing_plot(results_data)


if __name__ == "__main__":
    args = parse_args()
    try:
        main_coro = main(args.hosts, args.sleep_ms)
        asyncio.run(capnp.run(main_coro))
    except KeyboardInterrupt:
        print("\nClient stopped by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
