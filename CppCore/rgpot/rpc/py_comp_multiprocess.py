#!/usr/bin/env python3


# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pycapnp",
#   "ase",
# ]
# ///

# Usage (needs the environment setup):
# 1. Start servers (e.g. in tmux)
# ./bbdir/CppCore/rgpot/rpc/potserv 12345 LJ
# ./bbdir/CppCore/rgpot/rpc/potserv 12346 LJ
# ./bbdir/CppCore/rgpot/rpc/potserv 12347 LJ
# 2. Run client
# uv run CppCore/rgpot/rpc/py_parclient.py \
#        localhost:12345 \
#        localhost:12346 \
#        localhost:12347

import argparse
import asyncio
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import capnp
import matplotlib.pyplot as plt
import Potentials_capnp
from ase.build import bulk, molecule
from ase.calculators.lj import LennardJones


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compares parallel local computation (MPI-style) with parallel RPC to a server pool."
    )
    parser.add_argument(
        "hosts",
        nargs="+",
        help="List of server addresses in HOST:PORT format for the RPC pool.",
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


# --- Functions for the "Traditional" MPI-style Parallelism ---


def run_local_lj_calculation(atoms):
    """A simple function to be run in a separate process."""
    calc = LennardJones()
    atoms.calc = calc
    return atoms.get_potential_energy()


# --- Functions for the RPC/Service Model Parallelism ---


async def run_single_rpc_calculation_async(host, port, atoms_obj, structure_name):
    """The core async function to run one RPC calculation."""
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
        return structure_name, energy
    except Exception as e:
        print(f"  => RPC Error for {structure_name} on {host}:{port}: {e}")
        return structure_name, None


async def run_all_rpc_calculations_concurrently(hosts, structures):
    """Main coroutine for the RPC benchmark."""
    server_addresses = [host.split(":") for host in hosts]
    tasks = []
    for i, (name, atoms) in enumerate(structures.items()):
        host, port = server_addresses[i % len(server_addresses)]
        task = asyncio.create_task(
            run_single_rpc_calculation_async(host, int(port), atoms, name)
        )
        tasks.append(task)

    results = await asyncio.gather(*tasks)
    return dict(results)


# --- Plotting ---


def create_plots(labels, mpi_times, rpc_times):
    """Generates a plot comparing the total execution time of the two parallel models."""
    fig, ax = plt.subplots(figsize=(10, 7))

    approaches = ["ProcessPool", "Parallel RPC Pool"]
    times = [sum(mpi_times), sum(rpc_times)]

    ax.bar(approaches, times, color=["#1f77b4", "#ff7f0e"])

    ax.set_ylabel("Total Execution Time (seconds)", fontsize=12)
    ax.set_title("Performance Comparison: MPI-style vs. RPC Server Pool", fontsize=14)
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)

    for i, v in enumerate(times):
        ax.text(
            i,
            v + 0.01 * max(times),
            f"{v:.4f} s",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    fig.tight_layout()
    plt.savefig("parallel_model_comparison_processpool.png")
    print("\nSaved parallel model comparison plot to 'parallel_model_comparison_processpool.png'")


# --- Main Orchestration ---


def main():
    """Main synchronous function to orchestrate and compare the two parallel benchmarks."""
    args = parse_args()

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
    for _, atoms in structures.items():
        atoms.set_cell([90, 90, 90])
        atoms.center()
    print(f"Generated {len(structures)} structures to calculate.\n")

    # --- Benchmark 1: Parallel Local Calculation (MPI-style) ---
    print(
        f"--- BENCHMARK 1: Running {len(structures)} local calculations in a Process Pool ---"
    )
    start_time_mpi = time.monotonic()
    local_results = {}
    with ProcessPoolExecutor(max_workers=4) as executor:
        future_to_name = {
            executor.submit(run_local_lj_calculation, atoms): name
            for name, atoms in structures.items()
        }
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                energy = future.result()
                local_results[name] = energy
                print(f"  Local Result for {name}: Energy = {energy:.4f}")
            except Exception as exc:
                print(f"  Local task for {name} generated an exception: {exc}")
    end_time_mpi = time.monotonic()
    total_time_mpi = end_time_mpi - start_time_mpi
    print(
        f"\nParallel Local (MPI-style) execution took: {total_time_mpi:.4f} seconds\n"
    )

    # --- Benchmark 2: Parallel RPC to Server Pool ---
    print(
        f"--- BENCHMARK 2: Running {len(structures)} calculations concurrently against RPC Pool ---"
    )
    start_time_rpc = time.monotonic()
    # This pattern correctly starts the event loop for the RPC calls.
    main_coro = run_all_rpc_calculations_concurrently(args.hosts, structures)
    rpc_results = asyncio.run(capnp.run(main_coro))
    end_time_rpc = time.monotonic()
    total_time_rpc = end_time_rpc - start_time_rpc
    print(f"\nParallel RPC Pool execution took: {total_time_rpc:.4f} seconds\n")

    # --- Summary and Plotting ---
    print("--- Summary ---")
    print(f"Parallel Local (ProcessPoolExecutor) Time: {total_time_mpi:.4f} s")
    print(f"Parallel RPC Pool Time:          {total_time_rpc:.4f} s")
    if total_time_rpc > 0:
        if total_time_rpc < total_time_mpi:
            speedup = total_time_mpi / total_time_rpc
            print(f"RPC Pool was {speedup:.2f}x faster.")
        else:
            slowdown = total_time_rpc / total_time_mpi
            print(f"RPC Pool was {slowdown:.2f}x slower.")

    create_plots(["ProcessPoolExecutor", "RPC Pool"], [total_time_mpi], [total_time_rpc])


if __name__ == "__main__":
    main()
