#!/usr/bin/env python3

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pycapnp",
#   "ase",
#   "mpi4py",
# ]
# ///

import argparse
import asyncio
import time

import capnp
import matplotlib.pyplot as plt
import Potentials_capnp
from ase.build import bulk, molecule
from ase.calculators.lj import LennardJones
from mpi4py import MPI


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compares parallel local computation (MPI) with parallel RPC to a server pool."
    )
    parser.add_argument(
        "hosts",
        nargs="*",
        default=[],
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
    """A simple function to be run in a separate MPI process."""
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


def create_plots(mpi_time, rpc_time):
    """Generates a plot comparing the total execution time of the two parallel models."""
    fig, ax = plt.subplots(figsize=(10, 7))

    approaches = ["Parallel Local (mpi4py)", "Parallel RPC Pool"]
    times = [mpi_time, rpc_time]

    bars = ax.bar(approaches, times, color=["#1f77b4", "#ff7f0e"])

    ax.set_ylabel("Total Execution Time (seconds)", fontsize=12)
    ax.set_title("Performance Comparison: mpi4py vs. RPC Server Pool", fontsize=14)
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)

    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval,
            f"{yval:.4f} s",
            va="bottom",
            ha="center",
        )

    fig.tight_layout()
    plt.savefig("parallel_model_comparison_mpi.png")
    print("\nSaved parallel model comparison plot to 'parallel_model_comparison_mpi.png'")


# --- Main Orchestration ---


def main():
    """Main function to orchestrate and compare the two parallel benchmarks."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Rank 0 (Master) prepares and distributes tasks
    if rank == 0:
        args = parse_args()

        print("--- [Rank 0] Creating ASE structures for workload ---")
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
        print("--- [Rank 0] Ensuring all structures have a valid cell ---")
        for _, atoms in structures.items():
            atoms.set_cell([90, 90, 90])
            atoms.center()
        print(f"Generated {len(structures)} structures to calculate.\n")

        # Manually split tasks for each process. This avoids the numpy error.
        tasks = list(structures.items())
        # This creates a list of lists, e.g., [[task1, task2], [task3, task4], ...]
        chunked_tasks = [tasks[i::size] for i in range(size)]
    else:
        # Worker processes have no initial data
        args = None
        structures = None
        tasks = None
        chunked_tasks = None

    # --- Benchmark 1: Parallel Local Calculation (mpi4py) ---
    if rank == 0:
        print(
            f"--- [Rank 0] BENCHMARK 1: Distributing {len(tasks)} tasks to {size} MPI processes ---"
        )
        start_time_mpi = MPI.Wtime()

    # Each process receives its chunk of tasks from rank 0
    local_tasks = comm.scatter(chunked_tasks, root=0)

    # All processes (including rank 0) do their share of the work
    local_results = {
        name: run_local_lj_calculation(atoms) for name, atoms in local_tasks
    }

    # Gather results back to the master process
    all_results_list = comm.gather(local_results, root=0)

    if rank == 0:
        end_time_mpi = MPI.Wtime()
        total_time_mpi = end_time_mpi - start_time_mpi
        # Combine results from all processes into a single dictionary
        all_mpi_results = {}
        if all_results_list:
            for res_dict in all_results_list:
                all_mpi_results.update(res_dict)
        print(
            f"\nParallel Local (mpi4py) execution took: {total_time_mpi:.4f} seconds\n"
        )

        # --- Benchmark 2: Parallel RPC to Server Pool (only on Rank 0) ---
        if not args.hosts:
            print("--- SKIPPING BENCHMARK 2: No RPC server hosts provided. ---")
            total_time_rpc = 0
        else:
            print(
                f"--- [Rank 0] BENCHMARK 2: Running {len(tasks)} calculations concurrently against RPC Pool ---"
            )
            start_time_rpc = time.monotonic()
            main_coro = run_all_rpc_calculations_concurrently(args.hosts, structures)
            rpc_results = asyncio.run(capnp.run(main_coro))
            end_time_rpc = time.monotonic()
            total_time_rpc = end_time_rpc - start_time_rpc
            print(f"\nParallel RPC Pool execution took: {total_time_rpc:.4f} seconds\n")

        # --- Summary and Plotting (only on Rank 0) ---
        print("--- Summary ---")
        print(f"Parallel Local (mpi4py) Time: {total_time_mpi:.4f} s")
        if total_time_rpc > 0:
            print(f"Parallel RPC Pool Time:       {total_time_rpc:.4f} s")
            if total_time_rpc < total_time_mpi:
                speedup = total_time_mpi / total_time_rpc
                print(f"RPC Pool was {speedup:.2f}x faster.")
            else:
                slowdown = total_time_rpc / total_time_mpi
                print(f"RPC Pool was {slowdown:.2f}x slower.")

        create_plots(total_time_mpi, total_time_rpc)


if __name__ == "__main__":
    main()
