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

import argparse
import asyncio
import time

import capnp
from ase.build import bulk, molecule
from ase.calculators.lj import LennardJones

import Potentials_capnp


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


async def main(hosts):
    """Main coroutine to orchestrate sequential and parallel calculations."""
    server_addresses = [host.split(":") for host in hosts]

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

    # Molecules created with ase.build.molecule() have a zero cell by default.
    print("--- Ensuring all structures have a valid cell ---")
    for name, atoms in structures.items():
        atoms.set_cell([90, 90, 90])
        atoms.center()

    print(f"Generated {len(structures)} structures to calculate.\n")

    # 2. Run calculations sequentially for baseline timing and correctness check
    print(
        f"--- Running {len(structures)} calculations sequentially on {hosts[0]} for comparison ---"
    )
    ase_lj_calc = LennardJones()
    start_time_seq = time.monotonic()
    for name, atoms in structures.items():
        print(f"\nProcessing {name}...")
        atoms.calc = ase_lj_calc
        ref_energy = atoms.get_potential_energy()
        print(f"  Reference ASE LJ Energy: {ref_energy:.4f}")

        host, port = server_addresses[0]
        await run_single_calculation(host, int(port), atoms, name)
    end_time_seq = time.monotonic()
    total_time_seq = end_time_seq - start_time_seq
    print(f"\nSequential execution took: {total_time_seq:.4f} seconds\n")

    # 3. Run calculations in parallel
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

    # 4. Show the results
    print("--- Summary ---")
    print(f"Sequential time: {total_time_seq:.4f} s")
    print(f"Parallel time:   {total_time_para:.4f} s")
    if total_time_para > 0:
        speedup = total_time_seq / total_time_para
        print(f"Speed-up:        {speedup:.2f}x")


if __name__ == "__main__":
    args = parse_args()
    try:
        # This pattern correctly starts the event loop.
        # 1. We call main() which returns a coroutine object.
        # 2. We pass that single coroutine object to capnp.run().
        # 3. We pass the result of that to asyncio.run().
        main_coro = main(args.hosts)
        asyncio.run(capnp.run(main_coro))
    except KeyboardInterrupt:
        print("\nClient stopped by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
