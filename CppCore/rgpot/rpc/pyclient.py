#!/usr/bin/env python3

import argparse
import asyncio
import capnp
import numpy as np

import Potentials_capnp


def parse_args():
    parser = argparse.ArgumentParser(
        usage="Connects to the Calculator server at the given address and does some RPCs"
    )
    parser.add_argument("host", help="HOST:PORT")

    return parser.parse_args()


async def main(connection):
    client = capnp.TwoPartyClient(connection)
    calculator = client.bootstrap().cast_as(Potentials_capnp.CuH2Pot)

    print("Connection made ", end="")

    # Create the atomic positions, atom types, and box matrix
    positions = [
        0.63940268750835,
        0.90484742551374,
        6.97516498544584,  # Cu
        3.19652040936288,
        0.90417430354811,
        6.97547796369474,  # Cu
        8.98363230369760,
        9.94703496017833,
        7.83556854923689,  # H
        7.64080177576300,
        9.94703114803832,
        7.83556986121272,  # H
    ]

    atom_types = [29, 29, 1, 1]  # Atomic types for Cu and H

    box_matrix = [15.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 30.0]

    # Create the ForceInput message
    force_input = Potentials_capnp.ForceInput.new_message()
    force_input.natm = len(atom_types)  # Number of atoms

    # Set positions
    pos_list = force_input.init("pos", len(positions))
    for idx, pos in enumerate(positions):
        pos_list[idx] = pos

    # Set atom types
    atom_list = force_input.init("atmnrs", len(atom_types))
    for idx, atom in enumerate(atom_types):
        atom_list[idx] = atom

    # Set the box matrix
    box_list = force_input.init("box", len(box_matrix))
    for idx, val in enumerate(box_matrix):
        box_list[idx] = val

    # Call the CuH2Pot.calculate method with ForceInput
    response = calculator.calculate(force_input)

    # Await and print the result
    result = await response
    result_dict = result.to_dict()
    print("Energy:", result_dict["result"]["energy"])
    print("Forces:", result_dict["result"]["forces"])


async def cmd_main(host):
    host, port = host.split(":")
    await main(await capnp.AsyncIoStream.create_connection(host=host, port=port))


if __name__ == "__main__":
    asyncio.run(capnp.run(cmd_main(parse_args().host)))
