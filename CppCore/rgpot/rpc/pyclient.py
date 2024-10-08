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
    # Create a request to call the calculate method
    positions = [
        {"x": 0.63940268750835, "y": 0.90484742551374, "z": 6.97516498544584},  # Cu
        {"x": 3.19652040936288, "y": 0.90417430354811, "z": 6.97547796369474},  # Cu
        {"x": 8.98363230369760, "y": 9.94703496017833, "z": 7.83556854923689},  # H
        {"x": 7.64080177576300, "y": 9.94703114803832, "z": 7.83556986121272},  # H
    ]

    atom_types = [29, 29, 1, 1]  # Atomic types for Cu and H
    box_matrix = [
        {"x": 15.0, "y": 0.0, "z": 0.0},
        {"x": 0.0, "y": 20.0, "z": 0.0},
        {"x": 0.0, "y": 0.0, "z": 30.0},
    ]
    # Prepare the request
    pos = Potentials_capnp.AtomMatrixRPC.new_message()
    pos.init("positions", 4)
    for idx, item in enumerate(positions):
        pos_ = Potentials_capnp.Position.new_message()
        pos_.x = positions[idx]["x"]
        pos_.y = positions[idx]["y"]
        pos_.z = positions[idx]["z"]
        pos.positions[idx] = pos_

    # Prepare the request for AtomTypes
    atom_types_msg = Potentials_capnp.AtomTypes.new_message()
    atom_types_list = atom_types_msg.init("atomTypes", len(atom_types))
    for idx, atom in enumerate(atom_types):
        atom_types_list[idx] = atom

    # Prepare the request for BoxMatrix
    box_matrix_msg = Potentials_capnp.BoxMatrix.new_message()
    box_matrix_list = box_matrix_msg.init("box", len(box_matrix))
    for idx, box in enumerate(box_matrix):
        box_matrix_list[idx].x = box["x"]
        box_matrix_list[idx].y = box["y"]
        box_matrix_list[idx].z = box["z"]

    # Call the CuH2Pot.calculate method
    response = calculator.calculate(pos, atom_types_msg, box_matrix_msg)

    # Extract the result from the response
    result = await response
    print(result.to_dict())


async def cmd_main(host):
    host, port = host.split(":")
    await main(await capnp.AsyncIoStream.create_connection(host=host, port=port))


if __name__ == "__main__":
    asyncio.run(capnp.run(cmd_main(parse_args().host)))
