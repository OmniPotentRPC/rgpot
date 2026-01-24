#!/usr/bin/env python3
import asyncio
import argparse
import subprocess
import sys
import os
import capnp
import numpy as np

# Load schema relative to this script
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SCHEMA_PATH = os.path.join(SCRIPT_DIR, "../CppCore/rgpot/rpc/Potentials.capnp")
pot_capnp = capnp.load(SCHEMA_PATH)


async def run_client(port):
    # Retry connection a few times to allow server startup
    for _ in range(10):
        try:
            # Create connection (this requires the KJ loop to be active)
            connection = await capnp.AsyncIoStream.create_connection(
                host="localhost", port=port
            )
            break
        except OSError:
            await asyncio.sleep(0.5)
    else:
        print("Failed to connect to server.")
        return False

    client = capnp.TwoPartyClient(connection)
    pot = client.bootstrap().cast_as(pot_capnp.Potential)

    # Test Case: CuH2 minimal system
    fip = pot_capnp.ForceInput.new_message()
    fip.natm = 2

    # Cu at 0,0,0 and H at 1.5, 0, 0 (Approx distance)
    # Using semantic types from your "Pro" schema if you applied it,
    # but based on your recent "simple" checks, we'll stick to the flat list
    # unless you updated the schema.
    # Assuming you kept the struct/flat hybrid from previous steps:
    pos_data = [0.0, 0.0, 0.0, 1.5, 0.0, 0.0]
    fip.init("pos", len(pos_data))
    for i, p in enumerate(pos_data):
        fip.pos[i] = p

    fip.init("atmnrs", 2)
    fip.atmnrs[0] = 29  # Cu
    fip.atmnrs[1] = 1  # H

    box_data = [10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0]
    fip.init("box", len(box_data))
    for i, b in enumerate(box_data):
        fip.box[i] = b

    print("Sending calculation request...")
    # This await requires the KJ loop
    result = await pot.calculate(fip)

    print(f"Received Energy: {result.result.energy}")
    print(f"Received Forces: {list(result.result.forces)}")
    assert result.result.energy == -0.67880756881223303
    assert result.result.forces[0] == -7.556524918281001
    assert result.result.forces[3] == 7.556524918281001

    # Basic physical sanity check
    if result.result.energy == 0.0 or np.isnan(result.result.energy):
        print("Error: Energy is zero or NaN")
        return False

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server-bin", required=True, help="Path to potserv executable"
    )
    parser.add_argument("--port", type=int, default=12345)
    args = parser.parse_args()

    # 1. Start Server
    print(f"Starting server: {args.server_bin} {args.port} CuH2")
    server_proc = subprocess.Popen(
        [args.server_bin, str(args.port), "CuH2"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        #  Run Client
        success = asyncio.run(capnp.run(run_client(args.port)))
    except Exception as e:
        print(f"Exception during test: {e}")
        # Print server stderr to help debug
        print("Server stderr:")
        print(server_proc.stderr.read().decode())
        success = False
    finally:
        # Cleanup
        print("Killing server...")
        server_proc.kill()
        server_proc.wait()

    if not success:
        sys.exit(1)
    print("Integration test passed.")


if __name__ == "__main__":
    main()
