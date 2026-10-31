"""Send one geometry to a running potserv and print the reply.

    ./bbdir/CppCore/potserv 12345 LJ
    python docs/orgmode/tutorials/pes_scan/evaluate_once.py 12345
"""

import asyncio
import sys
from pathlib import Path

import capnp
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
schema = capnp.load(str(ROOT / "CppCore/rgpot/rpc/Potentials.capnp"))


async def connect(port):
    """Return a client for the Potential interface served on port."""
    for _ in range(100):
        try:
            stream = await capnp.AsyncIoStream.create_connection(host="localhost", port=port)
            client = capnp.TwoPartyClient(stream)
            return client, client.bootstrap().cast_as(schema.Potential)
        except OSError:
            await asyncio.sleep(0.05)
    raise ConnectionError(f"nothing is listening on port {port}")


async def calculate(potential, positions, numbers, box):
    """Send one geometry; return the energy and an (N, 3) force array."""
    request = schema.ForceInput.new_message()
    request.pos = np.ravel(positions).tolist()
    request.atmnrs = list(numbers)
    request.box = np.ravel(box).tolist()
    reply = await potential.calculate(request)
    return reply.result.energy, np.reshape(reply.result.forces, (-1, 3))


async def main(port):
    _client, potential = await connect(port)
    positions = [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]
    energy, forces = await calculate(potential, positions, [0, 0], 20.0 * np.eye(3))
    print(f"energy: {energy:.6f} eV")
    for i, f in enumerate(forces):
        print(f"force on atom {i}: [{f[0]:+.6f}, {f[1]:+.6f}, {f[2]:+.6f}] eV/Angstrom")


if __name__ == "__main__":
    asyncio.run(capnp.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 12345)))
