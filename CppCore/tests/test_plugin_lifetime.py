#!/usr/bin/env python3
"""Check plugin callbacks remain live through a caller's global teardown."""
import subprocess
import sys

result = subprocess.run(sys.argv[1:], capture_output=True, text=True, timeout=20)
output = result.stdout + result.stderr
print(output, end="")
if result.returncode:
    raise RuntimeError(f"plugin lifetime driver returned {result.returncode}")
markers = ["lifetime fixture evaluated", "plugin instance destroyed", "owner destructors complete"]
positions = [output.find(marker) for marker in markers]
if any(position < 0 for position in positions) or positions != sorted(positions):
    raise RuntimeError("plugin lifetime callbacks did not finish in ownership order")
