"""Run capnp compile and rename .c++ output to .cpp for MSVC compatibility.

Usage: python capnp_compile.py <capnp> <outdir> <src_prefix> <input>...

MSVC cl.exe does not recognise the .c++ extension that capnp emits.
This wrapper renames the generated sources so that all compilers can
consume them without extra flags. Every schema is compiled in one run, so
an input that imports another (Potentials.capnp imports common.capnp)
generates both.
"""
import os
import shutil
import subprocess
import sys

capnp, outdir, src_prefix, *schemas = sys.argv[1:]
if not schemas:
    raise SystemExit("usage: capnp_compile.py <capnp> <outdir> <src_prefix> <input>...")
subprocess.check_call([capnp, "compile", f"-oc++:{outdir}", f"--src-prefix={src_prefix}", *schemas])

for schema in schemas:
    base = os.path.splitext(os.path.basename(schema))[0]  # e.g. "Potentials"
    old = os.path.join(outdir, f"{base}.capnp.c++")
    new = os.path.join(outdir, f"{base}.capnp.cpp")
    if os.path.exists(old):
        shutil.move(old, new)
