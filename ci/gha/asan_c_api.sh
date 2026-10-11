#!/bin/sh
# AddressSanitizer and UBSan on the C ABI stubs.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
CC=${CC:-gcc}
CFLAGS="-std=c11 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer"
out=$(mktemp -d)
"$CC" $CFLAGS \
  -I CppCore/rgpot/CPMDPot \
  CppCore/rgpot/CPMDPot/cpmd_c_abi_stub.c \
  CppCore/tests/test_cpmd_abi_main.c \
  -o "$out/cpmd"
"$out/cpmd"
"$CC" $CFLAGS \
  -I CppCore/rgpot/NWChemPot \
  CppCore/rgpot/NWChemPot/nwchem_c_abi_stub.c \
  CppCore/tests/test_nwchem_abi_main.c \
  -o "$out/nwchem"
"$out/nwchem"
"$CC" $CFLAGS \
  -I CppCore/rgpot/MOPACPot \
  CppCore/rgpot/MOPACPot/mopac_c_abi_stub.c \
  CppCore/tests/test_mopac_abi_main.c \
  -o "$out/mopac"
"$out/mopac"
rm -rf "$out"
echo "asan c api ok"
