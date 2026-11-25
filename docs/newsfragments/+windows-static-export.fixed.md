A static Windows `librgpot` now defines `RGPOT_STATIC` for in-tree targets.
Targets in the same CMake build no longer compile its headers as dllimport.
`scripts/abi_symbols.py` reads a PE export table with no names as empty.
It used to fail on the zero name table address of such a table.
