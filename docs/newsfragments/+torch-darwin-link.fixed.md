The metatomic torch dependency links on macOS. Its link arguments carried
`-rpath-link` and `--as-needed`, which are GNU ld only: Apple's linker
rejects them outright, so any Darwin build with `-Dwith_metatomic=true`
died with "unknown options" while producing `librgpot.3.dylib`. Those
flags now apply on GNU linkers only; Apple's resolves the dylibs from the
library path and their install names.
