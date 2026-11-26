The documentation now covers every public potential, C and C++ header,
Rust item, Cap'n Proto field, Python name and command line program. Each
potential has a how-to with its functional form, parameter sources, units,
a runnable example and its literature citation. `llms.txt` and
`llms-full.txt` are generated from the sources by `scripts/gen_llms.py`,
and `doccheck` fails when a public item has no entry or those files are
stale.
