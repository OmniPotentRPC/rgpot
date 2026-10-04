`scripts/export_uma_aoti.py` records which export path produced a package
(`export_path`), documents that torch 2.13 with fairchem-core 2.23 always takes
the static `make_fx` fallback, and warns when that fallback runs without
`--molecular-box`.
