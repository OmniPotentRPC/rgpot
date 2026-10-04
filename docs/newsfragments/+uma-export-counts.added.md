`scripts/export_uma_aoti.py` embeds `natoms`, per-element `counts` and the
`model`, `torch_version` and `fairchem_version` that produced the package.
UmaPot checks `natoms` and `counts` against every input, so a package for
C2H2 refuses C2H4 although both share `z_set` `[1, 6]`.
