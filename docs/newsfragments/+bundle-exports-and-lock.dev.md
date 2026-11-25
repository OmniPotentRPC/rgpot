The version-drift check fails when `rgpot-bundle/Cargo.lock` has lost the cog
markers around the `rgpot-core` version. The ABI stamp page records that a
whole-archive bundle exports 127 `rgpot_*` symbols, and `rgpot_so_exports` pins
that count next to the 73 of a normal link. Of those 73, two are
`rgpot_builtin_muller_brown_new` and `rgpot_muller_brown_config_default`, and
one is `rgpot_lammpc_potential`.
