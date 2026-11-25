<!-- Copyright 2023--present rgpot developers -->

The C ABI minor of `rgpot-core` is now 14 in `RGPOT_ABI_VERSION_MINOR` alongside
the `linkcell::SkinList` Verlet-skin backing and bundled `Cu_u3.eam` constructors
for `EamTablePotential`. A caller built against minor 0 to 13 still matches; a
caller built against minor 14 is refused by an older library.
