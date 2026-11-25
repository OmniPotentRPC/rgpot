ORCAPot runs ORCA energies and gradients through liborcac, the standalone
orcac library, opened at run time from ORCAC_LIBRARY or RGPOT_ORCAC_ENGINE and
checked against the vendored orcac.h, frozen at orcac ABI 1. A library with
another ABI version or struct sizes below the header's minimums is refused
before any run. It is exposed through the orca arm of PotentialConfig with
rgconf's ORCAParams on potserv ORCA, the liborca_engine plugin, rgpot.ORCAPot
and the ASE kind orca, which accepts the padded cluster box. As in
ORCALegacyPot, each call starts from ORCA's own guess unless guess_policy is
MoRead (restart in ORCAParams), which leaves the result cache unused, a box that would wrap the atoms is refused, and the orca file
first resolved is pinned, so a call after it changes throws.
