The OpenMP team activation threshold of calls without one of their own, such as
a neighbour-list build or `EMTPot`, can now be configured with Meson's
`team_min_atoms` option or the `RGPOT_TEAM_MIN_ATOMS` compile definition while
retaining 8192 atoms as the default; each classical potential keeps its own
counts as `kTeamFrom`.
